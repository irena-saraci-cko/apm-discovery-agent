import argparse
import chromadb
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import google.auth
from google.cloud import translate_v2 as translate
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding

# --- Global Configuration ---
# A set of patterns to ignore during web crawling to avoid irrelevant links.
IGNORE_PATTERNS = [
    "/login", "/signup", "/edit", "cdn-cgi", "?", "#", ".pdf", ".zip", ".jpg", ".png"
]

# We initialize the translator client only when it's first needed.
translate_client = None
# --- End Global Configuration ---


def fetch_sitemap_urls(base_url: str) -> list[str]:
    """
    Fetches and parses a sitemap.xml to extract all URLs.

    This is the preferred method for URL discovery as it's efficient and
    respects the website's own declared structure.

    Args:
        base_url: The base URL of the website (e.g., https://example.com).

    Returns:
        A list of URLs found in the sitemap, or an empty list if not found.
    """
    sitemap_url = urljoin(base_url, "sitemap.xml")
    urls = []
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall("ns:url/ns:loc", namespace):
            urls.append(loc.text)

        if urls:
            print(f"✅ Found {len(urls)} URLs in sitemap.xml")
            return urls
    except (requests.RequestException, ET.ParseError) as e:
        print(f"Could not fetch or parse sitemap.xml: {e}")

    return []


def translate_text(text: str, target_language: str) -> str:
    """
    Translates a given text to the target language using Google Cloud Translate.

    Initializes the client on the first call to avoid unnecessary setup.
    If translation fails, it returns the original text.

    Args:
        text: The text to translate.
        target_language: The ISO 639-1 code for the target language (e.g., "en").

    Returns:
        The translated text, or the original text if translation fails.
    """
    global translate_client

    if translate_client is None:
        try:
            print("Initializing Google Translate client...")
            translate_client = translate.Client()
            print("✅ Translate client initialized.")
        except Exception as e:
            print(f"Could not initialize Google Translate client: {e}")
            print("Translation will be skipped.")
            return text  # Return original text if client fails to init

    try:
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        print(f"Could not translate text: {e}")
        return text  # Return original text on failure


def process_url(url: str, translate_to: str = None) -> Document | None:
    """
    Scrapes a single URL, extracts its text content, and optionally translates it.

    Args:
        url: The URL to scrape.
        translate_to: The target language for translation. If None, no translation.

    Returns:
        A LlamaIndex Document object, or None if scraping fails.
    """
    print(f"Scraping: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        page_text = soup.get_text()
        if translate_to:
            print(f"Translating content to '{translate_to}'...")
            page_text = translate_text(page_text, translate_to)
            print("✅ Translation complete.")

        return Document(text=page_text, extra_info={"url": url})
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None


def crawl_and_scrape(
    urls: list[str], recursive: bool = False, translate_to: str = None
) -> list[Document]:
    """
    Scrapes a list of URLs, optionally crawling recursively and translating content.

    Args:
        urls: A list of starting URLs.
        recursive: If True, recursively follows links on the same domain.
        translate_to: The target language for translation.

    Returns:
        A list of LlamaIndex Document objects.
    """
    visited = set()
    documents = []
    queue = list(urls)
    base_domain = urlparse(urls[0]).netloc

    print("Starting scrape...")
    while queue:
        current_url = queue.pop(0)

        if current_url in visited or any(pattern in current_url for pattern in IGNORE_PATTERNS):
            continue

        if urlparse(current_url).netloc != base_domain:
            continue

        visited.add(current_url)
        doc = process_url(current_url, translate_to)
        if doc:
            documents.append(doc)

        if recursive:
            try:
                response = requests.get(current_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a", href=True):
                    absolute_link = urljoin(current_url, link["href"])
                    if absolute_link not in visited:
                        queue.append(absolute_link)
            except requests.RequestException as e:
                print(f"Could not fetch links from {current_url}: {e}")

    print(f"✅ Scrape complete. Found {len(documents)} pages.")
    return documents


def load_documents_from_sources(args) -> list[Document]:
    """
    Loads documents from web URLs or local PDFs based on provided arguments.
    """
    documents = []
    # 1. Load from URLs if provided
    if args.urls:
        sitemap_urls = fetch_sitemap_urls(args.urls[0])
        scrape_urls = sitemap_urls or args.urls
        documents.extend(
            crawl_and_scrape(scrape_urls, args.recursive, args.translate_to)
        )

    # 2. Load from PDFs if provided
    if args.pdfs:
        print(f"Reading {len(args.pdfs)} PDF file(s) using UnstructuredReader...")
        pdf_reader = UnstructuredReader()
        pdf_documents = pdf_reader.load_data(
            file=args.pdfs[0], unstructured_kwargs={"strategy": "hi_res"}
        )
        documents.extend(pdf_documents)
        print("✅ PDFs read successfully.")
    
    return documents


def build_and_save_index(name: str, documents: list[Document], overwrite: bool = False):
    """
    Builds a vector index from the documents and saves it to ChromaDB.
    """
    print("\nSetting up ChromaDB vector store...")
    db = chromadb.PersistentClient(path="./chroma_db")

    collection_name = f"{name.lower()}_docs"
    if overwrite:
        print(f"Overwrite flag is set. Deleting collection '{collection_name}' if it exists...")
        db.delete_collection(name=collection_name)
        print(f"✅ Collection '{collection_name}' deleted.")

    chroma_collection = db.get_or_create_collection(collection_name)
    print(f"✅ Using collection: '{collection_name}'")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Initializing the embedding model...")
    credentials, _ = google.auth.default()
    embed_model = VertexTextEmbedding(
        model_name="text-embedding-004", credentials=credentials
    )
    print("✅ Embedding model initialized.")

    print("Creating the index. This may take a few minutes...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"🎉 Success! Knowledge base for '{name}' has been created.")


def main():
    """Main function to parse arguments and run the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Create a knowledge base for a new payment method.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--name", required=True, help="The unique name for the knowledge base (e.g., 'alma')."
    )
    parser.add_argument(
        "--urls", nargs="*", default=[], help="List of website URLs to scrape."
    )
    parser.add_argument(
        "--pdfs", nargs="*", default=[], help="List of local PDF file paths to read."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively scrape all links from the provided URLs."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete any existing knowledge base with the same name.",
    )
    parser.add_argument(
        "--translate-to",
        type=str,
        default=None,
        help="Optional ISO 639-1 language code to translate content to (e.g., 'en').",
    )
    args = parser.parse_args()

    print(f"--- Starting Ingestion for: {args.name} ---")

    documents = load_documents_from_sources(args)

    if not documents:
        print("\nError: No documents were loaded. Please check your sources.")
        return

    build_and_save_index(args.name, documents, args.overwrite)


if __name__ == "__main__":
    main()