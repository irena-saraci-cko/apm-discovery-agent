import argparse
import chromadb
import os
from dotenv import load_dotenv
import google.auth

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.readers.confluence import ConfluenceReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding

# Import the refactored web scraping functions from our other script
from create_knowledge_base import fetch_sitemap_urls, crawl_and_scrape

# Define the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Check if the file exists before trying to load it
if os.path.exists(dotenv_path):
    print(f"Found .env file at: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}")


def load_confluence_documents(base_url, space_key):
    """Loads all documents from a specified Confluence space."""
    print(f"\n--- Loading Documents from Confluence space: {space_key} ---")
    
    # Explicitly load credentials from environment variables
    username = os.getenv("CONFLUENCE_USERNAME")
    api_key = os.getenv("CONFLUENCE_API_KEY")

    if not username or not api_key:
        print("❌ Error: CONFLUENCE_USERNAME and CONFLUENCE_API_KEY must be set in your .env file.")
        return []

    try:
        reader = ConfluenceReader(
            base_url=base_url,
            user_name=username,
            api_token=api_key
        )
        documents = reader.load_data(space_key=space_key, include_attachments=False)
        print(f"✅ Found {len(documents)} documents in Confluence.")
        return documents
    except Exception as e:
        print(f"❌ Failed to load from Confluence: {e}")
        return []

def load_api_reference_documents(url):
    """Loads documents from the API reference website."""
    print(f"\n--- Loading Documents from API Reference: {url} ---")
    try:
        sitemap_urls = fetch_sitemap_urls(url)
        scrape_urls = sitemap_urls or [url]
        documents = crawl_and_scrape(scrape_urls, recursive=not sitemap_urls)
        return documents
    except Exception as e:
        print(f"❌ Failed to load from API Reference: {e}")
        return []

def build_core_index(documents: list[Document]):
    """Builds and saves the 'core_knowledge' index in ChromaDB."""
    if not documents:
        print("No documents were loaded, skipping index creation.")
        return

    print("\n--- Building Core Knowledge Base Index ---")
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "core_knowledge"
    
    # Check if the collection exists before trying to delete it
    existing_collections = [c.name for c in db.list_collections()]
    if collection_name in existing_collections:
        print(f"Deleting old '{collection_name}' collection...")
        db.delete_collection(name=collection_name)
        print("✅ Old collection deleted.")

    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Initializing embedding model...")
    credentials, _ = google.auth.default()
    embed_model = VertexTextEmbedding(
        model_name="text-embedding-004", credentials=credentials
    )
    print("✅ Embedding model initialized.")

    print("Creating the index...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"🎉 Success! Core knowledge base '{collection_name}' has been created.")


def main():
    """Main function to build the core knowledge base."""
    # load_dotenv() # Load environment variables from .env file

    parser = argparse.ArgumentParser(
        description="Create the core knowledge base from Confluence and API references."
    )
    parser.add_argument("--confluence-url", required=True, help="Base URL of your Confluence instance.")
    parser.add_argument("--confluence-space", required=True, help="The Confluence space key to ingest.")
    parser.add_argument("--api-ref-url", default="https://api-reference.checkout.com/", help="URL for the API reference documentation.")
    args = parser.parse_args()

    # Load documents from all sources
    all_documents = []
    all_documents.extend(load_confluence_documents(args.confluence_url, args.confluence_space))
    all_documents.extend(load_api_reference_documents(args.api_ref_url))

    # Build and save the final index
    build_core_index(all_documents)

if __name__ == "__main__":
    main()
