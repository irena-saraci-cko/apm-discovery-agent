# APM Discovery Agent - Phase 2: Knowledge Base Ingestion

This project contains the scripts for building and verifying the knowledge core for an AI agent designed to automate the technical discovery of new payment methods.

This phase focuses on creating a robust and reusable data ingestion pipeline that can process various types of source documentation (websites, PDFs) and store them in a local vector database (`ChromaDB`).

## Core Features

-   **Multi-Source Ingestion:** Process data from web URLs and local PDF files.
-   **Intelligent Web Scraping:** Automatically uses sitemaps or recursive crawling with smart filters to gather web content.
-   **On-the-Fly Translation:** Can translate scraped web content to a target language (e.g., "en") using the Google Cloud Translation API.
-   **Advanced PDF Parsing:** Utilizes the `unstructured` library with a high-resolution OCR strategy (`hi_res`) to extract text and table data from complex, image-based, or scanned PDFs.
-   **Vectorization & Storage:** Uses Google's Vertex AI embedding models to create vector representations of the text and stores them in a local `ChromaDB` instance.

## How It Works

This section provides a deeper look into the technical processes behind the knowledge base creation.

### 1. Data Ingestion

The first step is to gather raw textual data from the provided sources.

#### Website Scraping

The scraper is designed to be both efficient and intelligent to avoid ingesting irrelevant content.

-   **Sitemap-First Strategy:** The script first attempts to find and parse a `sitemap.xml` file from the target domain. This is the preferred method as it provides a structured list of all important URLs directly from the website owner, ensuring comprehensive coverage without crawling unnecessary pages.
-   **Recursive Crawling:** If a sitemap is not found or the `--recursive` flag is used, the scraper will recursively crawl the website. It starts with the initial URLs, extracts all links, and adds same-domain URLs to a queue for scraping.
-   **Intelligent Filtering:** To keep the knowledge base clean and relevant, the scraper applies a set of ignore patterns to filter out URLs that are typically not useful for documentation. This includes login pages, links with query parameters (`?`), page fragments (`#`), and direct links to files (`.zip`, `.css`, etc.).
-   **Content Extraction:** For each valid page, the script uses `BeautifulSoup` to parse the HTML and extracts the core text content, stripping away tags, scripts, and styles.

#### PDF Processing

The system is equipped to handle complex PDFs that go beyond simple text, such as scanned documents, spec sheets with tables, or guides with diagrams.

-   **Unstructured Reader:** Instead of a basic text extractor, the pipeline uses the `UnstructuredReader` from the `llama-index-readers-file` package. This powerful library is capable of parsing complex layouts.
-   **High-Resolution OCR Strategy:** The reader is configured with the `strategy="hi_res"` (high resolution). This strategy instructs `unstructured` to use Optical Character Recognition (OCR) via the Tesseract engine when it encounters images or non-selectable text. This is crucial for accurately extracting text from tables, diagrams, and scanned pages within a PDF.
-   **Dependencies:** This advanced capability relies on system-level libraries (`poppler` for rendering and `tesseract` for OCR) and specific NLTK data packages for natural language processing tasks.

### 2. Data Augmentation

#### On-the-Fly Translation

When a target language is specified with the `--translate-to` argument, the pipeline can translate web content before ingestion.

-   **Process:** After fetching the raw HTML of a page, the entire content is sent to the Google Cloud Translation API.
-   **Benefit:** The translated HTML is then used for text extraction. This approach helps preserve the context and structure of the document during the translation process.

### 3. Vectorization and Storage

Once clean text is extracted and augmented, it is converted into a format that machine learning models can understand.

#### Text Embedding

-   **What are Embeddings?** An embedding is a numerical representation (a vector) of text. These vectors capture the semantic meaning, allowing the system to understand relationships between different pieces of text based on their meaning, not just keywords.
-   **Model:** The pipeline uses Google Cloud's `text-embedding-004` model via the Vertex AI platform. This is a sophisticated model that generates high-quality, meaningful vectors.
-   **Process:** Each piece of processed text is sent to the Vertex AI API, which returns a high-dimensional vector. This requires active Google Cloud authentication.

#### Vector Storage

-   **ChromaDB:** The generated vectors, along with their original text and metadata, are stored locally in a `ChromaDB` vector database.
-   **Benefit:** `ChromaDB` allows for efficient similarity searches. When a user asks a question, their question is also converted into a vector, and `ChromaDB` can quickly find the text chunks with the most similar vectors (i.e., the most semantically relevant information) from the knowledge base.

## Setup

1.  **Install System Dependencies (for PDF Parsing):**
    If you plan to process PDFs, you need to install `poppler` and `tesseract`.
    ```bash
    brew install poppler tesseract
    ```

2.  **Set up Python Environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python Libraries:**
    Install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    The advanced PDF parser requires specific data packages from the Natural Language Toolkit (NLTK). Run the provided setup script to download them.
    ```bash
    ./.venv/bin/python3 scripts/setup_nltk.py
    ```

5.  **Google Cloud Authentication:**
    Ensure you are authenticated with Google Cloud and have the required APIs enabled.
    ```bash
    # Log in to your Google account
    gcloud auth login

    # Set up application default credentials
    gcloud auth application-default login

    # Enable the necessary APIs
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable translate.googleapis.com
    ```

## Scripts

### 1. `create_knowledge_base.py`

This script builds a knowledge base from the sources you provide.

**Usage Examples:**

-   **Scrape a Website (Recursive):**
    ```bash
    ./.venv/bin/python3 scripts/create_knowledge_base.py \
      --name "klarna" \
      --urls "https://docs.klarna.com/" \
      --recursive \
      --overwrite
    ```

-   **Scrape & Translate:**
    ```bash
    ./.venv/bin/python3 scripts/create_knowledge_base.py \
      --name "alma-en" \
      --urls "https://docs.almapay.com/reference/v10" \
      --recursive \
      --translate-to "en" \
      --overwrite
    ```

-   **Ingest a Complex PDF:**
    ```bash
    ./.venv/bin/python3 scripts/create_knowledge_base.py \
      --name "ppro" \
      --pdfs "data/ppro_integration.pdf" \
      --overwrite
    ```

### 2. `verify_chroma.py`

This script allows you to run a direct similarity search against a knowledge base to verify its contents.

**Usage Example:**
```bash
./.venv/bin/python3 scripts/verify_chroma.py \
  --name "ppro" \
  --query "What are the supported countries for SEPA?"
```


