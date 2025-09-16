import argparse
import chromadb
import google.auth
import vertexai

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.google_genai import GoogleGenAI

def main():
    parser = argparse.ArgumentParser(description="Query a knowledge base for a specific payment method.")
    parser.add_argument("--name", required=True, help="The unique name of the payment method (e.g., alma).")
    parser.add_argument("--query", required=True, help="The question you want to ask the knowledge base.")
    parser.add_argument("--location", default="us-central1", help="The Google Cloud location/region.")
    args = parser.parse_args()

    print(f"Querying knowledge base for '{args.name}'...")

    # 1. Authenticate with Google Cloud and configure models
    print("Authenticating with Google Cloud and configuring models...")
    credentials, project_id = google.auth.default()
    vertexai.init(project=project_id, location=args.location, credentials=credentials)

    # Configure the embedding model
    embed_model = VertexTextEmbedding(
        model_name="text-embedding-004",
        project=project_id,
        credentials=credentials,
    )

    # Configure the LLM for answer generation
    llm = GoogleGenAI(
        model_name="models/gemini-pro",
        vertex=vertexai,
    )

    # Set the global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("✅ Authentication and model configuration successful.")

    # 2. Connect to the ChromaDB vector store
    print("Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = f"{args.name.lower()}_docs"
    chroma_collection = db.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f"✅ Connected to collection '{collection_name}'.")

    # 3. Load the index from the vector store
    # NOTE: We no longer need to pass the embed_model here
    # as it's now in the global Settings.
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

    # 4. Create a query engine
    print("Creating query engine...")
    query_engine = index.as_query_engine()
    print("✅ Query engine created.")

    # 5. Execute the query
    print(f"\nAsking: \"{args.query}\"\n")
    response = query_engine.query(args.query)

    # 6. Print the results
    print("--- RESPONSE ---")
    print(response)
    print("\n--- SOURCES ---")
    for node in response.source_nodes:
        print(f"Source: {node.metadata.get('url', 'N/A')}")
        print(f"Score: {node.score:.4f}")
        # print(f"Text: {node.get_content()[:250]}...") # Uncomment for more detail
        print("-" * 20)

if __name__ == "__main__":
    main()
