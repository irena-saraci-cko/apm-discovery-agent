import argparse
import chromadb
import google.auth
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List

class ChromaDBReader(BaseReader):
    """
    A reader to directly query a ChromaDB collection and retrieve documents.

    This is a simplified reader that connects to an existing ChromaDB collection
    and performs a similarity search to retrieve the top N most relevant documents
    for a given query. It's primarily used for verification and direct querying,
    bypassing the complexity of a full query engine.
    """

    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """
        Initializes the ChromaDBReader.

        Args:
            collection_name: The name of the ChromaDB collection to query.
            persist_directory: The directory where ChromaDB data is stored.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def load_data(self, query_text: str, top_n: int = 3) -> List[Document]:
        """
        Performs a similarity search in the ChromaDB collection.

        Args:
            query_text: The text to search for.
            top_n: The number of top results to return.

        Returns:
            A list of LlamaIndex Document objects representing the results.
        """
        print(f"--- 1. Connecting to ChromaDB Collection: '{self.collection_name}' ---")
        collection = self.client.get_collection(self.collection_name)
        print("✅ Connected.")

        print("\n--- 2. Authenticating and Initializing Embedding Model ---")
        credentials, project_id = google.auth.default()
        embed_model = VertexTextEmbedding(
            model_name="text-embedding-004",
            project=project_id,
            credentials=credentials,
        )
        print("✅ Embedding model initialized.")

        print("\n--- 3. Generating Query Embedding ---")
        query_embedding = embed_model.get_text_embedding(query_text)
        print("✅ Query embedding created.")

        print(f"\n--- 4. Querying for Top {top_n} Results ---")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
        )
        print("✅ Query complete.")

        documents = []
        if results and results.get("documents"):
            for i, doc_text in enumerate(results["documents"][0]):
                doc = Document(
                    text=doc_text,
                    extra_info={
                        "id": results["ids"][0][i],
                        "distance": results.get("distances", [[None]])[0][i],
                    },
                )
                documents.append(doc)
        return documents

def main():
    """Main function to parse arguments and run the verification query."""
    parser = argparse.ArgumentParser(
        description="Directly query a ChromaDB collection for verification.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--name", required=True, help="The unique name of the knowledge base (e.g., 'alma')."
    )
    parser.add_argument(
        "--query", required=True, help="The question or text to search for."
    )
    parser.add_argument(
        "--top_n", type=int, default=3, help="Number of results to return."
    )
    args = parser.parse_args()

    collection_name = f"{args.name.lower()}_docs"
    reader = ChromaDBReader(collection_name=collection_name)
    documents = reader.load_data(query_text=args.query, top_n=args.top_n)

    print("\n--- 5. Top Results ---")
    if not documents:
        print("No results found.")
    else:
        for i, doc in enumerate(documents):
            print(f"\n--- Result {i+1} ---")
            print(f"Source ID: {doc.extra_info.get('id', 'N/A')}")
            print(f"Similarity Distance: {doc.extra_info.get('distance', 'N/A'):.4f}")
            print("Text:")
            print(doc.text)

if __name__ == "__main__":
    main()
