from pathlib import Path
import json
import pickle

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


class ProcessRetriever:
    """
    Retrieval layer for Agentic Process Intelligence.

    Loads:
    - RAG documents
    - TF-IDF vectorizer
    - Persistent ChromaDB collection

    and provides similarity-based retrieval.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)

        self.rag_dir = (
            self.project_root
            / "data"
            / "rag"
        )

        self.documents_path = (
            self.rag_dir
            / "rag_documents.json"
        )

        self.vectorizer_path = (
            self.rag_dir
            / "tfidf_vectorizer.pkl"
        )

        self.chroma_path = (
            self.rag_dir
            / "chroma_db"
        )

        # Load vectorizer
        with open(
            self.vectorizer_path,
            "rb"
        ) as f:
            self.vectorizer = pickle.load(f)

        # Initialize Chroma
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )

        self.collection = (
            self.client
            .get_collection(
                name="process_intelligence"
            )
        )

        # Load metadata documents
        with open(
            self.documents_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.documents = json.load(f)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Retrieve the most relevant process-intelligence
        documents for a user query.
        """

        query_vector = (
            self.vectorizer
            .transform([query])
            .toarray()[0]
        )

        results = self.collection.query(
            query_embeddings=[
                query_vector.tolist()
            ],
            n_results=top_k
        )

        retrieved_documents = []

        for i, document in enumerate(
            results["documents"][0]
        ):
            retrieved_documents.append({
                "document": document,
                "metadata": (
                    results["metadatas"][0][i]
                ),
                "distance": (
                    results["distances"][0][i]
                    if "distances" in results
                    else None
                )
            })

        return retrieved_documents
    