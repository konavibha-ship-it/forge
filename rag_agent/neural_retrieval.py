"""
Phase 5 upgrade: real neural embeddings, replacing TF-IDF.

retrieval.py's TF-IDF approach matches based on shared WORDS. It has a
real limitation you likely already saw: "What did the developer use?"
failed to match "Forge was built by a self-taught developer using free
tools" well enough, because TF-IDF has no idea "use" and "using" are
related, or that the sentence is actually about the right topic — it only
counts word overlap.

Neural embeddings fix this. A pretrained neural network (trained on huge
amounts of text) turns each sentence into a vector where MEANING — not
just word overlap — determines closeness. "What did the developer use?"
and "...using free tools" end up close together in vector space even
though the wording differs, because the model has learned what these
sentences are actually about.

This uses `sentence-transformers`, a free library that runs entirely on
your own machine — no API key, no cost. The first run downloads a small
pretrained model (~90MB, one-time, cached locally afterward).

The interface (NeuralVectorStore.add / .search) intentionally matches
retrieval.py's SimpleVectorStore, so agent.py can use either one
interchangeably — swap one line in app/main.py to switch between them.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np


class NeuralVectorStore:
    """Same interface as SimpleVectorStore (retrieval.py), but backed by
    real neural embeddings instead of TF-IDF."""

    # 'all-MiniLM-L6-v2' is a small, fast, well-regarded general-purpose
    # embedding model — a standard free choice for this kind of task.
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"Loading embedding model '{self.MODEL_NAME}' "
              f"(first run downloads it, ~90MB, one-time)...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None  # shape: (num_docs, embedding_dim)

    def add(self, document: str) -> None:
        self.documents.append(document)
        # Recomputing all embeddings on every add is simple and fine for a
        # small knowledge base. A production system would append instead —
        # left as a natural optimization once your knowledge base grows.
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.documents:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity between the query and every stored document,
        # vectorized with numpy (same math as retrieval.py's
        # cosine_similarity, just computed for all documents at once).
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = (self.embeddings @ query_embedding) / (doc_norms * query_norm + 1e-8)

        ranked_indices = np.argsort(-similarities)[:top_k]
        return [(self.documents[i], float(similarities[i])) for i in ranked_indices]


if __name__ == "__main__":
    # Same demo as retrieval.py, including the query that TF-IDF struggled
    # with — this time it should work correctly.
    knowledge_base = [
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Python is a popular programming language known for readability.",
        "Rust is a systems programming language focused on memory safety.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Transformers are a neural network architecture built around attention.",
        "Mount Everest is the tallest mountain above sea level on Earth.",
        "PyTorch is a deep learning framework widely used for research.",
        "The Amazon rainforest produces a significant share of Earth's oxygen.",
        "Forge was built by a self-taught developer using free tools.",
    ]

    store = NeuralVectorStore()
    for doc in knowledge_base:
        store.add(doc)

    queries = [
        "Which language is good for AI research?",
        "Tell me about a famous tower in France.",
        "What is the highest mountain?",
        "What did the developer use?",  # the one TF-IDF got wrong
    ]

    for query in queries:
        print(f"\nQuery: {query!r}")
        results = store.search(query, top_k=2)
        for doc, score in results:
            print(f"  score={score:.4f}  -> {doc}")

    print("\nCompare this to retrieval.py's output on the same last query —")
    print("neural embeddings should correctly match it to the Forge/developer")
    print("fact, where TF-IDF matched something unrelated.")