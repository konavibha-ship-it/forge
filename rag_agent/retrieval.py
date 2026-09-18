"""
Phase 5, part 1: Retrieval — turn text into vectors, search by similarity.

Real RAG systems (and Hugging Face's sentence-transformers) turn text into
"embeddings" — long numeric vectors where similar meanings end up close
together in vector space — using a trained neural network. That requires a
downloaded model. Here we build a simplified but genuine version entirely
from scratch, using no libraries and no downloads: TF-IDF vectors.

TF-IDF ("Term Frequency - Inverse Document Frequency") turns each document
into a vector where:
  - Words that appear OFTEN in a document score higher (term frequency)
  - Words that appear in MANY documents score lower (they're less
    distinctive — inverse document frequency)

It's a simpler, older technique than modern neural embeddings, but the
core idea is identical: turn text into vectors, then measure "similarity"
by comparing vectors (we use cosine similarity, the same metric real
embedding search uses). This is genuinely how search engines worked before
neural embeddings, and it's still used today as a fast baseline.

Once this works, the natural upgrade (noted at the bottom) is swapping in
real neural embeddings via Hugging Face's sentence-transformers — free,
runs locally, no API key — once you're ready to add that dependency.
"""

from __future__ import annotations
import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFVectorizer:
    """Learns a vocabulary from a set of documents, then turns any text
    into a TF-IDF vector over that vocabulary."""

    def __init__(self):
        self.vocab: list[str] = []
        self.word_to_idx: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> None:
        """Learn the vocabulary and IDF scores from a corpus of documents."""
        doc_token_sets = [set(tokenize(doc)) for doc in documents]

        # Vocabulary: every unique word across all documents.
        all_words = set()
        for tokens in doc_token_sets:
            all_words.update(tokens)
        self.vocab = sorted(all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # IDF: log(num_documents / num_documents_containing_word).
        # Rare words (appear in few docs) get a HIGH idf score.
        # Common words (appear in most docs) get a LOW idf score.
        num_docs = len(documents)
        doc_freq = Counter()
        for tokens in doc_token_sets:
            for word in tokens:
                doc_freq[word] += 1

        self.idf = {
            word: math.log(num_docs / doc_freq[word]) + 1.0  # +1 avoids zero
            for word in self.vocab
        }

    def transform(self, text: str) -> list[float]:
        """Turn one piece of text into a TF-IDF vector."""
        tokens = tokenize(text)
        term_counts = Counter(tokens)
        total_terms = len(tokens) if tokens else 1

        vector = [0.0] * len(self.vocab)
        for word, count in term_counts.items():
            if word in self.word_to_idx:
                tf = count / total_terms
                idx = self.word_to_idx[word]
                vector[idx] = tf * self.idf[word]
        return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar two vectors are, from -1 (opposite) to 1 (identical
    direction). This is the standard metric for comparing embeddings."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimpleVectorStore:
    """Stores documents + their vectors, and finds the most similar ones
    to a query — the core operation behind retrieval in RAG."""

    def __init__(self, vectorizer: TFIDFVectorizer):
        self.vectorizer = vectorizer
        self.documents: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, document: str) -> None:
        self.documents.append(document)
        self.vectors.append(self.vectorizer.transform(document))

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top_k documents most similar to the query, each
        paired with its similarity score."""
        query_vec = self.vectorizer.transform(query)
        scored = [
            (doc, cosine_similarity(query_vec, doc_vec))
            for doc, doc_vec in zip(self.documents, self.vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    # Demo: a tiny "knowledge base" of facts, then search it with queries
    # that don't exactly match the wording — proving this is doing real
    # semantic-ish matching, not just literal keyword search.

    knowledge_base = [
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Python is a popular programming language known for readability.",
        "Rust is a systems programming language focused on memory safety.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Transformers are a neural network architecture built around attention.",
        "Mount Everest is the tallest mountain above sea level on Earth.",
        "PyTorch is a deep learning framework widely used for research.",
        "The Amazon rainforest produces a significant share of Earth's oxygen.",
    ]

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(knowledge_base)

    store = SimpleVectorStore(vectorizer)
    for doc in knowledge_base:
        store.add(doc)

    print(f"Vocabulary size: {len(vectorizer.vocab)}\n")

    queries = [
        "Which language is good for AI research?",
        "Tell me about a famous tower in France.",
        "What is the highest mountain?",
    ]

    for query in queries:
        print(f"Query: {query!r}")
        results = store.search(query, top_k=2)
        for doc, score in results:
            print(f"  score={score:.4f}  -> {doc}")
        print()

    print("If the top results are genuinely relevant to each query even")
    print("though the wording doesn't match exactly, retrieval is working.")