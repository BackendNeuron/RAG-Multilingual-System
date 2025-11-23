# src/phase2/retrieval_ranking.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalRanker:
    """
    Handles relevance scoring and ranking of retrieved documents/chunks.
    Uses FAISS vector DB, multilingual embeddings, and metadata-aware hybrid ranking.
    """

    def __init__(self, faiss_index_path, metadata_path,
                 embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device="cpu"):
        """
        Args:
            faiss_index_path: path to FAISS .bin index
            metadata_path: path to metadata_mapping.json
            embedding_model_name: same model used for embeddings
            device: "cpu" or "cuda"
        """
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)

        # Load metadata mapping
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Initialize embedding model
        self.device = device
        self.model = SentenceTransformer(embedding_model_name)
        self.model.to(device)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert query to embedding vector.
        """
        return self.model.encode([query], convert_to_numpy=True, device=self.device)

    def retrieve_top_k(self, query: str, k: int = 5):
        """
        Retrieve top-K relevant chunks/documents for a query.
        Returns a list of dicts: {chunk_text, metadata, distance}
        """
        query_emb = self.embed_query(query)
        distances, indices = self.index.search(query_emb, k*3)  # retrieve more for re-ranking
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if str(idx) in self.metadata:
                item = self.metadata[str(idx)]
                results.append({
                    "chunk_text": item.get("chunk_text", ""),
                    "question": item.get("question", ""),
                    "short_answer": item.get("short_answer", ""),
                    "metadata": item.get("metadata", {}),
                    "faiss_distance": float(dist)
                })
        return results

    def rank_results(self, retrieved_docs: list, query: str, alpha=0.5, beta=0.3, gamma=0.2) -> list:
        """
        Hybrid ranking: combines FAISS distance, cosine similarity, and metadata score.
        Args:
            alpha: weight for FAISS score
            beta: weight for cosine similarity
            gamma: weight for metadata relevance
        """
        if not retrieved_docs:
            return []

        # Embed query and chunk_texts
        chunk_texts = [doc["chunk_text"] for doc in retrieved_docs]
        chunk_embs = self.model.encode(chunk_texts, convert_to_numpy=True, device=self.device)
        query_emb = self.embed_query(query)

        # Cosine similarity
        sims = cosine_similarity(query_emb, chunk_embs)[0]

        # Normalize FAISS distance to 0-1 (smaller distance = higher score)
        faiss_scores = np.array([1 / (1 + doc["distance"]) for doc in retrieved_docs])

        # Metadata scoring (example: match domain or question type if available)
        metadata_scores = []
        query_lower = query.lower()
        for doc in retrieved_docs:
            meta = doc.get("metadata") or {}  # ensures meta is always a dict
            score = 0
            # Example: +0.5 if domain matches query keywords
            domain = meta.get("domain", "").lower()
            if domain and domain in query_lower:
                score += 0.5
            # Example: +0.5 if question type matches (who, what, when, etc.)
            qtype = meta.get("question_type", "").lower()
            for keyword in ["who", "what", "when", "where", "how"]:
                if qtype == keyword and keyword in query_lower:
                    score += 0.5
            metadata_scores.append(score)
        metadata_scores = np.array(metadata_scores)

        # Hybrid score
        hybrid_scores = alpha * faiss_scores + beta * sims + gamma * metadata_scores

        # Sort retrieved_docs by hybrid score descending
        ranked_docs = [doc for _, doc in sorted(zip(hybrid_scores, retrieved_docs),
                                                key=lambda x: x[0], reverse=True)]
        return ranked_docs
