class Retriever:
    def __init__(self, faiss_index_path, metadata_path, embedding_model_name, device="cpu"):
        import json, faiss
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.model = SentenceTransformer(embedding_model_name)
        self.model.to(device)
        self.index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_mapping = json.load(f)
    
    def retrieve_top_k(self, query: str, top_k: int = 5):
        query_emb = self.model.encode([query], convert_to_numpy=True, device=self.device)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            entry = self.metadata_mapping[str(idx)]
            results.append({
                "chunk_text": entry.get("chunk_text", ""),
                "chunk_id": entry.get("chunk_id"),
                "question": entry.get("question"),
                "short_answer": entry.get("short_answer"),
                "metadata": entry.get("metadata"),
                "distance": float(dist)
            })
        return results
