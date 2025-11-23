# src/phase2/performance_utils.py

import time
import logging
from typing import Any, List, Dict, Optional
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class PerformanceUtils:
    """
    Handles persistent exact-match caching, batch processing,
    embedding management, logging, and performance metrics collection for RAG system.
    """

    def __init__(
        self,
        embedding_model_name: str,
        metadata_path: str,
        use_gpu: bool = False
    ):
        # ------------------- Logging -------------------
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            filename="query_logs.log",
            filemode="a"
        )

        # ------------------- Embedding (optional) -------------------
        self.device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(embedding_model_name)
        self.model.to(self.device)

        # ------------------- Cache -------------------
        self.metadata_path = metadata_path
        self.metadata: Dict[str, Dict[str, str]] = {}

        # Load existing cache
        if Path(metadata_path).exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logging.info(f"Exact-match cache loaded from {metadata_path}")

    # ------------------- Exact-match Caching -------------------
    def get_cached_response(self, query: str) -> Optional[str]:
        """Return cached response if query matches exactly."""
        if query in self.metadata:
            logging.info(f"Exact cache hit for query: {query}")
            return self.metadata[query]["response"]
        return None

    def add_to_cache(self, query: str, response: str, metadata: Optional[Dict[str, str]] = None):
        """Add query and response to exact-match cache and persist to disk."""
        if metadata is None:
            metadata = {}
        self.metadata[query] = {"response": response, **metadata}

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"Added query to exact-match cache: {query}")

    # ------------------- Logging -------------------
    def log_query(self, query: str, response: str, latency: float):
        logging.info(f"Query: {query} | Latency: {latency:.3f}s | Response length: {len(response)}")

    # ------------------- Batch Processing -------------------
    def batch_process(self, queries: List[str], process_fn, force_recompute: bool = False):
        """
        Process multiple queries with exact-match caching.

        Args:
            queries: list of query strings
            process_fn: function(query: str) -> response
            force_recompute: if True, always run process_fn ignoring cache
        Returns:
            List of dicts containing query, response, and latency
        """
        results = []
        start_batch = time.time()
        for q in queries:
            entry = {"query": q, "response": None, "latency": None}

            cached = None if force_recompute else self.get_cached_response(q)
            if cached:
                entry["response"] = cached
                entry["latency"] = 0.0  # cache hit is nearly instant
                results.append(entry)
                continue

            t0 = time.time()
            response = process_fn(q)
            latency = time.time() - t0

            entry["response"] = response
            entry["latency"] = round(latency, 3)

            self.add_to_cache(q, response)
            self.log_query(q, response, latency)
            results.append(entry)

        batch_latency = time.time() - start_batch
        logging.info(f"Processed batch of {len(queries)} queries in {batch_latency:.3f}s")
        return results

    # ------------------- Embedding (optional for other tasks) -------------------
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True, device=self.device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, device=self.device, batch_size=32)

    # ------------------- Performance Metrics -------------------
    def measure_latency(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        logging.info(f"Function {func.__name__} executed in {latency:.3f}s")
        return result, latency
