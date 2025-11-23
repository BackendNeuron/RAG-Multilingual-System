# src/phase2/response_filter_dynamic.py

import os
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.rag_core.llm_integration import LLMClient
import requests
import numpy as np

class CorrectiveResponseFilter:
    """
    Dynamic Corrective Response Filter:
    - Uses vector DB if relevant
    - Lets LLM answer unknown/generic queries naturally
    - Fallback to Tavily only if answer needs more grounding
    - Ensures context-aware query to Tavily when used
    - Optimized for minimal LLM calls
    """

    TAVILY_API_URL = "https://api.tavily.com/search"
    TOP_K_RESULTS = 3

    def __init__(
        self,
        llm: LLMClient,
        faiss_index_path: str,
        metadata_mapping_path: str,
        model_name: str,
        tavily_api_key: str = None,
        add_to_vector_db: bool = True
    ):
        self.llm = llm
        self.add_to_vector_db = add_to_vector_db
        self.faiss_index_path = faiss_index_path
        self.metadata_mapping_path = metadata_mapping_path
        self.model_name = model_name
        self.TAVILY_API_KEY = tavily_api_key

        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

        # Embedding model
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)

        # Load or init FAISS index
        if Path(self.faiss_index_path).exists():
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())

        # Load metadata
        if Path(self.metadata_mapping_path).exists():
            with open(self.metadata_mapping_path, "r", encoding="utf-8") as f:
                self.metadata_mapping = json.load(f)
        else:
            self.metadata_mapping = {}

    # ----------------- Utilities -----------------
    def _tavily_search(self, question: str) -> str:
        if not self.TAVILY_API_KEY:
            return ""
        try:
            r = requests.post(
                self.TAVILY_API_URL,
                headers={"Authorization": f"Bearer {self.TAVILY_API_KEY}", "Content-Type": "application/json"},
                json={"query": question, "max_results": self.TOP_K_RESULTS},
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            return "\n\n".join(
                f"Title: {res.get('title','')}\nContent: {res.get('content','')}" for res in results
            )
        except Exception as e:
            print(f"[Tavily] Search failed: {e}")
            return ""

    def _retrieve_docs(self, question: str, top_k=4):
        if self.index.ntotal == 0:
            return []
        query_emb = self.model.encode([question], convert_to_numpy=True, device=self.device)
        D, I = self.index.search(query_emb, top_k)
        docs = [self.metadata_mapping[str(idx)] for idx in I[0] if str(idx) in self.metadata_mapping]
        return docs

    def _is_context_relevant(self, question: str, context: str) -> bool:
        """LLM evaluates if retrieved context is relevant to question"""
        if not context.strip():
            return False
        prompt = f"""
You are an evaluator. Assess if the following context is relevant for answering the question.
Question: {question}
Context: {context}

Return 'yes' if relevant, otherwise 'no'.
"""
        result = self.llm.generate_answer(prompt).strip().lower()
        return result in ("yes", "y")

    def _needs_fallback(self, question: str, answer: str, context: str) -> bool:
        """LLM evaluates if answer is sufficiently grounded; fallback needed if not"""
        if not answer.strip():
            return True
        prompt = f"""
You are an evaluator. Assess if the answer below is fully grounded based on the context.
Question: {question}
Context: {context}
Answer: {answer}

Return 'yes' if grounded, otherwise 'no'.
"""
        result = self.llm.generate_answer(prompt).strip().lower()
        return result not in ("yes", "y")

    def make_context_aware_query(self, conversation_history: str, current_query: str) -> str:
        """
        Converts a context-dependent query into a self-contained query
        suitable for external search (like Tavily).
        """
        prompt = f"""You are an assistant. Given the conversation so far and the current user query,
    rewrite the query to be fully self-contained so that someone reading only this query
    would understand it. Avoid pronouns and ensure all references are explicit.

    Conversation history:
    {conversation_history}

    Current query:
    {current_query}

    Return only the rewritten query suitable for external search."""
        
        rewritten = self.llm.generate_answer(prompt).strip()
        # Safety fallback
        if not rewritten:
            rewritten = current_query
        return rewritten


    # ----------------- Main workflow -----------------
    def generate_answer(self, question: str, conversation_history: str = "") -> str:
        """
        Main entry:
        - Uses vector DB if relevant
        - LLM answers generic queries naturally
        - Calls Tavily fallback only if answer ungrounded
        - Context-aware query generated only when Tavily is called
        """
        # 1️⃣ Retrieve vector DB docs
        retrieved_docs = self._retrieve_docs(question)
        context_text = "\n".join([d.get("text","") for d in retrieved_docs])

        # 2️⃣ Decide whether to use vector DB context
        if self._is_context_relevant(question, context_text):
            answer = self.llm.generate_answer(question, context=context_text)
        else:
            answer = self.llm.generate_answer(question)  # LLM from its own knowledge

        # 3️⃣ Decide if fallback is needed
        if self.TAVILY_API_KEY and self._needs_fallback(question, answer, context_text):
            # Rewrite query for Tavily only when fallback is actually used
            context_for_tavily = conversation_history + "\n" + answer if conversation_history else answer
            context_aware_question = self.make_context_aware_query(context_for_tavily, question)
            fallback_text = self._tavily_search(context_aware_question)

            if fallback_text:
                # Optionally embed fallback into FAISS
                if self.add_to_vector_db:
                    emb = self.model.encode([fallback_text], convert_to_numpy=True, device=self.device)
                    idx = self.index.ntotal
                    self.index.add(emb)
                    self.metadata_mapping[str(idx)] = {"question": question, "text": fallback_text}
                    faiss.write_index(self.index, self.faiss_index_path)
                    with open(self.metadata_mapping_path, "w", encoding="utf-8") as f:
                        json.dump(self.metadata_mapping, f, ensure_ascii=False, indent=2)

                # Regenerate answer using Tavily content
                answer = self.llm.generate_answer(question, context=fallback_text)

        return answer
