

# src/phase2/response_filter_corrective.py

import os
import json
import time
import hashlib
from pathlib import Path

import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.rag_core.llm_integration import LLMClient

class CorrectiveResponseFilter:
    """
    Response filter with multi-step grading and fallback, inspired by Corrective RAG.
    Implements:
    - Retrieval relevance grading
    - Answer grounding (hallucination) grading
    - Conditional Tavily fallback
    - Optional embedding of fallback content into FAISS
    """

    # ----------------- Tavily config -----------------
    TAVILY_API_URL = "https://api.tavily.com/search"
    TOP_K_RESULTS = 3
    SLEEP_BETWEEN_CALLS = 1

    # ----------------- FAISS vector DB config -----------------


    MAX_LOOPS = 1

    def __init__(self,
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
        self.TAVILY_API_KEY = tavily_api_key  # <--- store it here

        # Device selection
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

        # Load embedding model
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)

        # Load or init FAISS index
        if Path(self.faiss_index_path).exists():
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())

        # Load metadata mapping
        if Path(self.metadata_mapping_path).exists():
            with open(self.metadata_mapping_path, "r", encoding="utf-8") as f:
                self.metadata_mapping = json.load(f)
        else:
            self.metadata_mapping = {}

    # ----------------- Utils -----------------
    @staticmethod
    def _hash_question(q: str) -> str:
        return hashlib.md5(q.encode("utf-8")).hexdigest()

    def _tavily_search(self, question: str) -> str:
        """Search via Tavily and return combined top results."""
        if not self.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not set in environment")

        headers = {
            "Authorization": f"Bearer {self.TAVILY_API_KEY}",
            "Content-Type": "application/json",
        }

        body = {"query": question, "max_results": self.TOP_K_RESULTS}

        try:
            r = requests.post(self.TAVILY_API_URL, json=body, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[Tavily] Search request failed: {e}")
            return ""

        data = r.json()
        results = data.get("results", [])
        if not results:
            return ""

        texts = []
        for res in results:
            title = res.get("title", "")
            content = res.get("content", "")
            texts.append(f"Title: {title}\nContent: {content}")

        return "\n\n".join(texts)

    # ----------------- Grading -----------------
    def grade_retrieval(self, question: str, retrieved_docs: list) -> bool:
        """Check if retrieved docs are relevant to the question"""
        if not retrieved_docs:
            return False
        context_text = "\n".join([d["chunk_text"] for d in retrieved_docs])
        prompt = f"""
You are a grader. Assess if the retrieved documents are relevant to the question.
Question: {question}
Documents: {context_text}

Return 'yes' if relevant, else 'no'.
"""
        try:
            result = self.llm.generate_answer(prompt).strip().lower()
            return result in ("yes", "y")
        except Exception as e:
            print(f"Retrieval grading failed: {e}")
            return False

    def grade_answer(self, question: str, answer: str, context: str) -> bool:
        """Check if answer is grounded in context and addresses the question"""
        if not answer.strip():
            return False
        prompt = f"""
You are a grader. Assess if the answer is grounded in the provided context and fully answers the question.
Question: {question}
Context: {context}
Answer: {answer}

Return 'yes' if grounded and correct, else 'no'.
"""
        try:
            result = self.llm.generate_answer(prompt).strip().lower()
            return result in ("yes", "y")
        except Exception as e:
            print(f"Answer grading failed: {e}")
            return False

    # ----------------- Fallback -----------------
    def fallback(self, question: str) -> str:
        """Fallback to Tavily and optionally store content in FAISS"""
        tavily_text = self._tavily_search(question)
        if not tavily_text:
            return "Sorry, could not fetch fallback results via Tavily."

        # Optionally embed fallback content into FAISS
        if self.add_to_vector_db:
            embedding = self.model.encode([tavily_text], convert_to_numpy=True, device=self.device)
            idx = self.index.ntotal
            self.index.add(embedding)
            self.metadata_mapping[str(idx)] = {"question": question, "text": tavily_text}
            faiss.write_index(self.index, self.faiss_index_path)
            with open(self.metadata_mapping_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata_mapping, f, ensure_ascii=False, indent=2)

        # Generate final answer
        try:
            return self.llm.generate_answer(question, context=tavily_text)
        except Exception as e:
            return f"LLM failed to generate answer: {e}"

    # ----------------- Main Validation + Fallback Loop -----------------
    def validate_and_fallback(self, question: str, retrieved_docs: list, previous_context: str = "") -> str:
        """
        Main workflow: looped validation, grading, and conditional fallback.
        Returns the final validated answer.
        """
        # Combine retrieved docs + previous context for LLM generation
        context_text = "\n".join([d["chunk_text"] for d in retrieved_docs])
        if previous_context:
            context_text = previous_context + "\n" + context_text

        # Initial answer generation
        answer = self.llm.generate_answer(question, context=context_text)

        state = {
            "answer": answer,
            "loop_count": 0,
            "retrieved_docs": retrieved_docs
        }

        while state["loop_count"] < self.MAX_LOOPS:
            state["loop_count"] += 1

            retrieval_ok = self.grade_retrieval(question, retrieved_docs)
            answer_ok = self.grade_answer(question, state["answer"], context_text)

            if retrieval_ok and answer_ok:
                break  # Accept current answer

            # Trigger fallback
            # 1️⃣ Rewrite query using full conversation context
            context_aware_question = self.make_context_aware_query(previous_context, question)

            # 2️⃣ Search Tavily with rewritten query
            tavily_text = self._tavily_search(context_aware_question)
            if not tavily_text:
                state["answer"] = "Sorry, could not fetch fallback results via Tavily."
            else:
                # Optionally embed fallback content into FAISS
                if self.add_to_vector_db:
                    embedding = self.model.encode([tavily_text], convert_to_numpy=True, device=self.device)
                    idx = self.index.ntotal
                    self.index.add(embedding)
                    self.metadata_mapping[str(idx)] = {"question": context_aware_question, "text": tavily_text}
                    faiss.write_index(self.index, self.faiss_index_path)
                    with open(self.metadata_mapping_path, "w", encoding="utf-8") as f:
                        json.dump(self.metadata_mapping, f, ensure_ascii=False, indent=2)

                # Generate final answer from LLM using Tavily content
                state["answer"] = self.llm.generate_answer(question, context=tavily_text)

            # Update context_text for next loop
            context_text = previous_context + "\n" + state["answer"] if previous_context else state["answer"]

        return state["answer"]


    def make_context_aware_query(self, conversation_history: str, current_query: str) -> str:
        """
        Converts a context-dependent query into a self-contained query
        suitable for external search (like Tavily).
        """
        prompt = f"""You are an assistant. Given the conversation so far and the current user query,
    rewrite the query to be fully self-contained so that someone reading only this query
    would understand it.

    Conversation so far:
    {conversation_history}

    Current query:
    {current_query}

    Return only the rewritten, self-contained query."""
        rewritten_query = self.llm.generate_answer(prompt)
        return rewritten_query.strip()
