# src/services/rag_service.py

from pathlib import Path
from typing import List, Dict, Optional

from src.retrieval.retriever import Retriever
from src.rag_core.query_preprocessing import (
    TextNormalizer, SpellCorrector, ContextHandler, QueryExpander, QueryProcessor
)
from src.rag_core.retrieval_ranking import RetrievalRanker
from src.rag_core.llm_integration import LLMClient
from src.rag_core.response_filtering import CorrectiveResponseFilter
from src.rag_core.performance_utils import PerformanceUtils
import os


class RAGService:
    """
    Wrapper for the full RAG pipeline including retrieval, ranking,
    LLM answer generation, dynamic corrective filtering, and performance caching.
    """

    def __init__(
        self,
        retriever_faiss_path: str,
        retriever_metadata_path: str,
        embedding_model_name: str,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        performance_cache_path: str = "perf_cache.json",
        use_gpu: bool = False,
        context_history_max: int = 3,
        add_to_vector_db: bool = True,
        spell_corrector_language: str = "en",
        query_expander_top_k: int = 5,
        retriever_top_k: int = 5,
        tavily_api_key: Optional[str] = None,
    ):
        device = "cuda" if use_gpu else "cpu"

        # -------------------------------
        # Retriever
        # -------------------------------
        self.retriever = Retriever(
            faiss_index_path=retriever_faiss_path,
            metadata_path=retriever_metadata_path,
            embedding_model_name=embedding_model_name,
            device=device
        )

        # -------------------------------
        # Query Preprocessing
        # -------------------------------
        self.normalizer = TextNormalizer()
        self.spell_corrector = SpellCorrector(language=spell_corrector_language)
        self.context_handler = ContextHandler(max_history=context_history_max)
        self.expander = QueryExpander(top_k=query_expander_top_k, device=device)
        self.query_processor = QueryProcessor(
            self.normalizer, self.spell_corrector, self.context_handler, self.expander
        )

        # -------------------------------
        # Ranker
        # -------------------------------
        self.ranker = RetrievalRanker(
            faiss_index_path=retriever_faiss_path,
            metadata_path=retriever_metadata_path,
            device=device
        )

        # -------------------------------
        # LLM Client
        # -------------------------------
        if not llm_api_key:
            llm_api_key = os.environ.get("GROQ_API_KEY")
        if not llm_model_name:
            llm_model_name = "llama-3.3-70b-versatile"

        self.llm = LLMClient(
            api_key=llm_api_key,
            model_name=llm_model_name
        )

        # -------------------------------
        # Dynamic Response Filter
        # -------------------------------
        if not tavily_api_key:
            tavily_api_key = os.environ.get("TAVILY_API_KEY")

        self.response_filter = CorrectiveResponseFilter(
            llm=self.llm,
            faiss_index_path=retriever_faiss_path,
            metadata_mapping_path=retriever_metadata_path,
            model_name=embedding_model_name,
            tavily_api_key=tavily_api_key,
            add_to_vector_db=add_to_vector_db
        )

        # -------------------------------
        # Performance Utils
        # -------------------------------
        self.perf_utils = PerformanceUtils(
            embedding_model_name=embedding_model_name,
            metadata_path=performance_cache_path,
            use_gpu=use_gpu
        )

        # Conversation context
        self.conversation_history: str = ""
        self.retriever_top_k = retriever_top_k

    def ask_question(self, user_query: str) -> Dict[str, Optional[str]]:
        """
        Process a single query through the full RAG pipeline.
        Returns a dict containing final answer, latency, and cache info.
        """

        def _process(q: str) -> str:
            # Step 1: Preprocess query
            processed_query = self.query_processor.process(q)

            # Step 2: Retrieve top-K documents
            retrieved_docs = self.retriever.retrieve_top_k(processed_query, top_k=self.retriever_top_k)

            # Step 3: Rank results
            ranked_docs = self.ranker.rank_results(retrieved_docs, processed_query)

            # Step 4: Generate final answer using DynamicResponseFilter
            final_answer = self.response_filter.generate_answer(processed_query)

            # Step 5: Update conversation context
            self.conversation_history += f"User: {q}\nAI: {final_answer}\n"
            self.context_handler.update_context(q, final_answer)

            return final_answer

        # Use PerformanceUtils for caching + latency measurement
        results = self.perf_utils.batch_process([user_query], _process)
        return results[0]  # returns dict with keys: query, response, latency


