# src/phase2/query_processing/query_preprocessing.py

import re
from nltk.corpus import stopwords
import nltk
from langdetect import detect, DetectorFactory
from typing import Optional

# Ensure deterministic language detection
nltk.download("stopwords", quiet=True)
DetectorFactory.seed = 0

try:
    from camel_tools.utils.charmap import CharMapper
except ImportError:
    CharMapper = None
    print("Camel Tools not installed. Arabic-specific normalization will be skipped.")


class TextNormalizer:
    """
    Dummy normalizer that skips any modification to the text.
    """
    def __init__(self):
        self.language: Optional[str] = None

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
        except:
            lang = "en"
        self.language = lang
        return lang

    def load_resources(self):
        pass

    def normalize(self, text: str) -> str:
        return text


class SpellCorrector:
    """
    Handles typo and spelling correction (placeholder, unchanged).
    """
    def __init__(self, language="en"):
        self.language = language
        # Load spellcheck dictionary or model if implemented

    def correct(self, text: str) -> str:
        # Placeholder: return text as-is
        return text


class ContextHandler:
    """
    Handles multi-turn conversation context.
    Stores previous question/answer pairs and
    appends context to the current query.
    """
    def __init__(self, max_history=3):
        self.history = []  # list of tuples: [(question, answer), ...]
        self.max_history = max_history

    def update_context(self, question: str, answer: str):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_contextual_query(self, question: str, max_tokens: int = 1000) -> str:
        context_parts = [f"Q: {q}\nA: {a}" for q, a in self.history]
        context_str = "\n".join(context_parts)
        full_query = f"{context_str}\nQ: {question}" if context_str else f"Q: {question}"

        # Limit tokens if needed
        tokens = full_query.split()
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]

        return " ".join(tokens)


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class QueryExpander:
    """
    Expands a query using semantically similar words/phrases from a pre-defined vocabulary.
    """
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 vocabulary=None, top_k=5, device="cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.top_k = top_k

        self.vocabulary = vocabulary if vocabulary else []
        self.vocab_embeddings = None
        if self.vocabulary:
            self._embed_vocabulary()

    def _embed_vocabulary(self):
        print(f"Embedding vocabulary of {len(self.vocabulary)} terms...")
        self.vocab_embeddings = self.model.encode(self.vocabulary,
                                                  convert_to_numpy=True,
                                                  device=self.device)
        print("Vocabulary embeddings complete.")

    def expand(self, query: str) -> str:
        if not self.vocabulary or self.vocab_embeddings is None:
            return query

        query_emb = self.model.encode([query], convert_to_numpy=True, device=self.device)
        sims = cosine_similarity(query_emb, self.vocab_embeddings)[0]
        top_indices = sims.argsort()[::-1]

        expansion_terms = []
        count = 0
        for idx in top_indices:
            term = self.vocabulary[idx]
            if term.lower() not in query.lower():
                expansion_terms.append(term)
                count += 1
            if count >= self.top_k:
                break

        return query + " " + " ".join(expansion_terms) if expansion_terms else query


class QueryProcessor:
    """
    Full query preprocessing pipeline WITHOUT normalization.
    Keeps spell correction, query expansion, and conversation context.
    """
    def __init__(self, normalizer: TextNormalizer, spell_corrector: SpellCorrector,
                 context_handler: ContextHandler, expander: QueryExpander):
        self.normalizer = normalizer
        self.spell_corrector = spell_corrector
        self.context_handler = context_handler
        self.expander = expander

    def process(self, user_query: str) -> str:
        # Skip normalization entirely
        cleaned_query = user_query
        print(f"[QueryProcessor] Original query (skipped normalization): {cleaned_query}")

        # Spell correction placeholder (no-op)
        corrected_query = self.spell_corrector.correct(cleaned_query)
        print(f"[QueryProcessor] After spell correction: {corrected_query}")

        # Query expansion
        expanded_query = self.expander.expand(corrected_query)
        print(f"[QueryProcessor] After expansion: {expanded_query}")

        # Add conversation context
        contextual_query = self.context_handler.get_contextual_query(expanded_query)
        print(f"[QueryProcessor] After adding conversation context: {contextual_query}")

        return contextual_query
