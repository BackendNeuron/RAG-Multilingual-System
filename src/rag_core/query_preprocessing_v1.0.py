# src/phase2/query_processing/query_preprocessing.py

import re
import unicodedata
import spacy
from nltk.corpus import stopwords
import nltk
from langdetect import detect, DetectorFactory

# Ensure deterministic language detection
nltk.download("stopwords", quiet=True)
DetectorFactory.seed = 0

try:
    from camel_tools.utils.charmap import CharMapper
except ImportError:
    CharMapper = None
    print("Camel Tools not installed. Arabic-specific normalization will be skipped.")

class TextNormalizer:
    # ---------------------------
    # Class-level caches to avoid re-loading every query
    # ---------------------------
    spacy_models_cache = {}
    arabic_normalizer_cache = None

    def __init__(self, remove_accents=False):
        self.remove_accents_flag = remove_accents
        self.language = None
        self.nlp = None
        self.stopwords = set()
        self.arabic_normalizer = None

    # ---------------------------
    # Language detection
    # ---------------------------
    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
        except:
            lang = "en"
        self.language = lang
        return lang

    # ---------------------------
    # Load resources (SpaCy, stopwords, Arabic normalizer)
    # ---------------------------
    def load_resources(self):
        lang = self.language

        # ---------------------------
        # Load SpaCy model (cached)
        # ---------------------------
        if lang in self.spacy_models_cache:
            self.nlp = self.spacy_models_cache[lang]
        else:
            try:
                if lang == "en":
                    nlp = spacy.load("en_core_web_sm")
                elif lang == "fr":
                    nlp = spacy.load("fr_core_news_sm")
                elif lang == "de":
                    nlp = spacy.load("de_core_news_sm")
                else:
                    nlp = spacy.load("xx_ent_wiki_sm")
            except OSError:
                print(f"Warning: SpaCy model for '{lang}' not found. Using basic tokenization.")
                nlp = None

            self.nlp = nlp
            self.spacy_models_cache[lang] = nlp

        # ---------------------------
        # Load stopwords
        # ---------------------------
        try:
            if lang == "ar":
                self.stopwords = set(stopwords.words("arabic"))
            else:
                self.stopwords = set(stopwords.words(lang))
        except OSError:
            self.stopwords = set()

        # ---------------------------
        # Load Arabic-specific normalizer if needed
        # ---------------------------
        if lang == "ar" and CharMapper:
            if self.arabic_normalizer_cache is None:
                try:
                    self.arabic_normalizer_cache = CharMapper.builtin_mapper("arclean")
                except Exception as e:
                    print(f"Warning: Arabic normalization mapping failed: {e}")
                    self.arabic_normalizer_cache = None
            self.arabic_normalizer = self.arabic_normalizer_cache

    # ---------------------------
    # Text normalization steps
    # ---------------------------
    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def clean_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_accents(self, text: str) -> str:
        if not self.remove_accents_flag:
            return text
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    # ---------------------------
    # Tokenization and lemmatization
    # ---------------------------
    def tokenize_and_lemmatize(self, text: str):
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_space]
        else:
            tokens = text.split()
        return tokens

    # ---------------------------
    # Stopword removal
    # ---------------------------
    def remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t.lower() not in self.stopwords]

    # ---------------------------
    # Full normalization pipeline
    # ---------------------------
    def normalize(self, text: str) -> str:
        if self.language is None:
            self.detect_language(text)
            self.load_resources()

        if self.language == "ar" and self.arabic_normalizer:
            #text = self.arabic_normalizer.map_string(text)
            # For Arabic language , skip aggressive stopwords and lemmatization
            text = self.clean_whitespace(text)
            return text

        # Standard pipeline for other languages
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.clean_whitespace(text)
        text = self.remove_accents(text)
        tokens = self.tokenize_and_lemmatize(text)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)

class SpellCorrector:
    """
    Handles typo and spelling correction.
    """
    def __init__(self, language="en"):
        self.language = language
        # Load spellcheck dictionary or model
    
    def correct(self, text: str) -> str:
        """
        Placeholder: correct spelling mistakes.
        """
        # TODO: implement spell correction
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
        """
        Store the latest Q&A pair and manage history size.
        """
        self.history.append((question, answer))
        # Keep only the last max_history items
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_contextual_query(self, question: str, max_tokens: int = 1000) -> str:
        """
        Append previous context to the current query.
        Optionally limit the total length in tokens.
        """

        context_parts = []
        for q, a in self.history:
            context_parts.append(f"Q: {q}\nA: {a}")

        context_str = "\n".join(context_parts)

        full_query = f"{context_str}\nQ: {question}" if context_str else f"Q: {question}"

        # Token limits (optional)
        tokens = full_query.split()
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]

        return " ".join(tokens)
 
   
"""    def get_contextual_query(self, question: str, max_tokens: int = 1000) -> str:
        # Flatten previous Q&A pairs into a string
        context_str = ""
        for q, a in self.history:
            context_str += f"Q: {q} A: {a} "
        
        # Concatenate context with current query
        full_query = context_str + f"Q: {question}"
        
        # Simple truncation to avoid too long queries
        tokens = full_query.split()
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
        
        return " ".join(tokens)
"""


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class QueryExpander:
    """
    Expands a query using semantically similar words/phrases from a pre-defined vocabulary.
    Uses a multilingual Sentence Transformer to get embeddings.
    """

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 vocabulary=None, top_k=5, device="cpu"):
        """
        Args:
            model_name: HuggingFace/SentenceTransformer model.
            vocabulary: List of words/phrases to expand from.
            top_k: Number of expansion terms to add.
            device: "cpu" or "cuda".
        """
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.top_k = top_k

        # Vocabulary setup
        if vocabulary is None:
            self.vocabulary = []  # Default empty
        else:
            self.vocabulary = vocabulary

        self.vocab_embeddings = None
        if self.vocabulary:
            self._embed_vocabulary()

    def _embed_vocabulary(self):
        """Precompute embeddings for the vocabulary."""
        print(f"Embedding vocabulary of {len(self.vocabulary)} terms...")
        self.vocab_embeddings = self.model.encode(self.vocabulary,
                                                  convert_to_numpy=True,
                                                  device=self.device)
        print("Vocabulary embeddings complete.")

    def expand(self, query: str) -> str:
        """
        Expand query with top-K semantically similar terms from vocabulary.
        Returns the query with expansion terms appended.
        """
        if not self.vocabulary or self.vocab_embeddings is None:
            # Nothing to expand from
            return query

        # Embed the query
        query_emb = self.model.encode([query], convert_to_numpy=True, device=self.device)

        # Compute cosine similarity with vocabulary
        sims = cosine_similarity(query_emb, self.vocab_embeddings)[0]

        # Get top-K indices (excluding exact match)
        top_indices = sims.argsort()[::-1]
        expansion_terms = []
        count = 0
        for idx in top_indices:
            term = self.vocabulary[idx]
            if term.lower() not in query.lower():  # avoid duplicates
                expansion_terms.append(term)
                count += 1
            if count >= self.top_k:
                break

        # Return query + expansions
        expanded_query = query + " " + " ".join(expansion_terms) if expansion_terms else query
        return expanded_query


class QueryProcessor:
    """
    Main wrapper that runs the full query preprocessing pipeline:
    normalization, spell correction, context handling, query expansion.
    """
    def __init__(self, normalizer: TextNormalizer, spell_corrector: SpellCorrector,
                 context_handler: ContextHandler, expander: QueryExpander):
        self.normalizer = normalizer
        self.spell_corrector = spell_corrector
        self.context_handler = context_handler
        self.expander = expander
    
        """    
        def process(self, query: str) -> str:

        query = self.normalizer.normalize(query)
        #query = self.spell_corrector.correct(query)
        query = self.context_handler.get_contextual_query(query)
        query = self.expander.expand(query)
        return query
        """

    def process(self, user_query: str) -> str:
        """
        Run the full preprocessing pipeline on the input query.
        """
        # Normalize ONLY the new question
        cleaned_query = self.normalizer.normalize(user_query)
        print(f"[QueryProcessor] After normalization: {cleaned_query}")

        # Expand ONLY the cleaned new query
        expanded_query = self.expander.expand(user_query)
        print(f"[QueryProcessor] After expansion: {expanded_query}")

        # Only now attach the conversation history
        contextual_query = self.context_handler.get_contextual_query(expanded_query)
        print(f"[QueryProcessor] After adding conversation context: {contextual_query}")

        return contextual_query
    
   