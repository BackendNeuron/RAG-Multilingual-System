# 📘 Multilingual RAG System

A high-performance, multilingual Retrieval-Augmented Generation (RAG) system combining FAISS, Groq LLaMA models, Tavily web search, and robust preprocessing + corrective filtering to deliver fast, accurate, and context-grounded answers.

Designed as part of an advanced AI hiring task, this system demonstrates production-level architectural design, retrieval quality control, and multilingual capability.

# 🌟 Features

Multilingual Query Support
Automatically detects and processes queries in multiple languages.

High-speed Vector Retrieval using FAISS
Stores embeddings locally for fast, offline similarity search.

Groq-accelerated LLaMA Generation
Ultra-fast inference with LLaMA 3.x on Groq API.

Preprocessing Pipeline
Includes text cleaning, chunking, metadata processing, and embeddings generation.

Corrective Filtering Stages

Query preprocessing

Response filtering

Retrieval ranking

Metadata alignment

Context re-weighting

Local Caching System
Uses exact_cache.json for performance optimizations.

Web Search Fallback (Tavily)
When vector retrieval is insufficient or unclear, the system expands context using external web search.

Full FastAPI Backend
Clean REST API with clear request/response schemas.

ElevenLabs TTS Integration for responses

Lightweight HTML Frontend
Inline CSS+JS for simple interaction with the RAG backend.

# 🧠 Question Difficulty Classification Model (spaCy) for metadata

Inside the models/ directory, the project includes a custom fine-tuned spaCy text classification model.
This model predicts the difficulty level of a user question across five classes:

Too Easy
Easy
Medium
Hard
Too Hard

📚 Fine-Tuning Data

The model was fine-tuned using a Kaggle dataset of natural-language questions.
Before training, the dataset was cleaned and standardized inside the data/processed/ folder.

# 🧪 Fine-Tuning Notebook

The full training workflow—including preprocessing, labeling, fine-tuning, and evaluation—is documented in the Jupyter notebook located inside notebooks folder

# 🏗 Architecture

Overall multi-stage flow inspired by advanced modern RAG systems:

                    ┌─────────────────────────┐
                    │       User Query        │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │  Query Preprocessing   │
                     │  (language, cleaning)  │
                     └─────────────┬──────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │      FAISS Vector Retrieval            │
              │ (top-k semantically similar chunks)    │
              └─────────────┬─────────────────────────┘
                            │
                            ▼
                    ┌─────────────────────┐
                    │  Retrieval Ranking   │
                    │ + Metadata Filtering │
                    └─────────────┬────────┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   Groq LLaMA Gen    │
                      └─────────────┬────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Response Filtering   │
                         │ (grounding + quality)│
                         └─────────────┬────────┘
                                       │
                          ┌────────────┴─────────────┐
                          ▼                          ▼
                 ┌─────────────────┐       ┌────────────────────┐
                 │  Final Answer   │       │ Tavily Web Search   │
                 └─────────────────┘       └────────────┬────────┘
                                                         │
                                                         ▼
                                      ┌──────────────────────────┐
                                      │  Re-run Generation + QC  │
                                      └──────────────────────────┘

# 📁 Project Structure (Real Directory Tree)

Based on your screenshot:

rag_system/
│
├── backend/
│   ├── __pycache__/
│   ├── app.py
│   ├── elevenlabs_tts.py
│   ├── evaluate.py
│   └── service.py
│
├── data/
│   ├── chunks/
│   │   └── semantic_chunked_dataset.json
│   ├── embeddings/
│   │   ├── faiss_index.bin
│   │   └── metadata_mapping.json
│   ├── performance/
│   │   └── exact_cache.json
│   ├── processed/
│   │   ├── Natural-Questions-Cleaned.csv
│   │   └── Natural-Questions-Filtered.csv
│   └── raw/
│       └── Natural-Questions-Filtered.csv
│
├── models/
│
├── notebooks/
│   └── fine_tune_question_difficulty_spacy.ipynb
│
├── src/
│   ├── embeddings/
│   │   └── embedding.py
│   ├── preprocessing/
│   │   ├── chunking.py
│   │   ├── load_data.py
│   │   ├── metadata.py
│   │   └── preprocess.py
│   ├── rag_core/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── lm_integration.py
│   │   ├── performance_utils.py
│   │   ├── query_preprocessing_v1.0.py
│   │   ├── query_preprocessing.py
│   │   ├── response_filtering_v1.0.py
│   │   ├── response_filtering.py
│   │   ├── response_filtering_draft.py
│   │   └── retrieval_ranking.py
│   └── retrieval/
│       ├── __pycache__/
│       └── retriever.py
│
└── static/
    └── index.html  (inline CSS + JS)

# 📋 Prerequisites

Python 3.10+

FastAPI

FAISS

Groq API Key

Tavily API Key

Pydantic

Uvicorn

# 🚀 Installation
1. Clone Project
git clone repo.
cd rag_system

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Set Environment Variables

Create .env:

GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
ELEVENLABS_API_KEY=your_key 

MODEL_NAME=llama-3.3-70b-versatile
Copy env_template.txt to .env and fill in your real API keys and paths before running the project.

# ▶️ Running the Backend

Start FastAPI server:

uvicorn backend.app:app --reload


The API will run at:

http://127.0.0.1:8000

PORT 8000 (optional you can try any available port)
Open the frontend:

rag_system/static/index.html

# 🎯 API Endpoints
POST /ask-question
GET /health
Send a question and receive an answer:

{
  "query": "Who's the Weeknd?!"
}

Response
{
  "answer": "...",
  "context_used": [...],
  "retrieval_time_s": 13
}

🔍 How It Works
🔹 1. Query Preprocessing

Cleans text, normalizes multilingual input, and extracts intent.

🔹 2. FAISS Retrieval

Loads faiss_index.bin + metadata_mapping.json for fast vector search.

🔹 3. Document Ranking

Scores retrieved chunks using semantic + metadata rules.

🔹 4. LLaMA Generation

Uses Groq LLaMA for extremely fast inference.

🔹 5. Response Filtering

Multiple safety and grounding checks before final answer.

🔹 6. Tavily Fallback

Searches the web if retrieved knowledge is insufficient.

# 🧪 Example Usage (Python)
from backend.service import RAGService

rag = RAGService()

result = rag.run_query("Explain photosynthesis in simple terms")

print(result.answer)

# 🛠 Troubleshooting
❌ FAISS index not found

Rebuild embeddings:

python src/embeddings/embedding.py

❌ Slow responses

Delete cache to refresh:

data/performance/exact_cache.json

❌ Tavily key missing

Double-check .env

# 🎨 Frontend

Your frontend uses:

Pure HTML

Inline CSS

Inline JS (fetch → POST → render response)

Works directly with FastAPI.

# 📈 Performance Notes

Local caching drastically improves repeated queries

FAISS index keeps retrieval <10ms

Groq inference delivers ~10–20 tokens/ms

Chunk size & overlap tuned for multilingual content

Preprocessing pipeline reduces noise and improves grounding

# 🧩 Future Expansions

Add streaming responses: Enable partial responses for faster user feedback on long outputs.

Add JWT-authenticated endpoints: Secure API access with token-based authentication.

Optimize semantic retrieval latency: Explore approximate nearest neighbor (ANN) indexing, vector pruning, or lightweight embedding models to reduce response time without sacrificing accuracy.

Adaptive semantic caching: Store embeddings or intermediate results for frequently asked queries to reduce repeated computation.

Batch processing for embedding queries: Process multiple queries together to leverage vectorization and speed up retrieval.

Parallelize expensive operations: Use async or multithreading where safe for model inference or external API calls.

Hybrid retrieval: Combine dense vector search with selective metadata filtering to limit candidate results early and speed up overall latency.