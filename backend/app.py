
# app.py
from dotenv import load_dotenv
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
from fastapi.responses import HTMLResponse
from .service import RAGService  
from fastapi import FastAPI, HTTPException

from fastapi import Body
from .evaluate import Evaluator
from typing import List
from fastapi import Query

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# Access variables
retriever_faiss_path = os.getenv("RETRIEVER_FAISS_PATH")
retriever_metadata_path = os.getenv("RETRIEVER_METADATA_PATH")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")
llm_model_name = os.getenv("LLM_MODEL_NAME")
performance_cache_path = os.getenv("PERFORMANCE_CACHE_PATH")
tavily_api_key = os.getenv("TAVILY_API_KEY")
use_gpu = int(os.getenv("USE_GPU", 0))
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
csv_path = os.getenv("CSV_PATH", r"D:\DIGITISED TASK\rag_system\data\processed\Natural-Questions-with-metadata.csv")

print("RETRIEVER_FAISS_PATH:", retriever_faiss_path)
print("RETRIEVER_METADATA_PATH:", retriever_metadata_path)
print("EMBEDDING_MODEL_NAME:", embedding_model_name)
print("GROQ_API_KEY:", groq_api_key)
print("LLM_MODEL_NAME:", llm_model_name)
print("PERFORMANCE_CACHE_PATH:", performance_cache_path)
print("TAVILY_API_KEY:", tavily_api_key)
print("USE_GPU:", use_gpu)
print("ELEVENLABS_API_KEY:", elevenlabs_api_key)
print("CSV WITH METADATA PATH:", csv_path)

import base64
from .elevenlabs_tts import generate_tts_audio


app = FastAPI(title="Multilingual RAG System", version="1.0")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# -------------------------------
# Pydantic models for API input/output
# -------------------------------

class QuestionRequest(BaseModel):
    query: str
    top_k: Optional[int] = None  # optional override for retrieval top_k


class QuestionResponse(BaseModel):
    query: str
    response: Optional[str]
    latency: Optional[float]
    audio_base64: Optional[str] = None


class EvalQuery(BaseModel):
    query: str
    expected_answer: str

class EvalRequest(BaseModel):
    dataset: List[EvalQuery]
    random_sample: bool = False
    num_samples: int = 10

# -------------------------------
# Initialize RAGService
# -------------------------------
rag_service = RAGService(
    retriever_faiss_path=retriever_faiss_path,
    retriever_metadata_path=retriever_metadata_path,
    embedding_model_name=embedding_model_name,
    llm_api_key=groq_api_key,
    llm_model_name=llm_model_name,
    performance_cache_path=performance_cache_path,
    tavily_api_key=tavily_api_key,
    use_gpu=use_gpu == 1  # Convert int 0/1 to boolean
)




# -------------------------------
# API Endpoints
# -------------------------------


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = Path("static/index.html")
    if not index_path.exists():
        return HTMLResponse("<h1>Index file not found</h1>")
    return index_path.read_text(encoding="utf-8")

@app.post("/ask-question", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Override top_k if provided
        if request.top_k is not None:
            rag_service.retriever_top_k = request.top_k

        # Get RAG answer
        result = rag_service.ask_question(request.query)

        answer_text = result.get("response", "")

        # Generate TTS
        audio_bytes = generate_tts_audio(answer_text)

        # Encode audio as base64 for browser playback
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Add the audio string to response
        result["audio_base64"] = audio_b64

        if "latency" in result:
            result["latency"] = float(result["latency"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {e}")


@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify that the API and RAG system are running.
    Returns status and optional component checks.
    """
    try:
        # Check if the RAG service is initialized
        rag_status = "initialized" if rag_service else "not initialized"

    # Optionally, we can also test a quick retrieval call with a dummy query, maybe test_query = "ping" or " who's the weeknd?" 
        # to ensure the RAG system is responsive        
        # _ = rag_service.ask_question(test_query)

        return {
            "status": "ok",
            "rag_service": rag_status
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }



@app.get("/evaluate")
def evaluate(
    samples: int = 1000,
    answer_type: str = "both",
    random_sample: bool = False,
    similarity_threshold: float = 0.7  # for semantic matching threshold
):
    evaluator = Evaluator(
        rag_service=rag_service, 
        csv_path=csv_path,
        num_samples=samples,
        random_sample=random_sample,
        answer_type=answer_type,
        similarity_threshold=similarity_threshold
    )
    
    results = evaluator.run_evaluation()
    return results


