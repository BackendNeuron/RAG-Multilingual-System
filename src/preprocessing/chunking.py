# semantic_chunking.py
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
nltk.download("punkt")

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS_PER_CHUNK = 400
OVERLAP_TOKENS = 80
SIMILARITY_THRESHOLD = 0.85
INPUT_CSV = r"D:\DIGITISED TASK\rag_system\data\processed\Natural-Questions-with-metadata.csv"
OUTPUT_JSON = r"D:/DIGITISED TASK/rag_system/data/chunks/semantic_chunked_dataset.json"

# -------------------------------
# Initialize model
# -------------------------------
model = SentenceTransformer(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------------------
# Helper functions
# -------------------------------
def split_into_sentences(text):
    return nltk.sent_tokenize(text)


def embed_sentences(sentences):
    return model.encode(sentences, convert_to_numpy=True, device=device)


def chunk_by_semantic(sentences, max_tokens=MAX_TOKENS_PER_CHUNK, overlap=OVERLAP_TOKENS):
    embeddings = embed_sentences(sentences)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sent in enumerate(sentences):

        # Estimate token count using whitespace split
        sent_tokens = len(sent.split())

        # If single sentence too large, save as standalone chunk
        if sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(sent)
            current_chunk = []
            current_tokens = 0
            continue

        # If current chunk is not empty, check semantic similarity
        if current_chunk:
            last_embedding = embeddings[i - 1]
            curr_embedding = embeddings[i]
            sim = cosine_similarity([last_embedding], [curr_embedding])[0][0]

            # New chunk required
            if sim < SIMILARITY_THRESHOLD or current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))

                # Overlap last sentences
                overlap_sentences = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_sentences.copy()
                current_tokens = sum(len(s.split()) for s in current_chunk)

        # Add sentence
        current_chunk.append(sent)
        current_tokens += sent_tokens

    # Save last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(INPUT_CSV)
chunked_data = []

# -------------------------------
# Process rows
# -------------------------------
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    question = row.get("question", "")
    short_answer = row.get("short_answers", "")
    long_answer = row.get("long_answers", "")
    metadata = {k: row[k] for k in row.index if k not in ["question", "short_answers", "long_answers"]}

    # If long answer exists, chunk it
    if pd.notna(long_answer) and len(long_answer.strip()) > 0:
        sentences = split_into_sentences(long_answer)
        chunks = chunk_by_semantic(sentences)
    else:
        # Otherwise use short answer
        chunks = [short_answer] if pd.notna(short_answer) else []

    for i, chunk in enumerate(chunks):
        entry = {
            "question": question,
            "chunk_id": f"{idx}_{i}",
            "chunk_text": chunk,
            "short_answer": short_answer,
            "metadata": metadata
        }
        chunked_data.append(entry)

# -------------------------------
# Save JSON
# -------------------------------
import os
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=2)

print(f"Semantic chunking complete. {len(chunked_data)} chunks saved to {OUTPUT_JSON}")
