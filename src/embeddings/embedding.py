# embedding.py
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilingual model SAME AS used in chunking.py
INPUT_JSON = r"D:/DIGITISED TASK/rag_system/data/chunks/semantic_chunked_dataset.json"
FAISS_INDEX_PATH = r"D:/DIGITISED TASK/rag_system/data/embeddings/faiss_index.bin"
METADATA_MAPPING_PATH = r"D:/DIGITISED TASK/rag_system/data/embeddings/metadata_mapping.json"
BATCH_SIZE = 64  # number of chunks per batch during embedding

# -------------------------------
# Initialize model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME)
model.to(device)

# -------------------------------
# Load chunked JSON
# -------------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------------------------------
# Generate embeddings with progress bar
# -------------------------------
chunk_texts = [entry["chunk_text"] for entry in chunks]
embeddings = []

print(f"Generating embeddings for {len(chunk_texts)} chunks...")
for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), desc="Embedding batches"):
    batch_texts = chunk_texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, device=device)
    embeddings.append(batch_embeddings)

# Combine all batches into a single numpy array
embeddings = np.vstack(embeddings)
print(f"Embeddings generated: {embeddings.shape}")

# -------------------------------
# Create FAISS index
# -------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# -------------------------------
# Save FAISS index
# -------------------------------
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"FAISS index saved at: {FAISS_INDEX_PATH}")

# -------------------------------
# Save metadata mapping
# -------------------------------
metadata_mapping = {
    i: {
        "chunk_id": chunks[i]["chunk_id"],
        "question": chunks[i]["question"],
        "short_answer": chunks[i]["short_answer"],
        "metadata": chunks[i]["metadata"]
    }
    for i in range(len(chunks))
}

os.makedirs(os.path.dirname(METADATA_MAPPING_PATH), exist_ok=True)
with open(METADATA_MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_mapping, f, ensure_ascii=False, indent=2)
print(f"Metadata mapping saved at: {METADATA_MAPPING_PATH}")
