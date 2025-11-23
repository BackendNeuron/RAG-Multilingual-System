import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from wordfreq import word_frequency
import time

# -------------------------------
# 1️⃣ Load CSV
# -------------------------------
csv_path = r"D:\DIGITISED TASK\rag_system\data\processed\Natural-Questions-Cleaned.csv"
print("Loading CSV...")
df = pd.read_csv(csv_path)

if "question" not in df.columns:
    raise ValueError("Column 'question' not found in CSV")

questions = df["question"].tolist()
print(f"Loaded {len(questions)} questions from CSV.\n")

# -------------------------------
# 2️⃣ Load spaCy models
# -------------------------------
print("Loading spaCy models...")
nlp_type = spacy.load("xx_ent_wiki_sm")
nlp_diff = spacy.load(r"D:\DIGITISED TASK\rag_system\models\spacy_difficulty_model_fine_tuned_40iter")
print("SpaCy models loaded.\n")

# -------------------------------
# 3️⃣ Define Question Type Classifier
# -------------------------------
def classify_question_type(q):
    doc = nlp_type(q.strip())
    if not doc:
        return "other"
    first_token = doc[0].text.lower()
    if first_token in ["who", "what", "when", "where", "why", "how"]:
        return first_token
    else:
        return "other"

# -------------------------------
# 4️⃣ Define Difficulty + Rarity
# -------------------------------
def question_rarity_score(question):
    words = question.lower().split()
    rarities = [1 - word_frequency(w, 'en') for w in words if w.isalpha()]
    return sum(rarities)/len(rarities) if rarities else 0

def predict_difficulty_and_status(question):
    doc = nlp_diff(question)
    spacy_pred = max(doc.cats, key=doc.cats.get)
    rarity = question_rarity_score(question)
    status = ""
    if rarity > 0.6:
        status = "This question has rare words; might be harder than predicted!"
    return spacy_pred, status

# -------------------------------
# 5️⃣ Compute Question Type & Difficulty
# -------------------------------
print("Classifying question types and difficulty...")
q_types = []
difficulties = []
difficulty_statuses = []

for q in tqdm(questions, desc="Processing questions"):
    # Type
    q_types.append(classify_question_type(q))
    
    # Difficulty
    diff, status = predict_difficulty_and_status(q)
    difficulties.append(diff)
    difficulty_statuses.append(status)

df["Question_Type"] = q_types
df["Predicted_Difficulty"] = difficulties
df["Difficulty_Status"] = difficulty_statuses
print("Question type and difficulty classification completed.\n")

# -------------------------------
# 6️⃣ Domain Assignment (last step)
# Please Note: These domains are as found on official HuggingFace model nvidia/multilingual-domain-classifier
# -------------------------------
domains = [
    'Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 
    'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 
    'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 
    'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 
    'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 
    'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 
    'Shopping', 'Sports', 'Travel_and_Transportation'
]

print("Loading SentenceTransformer model for domain assignment...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Generating embeddings for domains...")
domain_embeddings = embed_model.encode(domains, convert_to_numpy=True)

print("Generating embeddings for questions...")
question_embeddings = []
for q in tqdm(questions, desc="Embedding questions"):
    emb = embed_model.encode(q, convert_to_numpy=True)
    question_embeddings.append(emb)
question_embeddings = np.array(question_embeddings)

print("Assigning domains based on cosine similarity...")
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

assigned_domains = []
for q_emb in tqdm(question_embeddings, desc="Assigning domain"):
    similarities = [cosine_similarity(q_emb, d_emb) for d_emb in domain_embeddings]
    closest_idx = np.argmax(similarities)
    assigned_domains.append(domains[closest_idx])

df["Assigned_Domain"] = assigned_domains
print("Domain assignment completed.\n")

# -------------------------------
# 7️⃣ Save CSV
# -------------------------------
output_path = r"D:\DIGITISED TASK\rag_system\data\processed\Natural-Questions-with-metadata.csv"
df.to_csv(output_path, index=False)
print(f"Enhanced CSV saved to {output_path}")
