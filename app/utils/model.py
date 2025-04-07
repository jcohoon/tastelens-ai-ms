import os
import json
import numpy as np
import openai

# Dummy vector store — replace with Supabase or DB client in production
USER_VECTORS_PATH = "data/user_vectors.json"
ITEM_VECTORS_PATH = "data/item_vectors.json"

def load_user_vector(user_id):
    try:
        with open(USER_VECTORS_PATH, "r") as f:
            vectors = json.load(f)
        return np.array(vectors.get(user_id))
    except:
        return None

def load_item_vector(item_id):
    try:
        with open(ITEM_VECTORS_PATH, "r") as f:
            vectors = json.load(f)
        return np.array(vectors.get(item_id))
    except:
        return None

def predict_dot(user_vec, item_vec):
    return float(np.dot(user_vec, item_vec))

def get_all_item_vectors():
    try:
        with open(ITEM_VECTORS_PATH, "r") as f:
            vectors = json.load(f)
        return {k: np.array(v) for k, v in vectors.items()}
    except:
        return {}

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_reviews(user_id, item_id):
    # Dummy prompt — customize this for better results
    prompt = f"Summarize why user {user_id} would like item {item_id} based on their taste."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response['choices'][0]['message']['content'].strip()