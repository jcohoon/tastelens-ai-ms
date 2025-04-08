# ai_service/app/utils/model.py

import os
import numpy as np
import openai
import pandas as pd
import json
from surprise import Dataset, Reader, SVD
from app.utils.supabase_client import supabase

openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------
# Supabase-based Training
# ------------------------
def train_model_from_supabase():
    response = supabase.table("ratings").select("user_id, item_id, rating").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        raise Exception("No ratings found")

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # Upsert user_vectors
    for uid in trainset.all_users():
        raw_id = trainset.to_raw_uid(uid)
        vector = algo.pu[uid].tolist()
        supabase.table("user_vectors").upsert({"user_id": raw_id, "vector": vector}).execute()

    # Upsert item_vectors
    for iid in trainset.all_items():
        raw_id = trainset.to_raw_iid(iid)
        vector = algo.qi[iid].tolist()
        supabase.table("item_vectors").upsert({"item_id": raw_id, "vector": vector}).execute()

# ------------------------
# Supabase-based Retrieval
# ------------------------
def load_user_vector(user_id):
    res = supabase.table("user_vectors").select("vector").eq("user_id", user_id).execute()
    if res.data:
        return np.array(res.data[0]["vector"])
    return None

def load_item_vector(item_id):
    res = supabase.table("item_vectors").select("vector").eq("item_id", item_id).execute()
    if res.data:
        return np.array(res.data[0]["vector"])
    return None

def get_all_item_vectors():
    res = supabase.table("item_vectors").select("item_id, vector").execute()
    if res.data:
        return {r["item_id"]: np.array(r["vector"]) for r in res.data}
    return {}

# ------------------------
# Prediction & Summarization
# ------------------------
def predict_dot(user_vec, item_vec):
    return float(np.dot(user_vec, item_vec))

def summarize_reviews(user_id, item_id):
    prompt = f"Summarize why user {user_id} would like item {item_id} based on their taste."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()
