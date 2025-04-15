# ai_service/app/utils/model.py

import os
import numpy as np
import openai
import pandas as pd
import json
from surprise import Dataset, Reader, SVD
from app.utils.supabase_client import supabase
import logging

openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------
# Supabase-based Training
# ------------------------
def train_model_from_supabase():
    level=logging.INFO,  # Or DEBUG if you want more
    logging.info("🔍 Pulling ratings from Supabase...")

    response = supabase.table("ratings").select("user_id, item_id, rating").execute()
    df = pd.DataFrame(response.data)
    logging.info(f"📊 Loaded {len(df)} ratings.")

    if df.empty:
        raise Exception("No ratings found")

    logging.info("⚙️ Training SVD model...")

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
    logging.info("✅ Upserting user and item vectors...")
    for iid in trainset.all_items():
        raw_id = trainset.to_raw_iid(iid)
        vector = algo.qi[iid].tolist()
        supabase.table("item_vectors").upsert({"item_id": raw_id, "vector": vector}).execute()

# ------------------------
# Supabase-based Retrieval
# ------------------------
def load_user_vector(user_id):
    logging.info(f"Loading user vector for user_id={user_id}")
    res = supabase.table("user_vectors").select("vector").eq("user_id", user_id).execute()
    if res.data:
        logging.info(f"User vector found for user_id={user_id}")
        return np.array(res.data[0]["vector"])
    logging.warning(f"No user vector found for user_id={user_id}")
    return None

def load_item_vector(item_id):
    logging.info(f"Loading item vector for item_id={item_id}")
    res = supabase.table("item_vectors").select("vector").eq("item_id", item_id).execute()
    if res.data:
        logging.info(f"Item vector found for item_id={item_id}")
        return np.array(res.data[0]["vector"])
    logging.warning(f"No item vector found for item_id={item_id}")
    return None

def get_all_item_vectors():
    logging.info("Loading all item vectors")
    res = supabase.table("item_vectors").select("item_id, vector").execute()
    if res.data:
        logging.info(f"Loaded {len(res.data)} item vectors")
        return {r["item_id"]: np.array(r["vector"]) for r in res.data}
    logging.warning("No item vectors found")
    return {}

def get_user_ratings(user_id):
    logging.info(f"Fetching ratings for user_id={user_id}")
    res = supabase.table("ratings") \
        .select("item_id, review_text, rating") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(5) \
        .execute()

    if not res.data:
        logging.warning(f"No ratings found for user_id={user_id}")
        return "No available ratings from this user."

    review_texts = []
    for r in res.data:
        rating_val = float(r['rating']) if r.get('rating') else None
        rating_str = f"{rating_val:.1f}" if rating_val is not None else "No rating"
        review_text = r.get('review_text') or "No review text"
        review_texts.append(
            f"- Item {r['item_id']}: rated {rating_str}/5 — \"{review_text}\""
        )

    logging.info(f"Fetched {len(review_texts)} ratings for user_id={user_id}")
    return "\n".join(review_texts)

def get_item_details(item_id):
    res = supabase.table("items") \
        .select("title, description, metadata") \
        .eq("id", item_id) \
        .single() \
        .execute()

    if not res.data:
        return "No information found for this item."

    item = res.data
    details = f"Title: {item.get('title', 'Unknown')}\nDescription: {item.get('description', 'No description.')}"
    if "metadata" in item and item["metadata"]:
        details += f"\nMetadata: {', '.join(item['metadata'])}"
    return details

# ------------------------
# Prediction & Summarization
# ------------------------
def predict_dot(user_vec, item_vec):
    raw_score = float(np.dot(user_vec, item_vec))
    # Clamp to 0-5 scale
    return max(0.0, min(5.0, raw_score))

def summarize_reviews(user_id, item_id):
    user_reviews = get_user_ratings(user_id)
    item_details = get_item_details(item_id)

    prompt = (
        f"Based on my rating and review history:\n{user_reviews}\n\n"
        f"And this item's description and metadata:\n{item_details}\n\n"
        "Explain why I might like this item.\n\n"
        "Be concise and only refer to my past reviews and ratings generally.\n\n"
        "For example, you can say something about the titles, themes or tones that the user has liked in the past.\n"
        "If the user has rated items with similar themes or tones, mention that.\n"
        "If the user has not provided any or very few reviews, don't mention it.\n"
        "If the user has only provided a few ratings, you can contextualize the recommendation with that.\n"
        
    )

    response = openai.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.7,
    )

    # Extract the actual text response
    summary_text = response.output[0].content[0].text
    return summary_text.strip()