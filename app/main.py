# ai_service/app/main.py
print("🚀 Starting main.py")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import redis
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

from app.utils.model import (
    load_user_vector,
    load_item_vector,
    predict_dot,
    get_all_item_vectors,
    summarize_reviews,
    train_model_from_supabase
)

load_dotenv()

# Redis setup
redis_url = os.getenv("REDIS_URL")
parsed_url = urlparse(redis_url)
redis_client = redis.Redis(
    host=parsed_url.hostname,
    port=parsed_url.port,
    username=parsed_url.username,
    password=parsed_url.password,
    decode_responses=True,
)

app = FastAPI()

# ----- Request Models -----
class RatingRequest(BaseModel):
    user_id: str
    item_id: str

class RecommendationsRequest(BaseModel):
    user_id: str
    top_k: int = 10

class SummaryRequest(BaseModel):
    user_id: str
    item_id: str

# ----- Routes -----
@app.get("/health")
def health_check():
    try:
        redis_client.set("healthcheck", "ok", ex=60)
        result = redis_client.get("healthcheck")
        return {"status": "ok", "redis": result == "ok"}
    except Exception as e:
        return {"status": "error", "redis_error": str(e)}

@app.post("/predict_rating")
def predict_rating(req: RatingRequest):
    cache_key = f"rating:{req.user_id}:{req.item_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"predicted_rating": float(cached)}

    user_vector = load_user_vector(req.user_id)
    item_vector = load_item_vector(req.item_id)

    if user_vector is None or item_vector is None:
        raise HTTPException(status_code=404, detail="User or item vector not found")

    rating = predict_dot(user_vector, item_vector)
    redis_client.setex(cache_key, 3600 * 6, rating)
    return {"predicted_rating": rating}

@app.post("/recommendations")
def recommend_items(req: RecommendationsRequest):
    cache_key = f"recs:{req.user_id}:{req.top_k}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"recommendations": cached.split(",")}

    user_vector = load_user_vector(req.user_id)
    if user_vector is None:
        raise HTTPException(status_code=404, detail="User vector not found")

    item_vectors = get_all_item_vectors()
    scored = [(item_id, predict_dot(user_vector, vec)) for item_id, vec in item_vectors.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_items = [item_id for item_id, _ in scored[:req.top_k]]

    redis_client.setex(cache_key, 3600 * 12, ",".join(top_items))
    return {"recommendations": top_items}

@app.post("/summarize")
def summarize(req: SummaryRequest):
    cache_key = f"summary:{req.user_id}:{req.item_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"summary": cached}

    summary = summarize_reviews(req.user_id, req.item_id)
    redis_client.setex(cache_key, 3600 * 24 * 7, summary)
    return {"summary": summary}

import logging
logging.basicConfig(level=logging.INFO)

@app.get("/startup-test")
def startup_test():
    return {"status": "booted"}

@app.post("/train_model")
def train_model():
    try:
        logging.info("🔥 Received /train_model call")

        try:
            from app.utils.model import train_model_from_supabase
            logging.info("📦 Successfully imported train_model_from_supabase")
        except Exception as import_err:
            logging.error(f"❌ Import failed: {import_err}")
            raise HTTPException(status_code=500, detail="Import error")

        train_model_from_supabase()
        logging.info("✅ Model training finished.")
        return {"status": "model updated"}

    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


# Supabase Endpoints
from app.utils.supabase_client import supabase

def get_ratings():
    response = supabase.table("ratings").select("*").execute()
    return response.data  # List of dicts with user_id, item_id, rating

def get_user_vector(user_id):
    res = supabase.table("user_vectors").select("vector").eq("user_id", user_id).execute()
    if res.data:
        return res.data[0]["vector"]
    return None

def get_item_vector(item_id):
    res = supabase.table("item_vectors").select("vector").eq("item_id", item_id).execute()
    if res.data:
        return res.data[0]["vector"]
    return None

def save_user_vector(user_id, vector):
    supabase.table("user_vectors").upsert({
        "user_id": user_id,
        "vector": vector
    }).execute()