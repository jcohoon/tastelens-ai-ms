print("🚀 Starting main.py")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import redis
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI()

@app.get("/startup-test")
def startup_test():
    return {"status": "booted"}

# ----- Redis setup -----
redis_url = os.getenv("REDIS_URL")
parsed_url = urlparse(redis_url)
redis_client = redis.Redis(
    host=parsed_url.hostname,
    port=parsed_url.port,
    username=parsed_url.username,
    password=parsed_url.password,
    decode_responses=True,
)

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
    from app.utils.model import load_user_vector, load_item_vector, predict_dot

    cache_key = f"rating:{req.user_id}:{req.item_id}"
    logging.info(f"Checking cache for predicted rating with key: {cache_key}")
    cached = redis_client.get(cache_key)
    if cached:
        logging.info(f"Cache hit for {cache_key}")
        return {"predicted_rating": float(cached)}

    logging.info(f"Cache miss for {cache_key}. Loading user and item vectors.")
    user_vector = load_user_vector(req.user_id)
    item_vector = load_item_vector(req.item_id)

    if user_vector is None or item_vector is None:
        logging.warning(f"User or item vector not found for user_id={req.user_id}, item_id={req.item_id}")
        raise HTTPException(status_code=404, detail="User or item vector not found")

    rating = predict_dot(user_vector, item_vector)
    logging.info(f"Predicted rating for user_id={req.user_id}, item_id={req.item_id}: {rating}")
    redis_client.setex(cache_key, 3600 * 6, rating)
    return {"predicted_rating": rating}

@app.post("/recommendations")
def recommend_items(req: RecommendationsRequest):
    from app.utils.model import load_user_vector, get_all_item_vectors, predict_dot

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
    from app.utils.model import summarize_reviews

    cache_key = f"summary:{req.user_id}:{req.item_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"summary": cached}

    try:
        summary = summarize_reviews(req.user_id, req.item_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    redis_client.setex(cache_key, 3600 * 24 * 7, summary)
    return {"summary": summary}

@app.post("/train_model")
def train_model():
    try:
        from app.utils.model import train_model_from_supabase
        logging.info("📦 Successfully imported train_model_from_supabase")
        train_model_from_supabase()
        logging.info("✅ Model training finished.")
        return {"status": "model updated"}
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")