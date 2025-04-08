print("🚀 Starting main.py")

from fastapi import FastAPI

app = FastAPI()

@app.get("/startup-test")
def startup_test():
    return {"status": "booted"}