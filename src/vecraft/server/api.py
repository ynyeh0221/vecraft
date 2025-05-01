
from fastapi import FastAPI

app = FastAPI(title="vecraft API")

@app.get("/health")
def health():
    return {"status": "ok"}
