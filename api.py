# api.py
"""
FastAPI service for Part 2.
Exposes:
 - POST /api/ingest
 - POST /api/query
 - POST /api/query/batch

Reuses Part 1 logic (ingest.py + query.py) without duplication.
"""

import asyncio
import json
import httpx
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# === IMPORT PART 1 LOGIC ===
from ingest import ingest  # async ingestion function
from query import api_answer_single, api_answer_batch



# =======================================================
#               FASTAPI INITIALIZATION
# =======================================================

app = FastAPI(title="TransFi RAG FastAPI Service")

# =====================================================================
#                         API MODELS
# =====================================================================
class IngestRequest(BaseModel):
    urls: list[str]
    callback_url: str


class QueryRequest(BaseModel):
    question: str
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3"
    topk: int = 5


class BatchQueryRequest(BaseModel):
    questions: list[str]
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3"
    topk: int = 5
    callback_url: str | None = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================================================
#               REQUEST MODELS
# =======================================================

class IngestRequest(BaseModel):
    urls: list[str]
    callback_url: str


class QueryRequest(BaseModel):
    question: str


class BatchQueryRequest(BaseModel):
    questions: list[str]
    callback_url: str | None = None


# =======================================================
#               WEBHOOK UTILITY
# =======================================================

async def send_webhook(callback_url: str, payload: dict):
    """
    Sends POST to webhook receiver.
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(callback_url, json=payload)
        print(f"Webhook sent → {callback_url}")
    except Exception as e:
        print(f"Webhook failed → {callback_url}: {e}")


# =======================================================
#               BACKGROUND INGESTION TASK
# =======================================================

async def run_ingestion(urls: list[str], callback_url: str):
    metrics_all = []

    for url in urls:
        print(f"Starting ingestion for → {url}")
        m = await ingest(start_url=url, concurrency=8, embed_model="nomic-embed-text")
        metrics_all.append({"url": url, "metrics": m})

    await send_webhook(callback_url, {"metrics": metrics_all})


# =======================================================
#               INGEST ENDPOINT
# =======================================================

@app.post("/api/ingest")
async def api_ingest(body: IngestRequest, bg: BackgroundTasks):

    # Start background job
    bg.add_task(run_ingestion, body.urls, body.callback_url)

    return {
        "message": "Ingestion started",
        "urls": body.urls,
        "will_callback_to": body.callback_url
    }


# =======================================================
#               SINGLE QUERY ENDPOINT
# =======================================================

@app.post("/api/query")
async def api_query(body: QueryRequest):
    result = await api_answer_single(body.question)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metrics": result["metrics"],
    }


# =======================================================
#               BATCH QUERY ENDPOINT
# =======================================================

@app.post("/api/query/batch")
async def api_query_batch(body: BatchQueryRequest, bg: BackgroundTasks):

    if body.callback_url:
        # async mode — respond immediately, compute later
        async def run_batch_and_send():
            result = await api_answer_batch(body.questions)
            await send_webhook(body.callback_url, {"metrics": result})

        bg.add_task(run_batch_and_send)

        return {
            "message": "Batch query started",
            "questions": len(body.questions),
            "will_callback_to": body.callback_url
        }

    # synchronous mode
    result = await api_answer_single(body.question)


    return result


# =======================================================
#               RUN SERVER
# =======================================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
