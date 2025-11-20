📌 Project Structure

TransFi/
│
├── core/
│ ├── ingest_core.py # Reusable ingestion pipeline (async)
│ ├── query_core.py # Reusable RAG + retrieval logic
│
├── data/
│ ├── raw_html/ # Saved HTML pages
│ ├── text/ # Cleaned text from pages
│
├── index/
│ ├── embeddings.npy # Vector index
│ ├── metadata.json # Chunk + doc metadata
│
├── scripts/
│ ├── api.py # FastAPI service
│ ├── webhook_receiver.py # Webhook callback server
│
├── ingest.py # CLI ingestion tool (Part 1)
├── query.py # CLI query tool (Part 1)
├── requirements.txt
└── README.md

---

⚡ Features

✅ Part 1 — CLI RAG Pipeline

Async web scraping using aiohttp

HTML → cleaned text → chunks

Batch async embeddings with Ollama

Vector search using cosine similarity

LLM answer generation with Ollama llama3

Pretty metrics + source citations

-

✅ Part 2 — FastAPI RAG Service

/api/ingest → background ingestion job

/api/query → single-question RAG

/api/query/batch → multi-question async RAG

Webhook-based ingestion completion callback

Fully async architecture using asyncio

---

🛠 Installation

## 1️⃣ Clone the repo

git clone <git@github.com:KeyurShelke/Tranfi_Rag.git>
cd TRANSFI_PROJECT

-

## 2️⃣ Create virtual environment

python3 -m venv .venv
source .venv/bin/activate # macOS/Linux

-

## 3️⃣ Install dependencies

## pip install -r requirements.txt

4️⃣ Install and run Ollama
Download Ollama → https://ollama.com/download
Then pull required models:

- ollama pull nomic-embed-text
  ollama pull llama3
- ***

🧩 PART 1 — CLI Ingestion & Querying

📥 1. Run Ingestion
This crawls TransFi pages, cleans text, chunks them, embeds using Ollama, and stores the index.

- python ingest.py --url https://www.transfi.com --concurrency 8
-

Example Output
=== Ingestion Metrics ===
Total Time (s): 82.01
Pages Scraped: 18
Pages Failed: 0
Total Chunks Created: 477
Embedding Generation Time (s): 79.24
Saved embeddings -> index/embeddings.npy
Saved metadata -> index/metadata.json

## 🔍 2. Run Query (Single)

## python query.py --question "What is BizPay?"

## 🔍 3. Run Query (Batch)

## python query.py --questions questions.txt --concurrent

Output Example
QUESTION: What is BizPay?

Based on the provided context...

--- SOURCES ---
[1] https://www.transfi.com/products/bizpay
Snippet: Unlock the world of borderless payments...

--- METRICS ---
Total Latency (s): 63.15
Embedding Time (s): 0.004
Retrieval Time (s): 0.002
LLM Time (s): 63.10

---

🚀 PART 2 — FastAPI Service

The system now exposes ingestion + query endpoints via REST APIs.

## 🖥️ Run Webhook Receiver (Terminal 1)

## python webhook_receiver.py --port 8001

Expected:

🚀 Webhook Receiver running on http://localhost:8001/webhook
Timestamp: ...
Payload: { "metrics": {...} }

---

## 🌐 Run FastAPI Server (Terminal 2)

## uvicorn api:app --port 8000 --reload

Expected:

Uvicorn running on http://127.0.0.1:8000

---

## 📡 Trigger Ingestion (Terminal 3)

curl -X POST http://localhost:8000/api/ingest \
 -H "Content-Type: application/json" \
 -d '{"urls": ["https://www.transfi.com"], "callback_url": "http://localhost:8001/webhook"}'

-

Immediate API response:

{"message": "Ingestion started", "will_callback_to": "http://localhost:8001/webhook"}

Later in Terminal 1:

Webhook received! { "metrics": {...} }

---

## Query Endpoint

curl -X POST http://localhost:8000/api/query \
 -H "Content-Type: application/json" \
 -d '{"question": "What is BizPay?"}'

-

Returns:

{
"answer": "...",
"sources": [...],
"metrics": {...}
}

---

🔥 Batch Query Endpoint
Sync mode:

- curl -X POST http://localhost:8000/api/query/batch \
   -H "Content-Type: application/json" \
   -d '{"questions": ["Q1","Q2"]}'
-

## Async mode (webhook):

curl -X POST http://localhost:8000/api/query/batch \
 -H "Content-Type: application/json" \
 -d '{"questions": ["Q1","Q2"], "callback_url": "http://localhost:8001/webhook"}'

- ***

Architecture Overview
🏗 Layered Design

API Layer (FastAPI)
↓
Service Layer (Part 2 logic)
↓
Core Logic (Part 1 ingestion + RAG)
↓
Ollama (Embeddings + LLM)

Async Everywhere

aiohttp for HTTP fetch + embeddings

asyncio for parallel tasks

FastAPI BackgroundTasks for ingestion

Webhooks for long-running job completion

🏁 Conclusion

This project implements a complete RAG pipeline with:

robust async ingestion

reproducible vector search

FastAPI microservice architecture

webhook-based decoupled execution
