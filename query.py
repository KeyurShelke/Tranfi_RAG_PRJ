
"""
Async-first query flow using Ollama embeddings + Ollama LLM (non-streaming).
Usage:
 python query.py --question "What is BizPay?"
 python query.py --questions questions.txt --concurrent
"""

import argparse
import asyncio
import aiohttp
import json
import numpy as np
import time
import textwrap

EMB_FILE = "index/embeddings.npy"
META_FILE = "index/metadata.json"

OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_ENDPOINT = "http://localhost:11434/api/generate"


# ----------------------------------------------------
# ASYNC OLLAMA HELPERS
# ----------------------------------------------------
async def embed_query(session, text, model="nomic-embed-text"):
    """Generate embedding using Ollama (async, JSON)."""
    payload = {"model": model, "prompt": text}

    async with session.post(
        OLLAMA_EMBED_ENDPOINT,
        json=payload,
        timeout=30
    ) as r:
        data = await r.json()
        emb = data["embedding"]
        return np.array(emb, dtype=np.float32)


async def generate_answer(session, prompt, model="llama3", max_tokens=512):
    """
    FINAL FIX: Correct NON-STREAMING Ollama call.
    - Use `stream=False` INSIDE JSON payload.
    - Do NOT append ?stream=false to URL.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False   # <-- CRITICAL FIX
    }

    async with session.post(
        OLLAMA_GEN_ENDPOINT,
        json=payload,
        timeout=120
    ) as r:
        data = await r.json()

        # Depending on model, output can be under different keys
        return (
            data.get("response")
            or data.get("output")
            or data.get("text")
            or json.dumps(data)
        )


# ----------------------------------------------------
# LOAD INDEX + RETRIEVAL
# ----------------------------------------------------
def load_index():
    emb = np.load(EMB_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # normalize embeddings for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms

    return emb_norm, meta


def search_topk(emb_matrix, qvec, topk=5):
    q_norm = qvec / (np.linalg.norm(qvec) + 1e-12)
    sims = emb_matrix.dot(q_norm)
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]


# ----------------------------------------------------
# PROMPT BUILDER
# ----------------------------------------------------
def build_prompt(question, retrieved_indices, meta):
    prompt = (
        "You are an assistant answering ONLY using the provided context.\n"
        "Use citations like [1], [2], etc.\n\n"
        f"Question:\n{question}\n\n"
        "Context:\n"
    )

    for i, idx in enumerate(retrieved_indices, start=1):
        url = meta["chunk_meta"][idx]["url"]

        snippet = ""
        for doc in meta["docs"]:
            if doc["url"] == url:
                snippet = doc["long_description"][:500].replace("\n", " ")
                break

        prompt += f"[{i}] Source: {url}\nSnippet: {snippet}\n\n"

    prompt += "\nAnswer (include citations like [1], [2]):\n"
    return prompt


# ----------------------------------------------------
# PROCESS SINGLE QUESTION
# ----------------------------------------------------
async def answer_single(question, embed_model, llm_model, topk):
    emb_norm, meta = load_index()

    async with aiohttp.ClientSession() as session:
        t_start = time.time()

        # 1. Embed query
        q_emb = await embed_query(session, question, model=embed_model)
        t_embed = time.time()

        # 2. Retrieve top-k
        idxs, sims = search_topk(emb_norm, q_emb, topk=topk)
        t_retr = time.time()

        # 3. Prepare prompt
        prompt = build_prompt(question, idxs, meta)

        # 4. Generate answer from Ollama
        answer = await generate_answer(session, prompt, model=llm_model)
        t_llm = time.time()

    metrics = {
        "Total Latency (s)": round(t_llm - t_start, 3),
        "Embedding Time (s)": round(t_embed - t_start, 3),
        "Retrieval Time (s)": round(t_retr - t_embed, 3),
        "LLM Time (s)": round(t_llm - t_retr, 3),
        "Top-K Returned": len(idxs),
    }

    sources = []
    for rank, idx in enumerate(idxs, start=1):
        url = meta["chunk_meta"][idx]["url"]
        snippet = ""
        for d in meta["docs"]:
            if d["url"] == url:
                snippet = d["long_description"][:300].replace("\n", " ")
                break

        sources.append({
            "rank": rank,
            "url": url,
            "snippet": snippet[:250] + "...",
            "score": float(sims[rank - 1])
        })

    return answer, sources, metrics


# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
async def main_loop(questions, embed_model, llm_model, topk, concurrent):
    if concurrent:
        tasks = [
            answer_single(q, embed_model, llm_model, topk)
            for q in questions
        ]
        results = await asyncio.gather(*tasks)

    else:
        results = []
        for q in questions:
            r = await answer_single(q, embed_model, llm_model, topk)
            results.append(r)

    # Pretty output
    for i, (answer, sources, metrics) in enumerate(results):
        print("\n====================================")
        print(f"QUESTION: {questions[i]}")
        print("====================================\n")

        print(textwrap.fill(answer, width=110))

        print("\n--- SOURCES ---")
        for s in sources:
            print(f"[{s['rank']}] {s['url']}")
            print(f"   Snippet: {s['snippet']}\n")

        print("--- METRICS ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--questions", type=str)
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text")
    parser.add_argument("--llm-model", type=str, default="llama3")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()

    # Load question(s)
    qlist = []
    if args.question:
        qlist = [args.question]
    elif args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            qlist = [line.strip() for line in f if line.strip()]
    else:
        print("❗ Please provide --question or --questions")
        exit()

    asyncio.run(
        main_loop(qlist, args.embed_model, args.llm_model, args.topk, args.concurrent)
    )

# =====================================================================
# FUNCTIONS FOR REUSE BY API (REQUIRED FOR PART 2)
# =====================================================================

async def api_answer_single(question: str, embed_model="nomic-embed-text", llm_model="llama3", topk=5):
    """
    Reuses the same logic as the CLI version of answer_single,
    but returns structured JSON for the FastAPI service.
    """
    answer, sources, metrics = await answer_single(
        question,
        embed_model=embed_model,
        llm_model=llm_model,
        topk=topk
    )

    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics
    }


async def api_answer_batch(questions: list[str], embed_model="nomic-embed-text", llm_model="llama3", topk=5):
    """
    Runs multiple questions concurrently.
    """
    tasks = [
        api_answer_single(q, embed_model, llm_model, topk)
        for q in questions
    ]
    results = await asyncio.gather(*tasks)

    return {
        "results": results,
        "total_questions": len(results)
    }
