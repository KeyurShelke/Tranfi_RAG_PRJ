import numpy as np
import aiohttp
import asyncio
import json
import time

OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"

# ------------------------------------
# LOAD INDEX
# ------------------------------------
def load_index():
    emb = np.load("index/embeddings.npy")
    with open("index/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, meta["docs"], meta["chunk_meta"]

# ------------------------------------
# COSINE SIMILARITY SEARCH
# ------------------------------------
def top_k_cosine(query_vec, doc_vecs, k=5):
    # Normalize
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    D = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

    sims = D @ q
    idxs = np.argsort(sims)[::-1][:k]
    return idxs, sims[idxs]

# ------------------------------------
# EMBED QUESTION
# ------------------------------------
async def embed_query(text, model=DEFAULT_EMBED_MODEL):
    async with aiohttp.ClientSession() as session:
        payload = {"model": model, "prompt": text}
        async with session.post(OLLAMA_EMBED_ENDPOINT, json=payload) as r:
            data = await r.json()
            emb = data.get("embedding") or data.get("embeddings")
            return np.array(emb, dtype=np.float32)

# ------------------------------------
# BUILD PROMPT
# ------------------------------------
def build_prompt(question, docs, metas, idxs):
    context_blocks = []

    for rank, i in enumerate(idxs):
        meta = metas[i]
        doc = docs[i]
        snippet = doc["long_description"][:400].replace("\n", " ")

        context_blocks.append(
            f"[{rank+1}] URL: {meta['url']}\nSnippet: {snippet}\n"
        )

    context = "\n".join(context_blocks)

    prompt = f"""
You are a helpful RAG assistant. Use ONLY the context below.

Question: {question}

Context:
{context}

Provide a concise, factual answer with citations [1], [2], etc.
"""
    return prompt

# ------------------------------------
# CALL LLM
# ------------------------------------
async def call_llm(prompt, model=DEFAULT_LLM_MODEL):
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        async with session.post(OLLAMA_CHAT_ENDPOINT, json=payload) as r:
            out = await r.json()
            return out.get("message", {}).get("content", "")

# ------------------------------------
# FULL QUERY PIPELINE
# ------------------------------------
async def answer_question(question, k=5):
    emb_matrix, docs, metas = load_index()

    metrics = {}
    t0 = time.time()

    # Embed query
    t_emb = time.time()
    qvec = await embed_query(question)
    metrics["embedding_time"] = round(time.time() - t_emb, 4)

    # Retrieve
    t_ret = time.time()
    idxs, sims = top_k_cosine(qvec, emb_matrix, k=k)
    metrics["retrieval_time"] = round(time.time() - t_ret, 4)

    # Prompt
    prompt = build_prompt(question, docs, metas, idxs)

    # LLM
    t_llm = time.time()
    answer = await call_llm(prompt)
    metrics["llm_time"] = round(time.time() - t_llm, 4)

    metrics["total_time"] = round(time.time() - t0, 4)
    metrics["top_k"] = k

    # Build sources
    sources = []
    for rank, i in enumerate(idxs):
        meta = metas[i]
        doc = docs[i]
        snippet = doc["long_description"][:300].replace("\n", " ")
        sources.append({
            "rank": rank+1,
            "url": meta["url"],
            "snippet": snippet,
            "similarity": float(sims[rank])
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "metrics": metrics
    }
