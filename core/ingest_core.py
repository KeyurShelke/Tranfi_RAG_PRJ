import asyncio
import aiohttp
import aiofiles
import os
import json
import time
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_BATCH = 8
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# -----------------------------
# HELPERS
# -----------------------------

def is_internal(url):
    return urlparse(url).netloc.endswith("transfi.com")

def clean_text(html):
    soup = BeautifulSoup(html, "lxml")

    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        t = tag.get_text(" ", strip=True)
        if t:
            parts.append(t)

    return title, "\n".join(parts)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    i = 0
    L = len(text)

    while i < L:
        j = min(i + chunk_size, L)
        chunks.append(text[i:j].strip())
        i = max(j - overlap, j)
        if j == L:
            break

    return chunks

# -----------------------------
# ASYNC HTTP FETCH
# -----------------------------
async def fetch_html(session, url, timeout=25):
    try:
        async with session.get(url, timeout=timeout) as resp:
            text = await resp.text()
            return url, text, None
    except Exception as e:
        return url, None, str(e)

# -----------------------------
# ASYNC EMBEDDING
# -----------------------------
async def embed_texts(session, texts, model=DEFAULT_EMBED_MODEL):
    results = [None] * len(texts)
    sem = asyncio.Semaphore(8)

    async def _call(i, txt):
        async with sem:
            payload = {"model": model, "prompt": txt}
            try:
                async with session.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=60) as r:
                    data = await r.json()
                    emb = data.get("embedding") or data.get("embeddings") or data
                    results[i] = np.array(emb, dtype=np.float32)
            except Exception:
                results[i] = np.zeros(768, dtype=np.float32)

    tasks = [asyncio.create_task(_call(i, t)) for i, t in enumerate(texts)]
    await asyncio.gather(*tasks)
    return results

# -----------------------------
# MAIN INGEST FUNCTION (API + CLI)
# -----------------------------
async def ingest_urls(urls, embed_model=DEFAULT_EMBED_MODEL, concurrency=8):
    """
    Reusable ingestion core used by both CLI and API.
    Returns: metrics dictionary
    """

    os.makedirs("data/raw_html", exist_ok=True)
    os.makedirs("data/text", exist_ok=True)
    os.makedirs("index", exist_ok=True)

    t0 = time.time()
    pages_scraped = 0
    pages_failed = 0
    errors = []
    docs = []
    all_chunks = []
    chunk_meta = []
    total_tokens_est = 0

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:

        # -------------------------
        # FETCH ALL PAGES
        # -------------------------
        fetch_tasks = [fetch_html(session, u) for u in urls]

        for fut in asyncio.as_completed(fetch_tasks):
            url, html, err = await fut

            if err or not html:
                pages_failed += 1
                errors.append({url: err})
                continue

            pages_scraped += 1

            # Save raw
            slug = url.replace("https://", "").replace("http://", "").replace("/", "_")
            raw_path = f"data/raw_html/{slug}.html"
            async with aiofiles.open(raw_path, "w", encoding="utf-8") as f:
                await f.write(html)

            # Parse + chunk
            title, text = clean_text(html)
            text_path = f"data/text/{slug}.txt"
            async with aiofiles.open(text_path, "w", encoding="utf-8") as f:
                await f.write(text)

            chunks = chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                chunk_meta.append({"url": url, "title": title})
                total_tokens_est += len(c.split())

            short = text[:300].replace("\n", " ").strip()
            docs.append({
                "title": title,
                "url": url,
                "short_description": short,
                "long_description": text,
            })

    if not all_chunks:
        return {"error": "No chunks extracted."}

    # -------------------------
    # EMBEDDING BATCH
    # -------------------------
    emb_start = time.time()
    final_vecs = []

    session = aiohttp.ClientSession()
    try:
        for i in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[i:i + EMBED_BATCH]
            vecs = await embed_texts(session, batch, model=embed_model)
            final_vecs.extend(vecs)
    finally:
        await session.close()

    emb_time = time.time() - emb_start

    # -------------------------
    # SAVE INDEX
    # -------------------------
    emb_matrix = np.vstack(final_vecs).astype(np.float32)
    np.save("index/embeddings.npy", emb_matrix)

    meta = {"docs": docs, "chunk_meta": chunk_meta}
    with open("index/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # -------------------------
    # METRICS
    # -------------------------
    total_time = time.time() - t0
    metrics = {
        "total_time": total_time,
        "pages_scraped": pages_scraped,
        "pages_failed": pages_failed,
        "chunks": len(all_chunks),
        "tokens_estimate": total_tokens_est,
        "embedding_time": emb_time,
        "errors": errors,
    }

    return metrics
 