# ingest.py
"""
Async-first ingestion for TransFi RAG using Ollama embeddings.
Saves:
 - data/raw_html/<slug>.html
 - data/text/<slug>.txt
 - index/embeddings.npy
 - index/metadata.json

Run:
 python ingest.py --url https://www.transfi.com --concurrency 8 --embed-model nomic-embed-text
"""
import asyncio
import aiohttp
import aiofiles
import time
import json
import os
import math
from bs4 import BeautifulSoup
import numpy as np
from urllib.parse import urljoin, urlparse
from tqdm.asyncio import tqdm

# Config (CLI overrides available)
BASE_START = "https://www.transfi.com"
OUT_RAW = "data/raw_html"
OUT_TEXT = "data/text"
INDEX_DIR = "index"
EMB_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
META_FILE = os.path.join(INDEX_DIR, "metadata.json")

CHUNK_SIZE = 800     # characters per chunk (coarse)
CHUNK_OVERLAP = 200
EMBED_BATCH = 8      # embed requests in parallel (not too large)
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
CRAWL_PATHS = ["/products", "/solutions"]  # pages to target

os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_TEXT, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# utils
def is_internal(url):
    return urlparse(url).netloc.endswith("transfi.com")

def clean_text(html):
    soup = BeautifulSoup(html, "lxml")
    # remove scripts/styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    # Collect paragraphs and headings
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

# async HTTP helpers
async def fetch_html(session, url, timeout=25):
    try:
        async with session.get(url, timeout=timeout) as resp:
            text = await resp.text()
            return url, text, None
    except Exception as e:
        return url, None, str(e)

async def gather_seed_urls(session, start_url, concurrency=8):
    # Get pages linked from start that likely contain products/solutions (one-level)
    urls = set()
    try:
        _, html, err = await fetch_html(session, start_url)
        if not html:
            return urls
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                full = urljoin(start_url, href)
            elif href.startswith("http"):
                full = href
            else:
                continue
            if is_internal(full) and any(p in full.lower() for p in CRAWL_PATHS):
                urls.add(full)
    except Exception:
        pass
    # Always include start page
    urls.add(start_url)
    return urls

# Ollama embedding (async)
async def embed_texts(session, texts, model=DEFAULT_EMBED_MODEL):
    """
    Accepts list of texts; sends multiple concurrent requests (bounded).
    Returns list of vectors (numpy float32 arrays).
    """
    results = [None] * len(texts)
    sem = asyncio.Semaphore(8)

    async def _call(i, txt):
        async with sem:
            payload = {"model": model, "prompt": txt}
            try:
                async with session.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=60) as r:
                    data = await r.json()
                    emb = data.get("embedding") or data.get("embeddings") or data
                    arr = np.array(emb, dtype=np.float32)
                    results[i] = arr
            except Exception as e:
                results[i] = None

    tasks = [asyncio.create_task(_call(i, t)) for i, t in enumerate(texts)]
    await asyncio.gather(*tasks)
    return results

async def ingest(start_url=BASE_START, concurrency=8, embed_model=DEFAULT_EMBED_MODEL):
    t0 = time.time()
    pages_scraped = 0
    pages_failed = 0
    errors = []
    docs = []
    all_chunks = []
    chunk_meta = []
    total_tokens_est = 0

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        # 1) discover candidate pages (shallow)
        print("Discovering product/solution pages...")
        seed_urls = await gather_seed_urls(session, start_url, concurrency)
        seed_urls = sorted(seed_urls)
        print("Candidate URLs:", len(seed_urls))

        # 2) fetch concurrently the candidate URLs
        fetch_tasks = [fetch_html(session, u) for u in seed_urls]
        for fut in tqdm(asyncio.as_completed(fetch_tasks), total=len(fetch_tasks)):
            try:
                url, html, err = await fut
            except Exception as e:
                continue
            if err or not html:
                pages_failed += 1
                errors.append({url: err})
                continue
            pages_scraped += 1
            # Save raw html
            slug = url.replace("https://", "").replace("http://", "").replace("/", "_")
            raw_path = os.path.join(OUT_RAW, f"{slug}.html")
            async with aiofiles.open(raw_path, "w", encoding="utf-8") as f:
                await f.write(html)

            # Parse and extract
            title, text = clean_text(html)
            short = text[:300].replace("\n", " ").strip()
            text_path = os.path.join(OUT_TEXT, f"{slug}.txt")
            async with aiofiles.open(text_path, "w", encoding="utf-8") as f:
                await f.write(text)

            # chunk
            chunks = chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                chunk_meta.append({"url": url, "title": title})
                total_tokens_est += len(c.split())

            docs.append({"title": title, "url": url, "short_description": short, "long_description": text})

    if not all_chunks:
        print("No chunks found — exiting.")
        return

    # 3) batch embeddings (async)
    print(f"Total chunks: {len(all_chunks)}. Generating embeddings (async)...")
    emb_start = time.time()
    # embed in batches to avoid giant parallelism
    B = EMBED_BATCH
    embeddings_out = []
    for i in range(0, len(all_chunks), B):
        batch = all_chunks[i:i+B]
        vecs = await embed_texts(aiohttp.ClientSession(), batch, model=embed_model)
        # embed_texts uses its own session; but we call with new session inside function for safety
        # however to avoid resource leak, embed_texts created session for each call above - OK
        # handle results
        for v in vecs:
            if v is None:
                # fallback zero vector
                v = np.zeros(768, dtype=np.float32)
            embeddings_out.append(v)
    emb_time = time.time() - emb_start

    # 4) indexing — simple numpy matrix
    print("Indexing embeddings...")
    idx_start = time.time()
    emb_matrix = np.vstack(embeddings_out).astype(np.float32)
    np.save(EMB_FILE, emb_matrix)
    meta = {"docs": docs, "chunk_meta": chunk_meta}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    idx_time = time.time() - idx_start

    total_time = time.time() - t0
    metrics = {
        "Total Time (s)": round(total_time, 2),
        "Pages Scraped": pages_scraped,
        "Pages Failed": pages_failed,
        "Total Chunks Created": len(all_chunks),
        "Total Tokens (approx)": total_tokens_est,
        "Embedding Generation Time (s)": round(emb_time, 2),
        "Indexing Time (s)": round(idx_time, 2),
        "Errors": errors,
        "Average Scraping Time per Page (s)": round(total_time / pages_scraped if pages_scraped else 0, 2)
    }

    print("\n=== Ingestion Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nSaved embeddings ->", EMB_FILE)
    print("Saved metadata ->", META_FILE)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=BASE_START)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()
    asyncio.run(ingest(start_url=args.url, concurrency=args.concurrency, embed_model=args.embed_model))
