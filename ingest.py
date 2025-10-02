from pathlib import Path
import json
from raglab.config import Settings
from raglab.scrape import crawl
from raglab.chunk import chunk_docs
from raglab.embed import Embedder
from raglab.index import FaissIndex
from tqdm import tqdm

import argparse

ap = argparse.ArgumentParser() 
ap.add_argument("--config", required=True)
ap = ap.parse_args()

cfg = Settings.load(ap.config)
seeds = [l.strip() for l in open(cfg.seeds_file, "r", encoding="utf-8") if l.strip()]

# 1. Crawl
docs = crawl(seeds, cfg.seed_domain, Path("data"), max_pages=cfg.scrape["max_pages"], same_domain_only=cfg.scrape["same_domain_only"])
Path("data/processed").mkdir(parents=True, exist_ok=True)
with open("data/processed/docs.jsonl", "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n") # one json doc per line

# 2. Chunk
chunks = chunk_docs(docs, target_tokens=cfg.chunk["target_tokens"], overlap_tokens=cfg.chunk["overlap_tokens"])
Path("data/chunks").mkdir(exist_ok=True)
with open("data/processed/chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n") # one json chunk per line

# 3. Embed
embedder = Embedder(cfg.embeddings["model_name"], cfg.embeddings["batch_size"]) # Initialize embedder
texts = [chunk["text"] for chunk in chunks] # Extract texts from chunks
X = embedder.encode(texts) # Get embeddings

# 4. Index
meta = [{"url": chunk["url"], "title": chunk.get("title", ""), "text": chunk["text"][:2000]} for chunk in chunks] # Extract metadata from chunks
ix = FaissIndex(cfg.index["faiss_path"], cfg.index["meta_path"]) # Initialize index
ix.build(X, meta) # Build index
ix.save() # Save index
print(f"Ingest complete. {len(meta)} chunks indexed")