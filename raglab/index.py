# Class for managing a FAISS index with metadata

import json, os
from pathlib import Path
import numpy as np
import faiss
from typing import List, Dict

class FaissIndex:
    def __init__(self, index_path: str, meta_path: str):
        self.index_path, self.meta_path = Path(index_path), Path(meta_path) # Paths for index and metadata
        self.index = None # FAISS index
        self.meta: list[Dict] = [] # List to hold metadata for each vector

    # Build the FAISS index from embeddings and associated metadata
    def build (self, embeddings: np.ndarray, meta: List[Dict]):
        dim = embeddings.shape[1] # Dimension of embeddings
        self.index = faiss.IndexFlatIP(dim) # Create FAISS index for inner product
        self.index.add(embeddings.astype('float32')) # Add embeddings to index
        self.meta = meta # Store metadata
    
    # Save the FAISS index and metadata to disk
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        faiss.write_index(self.index, str(self.index_path)) # Save FAISS index
        with open(self.meta_path, "w", encoding="utf-8") as f: # Save metadata as JSON
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Load the FAISS index and metadata from disk    
    def load(self):
        self.index = faiss.read_index(str(self.index_path)) # Load FAISS index
        self.meta = [json.loads(line) for line in open(self.meta_path, "r", encoding="utf-8")] # Load metadata
    
    # Search the index with a query vector and return top_k results with metadata
    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, List[Dict]]:
        D, I = self.index.search(query_vec.astype('float32'), top_k) # Search index
        selected = [self.meta[i] for i in I[0]] # Retrieve metadata for top results
        return D, selected # Return distances and metadata of top results
            