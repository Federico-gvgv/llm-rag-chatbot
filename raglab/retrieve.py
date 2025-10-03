# Search strategy: retrieve top-k documents using an index and an embedder, with optional MMR re-ranking

import numpy as np
from typing import List, Dict, Tuple

# Cosine similarity on normalized embeddings -> dot product

def mmr(doc_embs: np.ndarray, query_emb: np.ndarray, k: int, lam: float = 0.5) -> List[int]:
    idxs = list(range(len(doc_embs))) # candidate indices
    selected = [] # selected indices
    sims = doc_embs @ query_emb # cosine similarities
    while len(selected) < min(k, len(idxs)): # while we need more selections
        if not selected: # if nothing selected yet, pick the most similar
            i = int(np.argmax(sims))
            selected.append(i)
            continue
        # diversity term
        max_div = -1
        best = None
        for i in idxs:
            if i in selected: continue # already selected
            diversity = max(doc_embs[i] @ doc_embs[j] for j in selected) if selected else 0 # max similarity to selected
            score = lam * sims[i] - (1 - lam) * diversity # MMR score
            if score > max_div:
                max_div = score
                best = i
        selected.append(best) # add best to selected
    return selected[:k] # return top k selected

# Class for retrieving documents using an index and an embedder
class Retriever:

    def __init__(self, index, embedder, top_k=5, use_mmr=True, mmr_lambda=0.5):
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    # Retrieve documents for a given query
    def retrieve(self, query: str) -> Tuple[List[Dict], List[float]]:
        q = self.embedder.encode([query]) # encode query
        scores, docs = self.index.search(q, max(self.top_k * 4 if self.use_mmr else self.top_k, self.top_k)) # search index
        if self.use_mmr: # apply MMR re-ranking
            doc_embs = self.embedder.encode([doc["text"] for doc in docs]) # encode documents
            order = mmr(doc_embs, q[0], self.top_k, self.mmr_lambda) # get MMR order
            docs = [docs[i] for i in order] # reorder documents
            scores = [float((doc_embs[i] @ q[0])) for i in order] # recompute scores
        else: # truncate to top_k
            docs = docs[:self.top_k] # truncate documents
            scores = scores[:self.top_k] # truncate scores
        return docs, list(map(float, scores)) # return documents and scores
        