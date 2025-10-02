# Class for generating embeddings using a specified model

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    # Initialize the embedder with a specified model and batch size
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    # Encode a list of texts into embeddings
    def encode(self, texts: List[str]) -> np.array:
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True))