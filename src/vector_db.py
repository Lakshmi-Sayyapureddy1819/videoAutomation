import faiss
import numpy as np
import os
import json

class VectorDB:
    def __init__(self, index_path="data/vector_db/video_index.faiss", metadata_path="data/vector_db/metadata.json", dim=512):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.metadata = []
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Inner Product (Cosine Similarity if normalized)
            self.index = faiss.IndexFlatIP(dim)

    def add_item(self, embedding, meta):
        """
        embedding: numpy array of shape (1, dim)
        meta: dict containing info (url, path, description, etc.)
        """
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embedding.shape[1]}")
            
        self.index.add(embedding.astype(np.float32))
        self.metadata.append(meta)

    def search(self, query_embedding, k=5):
        """
        Returns list of (metadata, score)
        """
        if self.index.ntotal == 0:
            return []
            
        D, I = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)