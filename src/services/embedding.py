from sentence_transformers import SentenceTransformer
from typing import List

import torch


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    async def embed_document(self, document: str) -> List[float]:
        try:
            print(f"device: {self.device}")
            embedding = self.model.encode(document, device=self.device)
            return embedding.tolist()
        
        except Exception as e:
            raise e
        
