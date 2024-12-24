import torch
from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger
from utils.config import CFG


# ---- 문서 임베딩 생성 서비스 ---- #
class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(CFG.embedding_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    async def embed_document(self, document: str) -> List[float]:
        try:
            logger.info(f"device: {self.device}")
            logger.info(f"document: {document}")
            
            embedding = self.model.encode(document, device=self.device)
            return embedding.tolist()
        
        except Exception as e:
            raise e
        


if __name__ == "__main__":
    embedding_service = EmbeddingService()
    print(embedding_service.embed_document("안녕하세요, 반갑습니다."))
