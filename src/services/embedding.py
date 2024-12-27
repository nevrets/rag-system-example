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
        self.max_seq_length = CFG.max_seq_length
        
    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + len(word) + 1 > self.max_seq_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length + 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks if chunks else [text]    # 빈 list 방지
        
        
    async def embed_document(self, document: str) -> List[float]:
        try:
            logger.info(f"device: {self.device}")
            logger.info(f"document: {document}")
            
            chunks = self._split_text(document)
            logger.info(f"split into {len(chunks)} chunks")
            
            # 각 chunk 임베딩
            embeddings = []
            for chunk in chunks:
                embedding = self.model.encode(
                    chunk, 
                    device=self.device,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings.append(embedding)
                
            if embeddings:
                mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
                return mean_embedding.cpu().tolist()
       
            return []  
          
            # # 모든 chunk가 비어있는 경우 원본 문서 임베딩
            # embedding = self.model.encode(document, device=self.device)
            # return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise e
        


if __name__ == "__main__":
    import asyncio
    
    async def test_embedding():
        embedding_service = EmbeddingService()
        test_text = "안녕하세요, 반갑습니다."
        
        logger.info(f"Testing embedding for text: {test_text}")
        embedding = await embedding_service.embed_document(test_text)
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info(f"First few values: {embedding[:5]}")

    # 비동기 테스트 실행
    asyncio.run(test_embedding())