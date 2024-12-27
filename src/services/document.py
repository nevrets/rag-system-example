import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.llm_service import VLLMService
from typing import List, Optional
from services.embedding import EmbeddingService
from services.milvus import MilvusService
from pydantic import BaseModel
from loguru import logger

from utils.config import CFG


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = {}


class DocumentBatch(BaseModel):
    documents: List[Document]


class DocumentService:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.llm_service = VLLMService()
        self.embedding_service = EmbeddingService()
        self.milvus_service = MilvusService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CFG.chunk_size,
            chunk_overlap=CFG.chunk_overlap,
            length_function=len
        )
    
    
    # ---- 문서 삽입 ---- #
    async def process_document(self, 
                               documents: List[Document]
                               ):
        results = []
        
        for document in documents:
            if not document['id']:
                document['id'] = str(uuid.uuid4())
                
            # document embedding
            embedding = await self.embedding_service.embed_document(document['text'])
            
            entity = {
                "id": document['id'],  
                "text": document['text'],
                "embedding": embedding,
                "metadata": document['metadata']
            }

            results.append(entity)
            
        await self.milvus_service.insert_document(results)
        logger.info(f"Inserted {len(results)} documents")
        
        return results


    # ---- 유사한 문서 검색 ---- #
    async def search_similar_documents(self, 
                                       query: str, 
                                       limit: int = 5
                                       ):
        # 쿼리 텍스트 임베딩
        query_embedding = await self.embedding_service.embed_document(query)
        
        # Milvus에서 유사한 문서 검색
        results = await self.milvus_service.search_documents(
            query_embedding=query_embedding,
            limit=limit
        )
        
        return results


    # ---- 문서 삭제 ---- #
    async def delete_documents(self, 
                               doc_ids: List[str]
                               ) -> bool:
        try:
            await self.milvus_service.delete_documents(doc_ids)
            return True
        
        except Exception as e:
            logger.error(f"문서 삭제 중 오류 발생: {e}")
            raise e


    # ---- 문서 업데이트 ---- #
    async def update_document(self, 
                              doc_id: str, 
                              new_text: str, 
                              new_metadata: dict = None
                              ) -> bool:
        
        try:
            new_embedding = await self.embedding_service.embed_document(new_text)
            
            return await self.milvus_service.update_document(
                doc_id=doc_id, 
                text=new_text, 
                embedding=new_embedding, 
                metadata=new_metadata
            )
        
        except Exception as e:
            logger.error(f"문서 업데이트 중 오류 발생: {e}")
            raise e



if __name__ == "__main__":
    # 최소 필수 정보만
    doc1 = Document(text="Hello, world!")
    
    # 모든 필드 포함
    doc2 = Document(
        id="doc2",
        text="이 문서는 테스트를 위한 문서입니다.",
        metadata={
            "source": "test",
            "author": "John Doe",
            "language": "ko"
        }
    )

    # 여러 문서를 한번에 처리
    batch = DocumentBatch(documents=[
        Document(text="첫 번째 문서"),
        Document(
            id="doc2",
            text="두 번째 문서",
            metadata={"source": "test"}
        )
    ])