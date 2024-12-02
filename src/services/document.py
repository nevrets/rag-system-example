import uuid
from typing import List, Optional
from services.embedding import EmbeddingService
from services.milvus import MilvusService
from pydantic import BaseModel



class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = {}

class DocumentBatch(BaseModel):
    documents: List[Document]

class DocumentService:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.embedding_service = EmbeddingService()
        self.milvus_service = MilvusService()
        
    async def process_document(self, documents: List[Document]):
        results = []
        
        for document in documents:
            if not document.id:
                document.id = str(uuid.uuid4())
                
            # document embedding
            embedding = await self.embedding_service.embed_document(document.text)
            
            entity = {
                "id": document.id,  
                "text": document.text,
                "embedding": embedding,
                "metadata": document.metadata
            }

            results.append(entity)
            
        await self.milvus_service.insert_document(results)
            
        return results
