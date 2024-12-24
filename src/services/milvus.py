from pymilvus import Collection, MilvusClient, FieldSchema, DataType, CollectionSchema, connections
from typing import List, Dict
from loguru import logger
from utils.config import CFG


class MilvusService:
    def __init__(self):
        self.client = MilvusClient(uri=CFG.milvus_uri)
        self.collection_name = CFG.milvus_collection
        self.dimension = CFG.milvus_dimension
        self.database = CFG.milvus_db
        self.connect()
        self.init_collection()


    # ---- Milvus 연결 ---- #
    def connect(self):
        # 기존 연결 해제
        try:
            connections.disconnect(alias=self.db_alias)
        except:
            pass
        
        # db.use_database(self.database)    # 기본 데이터베이스(default) 변경
        
        # 새로운 연결 생성
        connections.connect(
            host=CFG.milvus_host,
            port=CFG.milvus_port,
            db_name=self.database
        )
        
        print(f"Connected to Milvus server with database alias: {self.database}")


    # ---- Milvus Collection 초기화 ---- #
    def init_collection(self):
        if not self.client.has_collection(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")
            
            # schema 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            # schema 생성
            schema = CollectionSchema(fields=fields, description="RAG collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # index 생성
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": CFG.milvus_dimension},
                "metric_type": "COSINE"
            }
            
            # index 생성
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
        
        else:
            self.collection = Collection(name=self.collection_name)


    # ---- Milvus 삽입 ---- #
    async def insert_document(self,
                              documents: List[Dict]
                              ):
        try:
            entities = [
                [document["id"] for document in documents],
                [document["text"] for document in documents],
                [document["embedding"] for document in documents],
                [document["metadata"] for document in documents]
            ]
        
            self.collection.insert(entities)
            return True
        
        except Exception as e:
            raise Exception(f"Error inserting document into Milvus: {e}")


    # ---- Milvus 검색 ---- #
    async def search_documents(self, 
                               query_embedding: List[float], 
                               limit: int = 5
                               ):
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["id", "text", "metadata"]
            )
            
            return [{
                "id": hit.id,
                "text": hit.text,
                "metadata": hit.metadata,
                "score": hit.score
            } for hit in results[0]]
            
        except Exception as e:
            raise Exception(f"Error searching documents in Milvus: {e}")
        
        
    # ---- Milvus 삭제 ---- #
    async def delete_documents(self, 
                               doc_ids: List[str]
                               ) -> bool:
        try:
            expr = f'id in ["{"","".join(doc_ids)}"]'
            logger.info(f"Deleting documents with expression: {expr}")
            self.collection.delete(expr)
            logger.info(f"Deleted documents with IDs: {doc_ids}")
            return True
        
        except Exception as e:
            raise Exception(f"Error deleting documents in Milvus: {e}")
        
    
    # ---- Milvus 업데이트 (삭제 후 재삽입) ---- #
    async def update_document(self, 
                              doc_id: str, 
                              text: str, 
                              embedding: List[float], 
                              metadata: dict = None
                              ) -> bool:
        try:
            # 기존 문서 삭제
            await self.delete_documents([doc_id])

            # 새 문서 삽입
            document = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
            await self.insert_document([document])
            logger.info(f"문서 ID {doc_id} 업데이트 완료")
            return True
        
        except Exception as e:
            logger.error(f"문서 업데이트 중 오류 발생: {e}")
            raise e
