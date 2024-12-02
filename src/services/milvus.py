from pymilvus import Collection, MilvusClient, FieldSchema, DataType, CollectionSchema, connections, db
from typing import List, Dict

from utils.config import CFG


class MilvusService:
    def __init__(self):
        self.client = MilvusClient(uri=CFG.milvus_uri)
        self.collection_name = CFG.milvus_collection
        self.dimension = CFG.milvus_dimension
        self.database = CFG.milvus_db
        self.connect()
        self.init_collection()
        
    def connect(self):
        # 기존 연결 해제
        try:
            connections.disconnect(alias=self.db_alias)
        except:
            pass
        
        # db.use_database(self.database)    # 기본 데이터베이스(default) 변경
        
        # 새로운 연결 생성
        connections.connect(
            host=CFG.milvus_uri,
            port=CFG.milvus_port,
            db_name=self.database
        )
        
        print(f"Connected to Milvus server with database alias: {self.database}")
        
        
    def init_collection(self):
        if not self.client.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields=fields, description="RAG collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": CFG.milvus_dimension},
                "metric_type": "COSINE"
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
        
        else:
            self.collection = Collection(name=self.collection_name)
        

    async def insert_document(self, documents: List[Dict]):
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
