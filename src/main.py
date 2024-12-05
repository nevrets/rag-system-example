import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections
from services.document import DocumentService, Document, DocumentBatch
from utils.config import CFG

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
document_service = DocumentService()


@app.on_event("startup")
async def startup():
    connections.connect(
        host=CFG.milvus_host,
        port=CFG.milvus_port,
        db_name=CFG.milvus_db
    )


@app.get("/health")         # 요청 url 경로
async def health_check():
    return {"status": "healthy"}     # 응답 데이터

# ---- 문서 일괄 등록 ---- #
@app.post("/documents/batch")
async def insert_documents(document_batch: DocumentBatch):
    try:
        results = await document_service.process_document(document_batch.documents)
        return {"status": "success", "results": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ---- 문서 단일 등록 ---- #
@app.post("/documents/single")
async def insert_document(document: Document):
    try:
        results = await document_service.process_document([document])
        return {"status": "success", "results": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- 문서 검색 ---- #
@app.get("/documents/search")
async def search_documents(query: str, limit: int = 5):
    try:
        results = await document_service.search_similar_documents(query, limit)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)