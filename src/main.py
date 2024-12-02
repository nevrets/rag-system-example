import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections
from services.document import DocumentService, Document, DocumentBatch
from utils.config import CFG

app = FastAPI()
document_service = DocumentService()


# @app.on_event("startup")    # 서버 시작 시 실행되는 함수
async def startup():
    # Milvus 연결 설정
    connections.connect(
        alias=CFG.milvus_db,
        host=CFG.milvus_uri,
        port=CFG.milvus_port
    )

    return {"status": "healthy"}


@app.post("/documents/batch")
async def insert_documents(document_batch: DocumentBatch):
    try:
        results = await document_service.process_document(document_batch.documents)
        return {"status": "success", "results": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/documents/single")
async def insert_document(document: Document):
    try:
        results = await document_service.process_document([document])
        return {"status": "success", "results": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")         # 요청 url 경로
async def health_check():
    return {"status": "healthy"}     # 응답 데이터



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)