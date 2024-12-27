import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections
from services.document import DocumentService, Document, DocumentBatch
from utils.config import CFG
from typing import List
from chains.rag_chain import RAGChain
from services.llm_service import VLLMService

import warnings
warnings.filterwarnings("ignore")


app = FastAPI()
document_service = DocumentService()
llm_service = VLLMService()
rag_chain = RAGChain()


@app.on_event("startup")
async def startup():
    connections.connect(
        host=CFG.milvus_host,
        port=CFG.milvus_port,
        db_name=CFG.milvus_db
    )


@app.get("/health")                  # 요청 url 경로
async def health_check():
    return {"status": "healthy"}     # 응답 데이터


# ---- 문서 단일 등록 ---- #
@app.post("/documents/single")
async def insert_document(document: Document):
    try:
        results = await document_service.process_document([document])
        return {"status": "success", "results": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ---- 문서 일괄 등록 ---- #
@app.post("/documents/batch")
async def insert_documents(document_batch: DocumentBatch):
    try:
        results = await document_service.process_document(document_batch.documents)
        return {"status": "success", "results": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 문서 검색 ---- #
@app.get("/documents/search")
async def search_documents(query: str, 
                           limit: int = 5
                           ):
    try:
        results = await document_service.search_similar_documents(query, limit)
        return {"status": "success", "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 문서 삭제 ---- #
@app.delete("/documents/delete")
async def delete_documents(doc_ids: List[str]):
    try:
        await document_service.delete_documents(doc_ids)
        return {"status": "success", "results": f"Deleted {len(doc_ids)} documents"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ---- 문서 업데이트 ---- #
@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, 
                          document: Document
                          ):
    try:
        result = await document_service.update_document(
            doc_id=doc_id,
            new_text=document.text,
            new_metadata=document.metadata
        )
        return {"status": "success", "results": f"Updated {doc_id} document"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 텍스트 생성 ---- #
@app.post("/llm/generate")
async def generate_text(prompt: str):
    try:
        response = await llm_service._call(prompt)
        return {"status": "success", "results": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 텍스트 배치 생성 ---- #
@app.post("/llm/generate_batch")
async def generate_batch(prompts: List[str]):
    try:
        response = await llm_service.agenerate(prompts)
        return {"status": "success", "results": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- RAG 체인 쿼리 ---- #
@app.post("/rag/query")
async def rag_query(question: str, max_docs: int = 3):
    try:
        response = await rag_chain.query(question, max_docs)
        return {"status": "success", "results": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)