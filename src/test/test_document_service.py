import asyncio
from services.document import DocumentService
from services.embedding import EmbeddingService
from loguru import logger

logger.add("logs/document_crud.log", rotation="100KB")


async def test_document_crud():
    document_service = DocumentService()
    embedding_service = EmbeddingService()
    
    # 테스트 문서 생성
    test_doc = {
        "id": "test_crud_1",
        "text": "이것은 CRUD 테스트를 위한 문서입니다.",
        "metadata": {"source": "test", "type": "crud_test"}
    }
        
    try:
        # 문서 삽입
        logger.info("문서 삽입 테스트")
        await document_service.process_document([test_doc])
        
        # 문서 업데이트
        logger.info("문서 업데이트 테스트")
        updated_text = "이것은 업데이트된 테스트 문서입니다."
        embedding = await embedding_service.embed_document(updated_text)

        await document_service.update_document(
            doc_id=test_doc['id'],
            text=updated_text,
            embedding=embedding,
            metadata={"source": "test", "type": "updated"}
        )
        
        # 문서 검색으로 업데이트 확인
        logger.info("업데이트 확인을 위한 검색")
        results = await document_service.search_similar_documents(updated_text, limit=1)
        logger.info(f"검색 결과: {results}")
        
        # 문서 삭제
        logger.info("문서 삭제 테스트")
        await document_service.delete_documents([test_doc["id"]])
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")



if __name__ == "__main__":
    asyncio.run(test_document_crud())