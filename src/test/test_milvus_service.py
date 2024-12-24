import asyncio
from services.milvus import MilvusService
from loguru import logger
from utils.config import CFG

logger.add("logs/test_insert.log", rotation="100KB")


async def test_milvus_crud():
    milvus_service = MilvusService()
    
    test_documents = [
        {
            "id": "test_1",
            "text": "안녕하세요, RAG!",
            "embedding": [0.1] * CFG.milvus_dimension,  # 임베딩 차원에 맞게 조정
            "metadata": {"source": "test"}
        },
        {
            "id": "test_2",
            "text": "이것은 테스트 문서입니다.",
            "embedding": [0.2] * CFG.milvus_dimension,
            "metadata": {"source": "test"}
        }
    ]
    
    try:
        # 문서 삽입
        await milvus_service.insert_document(test_documents)
        logger.info("문서 삽입 성공")
        
        # 삽입 확인을 위한 검색
        search_results = await milvus_service.search_documents(
            query_embedding=[0.1] * CFG.milvus_dimension,
            limit=2
        )
        
        logger.info(f"검색 결과: {search_results}")
        
        # 문서 업데이트
        await milvus_service.update_document(
            doc_id="test_1",
            text="다음에 봐요, RAG!",
            embedding=[0.9] * CFG.milvus_dimension,
            metadata={"source": "updated"}
        )
        
        # 업데이트 확인을 위한 검색
        search_results = await milvus_service.search_documents(
            query_embedding=[0.9] * CFG.milvus_dimension,
            limit=2
        )
        
        logger.info(f"검색 결과: {search_results}")
        
        # 문서 삭제
        await milvus_service.delete_documents(["test_1"])
        logger.info("문서 삭제 성공")
    
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")



if __name__ == "__main__":
    asyncio.run(test_milvus_crud())
