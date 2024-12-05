import asyncio
from services.document import DocumentService, Document
from loaders.wiki_loader import WikiLoader
from loguru import logger


async def load_wiki_data():
    wiki_loader = WikiLoader()
    document_service = DocumentService()
    
    # 검색할 주제 리스트 (query)
    search_queries = [
        "인공지능",
        "머신러닝",
        "딥러닝",
        "자연어처리",
        "빅데이터",
        "머신러닝 프레임워크",
        "딥러닝 프레임워크"
    ]
    
    total_documents = 0
    
    for query in search_queries:
        try:
            articles = await wiki_loader.load(
                query=query,
                language="ko",
                load_max_docs=10    # 각 주제별 최대 문서 수
            )
            
            # Document 객체로 변환
            documents = [
                Document(
                    text=article["text"],
                    metadata=article["metadata"]
                ) for article in articles
            ]
            
            # 배치 처리로 Milvus에 저장
            results = await document_service.process_document(documents)
            total_documents += len(results)
            
            logger.info(f"Inserted {len(results)} documents for query: {query}")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            continue
        
    logger.info(f"Total documents inserted: {total_documents}")
    
            
            
if __name__ == "__main__":
    asyncio.run(load_wiki_data())
    