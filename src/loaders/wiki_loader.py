from langchain_community.document_loaders import WikipediaLoader
from typing import List, Optional, Dict, Any
from loguru import logger
from loaders.base import BaseLoader


class WikiLoader(BaseLoader):
    def __init__(self):
        pass
        
    async def load(self,
                   query: str,
                   language: str = "ko",
                   load_max_docs: Optional[int] = None
                   ) -> List[Dict[str, Any]]:
        
        try:
            # WikipediaLoader 인스턴스 직접 생성
            loader = WikipediaLoader(
                query=query,
                lang=language,
                load_max_docs=load_max_docs
            )

            documents = loader.load()
            
            articles = []
            for document in documents:
                articles.append({
                    "text": document.page_content,
                    "metadata": {
                        "source": "wikipedia",
                        "title": document.metadata.get("title", ""),
                        "source": document.metadata.get("source", ""),
                        "lang": language,
                        "query": query,
                    }
                })
                
            logger.info(f"Loaded {len(articles)} articles for {query} from Wikipedia")
            return articles
        
        except Exception as e:
            logger.error(f"Error loading articles for {query} from Wikipedia: {e}")
            raise e