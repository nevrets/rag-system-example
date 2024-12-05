from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLoader(ABC):
    """ 모든 문서 loader의 기본 인터페이스 """
    
    @abstractmethod
    def load(self, 
             query: str, 
             language: str = "ko", 
             load_max_docs: Optional[int] = None
             ) -> List[Dict[str, Any]]:
        
        """ 문서를 로드하는 기본 메서드
        
        Args:
            query (str): 검색 쿼리
            language (str, optional): 문서 언어. Defaults to "ko".
            load_max_docs (Optional[int], optional): 최대 로드할 문서 수. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: 로드된 문서 리스트. 각 문서는 다음 형식을 따름:
            {
                "text": str,  # 문서 본문
                "metadata": {
                    "source": str,  # 문서 출처 (예: "wikipedia", "pdf" 등)
                    "title": str,   # 문서 제목
                    "language": str, # 문서 언어
                    ...             # 기타 메타데이터
                }
            }
            
        """
        
        pass