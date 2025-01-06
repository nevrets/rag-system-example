from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from services.vllm import VLLMService
from services.document import DocumentService
from typing import Dict, Any, List
from loguru import logger
from utils.config import CFG


class RAGChain:
    # ---- RAG (Retrieval Augmented Generation) 체인 ---- #
    def __init__(self):
        self.llm_service = VLLMService()
        self.document_service = DocumentService()
        self.max_context_length = CFG.max_seq_length
        
    # ---- 텍스트 길이 제한 ---- #
    def _truncate_context(self, 
                          text: str, 
                          max_length: int
                          ) -> str:
        if len(text) > max_length:
            return text
        return text[:max_length] + "..."

    def _create_prompt(self,
                       question: str,
                       contexts: List[str]
                       ) -> str:
        """
        Context 기반 프롬프트 생성
        
        Args:
            question (str): 사용자 질문
            contexts (List[str]): 관련 문서 목록
            
        Returns:
            str: 프롬프트
            
        """
        # 각 컨텍스트 길이 제한
        truncated_contexts = [
            self._truncate_context(context, self.max_context_length)
            for context in contexts
        ]
        
        # 컨텍스트 결합
        context_text = "\n\n".join(
            f"문서 {i+1}:\n{context}"
            for i, context in enumerate(truncated_contexts)
        )
        
        # 프롬프트 템플릿
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

{context_text}

질문: {question}

답변:"""
        
        return prompt


    async def query(self,
                    question: str,
                    max_docs: int = 3
                    ) -> Dict[str, Any]:
        """
        질문에 대한 RAG 처리
        
        Args:
            question (str): 사용자 질문
            max_docs (int): 검색할 최대 문서 수
            
        Returns:
            Dict[str, Any]: 응답 및 참조 문서
        """
        try:
            # ---- 1. 관련 문서 검색 ---- #
            relevant_docs = await self.document_service.search_similar_documents(
                query=question, 
                limit=max_docs
            )
            
            # ---- 2. 문서 컨텍스트 추출 ---- #
            contexts = [doc['text'] for doc in relevant_docs]
            
            # ---- 3. 프롬프트 생성 ---- #
            prompt = self._create_prompt(question, contexts)
            
            # ---- 4. LLM으로 질문에 대한 답변 생성 ---- #
            responses = await self.llm_service.agenerate([prompt])
            response = responses[0] if responses else ""
            
            return {
                "answer": response, 
                "context": contexts,
                "metadata": {
                    "num_docs": len(contexts),
                    "question": question,
                }
            }
        
        except Exception as e:
            logger.error(f"RAGChain error: {str(e)}")
            raise e
        
        


    async def batch_query(self,
                          questions: List[str],
                          max_docs: int = 3
                          ) -> List[Dict[str, Any]]:
        """
        여러 질문에 대한 RAG 처리
        
        Args:
            questions (List[str]): 질문 목록
            max_docs (int): 검색할 최대 문서 수
            
        Returns:
            List[Dict[str, Any]]: 응답 및 참조 문서
        """
        try:
            results = []
            for question in questions:
                result = await self.query(
                    question=question, 
                    max_docs=max_docs
                )
                results.append(result)
            return results
        
        except Exception as e:
            logger.error(f"RAGChain batch query error: {str(e)}")
            raise e
