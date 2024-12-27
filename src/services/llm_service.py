from vllm import LLM as VLLM, SamplingParams
from langchain.llms.base import LLM
from typing import Any, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from utils.config import CFG
from loguru import logger
from transformers import AutoTokenizer
import torch.distributed as dist


class TokenLimitError(Exception):
    """토큰 제한 초과 에러"""
    pass

class ModelError(Exception):
    """모델 관련 에러"""
    pass


class VLLMService(LLM, BaseModel):
    """LangChain과 호환되는 vLLM 서비스"""
    
    _instance = None
    _is_initialized = False
    
    class Config:
        arbitrary_types_allowed = True
    
    model_name: str = Field(default=CFG.vllm_model_name)
    max_input_tokens: int = Field(default=CFG.max_input_tokens)
    max_tokens: int = Field(default=CFG.max_tokens)
    
    _vllm_engine: VLLM = PrivateAttr()
    _sampling_params: SamplingParams = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def _llm_type(self) -> str:
        return "vllm"
    
    def __init__(self, **kwargs):
        if not self._is_initialized:
            super().__init__(**kwargs)
            
            try:                
                # 토크나이저 초기화
                self._tokenizer = AutoTokenizer.from_pretrained(
                    CFG.vllm_model_name,
                )
                # 생성 파라미터 설정
                self._sampling_params = SamplingParams(
                    max_tokens=CFG.max_tokens,
                    temperature=CFG.temperature,
                    top_p=CFG.top_p,
                    top_k=CFG.top_k
                )
                
                terminators = [
                    self._tokenizer.eos_token_id,
                    self._tokenizer.convert_tokens_to_ids("<|eot_id>")
                ]
                
                # vLLM 엔진 초기화
                self._vllm_engine = VLLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    dtype="auto",
                    tokenizer=CFG.vllm_model_name,
                    max_model_len=self.max_input_tokens,
                    # eos_token_id=terminators,
                    tensor_parallel_size=CFG.tensor_parallel_size,
                    gpu_memory_utilization=CFG.gpu_memory_utilization,
                    seed=CFG.seed
                )
                
                logger.info(
                    f"vLLM engine initialized successfully: {self.model_name}, "
                    f"Max_input_tokens: {self.max_input_tokens}, "
                    f"Max_tokens: {self.max_tokens}"
                )
                VLLMService._is_initialized = True
                
            except Exception as e:
                logger.error(f"vLLM initialize error: {e}")
                raise ModelError(f"Failed to initialize vLLM engine: {e}")


    # ---- 프롬프트 토큰 수 계산 ---- #
    def _count_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    
    # ---- 프롬프트 유효성 검사 및 전처리 ---- #
    def _validate_and_truncate_prompt(self, 
                                      prompt: str,
                                      max_tokens: int
                                      ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        token_limit = max_tokens or self.max_input_tokens
        token_count = self._count_tokens(prompt)
        
        if token_count > token_limit:
            logger.warning(
                f"Truncating prompt from {token_count} to {token_limit} tokens",
                "Truncating ... "
            )
            
            tokens = self._tokenizer.encode(prompt)[:token_limit]
            truncated_tokens = tokens[:token_limit]
            truncated_prompt = self._tokenizer.decode(truncated_tokens)
            
            logger.info(f"Truncated prompt: {truncated_prompt}")
            return truncated_prompt
        
        logger.info(f"Prompt: {prompt}")
        return prompt


    # ---- 배치 유효성 검사 및 전처리 ---- #
    def _validate_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            raise ValueError("Prompts cannot be empty")
        
        validated_prompts = []
        for idx, prompt in enumerate(prompts):
            try:
                validated_prompt = self._validate_and_truncate_prompt(prompt, self.max_input_tokens)
                validated_prompts.append(validated_prompt)
            except Exception as e:
                logger.error(f"Failed to validate prompt {idx}: {e}")
                validated_prompts.append("")
        
        return validated_prompts


    # ---- 모델 호출 ---- #
    async def _call(self,
                    prompt: str
                    ) -> List[str]:
        try:
            # 토큰 수 검증
            token_count = self._count_tokens(prompt)
            if token_count > self.max_input_tokens:
                raise TokenLimitError(
                    f"Input exceeds token limit: {token_count} > {self.max_input_tokens}"
                )
            
            # 텍스트 전처리
            validated_prompt = self._validate_and_truncate_prompt(prompt, self.max_input_tokens)
            
            # 생성 요청
            outputs = self._vllm_engine.generate(
                prompts=[validated_prompt],
                sampling_params=self._sampling_params
            )
            
            if not outputs or not outputs[0].outputs:
                raise ModelError("No outputs from vLLM engine")
            
            return outputs[0].outputs[0].text
        
        except TokenLimitError as e:
            logger.error(f"Token limit error: {e}")
            raise e
        
        except Exception as e:
            logger.error(f"VLLM async generate error: {e}")
            raise ModelError(f"Failed to generate text: {e}")

    
    # ---- 배치 호출 ---- #
    async def agenerate(self, 
                        prompts: List[str]
                        ) -> List[str]:
        try:
            # 배치 유효성 검사
            validated_prompts = self._validate_batch(prompts)
            
            # 생성 요청
            outputs = self._vllm_engine.generate(
                prompts=validated_prompts,
                sampling_params=self._sampling_params
            )
            
            # 결과 처리
            results = []
            for idx, output in enumerate(outputs):
                try:
                    if output and output.outputs:
                        results.append(output.outputs[0].text)
                    else:
                        logger.warning(f"No outputs from vLLM engine for prompt {idx}")
                        results.append("")
                except Exception as e:
                    logger.error(f"Failed to process output {idx}: {e}")
                    results.append("")

            return results
        
        except Exception as e:
            logger.error(f"VLLM async generate error: {e}")
            raise ModelError(f"Failed to generate batch text: {e}")