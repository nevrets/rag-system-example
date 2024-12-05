from setuptools import setup, find_packages

setup(
    name="rag-system",
    version="0.1.0",
    description="RAG(Retrieval-Augmented Generation) System",
    author="jklee",
    author_email="nevretnevret@gmail.com",
    package_dir={"": "src"},  # src 디렉토리를 루트로 설정
    packages=find_packages(where="src"),
    install_requires=[
        # API & 웹 프레임워크
        "fastapi==0.85.1",
        "uvicorn==0.27.1",
        "pydantic==1.10.13",
        "python-multipart==0.0.9",

        # 벡터 데이터베이스
        "pymilvus==2.4.9",

        # LangChain
        "langchain==0.1.9",
        "langchain-community==0.0.28",
        "langchain-core==0.1.35",

        # 임베딩 & ML
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        "sentence-transformers==2.5.1",

        # vLLM
        "vllm==0.2.5",
        "ray==2.7.1",
        "transformers==4.36.0",
        "accelerate==0.25.0",

        # 유틸리티
        "python-dotenv==1.0.1",
        "requests==2.31.0",
        "numpy==1.26.4",
        "pandas==2.2.1",
        "prometheus-client==0.20.0",
        "pytest==8.1.1",
        "httpx==0.27.0",
        "loguru==0.7.0",

        # 의존성 해결
        "jsonpatch==1.33",
        "packaging==23.2",
        "setuptools>=69.0.0",
        "certifi>=2023.7.22",
        "charset-normalizer>=3.2.0",
        "urllib3>=1.26.18",
        "typing-extensions==4.7.1",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "rag-server=main:app",
            "rag-load-wiki=scripts.load_wiki_data:load_wiki_data"
        ],
    }
) 