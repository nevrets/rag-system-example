FROM harbor.euso.kr/rag/rag-example:base

WORKDIR /app
COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



## docker build --no-cache -t harbor.euso.kr/rag/rag-example:latest .
## docker push harbor.euso.kr/rag/rag-example:latest

