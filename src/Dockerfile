FROM harbor.euso.kr/rag/fastapi:base

WORKDIR /app
COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



## docker build --no-cache -t harbor.euso.kr/rag/fastapi:latest .
## docker push harbor.euso.kr/rag/fastapi:latest

