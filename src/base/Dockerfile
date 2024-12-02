FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# docker build --no-cache -t harbor.euso.kr/rag/fastapi:base .
# docker push harbor.euso.kr/rag/fastapi:base