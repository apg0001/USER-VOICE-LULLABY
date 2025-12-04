# FROM python:3.10-slim

# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     build-essential \
#     ffmpeg \
#     libsndfile1 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # COPY requirements.txt .
# COPY app/requirements.txt /app_requirements.txt
# COPY applio/requirements.txt /applio_requirements.txt
# RUN pip install --no-cache-dir -r /app_requirements.txt
# RUN pip install --no-cache-dir -r /applio_requirements.txt
# RUN pip uninstall -y torch torchaudio torchvision
# RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# COPY . .
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Ubuntu 패키지 (ffmpeg, libsndfile 등 동일)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    gcc \
    g++ \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# requirements 동일 (torch 먼저 제거 후 CUDA 설치)
COPY app/requirements.txt /app_requirements.txt
COPY applio/requirements.txt /applio_requirements.txt
RUN pip install --no-cache-dir -r /app_requirements.txt
RUN pip install --no-cache-dir -r /applio_requirements.txt

# 기존 torch 제거 + CUDA 11.8 설치 (동일)
RUN pip uninstall -y torch torchaudio torchvision || true
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 프로젝트 복사
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
