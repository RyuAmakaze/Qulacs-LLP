FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopenblas-dev tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest==8.3.5 \
    matplotlib \
    torchvision==0.16.0 \
    tqdm \
    pylatexenc \
    scikit-learn==1.7.0\
    xformers \
    debugpy \
    dotenv \
    numpy==1.26.0

# GPU対応のQulacs（CUDA Toolkit パス明示）
RUN CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda pip install --no-cache-dir qulacs-gpu==0.3.1

COPY . /app

CMD ["python", "src/train.py"]
