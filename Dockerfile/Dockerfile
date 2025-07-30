FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopenblas-dev libboost-all-dev tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest==8.3.5 \
    matplotlib \
    torchvision==0.16.0 \
    tqdm \
    pylatexenc \
    scikit-learn==1.7.0 \
    xformers \
    debugpy \
    dotenv \
    numpy==1.26.0

# Qulacs ソースを clone してビルド
RUN git clone --recursive https://github.com/qulacs/qulacs.git /app/qulacs && \
    mkdir -p /app/qulacs/build && cd /app/qulacs/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DUSE_GPU=ON && \
    make -j

# Python バインディングを pip 経由でインストール（qulacs_core.so が含まれる）
RUN USE_GPU=ON pip install /app/qulacs

COPY . /app

CMD ["python", "src/train.py"]

RUN cp /app/qulacs/build/python/qulacs_core*.so $(python -c "import site; print(site.getsitepackages()[0])")
