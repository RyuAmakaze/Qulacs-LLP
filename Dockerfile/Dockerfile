FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# 🧱 システム依存ライブラリをインストール
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    tzdata \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 🐍 Python パッケージ
RUN pip install --upgrade pip
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

# CUDA パスの明示（nvccなど）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# GPU対応のQulacs（C++コンパイル＋Pythonバインディング）
RUN git clone --recursive https://github.com/qulacs/qulacs.git && \
    cd qulacs && mkdir build && cd build && \
    cmake .. -DUSE_GPU=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j && \
    cmake --install . && \
    cd ../python && \
    pip install .

# 💾 プロジェクトコードをコピー
COPY . /app

# 🏁 起動コマンド
CMD ["python", "src/train.py"]
