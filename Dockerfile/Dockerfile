FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopenblas-dev libboost-all-dev tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python packages
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

# Qulacs 最新版（backprop_inner_product 対応）をソースからビルド
RUN git clone --recursive https://github.com/qulacs/qulacs.git /app/qulacs && \
    mkdir -p /app/qulacs/build && cd /app/qulacs/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DUSE_GPU=ON -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j && make install

# Python バインディングをインストール
RUN pip install /app/qulacs

# 必要ファイルを追加
COPY . /app

# CMD（必要に応じて変更）
CMD ["python", "src/train.py"]
