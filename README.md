# Qulacs-LLP
Qulacsを使ったLLP

GPUアクセラレーションを利用するため、Docker イメージでは `qulacs-gpu`
をインストールしています。

`src/config.py` の `USE_GPU` を `True` にすると、量子状態の計算で GPU 実装
を使用します。CPU で実行する場合は `False` にしてください。

## Dockerで
1. Docker イメージをビルドします。
   ```bash
   docker build -t qulacs-llp -f Dockerfile/Dockerfile .
   ```
   `qulacs-gpu` が含まれているため、CUDA 対応の環境でビルドしてください。
2. 作業ディレクトリをコンテナにマウントして学習を実行します。GPU を利用する場合は `--gpus all` を指定します。
   ```bash
   docker run --rm --shm-size=8g --gpus all -v $(pwd):/app -w /app qulacs-llp python -u src/train_llp.py
   ```

Dockerに入るだけ
```bash
docker run --rm --shm-size=8g --gpus all -v $(pwd):/app -w /app -it qulacs-llp bash
```
