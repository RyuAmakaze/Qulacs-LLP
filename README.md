# Qulacs-LLP
Qulacsを使ったLLP

## Dockerで
1. Docker イメージをビルドします。
   ```bash
   docker build -t qulacs-llp -f Dockerfile/Dockerfile .
   ```
2. 作業ディレクトリをコンテナにマウントして学習を実行します。GPU を利用する場合は `--gpus all` を指定します。
   ```bash
   docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app qulacs-llp python src/train.py
   ```

Dockerに入るだけ
```bash
docker run --rm --gpus all -v $(pwd):/app -w /app -it qulacs-llp bash
```
