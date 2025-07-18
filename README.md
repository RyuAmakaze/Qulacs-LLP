# Qulacs-LLP
Qulacsを使ったLLP

GPUアクセラレーションを利用するため、Docker イメージでは Qulacs を
GPU 対応でビルドしています。

## Dockerで
1. Docker イメージをビルドします。
   ```bash
   docker build -t qulacs-llp -f Dockerfile/Dockerfile .
   ```
   Qulacs(GPU 対応版) をビルドするため、CUDA が利用できる環境で実行してください。
2. 作業ディレクトリをコンテナにマウントして学習を実行します。GPU を利用する場合は `--gpus all` を指定します。
   ```bash
   docker run --rm --shm-size=2g --gpus all -v $(pwd):/app -w /app qulacs-llp python -u src/train.py
   ```

Dockerに入るだけ
```bash
docker run --rm --gpus all -v $(pwd):/app -w /app -it qulacs-llp bash
```

Qulacs が GPU を認識しているかは次のコマンドで確認できます。
```bash
python -c "from qulacs import QuantumStateGpu; print(QuantumStateGpu(1).get_device_name())"
```
"cuda" と表示されれば GPU 版が正しくインストールされています。
