"""Configuration constants for training.

学習で利用する各種パラメータをまとめた設定ファイルです。値を変更することで、
データの分割方法や量子回路の規模、最適化の繰り返し回数などを調整できます。"""

# テストデータとして使用する割合
TEST_SIZE = 0.3
# 学習データとテストデータを分割する際の乱数シード
RANDOM_STATE = 0
# 量子回路の初期パラメータ生成に用いる乱数シード
SEED = 0
# 使用する量子ビット数
NQUBIT = 4
# 出力回路の深さ
C_DEPTH = 4
# パラメータ最適化の最大イテレーション回数
MAX_ITER = 100

# Number of instances per bag for LLP
BAG_SIZE = 100

# Paths to pre-extracted feature datasets
TRAIN_DATA_PATH = "data/CIFAR10_test_features.pt"
TEST_DATA_PATH = "data/CIFAR10_test_features.pt"

# Dimensionality after PCA compression
PCA_DIM = 4
