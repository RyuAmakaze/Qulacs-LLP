import numpy as np
from sklearn.metrics import accuracy_score

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PCA_DIM,
    SEED,
    NQUBIT,
    C_DEPTH,
    MAX_ITER,
)

from qcl_classification import QclClassification
from data_utils import load_pt_features
from qulacs import QuantumState, QuantumStateGpu
from config import USE_GPU


def main():
    state = QuantumStateGpu(NQUBIT) if USE_GPU else QuantumState(NQUBIT)
    print(state.get_device_name())

    # Load pre-extracted features
    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    num_class = len(np.unique(y_train_label))

    np.random.seed(SEED)

    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    qcl.fit_backprop_inner_product(x_train, y_train_label, n_iter=MAX_ITER, lr=0.1)

    pred_train = qcl.pred_amplitude(x_train)
    acc_train = accuracy_score(y_train_label, np.argmax(pred_train, axis=1))

    pred_test = qcl.pred_amplitude(x_test)
    acc_test = accuracy_score(y_test_label, np.argmax(pred_test, axis=1))

    print(f"train accuracy: {acc_train:.3f}")
    print(f"test accuracy: {acc_test:.3f}")


if __name__ == "__main__":
    main()
