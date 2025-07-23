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

    # Load features stored in .pt files
    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    # One-hot encode labels
    num_class = len(np.unique(y_train_label))
    y_train = np.eye(num_class)[y_train_label]
    y_test = np.eye(num_class)[y_test_label]

    # Initialize random seed used in QCL parameters
    np.random.seed(SEED)

    # Setup quantum circuit parameters
    nqubit = NQUBIT
    c_depth = C_DEPTH

    # Create QCL model and train
    qcl = QclClassification(nqubit, c_depth, num_class)
    _, _, theta_opt = qcl.fit(x_train, y_train, maxiter=MAX_ITER)

    # Evaluate accuracy
    qcl.set_input_state(x_train)
    pred_train = qcl.pred(theta_opt)
    acc_train = accuracy_score(y_train_label, np.argmax(pred_train, axis=1))

    qcl.set_input_state(x_test)
    pred_test = qcl.pred(theta_opt)
    acc_test = accuracy_score(y_test_label, np.argmax(pred_test, axis=1))

    print(f"train accuracy: {acc_train:.3f}")
    print(f"test accuracy: {acc_test:.3f}")


if __name__ == "__main__":
    main()
