import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PCA_DIM,
    SEED,
    NQUBIT,
    C_DEPTH,
    MAX_ITER,
    BAG_SIZE,
)

from qcl_classification import QclClassification
from data_utils import load_pt_features, create_fixed_proportion_batches
from qulacs import QuantumState, QuantumStateGpu
from config import USE_GPU


def main():
    state = QuantumStateGpu(NQUBIT) if USE_GPU else QuantumState(NQUBIT)
    print(state.get_device_name())

    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    num_class = len(np.unique(y_train_label))
    np.random.seed(SEED)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_label, dtype=torch.long),
    )

    num_bags = len(train_ds) // BAG_SIZE
    teacher_probs = np.random.dirichlet(np.ones(num_class), size=num_bags)
    bag_sampler = create_fixed_proportion_batches(train_ds, teacher_probs, BAG_SIZE, num_class)

    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    qcl.fit_llp_inner_product(
        x_train,
        bag_sampler,
        torch.tensor(teacher_probs, dtype=torch.float32),
        n_iter=MAX_ITER,
        lr=0.1,
        loss="ce",
        n_jobs=2,
    )

    pred_train = qcl.pred_amplitude(x_train)
    acc_train = accuracy_score(y_train_label, np.argmax(pred_train, axis=1))

    pred_test = qcl.pred_amplitude(x_test)
    acc_test = accuracy_score(y_test_label, np.argmax(pred_test, axis=1))

    print(f"train accuracy: {acc_train:.3f}")
    print(f"test accuracy: {acc_test:.3f}")


if __name__ == "__main__":
    main()
