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
    LOSS_TYPE,
    USE_GPU,
)

from qcl_classification import QclClassification
from data_utils import load_pt_features, create_fixed_proportion_batches
from qulacs import QuantumState, QuantumStateGpu


def main():
    state = QuantumStateGpu(NQUBIT) if USE_GPU else QuantumState(NQUBIT)
    print(state.get_device_name())

    # Load features stored in .pt files
    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    num_class = len(np.unique(y_train_label))

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_label, dtype=torch.long),
    )

    # Randomly generate teacher proportions for each bag
    num_bags = len(train_dataset) // BAG_SIZE
    np.random.seed(SEED)
    teacher_probs_list = np.random.dirichlet(np.ones(num_class), size=num_bags)

    sampler = create_fixed_proportion_batches(
        train_dataset, teacher_probs_list, BAG_SIZE, num_class
    )
    bag_indices = sampler.batches
    teacher_props = np.array(teacher_probs_list)

    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    _, _, theta_opt = qcl.fit_bags(
        x_train, bag_indices, teacher_props, maxiter=MAX_ITER, loss_type=LOSS_TYPE
    )

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

