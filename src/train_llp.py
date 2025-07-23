import numpy as np
import torch
from torch.utils.data import TensorDataset

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
from qulacs import QuantumStateGpu


def main():
    state = QuantumStateGpu(NQUBIT)
    print(state.get_device_name())

    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    num_class = len(np.unique(y_train_label))

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_label, dtype=torch.long),
    )

    full_len = len(train_dataset) - len(train_dataset) % BAG_SIZE
    n_bags = full_len // BAG_SIZE

    teacher_probs_list = []
    for i in range(n_bags):
        labels = y_train_label[i * BAG_SIZE : (i + 1) * BAG_SIZE]
        counts = np.bincount(labels, minlength=num_class).astype(float)
        teacher_probs_list.append(counts / counts.sum())

    sampler = create_fixed_proportion_batches(
        train_dataset, teacher_probs_list, BAG_SIZE, num_class
    )

    ordered_indices = [idx for batch in sampler for idx in batch]
    x_bag = x_train[ordered_indices]
    teacher_props = np.array(teacher_probs_list)

    np.random.seed(SEED)
    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    _, _, theta_opt = qcl.fit_bags(
        x_bag,
        teacher_props,
        BAG_SIZE,
        maxiter=MAX_ITER,
        loss="ce",
    )

    qcl.set_input_state(x_train[:full_len])
    bag_pred_train = qcl.bag_pred(theta_opt, BAG_SIZE)

    qcl.set_input_state(x_test[: len(x_test) - len(x_test) % BAG_SIZE])
    bag_pred_test = qcl.bag_pred(theta_opt, BAG_SIZE)

    print("train bag preds shape:", bag_pred_train.shape)
    print("test bag preds shape:", bag_pred_test.shape)


if __name__ == "__main__":
    main()
