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

    x_train, _, y_train_label, _ = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    num_class = len(np.unique(y_train_label))

    np.random.seed(SEED)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_label, dtype=torch.long),
    )

    num_bags = len(train_dataset) // BAG_SIZE
    teacher_probs = np.random.dirichlet(np.ones(num_class), size=num_bags)
    sampler = create_fixed_proportion_batches(
        train_dataset, teacher_probs.tolist(), BAG_SIZE, num_class
    )
    bags = sampler.batches

    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    _, _, theta_opt = qcl.fit_bags(
        x_train, bags, teacher_probs, maxiter=MAX_ITER, loss_type="cross_entropy"
    )

    preds = qcl.pred_bags(theta_opt, bags)
    print("First bag predicted proportions:", preds[0])
    print("First bag teacher proportions:", teacher_probs[0])


if __name__ == "__main__":
    main()
