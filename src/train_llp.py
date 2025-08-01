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
from data_utils import load_pt_features, create_random_bags
from qulacs import QuantumState
from config import USE_GPU


def main():
    state = QuantumState(NQUBIT)
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

    bag_sampler, teacher_probs = create_random_bags(train_ds, BAG_SIZE, num_class, shuffle=True)

    qcl = QclClassification(NQUBIT, C_DEPTH, num_class)
    qcl.fit_llp_inner_product(
        x_train,
        bag_sampler,
        teacher_probs,
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
    from dotenv import load_dotenv
    import debugpy
    import os

    load_dotenv()

    if os.getenv("DEBUGPY_STARTED") != "1":
        os.environ["DEBUGPY_STARTED"] = "1"
        port = int(os.getenv("DEBUG_PORT", 5611))
        print(f"🔍 Waiting for debugger attach on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
    main()
