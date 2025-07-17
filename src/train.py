import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qcl_classification import QclClassification


def main():
    # Load iris dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Split into train and test sets
    x_train, x_test, y_train_label, y_test_label = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )

    # One-hot encode labels
    num_class = len(np.unique(y))
    y_train = np.eye(num_class)[y_train_label]
    y_test = np.eye(num_class)[y_test_label]

    # Initialize random seed used in QCL parameters
    np.random.seed(0)

    # Setup quantum circuit parameters
    nqubit = 4
    c_depth = 4

    # Create QCL model and train
    qcl = QclClassification(nqubit, c_depth, num_class)
    _, _, theta_opt = qcl.fit(x_train, y_train, maxiter=100)

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
