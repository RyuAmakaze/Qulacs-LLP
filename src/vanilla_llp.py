import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score

from data_utils import load_pt_features, create_random_bags
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PCA_DIM,
    SEED,
    BAG_SIZE,
)

def build_model(input_dim: int, num_class: int, hidden_dim: int) -> nn.Module:
    """Return a simple 1-hidden-layer MLP."""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_class),
    )
    return model

def compute_hidden_dim(param_count: int, input_dim: int, num_class: int) -> int:
    """Approximate hidden dimension to match desired parameter count."""
    # parameters = in*hidden + hidden*num_class + hidden + num_class
    hd = int((param_count - num_class) / (input_dim + num_class + 1))
    return max(1, hd)

def main(args):
    x_train, x_test, y_train_label, y_test_label = load_pt_features(
        TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM
    )

    input_dim = x_train.shape[1]
    num_class = len(np.unique(y_train_label))

    if args.num_params is not None:
        hidden_dim = compute_hidden_dim(args.num_params, input_dim, num_class)
    else:
        hidden_dim = args.hidden_dim

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = build_model(input_dim, num_class, hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model hidden_dim={hidden_dim}, params={total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_label, dtype=torch.long),
    )

    bag_sampler, teacher_probs = create_random_bags(train_ds, BAG_SIZE, num_class, shuffle=True)
    bag_list = list(bag_sampler)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for bag_idx, indices in enumerate(bag_list):
            batch_x = train_ds.tensors[0][indices]
            optimizer.zero_grad()
            logits = model(batch_x)
            probs = logits.softmax(dim=1).mean(dim=0)
            teacher = teacher_probs[bag_idx]
            loss = -(teacher * torch.log(probs + 1e-12)).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            avg_loss = total_loss / len(bag_list)
            print(f"[{epoch:03d}] loss={avg_loss:.4f}")

    with torch.no_grad():
        train_pred = model(torch.tensor(x_train, dtype=torch.float32)).softmax(dim=1)
        test_pred = model(torch.tensor(x_test, dtype=torch.float32)).softmax(dim=1)

    acc_train = accuracy_score(y_train_label, train_pred.argmax(1).numpy())
    acc_test = accuracy_score(y_test_label, test_pred.argmax(1).numpy())
    print(f"train accuracy: {acc_train:.3f}")
    print(f"test accuracy: {acc_test:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple neural network LLP trainer")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--num-params", type=int, default=None, help="Approximate total parameter count")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    args = parser.parse_args()
    main(args)
