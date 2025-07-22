import types
import torch
import random
import math
from typing import Sequence, List
from torch.utils.data import Sampler
from tqdm import tqdm
from sklearn.decomposition import PCA

def compute_proportions(labels, num_classes):
    """Compute normalized label counts for a batch."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts / counts.sum()

class FixedBatchSampler(Sampler[List[int]]):
    """Yield predefined lists of indices as batches."""

    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def create_fixed_proportion_batches(dataset, teacher_probs_list, bag_size, num_classes):
    """Return a FixedBatchSampler where each batch matches the given proportions."""
    dataset_indices = list(range(len(dataset)))

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in dataset_indices:
        root_idx = idx
        ds = dataset
        # Resolve the index through potentially nested Subset objects
        while hasattr(ds, "indices"):
            root_idx = ds.indices[root_idx]
            ds = ds.dataset
        label = int(targets[root_idx])
        if label < num_classes:
            # store dataset-relative index
            class_to_indices[label].append(idx)

    for idx_list in class_to_indices.values():
        random.shuffle(idx_list)

    batches = []
    for probs in teacher_probs_list:
        raw = [p * bag_size for p in probs]
        counts = [math.floor(c) for c in raw]
        remaining = bag_size - sum(counts)
        fractions = [r - math.floor(r) for r in raw]
        for cls in sorted(range(num_classes), key=lambda i: fractions[i], reverse=True)[:remaining]:
            counts[cls] += 1

        batch = []
        for cls, count in enumerate(counts):
            batch.extend(class_to_indices[cls][:count])
            class_to_indices[cls] = class_to_indices[cls][count:]
        batches.append(batch)

    return FixedBatchSampler(batches)


def create_random_bags(dataset, bag_size, num_classes, shuffle=True):
    """Create random bags and return a sampler and teacher label proportions."""
    dataset_indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(dataset_indices)

    # Walk to the root dataset to access labels
    base_dataset = dataset
    while hasattr(base_dataset, "indices"):
        base_dataset = base_dataset.dataset

    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        targets = getattr(base_dataset, "labels", None)
    if targets is None and isinstance(base_dataset, torch.utils.data.TensorDataset):
        if len(base_dataset.tensors) < 2:
            raise ValueError(
                "TensorDataset must contain at least two tensors to provide labels"
            )
        targets = base_dataset.tensors[1]
    if targets is None:
        raise ValueError(
            "Could not locate labels. Provide 'targets', 'labels', or use a TensorDataset with labels"
        )

    batches = []
    teacher_props = []
    # ignore last incomplete batch
    full_len = len(dataset_indices) - len(dataset_indices) % bag_size
    for start in range(0, full_len, bag_size):
        batch_indices = dataset_indices[start : start + bag_size]
        batches.append(batch_indices)

        labels = []
        for idx in batch_indices:
            root_idx = idx
            ds = dataset
            while hasattr(ds, "indices"):
                root_idx = ds.indices[root_idx]
                ds = ds.dataset
            label = int(targets[root_idx])
            if label < num_classes:
                labels.append(label)
        teacher_props.append(compute_proportions(torch.tensor(labels), num_classes))

    sampler = FixedBatchSampler(batches)
    teacher_tensor = torch.stack(teacher_props)
    return sampler, teacher_tensor


def load_pt_features(train_path: str, test_path: str, pca_dim: int | None = None):
    """Load feature tensors from ``.pt`` files and optionally apply PCA."""
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    X_train = train_data["features"].cpu().numpy()
    y_train = train_data["labels"].cpu().numpy()
    X_test = test_data["features"].cpu().numpy()
    y_test = test_data["labels"].cpu().numpy()

    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"PCA reduced shape: {X_train.shape}")

    return X_train, X_test, y_train, y_test
