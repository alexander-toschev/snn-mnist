"""Vision dataset helpers.

This project historically used `bindsnet.datasets.MNIST`, which yields items as
`{"image": Tensor, "label": int}`.

To support torchvision datasets (e.g. FashionMNIST) without touching the rest of
the pipeline, we wrap torchvision datasets to the same dict-shaped interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR100


class DictDataset(Dataset):
    """Wrap a torchvision-style dataset returning (x, y) -> {image, label}."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        # Ensure a consistent type: image tensor, label int
        if torch.is_tensor(y):
            try:
                y = int(y.item())
            except Exception:
                y = int(y)
        return {"image": x, "label": y}


def _parse_dataset_name(name: str) -> Tuple[str, Optional[str]]:
    """Return (base_name, split).

    Supported:
      - mnist
      - fashion / fashion-mnist
      - kmnist
      - emnist[:split]  (default split=balanced)
      - cifar100[:K]    (optional K=int → keep only labels 0..K-1)
    """
    n = (name or "mnist").strip().lower()
    if n.startswith("emnist"):
        if ":" in n:
            _, split = n.split(":", 1)
            split = (split or "balanced").strip()
        else:
            split = "balanced"
        return "emnist", split
    if n in {"fashion", "fashionmnist", "fashion-mnist"}:
        return "fashion", None
    if n.startswith("cifar100") or n.startswith("cifar-100"):
        if ":" in n:
            _, k = n.split(":", 1)
            k = (k or "").strip()
            return "cifar100", k or None
        return "cifar100", None
    return n, None


class FilterLabelsDataset(Dataset):
    """Filter a torchvision-style dataset to a subset of labels.

    Keeps samples whose original label is in `keep_labels`.
    If `remap=True`, remaps labels to 0..K-1 in the order of keep_labels.
    """

    def __init__(self, base: Dataset, keep_labels: list[int], remap: bool = True):
        self.base = base
        self.keep_labels = list(map(int, keep_labels))
        self.keep_set = set(self.keep_labels)
        self.remap = bool(remap)
        self._map = {int(lbl): i for i, lbl in enumerate(self.keep_labels)} if self.remap else None

        # Try fast path via .targets if available (torchvision datasets)
        targets = getattr(base, "targets", None)
        if targets is None:
            # Fallback: scan dataset (slower)
            self.indices = [i for i in range(len(base)) if int(base[i][1]) in self.keep_set]
        else:
            self.indices = [i for i, y in enumerate(targets) if int(y) in self.keep_set]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        x, y = self.base[self.indices[j]]
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        if self._map is not None:
            y = self._map[y]
        return x, y


def make_vision_datasets(
    *,
    dataset: str = "mnist",
    root: str = "./data",
    transform: Optional[transforms.Compose] = None,
    download: bool = True,
):
    """Return (train_ds, test_ds) with dict-shaped items."""
    base, split = _parse_dataset_name(dataset)

    # Default transform
    # - MNIST-like datasets already return 1×28×28
    # - CIFAR returns 3×32×32 (RGB). We keep it as-is for richer experiments.
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    if base == "mnist":
        train = MNIST(root=root, train=True, download=download, transform=transform)
        test = MNIST(root=root, train=False, download=download, transform=transform)
    elif base == "fashion":
        train = FashionMNIST(root=root, train=True, download=download, transform=transform)
        test = FashionMNIST(root=root, train=False, download=download, transform=transform)
    elif base == "kmnist":
        train = KMNIST(root=root, train=True, download=download, transform=transform)
        test = KMNIST(root=root, train=False, download=download, transform=transform)
    elif base == "emnist":
        split = split or "balanced"
        train = EMNIST(root=root, split=split, train=True, download=download, transform=transform)
        test = EMNIST(root=root, split=split, train=False, download=download, transform=transform)
    elif base == "cifar100":
        train = CIFAR100(root=root, train=True, download=download, transform=transform)
        test = CIFAR100(root=root, train=False, download=download, transform=transform)
        # Optional subset: cifar100:K keeps only fine labels 0..K-1.
        if split is not None and str(split).isdigit():
            k = int(split)
            keep = list(range(k))
            train = FilterLabelsDataset(train, keep_labels=keep, remap=True)
            test = FilterLabelsDataset(test, keep_labels=keep, remap=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DictDataset(train), DictDataset(test)
