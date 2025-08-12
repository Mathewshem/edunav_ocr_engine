import json
import os
import random
import numpy as np
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_labels(labels, path):
    # `labels` should be a list in class-index order
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(labels), f, ensure_ascii=False, indent=2)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[1])  # sort by id
        return [name for name, _ in items]

    raise ValueError("labels.json must be a list or a dict {name: id}.")


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
