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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f)

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
