import os
import torch
from torch import nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
from src.preprocess import get_eval_transform
from src.utils import project_root, load_labels, get_device

def evaluate(
    data_dir=os.path.join(project_root(), "data"),
    model_dir=os.path.join(project_root(), "models"),
    batch_size=16
):
    device = get_device()
    labels = load_labels(os.path.join(model_dir, "labels.json"))
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(labels))
    model.load_state_dict(torch.load(os.path.join(model_dir, "mobilenet_v2_best.pt"), map_location=device))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            correct += (logits.argmax(1) == targets).sum().item()
            total += imgs.size(0)

    acc = correct / total if total else 0.0
    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__":
    evaluate()
