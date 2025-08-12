import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from src.preprocess import get_train_transform, get_eval_transform
from src.utils import get_device, set_seed, save_labels, project_root

def train(
    data_dir=os.path.join(project_root(), "data"),
    out_dir=os.path.join(project_root(), "models"),
    epochs=8,
    batch_size=16,
    lr=3e-4
):
    set_seed(42)
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_train_transform())
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=get_eval_transform())
    labels = train_ds.classes  # e.g. ["Diagram","Handwritten","Printed"]

    save_labels(labels, os.path.join(out_dir, "labels.json"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Backbone: MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(labels))
    model = model.to(device)

    # Freeze most layers for speed/small dataset
    for name, param in model.features.named_parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
    best_path = os.path.join(out_dir, "mobilenet_v2_best.pt")

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(1) == targets).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model to {best_path} (val_acc={best_acc:.3f})")

    print("Training complete.")

if __name__ == "__main__":
    train()
