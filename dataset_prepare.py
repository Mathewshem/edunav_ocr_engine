import argparse, shutil, random, json, os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(folder: Path):
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_list(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def copy_set(files, dst_dir: Path, move=False):
    ensure_dir(dst_dir)
    for f in files:
        target = dst_dir / f.name
        if move:
            shutil.move(str(f), str(target))
        else:
            shutil.copy2(str(f), str(target))

def main():
    ap = argparse.ArgumentParser(description="Split dataset into train/val/test for OCR classifier.")
    ap.add_argument("--raw_dir", default="data_raw", help="Folder with class subfolders (Printed/Handwritten/Diagram)")
    ap.add_argument("--out_dir", default="data", help="Output folder to create train/val/test splits")
    ap.add_argument("--train", type=float, default=0.7, help="Train ratio")
    ap.add_argument("--val", type=float, default=0.15, help="Val ratio")
    ap.add_argument("--test", type=float, default=0.15, help="Test ratio")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    random.seed(args.seed)

    classes = []
    class_counts = {}

    for cls in ["Printed", "Handwritten", "Diagram"]:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"⚠️  Skipping missing class folder: {cls_dir}")
            continue
        classes.append(cls)
        imgs = list_images(cls_dir)
        random.shuffle(imgs)

        train, val, test = split_list(imgs, args.train, args.val)

        copy_set(train, out_dir / "train" / cls, move=args.move)
        copy_set(val,   out_dir / "val" / cls,   move=args.move)
        copy_set(test,  out_dir / "test" / cls,  move=args.move)

        class_counts[cls] = {"total": len(imgs), "train": len(train), "val": len(val), "test": len(test)}

    # Save labels.json alongside models later; for now put a copy in data/
    ensure_dir(out_dir)
    labels_path = out_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(classes, f)
    print(f"\n✅ Labels saved to {labels_path} -> {classes}")

    print("\n📊 Split summary:")
    for cls, c in class_counts.items():
        print(f"  {cls:12s} total={c['total']:3d} | train={c['train']:3d} val={c['val']:3d} test={c['test']:3d}")

    print(f"\n📁 Output created in: {out_dir.resolve()}")
    print("   └── train/ Printed|Handwritten|Diagram")
    print("   └── val/   Printed|Handwritten|Diagram")
    print("   └── test/  Printed|Handwritten|Diagram")

if __name__ == "__main__":
    main()

#checking
#python dataset_prepare.py --raw_dir data_raw --out_dir data
#on rasp
#python3 dataset_prepare.py --raw_dir data_raw --out_dir data
#train for windows
#python -m src.train
