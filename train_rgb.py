"""Fine-tune MobileNetV3-Small on the cleaned six-class RGB solar-panel set.

    python train_rgb.py --data data/rgb --epochs 15
Outputs (in models/): rgb_best.pt, rgb_classes.json, rgb_metrics.json,
                      rgb_confusion.png (if matplotlib is installed)
Tip: use a GPU (Colab) if the laptop is slow; CPU works but takes longer.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def loaders(root, size, bs, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),   # helps with sun/shade changes
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = {s: datasets.ImageFolder(Path(root) / s, train_tf if s == "train" else eval_tf)
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(ds[s], batch_size=bs, shuffle=(s == "train"),
                        num_workers=workers) for s in ds}
    return ds, dl


def evaluate(model, dl, dev):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(model(x.to(dev)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rgb")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True)
    ds, dl = loaders(a.data, a.size, a.bs, a.workers)
    classes = ds["train"].classes
    print("device:", dev, "| classes:", classes,
          "| sizes:", {k: len(v) for k, v in ds.items()})

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as e:   # no internet for the weight download
        print("WARNING: could not load ImageNet weights, training from scratch:", repr(e)[:80])
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(dev)

    # class weights: protects against imbalanced classes
    counts = np.bincount([y for _, y in ds["train"].samples], minlength=len(classes))
    w = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl["train"]))

    best, t0 = 0.0, time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x)
        yv, pv = evaluate(model, dl["val"], dev)
        acc = float((yv == pv).mean())
        print(f"epoch {ep+1:02d}/{a.epochs} loss {tot/len(ds['train']):.4f} "
              f"val_acc {acc:.4f} ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "rgb_best.pt")

    model.load_state_dict(torch.load(out / "rgb_best.pt", map_location=dev))
    yt, pt = evaluate(model, dl["test"], dev)
    rep = classification_report(yt, pt, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(yt, pt).tolist()
    print(classification_report(yt, pt, target_names=classes, zero_division=0))
    (out / "rgb_classes.json").write_text(json.dumps(classes))
    (out / "rgb_metrics.json").write_text(json.dumps(
        {"best_val_acc": best, "test_acc": float((yt == pt).mean()),
         "report": rep, "confusion": cm, "classes": classes}, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i][j], ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout()
        fig.savefig(out / "rgb_confusion.png", dpi=150)
    except ImportError:
        pass
    print(f"TEST accuracy: {(yt == pt).mean():.4f}")


if __name__ == "__main__":
    main()
