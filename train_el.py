"""Train a small CNN on ELPV: functional (0) vs defective (1) solar cells.

    python train_el.py --epochs 20
ELPV labels are defect probabilities {0, 1/3, 2/3, 1}. We call a cell DEFECTIVE
when p >= 0.5 (i.e. 2/3 and 1) - the usual binary convention; change --thr to
alter it. Split: fixed stratified 70/15/15 (seeded, so results are reproducible).
Outputs (models/): el_best.pt, el_metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models


def load_elpv(thr, seed):
    from elpv_dataset.utils import load_dataset
    out = load_dataset()
    images, proba = out[0], out[1]
    y = (np.asarray(proba) >= thr).astype(np.int64)
    idx = np.arange(len(y))
    # Own fixed, seeded, stratified split: the installed package version may or may
    # not ship the official train/test mask, so we don't depend on it.
    trainval, test_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
    tr, va = train_test_split(trainval, test_size=0.15 / 0.85, stratify=y[trainval],
                              random_state=seed)
    return images, y, tr, va, test_idx


class ELSet(Dataset):
    def __init__(self, images, y, idx, train):
        self.x, self.y, self.idx, self.train = images, y, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        im = torch.from_numpy(self.x[j].astype(np.float32) / 255.0)[None]  # 1x300x300
        if self.train:
            if torch.rand(1) < 0.5: im = im.flip(2)
            if torch.rand(1) < 0.5: im = im.flip(1)
            im = im * (0.85 + 0.3 * torch.rand(1))          # brightness jitter
            im = im + 0.02 * torch.randn_like(im)           # sensor-noise robustness
        im = torch.nn.functional.interpolate(im[None], size=224, mode="bilinear",
                                             align_corners=False)[0]
        im = (im - 0.449) / 0.226
        return im.repeat(3, 1, 1), int(self.y[j])


def evaluate(model, dl, dev):
    model.eval(); ys, ps, pr = [], [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(model(x.to(dev)), 1)[:, 1].cpu()
            pr.append(p); ps.append((p > 0.5).long()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(pr).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train samples")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True)
    images, y, tr, va, te = load_elpv(a.thr, a.seed)
    if a.limit:
        tr = tr[:a.limit]
    print(f"device {dev} | train {len(tr)} val {len(va)} test {len(te)} | "
          f"defective share train {y[tr].mean():.2f} test {y[te].mean():.2f}")

    mk = lambda idx, t: DataLoader(ELSet(images, y, idx, t), batch_size=a.bs, shuffle=t)
    dtr, dva, dte = mk(tr, True), mk(va, False), mk(te, False)

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as e:   # no internet for the weight download
        print("WARNING: could not load ImageNet weights, training from scratch:", repr(e)[:80])
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(dev)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * len(dtr))
    best = 0.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, t in dtr:
            x, t = x.to(dev), t.to(dev)
            opt.zero_grad(); loss = crit(model(x), t); loss.backward()
            opt.step(); sched.step(); tot += loss.item() * len(x)
        yv, pv, prv = evaluate(model, dva, dev)
        acc = float((yv == pv).mean())
        print(f"epoch {ep+1:02d}/{a.epochs} loss {tot/len(tr):.4f} val_acc {acc:.4f}")
        if acc >= best:
            best = acc; torch.save(model.state_dict(), out / "el_best.pt")

    model.load_state_dict(torch.load(out / "el_best.pt", map_location=dev))
    yt, pt, prt = evaluate(model, dte, dev)
    names = ["functional", "defective"]
    print(classification_report(yt, pt, target_names=names, zero_division=0))
    auc = float(roc_auc_score(yt, prt))
    print(f"TEST accuracy {(yt == pt).mean():.4f} | ROC-AUC {auc:.4f}")
    (out / "el_metrics.json").write_text(json.dumps({
        "best_val_acc": best, "test_acc": float((yt == pt).mean()), "roc_auc": auc,
        "confusion": confusion_matrix(yt, pt).tolist(),
        "report": classification_report(yt, pt, target_names=names, output_dict=True,
                                        zero_division=0)}, indent=2))


if __name__ == "__main__":
    main()
