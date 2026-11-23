"""Train the PVEL-AD cell classifier: MobileNetV3-Small, 1-channel EL input, multi-label.

    python train_pvel.py --data data\\pvel --epochs 12

Outputs 8 logits: 7 defect groups (crack, finger, black_core, thick_line, short_circuit,
dislocation, other) + 1 'defective' (any defect). Best epoch = highest mean validation ROC-AUC.
Per-class decision thresholds are tuned on VALIDATION only, then applied once to TEST.
Saves in models/: pvel_best.pt, pvel_thresholds.json, pvel_metrics.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

MEAN, STD = 0.449, 0.226


class ELDS(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train
        self.y8 = np.concatenate([y, (y.sum(1, keepdims=True) > 0).astype(np.uint8)], 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        im = torch.from_numpy(self.x[i].astype(np.float32) / 255.0)[None]      # 1xSxS
        if self.train:
            if torch.rand(1) < 0.5: im = im.flip(2)
            if torch.rand(1) < 0.5: im = im.flip(1)
            im = im.clamp(1e-4, 1) ** float(0.7 + 0.6 * torch.rand(1))           # gamma (EL brightness varies)
            im = im * (0.8 + 0.4 * torch.rand(1))
            im = im + 0.015 * torch.randn_like(im)
        im = (im - MEAN) / STD
        return im, torch.from_numpy(self.y8[i].astype(np.float32))


def build_model(n_out):
    try:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as e:
        print("WARNING: no ImageNet weights, training from scratch:", repr(e)[:80])
        m = models.mobilenet_v3_small(weights=None)
    old = m.features[0][0]
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    new.weight.data = old.weight.data.sum(1, keepdim=True)        # same response to grey input
    m.features[0][0] = new
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_out)
    return m


def predict(model, dl, dev):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.sigmoid(model(x.to(dev))).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def best_thr(y, p):
    if y.sum() == 0:
        return 0.5
    ts = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pvel")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: N train images")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, out = Path(a.data), Path(a.out); out.mkdir(exist_ok=True)
    classes = json.loads((d / "classes.json").read_text()) + ["DEFECTIVE"]
    ld = lambda n: (np.load(d / f"{n}_x.npy", mmap_mode="r"), np.load(d / f"{n}_y.npy"))
    (xtr, ytr), (xva, yva) = ld("train"), ld("val")
    has_test = (d / "test_x.npy").exists()
    xte, yte = ld("test") if has_test else (None, None)
    if a.limit:
        sel = np.random.permutation(len(ytr))[:a.limit]; xtr, ytr = np.asarray(xtr)[sel], ytr[sel]
    xtr, xva = np.asarray(xtr), np.asarray(xva)
    print(f"device {dev} | image {xtr.shape[1]}px | train {len(ytr)} val {len(yva)} "
          f"test {len(yte) if has_test else 0}")

    mk = lambda x, y, t: DataLoader(ELDS(x, y, t), batch_size=a.bs, shuffle=t, num_workers=a.workers)
    dtr, dva = mk(xtr, ytr, True), mk(xva, yva, False)

    model = build_model(len(classes)).to(dev)
    y8 = np.concatenate([ytr, (ytr.sum(1, keepdims=True) > 0)], 1)
    pos = y8.sum(0).clip(min=1)
    pw = torch.tensor(np.clip((len(y8) - pos) / pos, 1.0, 6.0), dtype=torch.float32).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dtr))

    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, t in dtr:
            x, t = x.to(dev), t.to(dev)
            opt.zero_grad(); loss = crit(model(x), t); loss.backward()
            opt.step(); sched.step(); tot += loss.item() * len(x)
        yv, pv = predict(model, dva, dev)
        aucs = [safe_auc(yv[:, k], pv[:, k]) for k in range(len(classes))]
        score = float(np.nanmean(aucs))
        print(f"epoch {ep+1:02d}/{a.epochs} loss {tot/len(ytr):.4f} | val mean-AUC {score:.4f} "
              f"| DEFECTIVE auc {aucs[-1]:.3f} | {(time.time()-t0)/60:.1f} min", flush=True)
        if ep >= a.epochs // 3 and score >= best:
            best = score; torch.save(model.state_dict(), out / "pvel_best.pt")
            print("   -> saved (best so far)")

    model.load_state_dict(torch.load(out / "pvel_best.pt", map_location=dev))
    yv, pv = predict(model, dva, dev)
    thr = [best_thr(yv[:, k], pv[:, k]) for k in range(len(classes))]
    (out / "pvel_thresholds.json").write_text(json.dumps(dict(zip(classes, thr)), indent=2))
    res = {"best_val_mean_auc": best, "thresholds": dict(zip(classes, thr))}

    for name, (X, Y) in (("val", (xva, yva)), ("test", (xte, yte))):
        if X is None:
            continue
        yy, pp = predict(model, mk(np.asarray(X), Y, False), dev)
        rows = {}
        print(f"\n=== {name.upper()} (n={len(yy)}) | thresholds tuned on val ===")
        print(f"{'class':14s}{'support':>8s}{'AUC':>7s}{'AP':>7s}{'prec':>7s}{'recall':>7s}{'F1':>7s}")
        for k, c in enumerate(classes):
            yk, pk = yy[:, k], pp[:, k]
            pred = pk >= thr[k]
            tp = int((pred & (yk == 1)).sum()); fp = int((pred & (yk == 0)).sum()); fn = int((~pred & (yk == 1)).sum())
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
            auc = safe_auc(yk, pk); ap_ = float(average_precision_score(yk, pk)) if yk.sum() else float("nan")
            rows[c] = {"support": int(yk.sum()), "auc": auc, "ap": ap_, "precision": pr, "recall": rc, "f1": f1}
            print(f"{c:14s}{int(yk.sum()):8d}{auc:7.3f}{ap_:7.3f}{pr:7.3f}{rc:7.3f}{f1:7.3f}")
        k = len(classes) - 1
        pred = pp[:, k] >= thr[k]; yk = yy[:, k]
        acc = float((pred == (yk == 1)).mean())
        print(f"defective-vs-good accuracy {acc:.4f} | false alarms {int((pred & (yk==0)).sum())} "
              f"of {int((yk==0).sum())} good | missed {int((~pred & (yk==1)).sum())} of {int((yk==1).sum())} defective")
        rows["defective_vs_good_accuracy"] = acc
        res[name] = rows
    (out / "pvel_metrics.json").write_text(json.dumps(res, indent=2))
    print("\nsaved", out / "pvel_best.pt", "| pvel_thresholds.json | pvel_metrics.json")


if __name__ == "__main__":
    main()
