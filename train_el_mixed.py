"""Train the EL model on ELPV + Roboflow crack set (mixed sources) to reduce domain shift.

    python train_el_mixed.py --rf extra_el\\roboflow_crack --epochs 20

Data use (no leakage):
  train : ELPV train split  + Roboflow train/
  val   : ELPV val split    + Roboflow valid/   (used only to pick the best epoch)
  test  : ELPV test split   and   Roboflow test/  (reported SEPARATELY, never trained on)
The ELPV split is identical to train_el.py (same seed), so numbers are comparable.
Saves models/el_mixed_best.pt (el_best.pt is NOT overwritten) and models/el_mixed_metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from train_el import load_elpv

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def to224(arr_or_img):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    return np.asarray(im.convert("L").resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def load_rf(root, split, pos, neg):
    xs, ys = [], []
    for label, cls in ((1, pos), (0, neg)):
        d = Path(root) / split / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXT:
                try:
                    xs.append(to224(Image.open(p)))
                    ys.append(label)
                except Exception as e:
                    print("skip", p.name, repr(e)[:60])
    return np.stack(xs), np.array(ys, dtype=np.int64)


class DS(Dataset):
    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = torch.from_numpy(self.x[i].astype(np.float32) / 255.0)[None]
        if self.train:
            if torch.rand(1) < 0.5: im = im.flip(2)
            if torch.rand(1) < 0.5: im = im.flip(1)
            im = im * (0.8 + 0.4 * torch.rand(1))                    # brightness
            im = (im - im.mean()) * (0.8 + 0.4 * torch.rand(1)) + im.mean()  # contrast
            im = im + 0.02 * torch.randn_like(im)
        im = (im - 0.449) / 0.226
        return im.repeat(3, 1, 1), int(self.y[i])


def evaluate(model, dl, dev):
    model.eval(); ys, ps, pr = [], [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(model(x.to(dev)), 1)[:, 1].cpu()
            pr.append(p); ps.append((p > 0.5).long()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(pr).numpy()


def report(name, yt, pt, prt):
    auc = float(roc_auc_score(yt, prt))
    acc = float((yt == pt).mean())
    print(f"\n=== {name} (n={len(yt)}) ===")
    print(classification_report(yt, pt, target_names=["functional", "defective"], zero_division=0))
    print(f"confusion [[TN FP] [FN TP]]: {confusion_matrix(yt, pt).tolist()}")
    print(f"{name}: accuracy {acc:.4f} | ROC-AUC {auc:.4f}")
    return {"n": int(len(yt)), "acc": acc, "auc": auc, "confusion": confusion_matrix(yt, pt).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", default="extra_el/roboflow_crack")
    ap.add_argument("--pos", default="DEFECT")
    ap.add_argument("--neg", default="NORMAL")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use N train samples")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(exist_ok=True)

    images, y, tr, va, te = load_elpv(0.5, a.seed)
    el = lambda idx: (np.stack([to224(images[j]) for j in idx]), y[idx])
    ex_tr, ey_tr = el(tr); ex_va, ey_va = el(va); ex_te, ey_te = el(te)
    rx_tr, ry_tr = load_rf(a.rf, "train", a.pos, a.neg)
    rx_va, ry_va = load_rf(a.rf, "valid", a.pos, a.neg)
    rx_te, ry_te = load_rf(a.rf, "test", a.pos, a.neg)

    Xtr = np.concatenate([ex_tr, rx_tr]); Ytr = np.concatenate([ey_tr, ry_tr])
    Xva = np.concatenate([ex_va, rx_va]); Yva = np.concatenate([ey_va, ry_va])
    if a.limit:
        sel = np.random.permutation(len(Ytr))[:a.limit]; Xtr, Ytr = Xtr[sel], Ytr[sel]
    print(f"device {dev} | train {len(Ytr)} (ELPV {len(ey_tr)} + RF {len(ry_tr)}) | "
          f"val {len(Yva)} | test ELPV {len(ey_te)}, RF {len(ry_te)} | "
          f"defective share train {Ytr.mean():.2f}")

    mk = lambda X, Y, t: DataLoader(DS(X, Y, t), batch_size=a.bs, shuffle=t, num_workers=a.workers)
    dtr, dva = mk(Xtr, Ytr, True), mk(Xva, Yva, False)
    d_el, d_rf = mk(ex_te, ey_te, False), mk(rx_te, ry_te, False)

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as e:
        print("WARNING: no ImageNet weights, training from scratch:", repr(e)[:80])
        model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(dev)
    cnt = np.bincount(Ytr, minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dtr))

    best = -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, t in dtr:
            x, t = x.to(dev), t.to(dev)
            opt.zero_grad(); loss = crit(model(x), t); loss.backward()
            opt.step(); sched.step(); tot += loss.item() * len(x)
        yv, pv, _ = evaluate(model, dva, dev)
        acc = float((yv == pv).mean())
        print(f"epoch {ep+1:02d}/{a.epochs} loss {tot/len(Ytr):.4f} val_acc {acc:.4f}")
        if acc >= best:
            best = acc; torch.save(model.state_dict(), out / "el_mixed_best.pt")

    model.load_state_dict(torch.load(out / "el_mixed_best.pt", map_location=dev))
    res = {"best_val_acc": best}
    res["elpv_test"] = report("ELPV test", *evaluate(model, d_el, dev))
    if len(ry_te):
        res["roboflow_test"] = report("Roboflow test", *evaluate(model, d_rf, dev))
    (out / "el_mixed_metrics.json").write_text(json.dumps(res, indent=2))
    print("\nsaved", out / "el_mixed_best.pt", "and el_mixed_metrics.json")


if __name__ == "__main__":
    main()
