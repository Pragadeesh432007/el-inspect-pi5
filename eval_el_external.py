"""Test the trained EL model (models/el_best.pt) on an EXTERNAL dataset laid out as
    <data>/<split>/DEFECT/*.jpg   and   <data>/<split>/NORMAL/*.jpg
(e.g. the Roboflow SOLAR-CRACK-DETECT folder export).

This is a cross-dataset test: the model never saw these images, so the result shows
how well it generalises. Preprocessing is identical to train_el.py
(grayscale -> 224x224 -> normalise -> 3 channels).

    python eval_el_external.py --data extra_el\\roboflow_crack --splits valid test

Outputs: printed metrics + models/el_external_<name>.json + a CSV of every prediction
(models/el_external_<name>_predictions.csv) for error analysis.
Folders with other names (e.g. "DEFECT NORMAL" - ambiguous labels) are ignored.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve)
from torchvision import models

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_img(path):
    im = Image.open(path).convert("L").resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
    x = (x - 0.449) / 0.226
    return x.repeat(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--pos", default="DEFECT", help="folder name of the defective class")
    ap.add_argument("--neg", default="NORMAL", help="folder name of the functional class")
    ap.add_argument("--model", default="models/el_best.pt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()

    root = Path(a.data)
    items = []                      # (path, label, split)
    for split in a.splits:
        for name, label in ((a.neg, 0), (a.pos, 1)):
            d = root / split / name
            if not d.is_dir():
                print("missing folder, skipped:", d)
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in EXTS:
                    items.append((p, label, split))
    if not items:
        raise SystemExit("No images found - check --data / --splits / folder names")
    print(f"{len(items)} images | defective {sum(l for _, l, _ in items)} | "
          f"functional {sum(1 - l for _, l, _ in items)}")

    try:
        net = models.mobilenet_v3_small(weights=None)
    except TypeError:
        net = models.mobilenet_v3_small(pretrained=False)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
    net.load_state_dict(torch.load(a.model, map_location="cpu"))
    net.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(items), a.bs):
            xb = torch.stack([load_img(p) for p, _, _ in items[i:i + a.bs]])
            probs.extend(torch.softmax(net(xb), 1)[:, 1].tolist())
    y = np.array([l for _, l, _ in items])
    pr = np.array(probs)
    pred = (pr > 0.5).astype(int)

    names = ["functional", "defective"]
    print("\n=== all selected splits, threshold 0.5 ===")
    print(classification_report(y, pred, target_names=names, zero_division=0))
    auc = float(roc_auc_score(y, pr)) if len(set(y)) == 2 else float("nan")
    cm = confusion_matrix(y, pred).tolist()
    print("confusion matrix [[TN FP] [FN TP]]:", cm)
    print(f"accuracy {(y == pred).mean():.4f} | ROC-AUC {auc:.4f}")

    # best threshold (Youden) - informative only: it is chosen on these same images
    fpr, tpr, thr = roc_curve(y, pr)
    best = int(np.argmax(tpr - fpr))
    bt = float(thr[best])
    if not np.isfinite(bt):
        bt = 0.5
    acc_best = float(((pr > bt).astype(int) == y).mean())
    print(f"(info) best threshold on this set {bt:.2f} -> accuracy {acc_best:.4f}; "
          "if you use it, pick it on 'valid' and report on 'test' only")

    per_split = {}
    for s in a.splits:
        m = np.array([sp == s for _, _, sp in items])
        if m.sum() and len(set(y[m])) == 2:
            per_split[s] = {"n": int(m.sum()),
                            "accuracy": float((y[m] == pred[m]).mean()),
                            "roc_auc": float(roc_auc_score(y[m], pr[m]))}
            print(f"{s}: n={per_split[s]['n']} acc={per_split[s]['accuracy']:.4f} "
                  f"auc={per_split[s]['roc_auc']:.4f}")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    tag = root.name
    (out / f"el_external_{tag}.json").write_text(json.dumps({
        "dataset": str(root), "n": len(items), "accuracy": float((y == pred).mean()),
        "roc_auc": auc, "confusion": cm, "per_split": per_split,
        "info_best_threshold": bt, "info_accuracy_at_best_threshold": acc_best}, indent=2))
    with open(out / f"el_external_{tag}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "true_label", "p_defective", "pred"])
        for (p, l, s), q, d in zip(items, pr, pred):
            w.writerow([str(p), s, l, f"{q:.4f}", int(d)])
    print(f"\nsaved models\\el_external_{tag}.json and ..._predictions.csv")


if __name__ == "__main__":
    main()
