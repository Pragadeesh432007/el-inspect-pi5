"""Evaluate a trained PVEL-AD model on a prepared test split (no training, no tuning).

    python eval_pvel.py --data data\\pvel_test

Uses models/pvel_best.pt and the per-class thresholds saved in models/pvel_thresholds.json
(tuned on validation during training). Prints per-class AUC/AP/precision/recall/F1 and the
defective-vs-good result, and saves models/pvel_eval_<dataname>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from train_pvel import ELDS, build_model, predict, safe_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pvel_test")
    ap.add_argument("--model", default="models/pvel_best.pt")
    ap.add_argument("--thr", default="models/pvel_thresholds.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    d = Path(a.data)
    classes = json.loads((d / "classes.json").read_text()) + ["DEFECTIVE"]
    thr_map = json.loads(Path(a.thr).read_text())
    thr = [thr_map[c] for c in classes]
    X = np.asarray(np.load(d / f"{a.split}_x.npy"))
    Y = np.load(d / f"{a.split}_y.npy")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(classes)).to(dev)
    model.load_state_dict(torch.load(a.model, map_location=dev))
    yy, pp = predict(model, DataLoader(ELDS(X, Y, False), batch_size=32), dev)

    rows = {}
    print(f"=== {a.split.upper()} from {d} (n={len(yy)}) | thresholds from validation ===")
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
    fa, ng = int((pred & (yk == 0)).sum()), int((yk == 0).sum())
    ms, nd = int((~pred & (yk == 1)).sum()), int((yk == 1).sum())
    print(f"\ndefective-vs-good: accuracy {acc:.4f} | false alarms {fa} of {ng} good "
          f"({100*fa/max(ng,1):.1f}%) | missed {ms} of {nd} defective ({100*ms/max(nd,1):.1f}%)")
    rows["defective_vs_good"] = {"accuracy": acc, "false_alarms": fa, "n_good": ng,
                                 "missed": ms, "n_defective": nd}
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / f"pvel_eval_{d.name}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
