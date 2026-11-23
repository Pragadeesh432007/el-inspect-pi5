"""Evaluate FP32 and INT8 ONNX models on the full prepared test split (same table as eval_pvel.py).

    python eval_onnx.py --data data\\pvel_test --bundle pi_bundle

Prints, for each model, per-class precision/recall/F1 and defective-vs-good accuracy, false alarms
and misses, using the validation-tuned thresholds. Saves models/onnx_eval_<model>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score, roc_auc_score

MEAN, STD = 0.449, 0.226


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pvel_test")
    ap.add_argument("--bundle", default="pi_bundle")
    ap.add_argument("--models", nargs="+", default=["pvel_fp32.onnx", "pvel_int8.onnx"])
    ap.add_argument("--thr", default="models/pvel_thresholds.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    d, b = Path(a.data), Path(a.bundle)
    classes = json.loads((d / "classes.json").read_text()) + ["DEFECTIVE"]
    thr_fp32 = json.loads(Path(a.thr).read_text())
    X = np.load(d / "test_x.npy", mmap_mode="r"); Y = np.load(d / "test_y.npy")
    Y8 = np.concatenate([Y, (Y.sum(1, keepdims=True) > 0)], 1)
    Path(a.out).mkdir(exist_ok=True)

    for m in a.models:
        t8 = b / "thresholds_int8.json"
        thr = json.loads(t8.read_text()) if ("int8" in m and t8.exists()) else thr_fp32
        print(f"\n{m}: using {'thresholds_int8.json' if thr is not thr_fp32 else a.thr}")
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        s = ort.InferenceSession(str(b / m), so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        P = np.zeros((len(X), len(classes)), np.float32)
        for i in range(len(X)):
            x = ((np.asarray(X[i], np.float32) / 255.0 - MEAN) / STD)[None, None]
            P[i] = 1 / (1 + np.exp(-s.run(None, {name: x})[0][0]))
            if (i + 1) % 1000 == 0:
                print(f"   {m}: {i+1}/{len(X)}", flush=True)
        rows = {}
        print(f"\n=== {m} on {d} (n={len(X)}) | thresholds from validation ===")
        print(f"{'class':14s}{'support':>8s}{'AUC':>7s}{'AP':>7s}{'prec':>7s}{'recall':>7s}{'F1':>7s}")
        for k, c in enumerate(classes):
            yk, pk = Y8[:, k], P[:, k]
            pred = pk >= thr[c]
            tp = int((pred & (yk == 1)).sum()); fp = int((pred & (yk == 0)).sum()); fn = int((~pred & (yk == 1)).sum())
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
            auc = float(roc_auc_score(yk, pk)) if 0 < yk.sum() < len(yk) else float("nan")
            ap_ = float(average_precision_score(yk, pk)) if yk.sum() else float("nan")
            rows[c] = {"support": int(yk.sum()), "auc": auc, "ap": ap_, "precision": pr, "recall": rc, "f1": f1}
            print(f"{c:14s}{int(yk.sum()):8d}{auc:7.3f}{ap_:7.3f}{pr:7.3f}{rc:7.3f}{f1:7.3f}")
        k = len(classes) - 1
        pred = P[:, k] >= thr["DEFECTIVE"]; yk = Y8[:, k]
        fa, ng = int((pred & (yk == 0)).sum()), int((yk == 0).sum())
        ms, nd = int((~pred & (yk == 1)).sum()), int((yk == 1).sum())
        acc = float((pred == (yk == 1)).mean())
        print(f"\ndefective-vs-good: accuracy {acc:.4f} | false alarms {fa} of {ng} good ({100*fa/max(ng,1):.1f}%)"
              f" | missed {ms} of {nd} defective ({100*ms/max(nd,1):.1f}%)")
        rows["defective_vs_good"] = {"accuracy": acc, "false_alarms": fa, "n_good": ng, "missed": ms, "n_defective": nd}
        (Path(a.out) / f"onnx_eval_{Path(m).stem}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
