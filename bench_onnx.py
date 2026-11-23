"""Benchmark ONNX models on any machine (PC or Raspberry Pi 5). Needs only numpy + onnxruntime.

    python bench_onnx.py --models pvel_fp32.onnx pvel_int8.onnx --threads 4

Reads sample_x.npy / sample_y.npy / thresholds.json / classes.json from the same folder.
Reports per model: file size, load time, latency (mean/p50/p95), FPS, process RAM, and
accuracy on the sample (DEFECTIVE flag at the validation threshold + AUC), plus per-class
agreement with the FP32 model. Results are also written to bench_<tag>.json.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MEAN, STD = 0.449, 0.226


def rss_mb():
    try:                                   # Linux / Raspberry Pi
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:                                   # Windows / macOS, if psutil is installed
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return float("nan")


def auc(y, p):
    """ROC-AUC via average ranks (ties handled); no sklearn needed on the Pi."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    u, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg_rank[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def prep(x):
    return ((x.astype(np.float32) / 255.0 - MEAN) / STD)[None, None]      # 1x1xSxS


def run_model(path, X, threads, warmup, repeat):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    name = sess.get_inputs()[0].name
    xs = [prep(x) for x in X]
    for i in range(warmup):
        sess.run(None, {name: xs[i % len(xs)]})
    lat = []
    for i in range(repeat):
        t = time.perf_counter()
        sess.run(None, {name: xs[i % len(xs)]})
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    logits = np.concatenate([sess.run(None, {name: a})[0] for a in xs])
    return {"load_s": load_s, "lat_ms_mean": float(lat.mean()), "lat_ms_p50": float(np.percentile(lat, 50)),
            "lat_ms_p95": float(np.percentile(lat, 95)), "fps": float(1000 / lat.mean()),
            "rss_mb": rss_mb(), "size_mb": os.path.getsize(path) / 1024 ** 2}, 1 / (1 + np.exp(-logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--n", type=int, default=0, help="use only the first N sample images")
    ap.add_argument("--tag", default=platform.node() or "device")
    a = ap.parse_args()
    d = Path(a.dir)
    X = np.load(d / "sample_x.npy"); Y = np.load(d / "sample_y.npy")
    if a.n:
        X, Y = X[:a.n], Y[:a.n]
    classes = json.loads((d / "classes.json").read_text()) + ["DEFECTIVE"]
    thr = json.loads((d / "thresholds.json").read_text())
    yd = (Y.sum(1) > 0).astype(int)
    print(f"{platform.platform()} | {platform.machine()} | onnxruntime {ort.__version__} | "
          f"threads {a.threads} | sample {len(X)} images ({int(yd.sum())} defective)")
    res, probs = {}, {}
    for m in a.models:
        r, p = run_model(d / m, X, a.threads, a.warmup, a.repeat)
        pdef = p[:, -1]
        r["defective_auc"] = auc(yd, pdef)
        r["defective_acc"] = float(((pdef >= thr["DEFECTIVE"]) == (yd == 1)).mean())
        res[m], probs[m] = r, p
        print(f"\n{m}: size {r['size_mb']:.2f} MB | load {r['load_s']:.2f} s | "
              f"latency mean {r['lat_ms_mean']:.1f} ms (p50 {r['lat_ms_p50']:.1f}, p95 {r['lat_ms_p95']:.1f}) | "
              f"{r['fps']:.1f} FPS | RAM {r['rss_mb']:.0f} MB | "
              f"DEFECTIVE acc {r['defective_acc']:.3f} AUC {r['defective_auc']:.3f}")
    if len(a.models) > 1:
        ref = probs[a.models[0]]
        for m in a.models[1:]:
            agree = {c: float(((ref[:, k] >= thr[c]) == (probs[m][:, k] >= thr[c])).mean())
                     for k, c in enumerate(classes)}
            res[m]["agreement_with_first"] = agree
            print(f"\n{m} agrees with {a.models[0]} on: " +
                  ", ".join(f"{c} {v*100:.1f}%" for c, v in agree.items()))
    res["_info"] = {"platform": platform.platform(), "machine": platform.machine(),
                    "ort": ort.__version__, "threads": a.threads, "n_sample": int(len(X))}
    (d / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved bench_{a.tag}.json")


if __name__ == "__main__":
    main()
