"""Export the trained PVEL-AD model to ONNX (FP32) and build
a small folder (pi_bundle) to copy to the Raspberry Pi.

    python export_onnx.py --data data\\pvel_test --calib data\\pvel

pi_bundle/ gets: pvel_fp32.onnx, sample_x.npy, sample_y.npy (a mixed good/defective
sample of the TEST split), thresholds.json, classes.json, bench_onnx.py, pi_check.py
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from train_pvel import MEAN, STD, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pvel_test", help="folder with test_x.npy / test_y.npy")
    ap.add_argument("--calib", default="data/pvel", help="folder with val_x.npy (INT8 calibration)")
    ap.add_argument("--model", default="models/pvel_best.pt")
    ap.add_argument("--thr", default="models/pvel_thresholds.json")
    ap.add_argument("--out", default="pi_bundle")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-sample", type=int, default=300, help="images copied for the Pi benchmark")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import onnx

    out = Path(a.out); out.mkdir(exist_ok=True)
    classes = json.loads((Path(a.data) / "classes.json").read_text())
    Xt = np.load(Path(a.data) / "test_x.npy", mmap_mode="r")
    Yt = np.load(Path(a.data) / "test_y.npy")
    size = Xt.shape[1]
    rng = np.random.RandomState(a.seed)

    # --- sample for the Pi: half defective, half good (if both exist) --------------------
    isdef = Yt.sum(1) > 0
    nd, ng = int(isdef.sum()), int((~isdef).sum())
    k = a.n_sample // 2 if ng else a.n_sample
    pick = list(rng.choice(np.where(isdef)[0], min(k, nd), replace=False))
    if ng:
        pick += list(rng.choice(np.where(~isdef)[0], min(a.n_sample - len(pick), ng), replace=False))
    pick = np.array(sorted(pick))
    np.save(out / "sample_x.npy", np.asarray(Xt[pick])); np.save(out / "sample_y.npy", Yt[pick])

    # --- FP32 ONNX ------------------------------------------------------------------------
    model = build_model(len(classes) + 1)
    model.load_state_dict(torch.load(a.model, map_location="cpu")); model.eval()
    dummy = torch.zeros(1, 1, size, size)
    fp32 = out / "pvel_fp32.onnx"
    try:
        torch.onnx.export(model, dummy, str(fp32), input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(model, dummy, str(fp32), input_names=["image"], output_names=["logits"],
                          opset_version=17)
    onnx.checker.check_model(onnx.load(str(fp32)))
    print(f"FP32 ONNX: {fp32} ({fp32.stat().st_size/1024**2:.2f} MB), input 1x1x{size}x{size}")

    # --- thresholds / classes / helper scripts -------------------------------------------
    shutil.copy(a.thr, out / "thresholds.json")
    (out / "classes.json").write_text(json.dumps(classes))
    here = Path(__file__).parent
    for f in ("bench_onnx.py", "pi_check.py"):
        if (here / f).exists():
            shutil.copy(here / f, out / f)

    print("\nNext: python quantize_sweep.py --val data\\pvel --test data\\pvel_test   (builds the INT8 model)")
    print(f"Folder for the Pi: {out}\\")


if __name__ == "__main__":
    main()
