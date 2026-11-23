"""Clean and split the RGB solar-panel fault dataset.

Input : a folder with one sub-folder per class (ImageFolder style), e.g.
          raw_rgb/Bird-drop/*.jpg  raw_rgb/Clean/*.jpg  raw_rgb/Dusty/*.jpg ...
        Works also if the download is already split (train/valid/test/<class>/...):
        the class is taken from the folder directly above each image, all splits
        are pooled, de-duplicated, then re-split.
Output: data/rgb/{train,val,test}/<class>/*.jpg  + data/rgb/report.json

Why de-duplicate before splitting: web/Roboflow sets contain many near-identical
images (augmentations, re-uploads). If copies land in both train and test, the
test accuracy is inflated. We hash every image (perceptual hash) and keep one
per near-duplicate group, THEN split.

Usage:
    python prep_rgb.py --src raw_rgb --dst data/rgb
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_class(name: str) -> str:
    """Make class names consistent across datasets (Bird-drop / bird drops ...)."""
    n = name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "bird-drop": "bird-drop", "bird-drops": "bird-drop", "bird-dropping": "bird-drop",
        "bird-droppings": "bird-drop", "birddrop": "bird-drop",
        "clean": "clean", "clean-panels": "clean", "clean-panel": "clean",
        "non-defective": "clean", "nondefective": "clean",
        "dusty": "dusty", "dust": "dusty",
        "electrical-damage": "electrical-damage", "electrical-damge": "electrical-damage",
        "physical-damage": "physical-damage", "physical-damge": "physical-damage",
        "snow": "snow-covered", "snow-covered": "snow-covered",
    }
    return aliases.get(n, n)


def collect(src: Path):
    """Return list of (path, class_name); class = folder just above the image."""
    items = []
    no_class = {"train", "valid", "val", "test", "validation", "images"}
    skipped = 0
    for p in src.rglob("*"):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        if p.parent.name.lower() in no_class:
            skipped += 1   # YOLO-style layout (labels in .txt) - no class folder
            continue
        items.append((p, norm_class(p.parent.name)))
    if skipped:
        print(f"Skipped {skipped} images with no class folder "
              "(detection-format exports; use the classification export instead)")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/rgb")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--hash-dist", type=int, default=4,
                    help="max perceptual-hash distance counted as duplicate")
    ap.add_argument("--min-side", type=int, default=64, help="drop tiny images")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    random.seed(a.seed)
    items = collect(src)
    if not items:
        raise SystemExit(f"No images found under {src}")

    kept = defaultdict(list)           # class -> [path]
    seen_by_cls = defaultdict(list)
    stats = {"found": len(items), "unreadable": 0, "too_small": 0, "duplicates": 0}

    random.shuffle(items)
    for p, cls in items:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if min(im.size) < a.min_side:
                    stats["too_small"] += 1
                    continue
                h = imagehash.phash(im)
        except Exception:
            stats["unreadable"] += 1
            continue
        # near-duplicate against everything already kept in ANY class
        # (the same image under two labels is also a problem)
        dup = False
        for c2, hs in seen_by_cls.items():
            if any(h - h2 <= a.hash_dist for h2 in hs):
                dup = True
                break
        if dup:
            stats["duplicates"] += 1
            continue
        seen_by_cls[cls].append(h)
        kept[cls].append(p)

    if dst.exists():
        shutil.rmtree(dst)
    counts = {}
    for cls, paths in kept.items():
        random.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(n * a.test))
        n_val = max(1, int(n * a.val))
        splits = {
            "test": paths[:n_test],
            "val": paths[n_test:n_test + n_val],
            "train": paths[n_test + n_val:],
        }
        counts[cls] = {k: len(v) for k, v in splits.items()}
        for split, plist in splits.items():
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(plist):
                shutil.copy2(p, out / f"{i:05d}{p.suffix.lower()}")

    report = {"stats": stats, "per_class": counts, "classes": sorted(kept)}
    (dst / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    small = [c for c, v in counts.items() if v["train"] < 100]
    if small:
        print("\nWARNING: few training images for:", small)


if __name__ == "__main__":
    main()
