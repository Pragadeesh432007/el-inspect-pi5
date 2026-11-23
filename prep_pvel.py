"""Prepare PVEL-AD (EL2021) for a multi-label defect classifier.

    python prep_pvel.py --root C:\\pvelad\\solar_cell_EL_image\\PVELAD\\EL2021 --out data\\pvel

What it does
  * reads every trainval XML -> image-level multi-hot labels (7 groups, see GROUPS)
  * adds N defect-free cells from othertypes/good (label = all zeros)
  * resizes to --size x --size grayscale uint8 and stores compact .npy arrays
  * trainval + good -> seeded 85/15 train/val split (stratified on each image's rarest class)
  * optional: --test-ann <folder with test XMLs>  -> 'test' split = random --test-n sample of the OFFICIAL
    test images (all defective) + --test-good defect-free cells that train/val never saw
Outputs in --out:  {train,val,test}_x.npy  {train,val,test}_y.npy  {split}_files.txt
                   classes.json  report.json
"""
import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["crack", "finger", "black_core", "thick_line", "short_circuit", "dislocation", "other"]
GROUPS = {
    "crack": "crack", "star_crack": "crack",
    "finger": "finger", "black_core": "black_core", "thick_line": "thick_line",
    "short_circuit": "short_circuit",
    "horizontal_dislocation": "dislocation", "vertical_dislocation": "dislocation",
    "corner": "other", "fragment": "other", "scratch": "other", "printing_error": "other",
}
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_xml(path):
    """-> (multi-hot vector over CLASSES, list of raw names)"""
    y = np.zeros(len(CLASSES), np.uint8)
    names = []
    for o in ET.parse(path).getroot().findall("object"):
        n = (o.findtext("name") or "").strip()
        names.append(n)
        g = GROUPS.get(n)
        if g is None:
            print("  unknown class name ignored:", n, "in", Path(path).name)
            continue
        y[CLASSES.index(g)] = 1
    return y, names


def load_img(path, size):
    im = Image.open(path).convert("L")
    return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def strat_key(y, freq):
    """rarest present class (global frequency) as the stratification key; 'good' = -1"""
    present = [i for i in range(len(CLASSES)) if y[i]]
    return min(present, key=lambda i: freq[i]) if present else -1


def split_idx(keys, val_frac, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(keys))
    cnt = Counter(keys)
    ok = all(c >= 2 for c in cnt.values())
    try:
        return train_test_split(idx, test_size=val_frac, random_state=seed,
                                stratify=keys if ok else None)
    except ValueError:
        return train_test_split(idx, test_size=val_frac, random_state=seed)


def save_split(out, name, files, size, labels, report):
    X = np.zeros((len(files), size, size), np.uint8)
    for i, f in enumerate(files):
        X[i] = load_img(f, size)
        if (i + 1) % 1000 == 0:
            print(f"   {name}: {i+1}/{len(files)}")
    Y = np.stack(labels).astype(np.uint8) if len(labels) else np.zeros((0, len(CLASSES)), np.uint8)
    np.save(out / f"{name}_x.npy", X)
    np.save(out / f"{name}_y.npy", Y)
    (out / f"{name}_files.txt").write_text("\n".join(str(f) for f in files), encoding="utf-8")
    per = {c: int(Y[:, i].sum()) for i, c in enumerate(CLASSES)}
    good = int((Y.sum(1) == 0).sum())
    report[name] = {"n": int(len(files)), "good": good, "defective": int(len(files) - good),
                    "per_class": per}
    print(f"{name}: n={len(files)} good={good} defective={len(files)-good} {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../PVELAD/EL2021")
    ap.add_argument("--out", default="data/pvel")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--good", type=int, default=3000, help="defect-free train/val images to add")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test-ann", default="", help="folder holding the official TEST xml files")
    ap.add_argument("--test-n", type=int, default=4000)
    ap.add_argument("--test-good", type=int, default=1500,
                    help="extra defect-free test cells from othertypes/good (disjoint from train/val; "
                         "the official test split contains only defective cells)")
    ap.add_argument("--only-test", action="store_true",
                    help="build ONLY the test split (use a different --out than the training data)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap each source at N images")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    zero = np.zeros(len(CLASSES), np.uint8)
    report = {"classes": CLASSES, "size": a.size, "groups": GROUPS}

    # defect-free cells: same seeded shuffle in every mode, so train/val goods and the
    # spare goods used for the test split never overlap
    goods = sorted(p for p in (root / "othertypes" / "good").iterdir() if p.suffix.lower() in EXT)
    rng.shuffle(goods)
    n_good = min(a.good, a.limit) if a.limit else a.good
    spare_goods = goods[n_good:]
    goods = goods[:n_good]

    if not a.only_test:
        items = []                                   # (image path, label vector)
        ann_dir, img_dir = root / "trainval" / "Annotations", root / "trainval" / "JPEGImages"
        xmls = sorted(ann_dir.glob("*.xml"))
        if a.limit:
            xmls = xmls[:a.limit]
        missing = 0
        for x in xmls:
            y, _ = read_xml(x)
            img = next((img_dir / (x.stem + e) for e in (".jpg", ".jpeg", ".png", ".bmp")
                        if (img_dir / (x.stem + e)).exists()), None)
            if img is None:
                missing += 1
                continue
            items.append((img, y))
        print(f"trainval: {len(items)} labelled images ({missing} xml without image)")
        items += [(p, zero.copy()) for p in goods]
        print(f"added {len(goods)} defect-free images from othertypes/good")

        Yall = np.stack([y for _, y in items])
        freq = Yall.sum(0)
        keys = [strat_key(y, freq) for _, y in items]
        tr, va = split_idx(keys, a.val, a.seed)
        for name, idx in (("train", tr), ("val", va)):
            save_split(out, name, [items[i][0] for i in idx], a.size,
                       [items[i][1] for i in idx], report)

    if a.test_ann:
        tdir = Path(a.test_ann)
        timgs = sorted(p for p in (root / "test" / "JPEGImages").iterdir() if p.suffix.lower() in EXT)
        rng.shuffle(timgs)
        timgs = timgs[:(a.limit if a.limit else a.test_n)]
        files, labels, noxml = [], [], 0
        index = {p.stem: p for p in tdir.rglob("*.xml")}
        print(f"test: {len(index)} xml files found under {tdir}")
        for p in timgs:
            x = index.get(p.stem)
            if x is None:
                noxml += 1
                y = zero.copy()                 # no annotation file = defect-free
            else:
                y, _ = read_xml(x)
            files.append(p); labels.append(y)
        tg = spare_goods[:a.test_good]
        files += tg; labels += [zero.copy() for _ in tg]
        print(f"test: + {len(tg)} defect-free cells from the unused part of othertypes/good")
        print(f"test: {noxml} of {len(timgs)} sampled images had no xml (treated as defect-free)")
        save_split(out, "test", files, a.size, labels, report)
        report["test_no_xml"] = noxml

    (out / "classes.json").write_text(json.dumps(CLASSES))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
