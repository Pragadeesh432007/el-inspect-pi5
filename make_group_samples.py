"""Make report figures: one EL example image for each of the 7 defect groups plus a good cell.
Uses the prepared arrays (no raw dataset needed). Needs only numpy and pillow.

    python make_group_samples.py --data data\\pvel --split train --out report_figs

Writes report_figs/group_samples.png (grid with labels) and one PNG per group
(sample_<group>.png, 384x384). It prefers an image that has ONLY that defect group; if none exists it
uses one that has the group together with others and says so. Change --seed for different examples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pvel")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default="report_figs")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = Path(a.data)
    X = np.load(d / f"{a.split}_x.npy", mmap_mode="r")
    Y = np.load(d / f"{a.split}_y.npy")
    classes = json.loads((d / "classes.json").read_text())
    rng = np.random.RandomState(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    picks = []                                            # (title, image, note)
    good = np.where(Y.sum(1) == 0)[0]
    if len(good):
        i = int(rng.choice(good))
        picks.append(("good (no defect)", np.asarray(X[i]), "", "good"))
    for k, c in enumerate(classes):
        only = np.where((Y[:, k] == 1) & (Y.sum(1) == 1))[0]
        anyc = np.where(Y[:, k] == 1)[0]
        if len(only):
            i, note = int(rng.choice(only)), ""
        elif len(anyc):
            i = int(rng.choice(anyc))
            note = "also has: " + ", ".join(classes[j] for j in range(len(classes)) if Y[i, j] and j != k)
        else:
            print(f"{c}: no image in the {a.split} split"); continue
        picks.append((c, np.asarray(X[i]), note, c))
        Image.fromarray(np.asarray(X[i])).save(out / f"sample_{c}.png")
        print(f"{c}: image {i} ({note or 'only this group'}) -> sample_{c}.png")
    if picks and picks[0][3] == "good":
        Image.fromarray(picks[0][1]).save(out / "sample_good.png")

    cols, cell, pad, head = 4, 300, 12, 34
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + head + pad) + pad + 40), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 8), "Examples of EL cell images, one per defect group (PVEL-AD, resized to 384x384)", fill=(30, 30, 30), font=font(17))
    for n, (t, im, note, _) in enumerate(picks):
        x = pad + (n % cols) * (cell + pad)
        y = 40 + pad + (n // cols) * (cell + head + pad)
        dr.text((x, y), t + (f"  ({note})" if note else ""), fill=(31, 111, 235), font=font(15))
        sheet.paste(Image.fromarray(im).convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y + head - 8))
    sheet.save(out / "group_samples.png")
    print(f"saved {out / 'group_samples.png'}")


if __name__ == "__main__":
    main()
