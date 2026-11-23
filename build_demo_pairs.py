"""Build demo cases from the Mendeley PV_EL_HDR_BW_Test_DB dataset.

Each damage condition was photographed three ways (EL, near-IR black/white, visible
HDR) together with the measured power loss. We use it ONLY as a demo/illustration set
(84 images, one lab cell) - not for training.

What it does
  * reads data/ELPV_DATA_FULL.CSV (';'-separated)
  * prints a summary (damage levels, power-loss range, how many damages are hidden in
    the visible image but visible in EL)
  * picks N representative conditions spread over the power-loss range, always
    including some "hidden in visible light" cases
  * writes resized EL + visible images and demo_cases.json (severity + suggested action)

Usage (Windows, from the solar_inspect folder, venv active):
    python build_demo_pairs.py --root "%USERPROFILE%\\Downloads\\PV_EL_HDR_BW_Test_DB\\PV_EL_HDR_BW_Test_DB"

The severity bands below are OUR choice for the prototype (not from the dataset):
    loss < low_thr  -> low, < high_thr -> moderate, else severe. Change with --low/--high.
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

COL_LOSS = "Power_output_loss [%] [V*I]"


def num(s):
    try:
        return float(str(s).replace(",", ".").strip())
    except ValueError:
        return None


def severity(loss, low, high):
    if loss is None:
        return "unknown"
    return "low" if loss < low else "moderate" if loss < high else "severe"


ACTION = {"low": "monitor / re-inspect later",
          "moderate": "schedule inspection or repair",
          "severe": "repair or replace soon",
          "unknown": "inspect"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing data\\ and images\\")
    ap.add_argument("--out", default="demo_pairs")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=640, help="longest side of saved images")
    ap.add_argument("--low", type=float, default=5.0)
    ap.add_argument("--high", type=float, default=15.0)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    csv_path = root / "data" / "ELPV_DATA_FULL.CSV"
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter=";"):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            if r.get("EL_file_name"):
                rows.append(r)
    if not rows:
        raise SystemExit(f"No rows read from {csv_path}")

    for r in rows:
        r["_loss"] = num(r.get(COL_LOSS))
        r["_hidden"] = r.get("incremental_damage_HDR_visible_type", "").lower() == "invisible"

    losses = [r["_loss"] for r in rows if r["_loss"] is not None]
    print(f"rows: {len(rows)} | power-loss range: {min(losses):.1f}% .. {max(losses):.1f}%")
    lv = {}
    for r in rows:
        lv[r.get("Degradation_level", "?")] = lv.get(r.get("Degradation_level", "?"), 0) + 1
    print("degradation levels:", lv)
    hidden = [r for r in rows if r["_hidden"]]
    print(f"damage invisible in visible HDR but present in EL: {len(hidden)} of {len(rows)}")

    # choose representative cases: evenly spaced over loss + a couple of hidden ones
    ordered = sorted([r for r in rows if r["_loss"] is not None], key=lambda r: r["_loss"])
    pick, step = [], max(1, len(ordered) // a.n)
    for i in range(0, len(ordered), step):
        pick.append(ordered[i])
    hid_sorted = sorted([r for r in hidden if r["_loss"] is not None], key=lambda r: r["_loss"])
    for r in hid_sorted[:: max(1, len(hid_sorted) // 3)][:3]:
        if r not in pick:
            pick.append(r)
    pick = pick[: a.n + 3]

    out.mkdir(exist_ok=True)
    cases = []
    for r in pick:
        el_name = r["EL_file_name"]
        vis_name = el_name.replace("_EL_", "_HDR_")
        el_p, vis_p = root / "images" / "EL" / el_name, root / "images" / "HDR" / vis_name
        if not el_p.exists() or not vis_p.exists():
            print("skip (missing file):", el_name)
            continue
        tag = r.get("images_set_name", Path(el_name).stem)
        for src, suffix in ((el_p, "EL"), (vis_p, "VIS")):
            im = Image.open(src).convert("RGB" if suffix == "VIS" else "L")
            im.thumbnail((a.size, a.size))
            im.save(out / f"{tag}_{suffix}.jpg", quality=90)
        sev = severity(r["_loss"], a.low, a.high)
        cases.append({
            "id": tag, "el_image": f"{tag}_EL.jpg", "visible_image": f"{tag}_VIS.jpg",
            "power_loss_percent": r["_loss"], "severity": sev, "suggested_action": ACTION[sev],
            "dataset_damage_level": r.get("Degradation_level"),
            "damage_kind": r.get("incremental_damage_kind"),
            "visible_in_rgb": not r["_hidden"],
            "visible_type_rgb": r.get("incremental_damage_HDR_visible_type"),
            "visible_type_el": r.get("incremental_damage_EL_visible_type"),
            "cause": r.get("incremental_damage_cause"),
            "description": r.get("incremental_damage_description_details"),
        })
    (out / "demo_cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\nsaved {len(cases)} cases to {out}\\ :")
    for c in cases:
        print(f"  {c['id']:>10}  loss {c['power_loss_percent']:>5}%  {c['severity']:<8} "
              f"visible-in-RGB: {c['visible_in_rgb']}")


if __name__ == "__main__":
    main()
