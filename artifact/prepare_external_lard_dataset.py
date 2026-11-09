#!/usr/bin/env python3.11
"""
Prepare an EXTERNAL runway-image dataset for the region-tagged Marabou workflow.

Dataset: LARD V2 (Landing Approach Runway Detection), DEEL-AI.
  source split used here: `ges` (Google Earth Studio aerial imagery)
  host:    https://huggingface.co/datasets/DEEL-AI/LARD_V2
  license: MIT (LARD code/labels). Underlying GES imagery subject to Google Earth terms;
           used here for non-commercial academic research only.

This dataset is INDEPENDENT of the VerifyGAN / TinyTaxiNet X-Plane data used in the main
case study: different generator, different airports/imagery source, different authors.

Label derivation (no model, purely from public annotations):
  Each row carries the four runway-corner pixel coordinates (x_TL,y_TL, x_TR,y_TR,
  x_BR,y_BR, x_BL,y_BL) and aircraft attitude (yaw, pitch, roll) plus lateral_path_angle.
  We derive an image-space lateral centerline-offset proxy (CTE analog):
        offset = mean(corner_x) / width - 0.5      in (-0.5, 0.5)
  offset < 0  => runway centroid left of image center  (label "L")
  offset ~ 0  => centered                              (label "C")
  offset > 0  => runway centroid right of image center (label "R")
  We also keep yaw (heading proxy / HE analog) and lateral_path_angle for cross-checks.

Output: external_lard_dataset.npz  +  external_lard_provenance.json
We store ONLY the resized grayscale arrays (24x48, [0,1]) and the continuous labels,
so the class binning can be chosen later without re-downloading.
"""
import urllib.request, urllib.error, csv, io, json, sys, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

BASE = "https://huggingface.co/datasets/DEEL-AI/LARD_V2/resolve/main/"
HDR  = {"User-Agent": "curl/8.0"}
ROOT = os.path.dirname(os.path.abspath(__file__))

SEED       = 0
SOURCE     = "ges"
H, W       = 24, 48           # resized grayscale (rows, cols) -> 1152-dim input
N_TRAIN    = 2000
N_TEST     = 700
ODD_FILTER = "IN_ODD"        # keep only in-operational-design-domain rows

# numeric geometry/attitude columns kept for target selection (no re-download needed later)
META_COLS = ["yaw", "pitch", "roll", "slant_distance", "along_track_distance",
             "height_above_runway", "lateral_path_angle", "vertical_path_angle"]


def fetch(url, timeout=90, retries=3):
    last = None
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=timeout).read()
        except Exception as e:  # noqa
            last = e
            time.sleep(1.0 + k)
    raise last


def load_meta(split):
    raw = fetch(BASE + f"metadata_{SOURCE}_{split}.csv").decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=';'))
    return [r for r in rows if r.get("runway_in_cone") == ODD_FILTER]


def offset_of(r):
    w = float(r["width"])
    xs = [float(r["x_TL"]), float(r["x_TR"]), float(r["x_BR"]), float(r["x_BL"])]
    return sum(xs) / 4.0 / w - 0.5


def sample(rows, n, rng):
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def download_one(r):
    try:
        data = fetch(BASE + r["image"])
        im = Image.open(io.BytesIO(data)).convert("L").resize((W, H))  # PIL size = (W,H)
        a = (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1)     # row-major HxW -> H*W
        meta = {k: float(r[k]) for k in META_COLS}
        meta["offset"] = offset_of(r)
        return (r["image"], a, meta, r["airport"], r["runway"])
    except Exception:
        return None


def build_split(rows, n, rng, tag):
    picks = sample(rows, min(n, len(rows)), rng)
    X, names, ap, rwy = [], [], [], []
    meta = {k: [] for k in META_COLS + ["offset"]}
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(download_one, r) for r in picks]
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if done % 200 == 0:
                print(f"  [{tag}] {done}/{len(picks)} attempted, {len(X)} ok", flush=True)
            if res is None:
                continue
            nm, a, m, air, rw = res
            X.append(a); names.append(nm); ap.append(air); rwy.append(rw)
            for k in meta:
                meta[k].append(m[k])
    out = {"X": np.asarray(X, np.float32), "names": np.asarray(names),
           "airport": np.asarray(ap), "runway": np.asarray(rwy)}
    for k in meta:
        out[k] = np.asarray(meta[k], np.float32)
    return out


def main():
    rng = np.random.default_rng(SEED)
    print("Loading metadata ...", flush=True)
    tr_rows = load_meta("train")
    te_rows = load_meta("test")
    print(f"train IN_ODD rows: {len(tr_rows)}   test IN_ODD rows: {len(te_rows)}", flush=True)

    print("Downloading TRAIN subset ...", flush=True)
    TR = build_split(tr_rows, N_TRAIN, rng, "train")
    print("Downloading TEST subset ...", flush=True)
    TE = build_split(te_rows, N_TEST, rng, "test")
    Xtr, Xte = TR["X"], TE["X"]
    print(f"got train={len(Xtr)} test={len(Xte)}", flush=True)

    save = {"Xtr": Xtr, "Xte": Xte, "H": H, "W": W,
            "names_tr": TR["names"], "names_te": TE["names"],
            "airport_tr": TR["airport"], "airport_te": TE["airport"],
            "runway_tr": TR["runway"], "runway_te": TE["runway"]}
    for k in META_COLS + ["offset"]:
        save[k + "_tr"] = TR[k]
        save[k + "_te"] = TE[k]
    np.savez_compressed(os.path.join(ROOT, "external_lard_dataset.npz"), **save)

    prov = {
        "dataset": "LARD V2",
        "source_split": SOURCE,
        "host": "https://huggingface.co/datasets/DEEL-AI/LARD_V2",
        "paper": "https://arxiv.org/pdf/2603.26748",
        "license": "MIT (LARD code/labels); GES imagery under Google Earth terms, academic use",
        "independent_of_verifygan": True,
        "independence_note": "Different generator (Google Earth Studio vs X-Plane/TinyTaxiNet/VerifyGAN), "
                             "different airports/imagery source, different authors.",
        "seed": SEED,
        "input": {"resized_rows": H, "resized_cols": W, "dim": H * W, "channels": "grayscale", "range": "[0,1]"},
        "counts": {"train": int(len(Xtr)), "test": int(len(Xte)),
                   "train_pool_IN_ODD": len(tr_rows), "test_pool_IN_ODD": len(te_rows)},
        "stored_geometry_labels": META_COLS + ["offset (image-space lateral centroid)"],
        "label_note": "Final classification target chosen empirically by select_external_lard_target.py.",
        "gpu_used": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(ROOT, "external_lard_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print("WROTE external_lard_dataset.npz and external_lard_provenance.json", flush=True)


if __name__ == "__main__":
    main()
