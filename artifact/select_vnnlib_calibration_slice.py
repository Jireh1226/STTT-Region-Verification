#!/usr/bin/env python3
"""Task A1: select a small, deterministic VNN-LIB expected-result calibration
slice from the released VNN-COMP expected-results table.

This is an external *known-label* sanity check on the solver/property
plumbing, not a semantic-realism oracle. We pick a handful of SAT and UNSAT
instances per preferred benchmark, with stable ordering and no randomness.

Inputs:
  review/sources/vnnlib_expected_results.csv   (columns: bench,onnx,vnnlib,result)

Outputs:
  vnnlib_calibration_slice.csv                 (selected rows + a repo_dir column)
  vnnlib_calibration_slice_summary.json        (per-benchmark and overall counts)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "review/sources/vnnlib_expected_results.csv"
OUT_CSV = ROOT / "vnnlib_calibration_slice.csv"
OUT_JSON = ROOT / "vnnlib_calibration_slice_summary.json"

# Benchmarks called out in the review, in priority order.
PREFERRED = [
    "acasxu_2023",
    "mnist_fc",
    "cifar2020",
    "dist_shift_2023",
    "cgan_2023",
    "collins_rul_cnn_2023",
]

# Map the expected-results "bench" id to the directory name used by the
# VNN-COMP 2023 benchmark repository (Christopher Brix mirror). The repo drops
# the year suffix for several benchmarks; A2 falls back to other candidates if
# the primary guess 404s.
REPO_DIR = {
    "acasxu_2023": "acasxu",
    "mnist_fc": "mnist_fc",
    "cifar2020": "cifar2020",
    "dist_shift_2023": "dist_shift",
    "cgan_2023": "cgan",
    "collins_rul_cnn_2023": "collins_rul_cnn",
}

PER_LABEL = 10  # up to this many SAT and this many UNSAT per benchmark


def load_rows():
    with SRC.open() as f:
        return list(csv.DictReader(f))


def select(rows):
    by_bench = defaultdict(lambda: {"sat": [], "unsat": []})
    for r in rows:
        b = r["bench"]
        if b in PREFERRED and r["result"] in ("sat", "unsat"):
            by_bench[b][r["result"]].append(r)

    selected = []
    per_bench = {}
    for b in PREFERRED:
        if b not in by_bench:
            per_bench[b] = {"available_sat": 0, "available_unsat": 0,
                            "selected_sat": 0, "selected_unsat": 0,
                            "note": "benchmark not present in expected-results table"}
            continue
        sats = sorted(by_bench[b]["sat"], key=lambda r: (r["onnx"], r["vnnlib"]))
        unsats = sorted(by_bench[b]["unsat"], key=lambda r: (r["onnx"], r["vnnlib"]))
        # Up to PER_LABEL of each; if one class is short, take all of it and
        # match the larger class to the same count where possible (keeps the
        # slice label-balanced per benchmark without inventing instances).
        n_sat_avail, n_unsat_avail = len(sats), len(unsats)
        smaller = min(n_sat_avail, n_unsat_avail)
        cap = min(PER_LABEL, smaller) if smaller > 0 else PER_LABEL
        take_sat = sats[: min(PER_LABEL, n_sat_avail) if smaller == 0 else cap]
        take_unsat = unsats[: min(PER_LABEL, n_unsat_avail) if smaller == 0 else cap]
        for r in take_sat + take_unsat:
            selected.append({**r, "repo_dir": REPO_DIR.get(b, b)})
        per_bench[b] = {
            "available_sat": n_sat_avail,
            "available_unsat": n_unsat_avail,
            "selected_sat": len(take_sat),
            "selected_unsat": len(take_unsat),
            "repo_dir": REPO_DIR.get(b, b),
        }

    # Global deterministic ordering: (bench, result, onnx, vnnlib).
    selected.sort(key=lambda r: (r["bench"], r["result"], r["onnx"], r["vnnlib"]))
    return selected, per_bench


def main():
    rows = load_rows()
    selected, per_bench = select(rows)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "repo_dir", "onnx",
                                          "vnnlib", "result"])
        w.writeheader()
        for r in selected:
            w.writerow({k: r[k] for k in w.fieldnames})

    summary = {
        "source": str(SRC.relative_to(ROOT)),
        "preferred_benchmarks": PREFERRED,
        "per_label_cap": PER_LABEL,
        "total_selected": len(selected),
        "selected_sat": sum(1 for r in selected if r["result"] == "sat"),
        "selected_unsat": sum(1 for r in selected if r["result"] == "unsat"),
        "per_benchmark": per_bench,
        "note": "External known-label slice for solver/property-plumbing "
                "calibration only; not a semantic-realism oracle.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Selected {summary['total_selected']} instances "
          f"({summary['selected_sat']} SAT, {summary['selected_unsat']} UNSAT)")
    for b, s in per_bench.items():
        if "selected_sat" in s:
            print(f"  {b:24s} sel {s.get('selected_sat',0)}S/"
                  f"{s.get('selected_unsat',0)}U  "
                  f"(avail {s.get('available_sat',0)}S/"
                  f"{s.get('available_unsat',0)}U)")
    print(f"Wrote {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
