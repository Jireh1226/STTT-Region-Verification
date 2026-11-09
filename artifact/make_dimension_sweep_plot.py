#!/usr/bin/env python3
"""Merge the TinyTaxiNet P3 dimension-sweep chunks and produce:
  - p3_dimension_sweep_results.json  (merged records + per-(basis,d) tallies)
  - p3_dimension_sweep.png           (verification rate vs subspace dimension,
                                       local PCA vs matched random, TinyTaxiNet)
Timeouts are reported and NOT counted as verified. Random is aggregated over seeds
(median with min/max band).
"""
import json, glob
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DIMS = [2, 4, 8, 16, 32, 64]


def merge_taxinet(chunk_glob):
    rec = []
    for fn in glob.glob(chunk_glob):
        try:
            rec += json.load(open(fn))["results"]
        except Exception:
            pass
    return [r for r in rec if "category" in r]


def tallies(rec):
    # per (basis,d): overall counts; for random, per-seed verified for min/max band
    by = defaultdict(Counter)
    seedver = defaultdict(lambda: defaultdict(Counter))  # (basis,d)->seed->Counter
    anchors = set()
    for r in rec:
        by[(r["basis"], r["d"])][r["category"]] += 1
        anchors.add(r["anchor_idx"])
        if r["basis"] == "random":
            seedver[("random", r["d"])][r["seed"]][r["category"]] += 1
    return by, seedver, len(anchors)


def rate(cnt):
    n = sum(cnt.values())
    return (100.0 * cnt["verified"] / n) if n else None, n


def main():
    rec = merge_taxinet(str(ROOT / "p3_dimsweep_chunk*.json"))
    by, seedver, nanch = tallies(rec)

    # save merged results + tallies
    tally_json = {}
    for basis in ("pca", "random"):
        tally_json[basis] = {}
        for d in DIMS:
            c = by[(basis, d)]
            r, n = rate(c)
            entry = {"verified": c["verified"], "sat": c["sat"], "timeout": c["timeout"],
                     "region_empty": c["region_empty"], "n": n,
                     "verified_pct": round(r, 2) if r is not None else None}
            if basis == "random":
                seedr = [100.0 * seedver[("random", d)][s]["verified"] / max(1, sum(seedver[("random", d)][s].values()))
                         for s in seedver[("random", d)]]
                if seedr:
                    entry["verified_pct_seed_min"] = round(min(seedr), 2)
                    entry["verified_pct_seed_max"] = round(max(seedr), 2)
            tally_json[basis][str(d)] = entry
    (ROOT / "p3_dimension_sweep_results.json").write_text(json.dumps(
        {"provenance": {"script": "make_dimension_sweep_plot.py", "n_anchors": nanch,
                        "note": "TinyTaxiNet P3, alpha=1sigma; timeouts not counted as verified"},
         "tallies": tally_json}, indent=2) + "\n")

    # plot
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5})
    fig, ax = plt.subplots(figsize=(4.4, 3.15))
    pv = [tally_json["pca"][str(d)]["verified_pct"] for d in DIMS]
    rv = [tally_json["random"][str(d)]["verified_pct"] for d in DIMS]
    ax.plot(DIMS, pv, "o-", color="#0072B2", lw=2, label="local PCA")
    ax.plot(DIMS, rv, "s--", color="#009E73", lw=2, label="matched random")
    # random seed band
    rmin = [tally_json["random"][str(d)].get("verified_pct_seed_min") for d in DIMS]
    rmax = [tally_json["random"][str(d)].get("verified_pct_seed_max") for d in DIMS]
    if all(x is not None for x in rmin + rmax):
        ax.fill_between(DIMS, rmin, rmax, color="#009E73", alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS); ax.set_xticklabels(DIMS)
    ax.set_xlabel("subspace dimension $d$")
    ax.set_ylabel("verification rate (\% verified)")
    ax.set_ylim(-3, 105)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "p3_dimension_sweep.png", dpi=300)
    print("PCA verified%%:", pv)
    print("RANDOM verified%%:", rv)
    print(f"Saved p3_dimension_sweep_results.json and p3_dimension_sweep.png (n_anchors={nanch})")


if __name__ == "__main__":
    main()
