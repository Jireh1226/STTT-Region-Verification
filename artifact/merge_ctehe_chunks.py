#!/usr/bin/env python3
"""Merge the per-chunk CTE/HE breakdown JSONs into the full-200 result.

Globs ctehe_chunk*.json in the working dir, EXCLUDING the stale ctehe_chunk18.json
(anchors 90-94 come from the single-anchor split files ctehe_chunk18a-e.json), and
sums the per-dimension counters into a single full-200 breakdown that matches the
format p3_dimsweep_ctehe_breakdown.py emits. Exact field names, no re-solving.
"""
import json, glob, os
from collections import Counter

DIMS = [2, 4, 8, 16, 32, 64]
FIELDS = ("n", "anchors_CTE_sat", "anchors_HE_sat", "anchors_sat",
          "anchors_verified", "anchors_timeout")


def main():
    files = [f for f in glob.glob("ctehe_chunk*.json")
             if os.path.basename(f) != "ctehe_chunk18.json"]
    by = {d: Counter() for d in DIMS}
    for f in sorted(files):
        d = json.load(open(f))
        for k, v in d["by_dimension"].items():
            dk = int(k)
            for fld in FIELDS:
                by[dk][fld] += v.get(fld, 0)
    merged = {
        "provenance": {
            "script": "p3_dimsweep_ctehe_breakdown.py", "basis": "pca",
            "candidate_subset_size": 200, "merged_from_chunk_files": len(files),
            "processed_anchor_count": by[DIMS[0]]["n"], "min_neighbors_required": 65,
            "dims": DIMS, "alpha": 1.0,
            "note": ("merged full-200 run; chunk18 split into single-anchor jobs "
                     "for anchors 90-94 (chunk18a-e); stale chunk18.json excluded"),
        },
        "by_dimension": {str(d): dict(by[d]) for d in DIMS},
    }
    open("p3_dimsweep_ctehe_breakdown.json", "w").write(json.dumps(merged, indent=2) + "\n")

    ns = [by[d]["n"] for d in DIMS]
    print("merged files:", len(files))
    print("processed anchors per dim (should be equal):", ns)
    hdr = "{:>4} {:>4} {:>8} {:>7} {:>9} {:>8}".format(
        "d", "n", "CTE_sat", "HE_sat", "verified", "timeout")
    print(hdr)
    for d in DIMS:
        c = by[d]
        print("{:>4} {:>4} {:>8} {:>7} {:>9} {:>8}".format(
            d, c["n"], c["anchors_CTE_sat"], c["anchors_HE_sat"],
            c["anchors_verified"], c["anchors_timeout"]))


if __name__ == "__main__":
    main()
