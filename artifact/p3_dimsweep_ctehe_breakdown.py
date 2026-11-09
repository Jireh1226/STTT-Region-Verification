#!/usr/bin/env python3
"""CTE/HE SAT breakdown supplement for the P3 dimension sweep. For a
deterministic 200-anchor candidate subset and the local-PCA basis, run the four
P3 sub-queries individually at each d for anchors with enough same-state
neighbors for d=64, and record which axis (CTE upper/lower, HE upper/lower) is
SAT. Answers 'is the high-d SAT growth CTE- or HE-driven?'. Exact status
parsing. Output: p3_dimsweep_ctehe_breakdown.json.
"""
import json, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import verify_p3_local_pca as p3

DIMS = [2, 4, 8, 16, 32, 64]
SUBQ = [("CTE_upper", 192, ">=", p3.P3_CTE_BOUND), ("CTE_lower", 192, "<=", -p3.P3_CTE_BOUND),
        ("HE_upper", 193, ">=", p3.P3_HE_BOUND), ("HE_lower", 193, "<=", -p3.P3_HE_BOUND)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--output", default="p3_dimsweep_ctehe_breakdown.json")
    args = ap.parse_args()
    images, labels = p3.load_data()
    cen = p3.centered_indices(labels)
    sub = cen[np.linspace(0, len(cen) - 1, 200, dtype=int)]   # deterministic 200-subset
    if args.count >= 0:
        sub = sub[args.start:args.start + args.count]
    cte, he = labels[:, 0], labels[:, 1]
    dmax = max(DIMS)
    by = defaultdict(Counter)   # d -> Counter of axis SAT counts + region SAT/verified
    t0 = time.time()

    def save():
        Path(args.output).write_text(json.dumps(
            {"provenance": {"script": "p3_dimsweep_ctehe_breakdown.py", "basis": "pca",
                            "candidate_subset_size": int(len(sub)),
                            "processed_anchor_count": int(by[DIMS[0]]["n"]),
                            "min_neighbors_required": int(dmax + 1),
                            "dims": DIMS, "alpha": 1.0},
             "by_dimension": {str(d): dict(by[d]) for d in DIMS}}, indent=2) + "\n")

    for k, idx in enumerate(sub):
        x0 = images[int(idx)]
        mask = (np.abs(cte - cte[idx]) <= p3.CTE_BIN_HALF) & (np.abs(he - he[idx]) <= p3.HE_BIN_HALF); mask[idx] = False
        nb = images[mask]
        if len(nb) < dmax + 1:
            continue
        Dm = nb - x0
        _, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        Vfull = Vt[:dmax]; sig = (Dm @ Vfull.T).std(0)
        for d in DIMS:
            V = Vfull[:d]; alphas = 1.0 * sig[:d]
            statuses = {}
            for name, var, direc, b in SUBQ:
                statuses[name] = p3.solve_one(x0, V, None, alphas, var, direc, b)["status"]
            by[d]["n"] += 1
            any_cte_sat = statuses["CTE_upper"] == "sat" or statuses["CTE_lower"] == "sat"
            any_he_sat = statuses["HE_upper"] == "sat" or statuses["HE_lower"] == "sat"
            any_to = any(s == "timeout" for s in statuses.values())
            if any_cte_sat:
                by[d]["anchors_CTE_sat"] += 1
            if any_he_sat:
                by[d]["anchors_HE_sat"] += 1
            if (any_cte_sat or any_he_sat):
                by[d]["anchors_sat"] += 1
            elif any_to:
                by[d]["anchors_timeout"] += 1
            else:
                by[d]["anchors_verified"] += 1
        if (k + 1) % 25 == 0:
            save()
            print(f"  {k+1}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)
    save()   # always persist final state, even when the last anchor(s) were skipped
    print("DONE ctehe breakdown", flush=True)


if __name__ == "__main__":
    main()
