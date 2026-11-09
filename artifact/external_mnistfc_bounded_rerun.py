#!/usr/bin/env python3
"""Re-run the R4+pixel-bounds @1sigma queries that timed out on the anchors
that matter most: anchors where R1@0.05 was SAT and R4+bounds@1sigma timed out
at 120 s. Longer timeout (default 600 s) to decide whether more broad-region
witnesses disappear under valid bounded local-PCA.

Output: external_mnistfc_bounded_rerun_results.json (incremental)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import verify_external_mnistfc_bounded_transfer as B

ROOT = Path(__file__).resolve().parent
SEVEN = [0, 1, 4, 6, 7, 18, 61]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--anchors", default=",".join(map(str, SEVEN)))
    args = ap.parse_args()
    B.TIMEOUT = args.timeout
    want = [int(x) for x in args.anchors.split(",")]

    Xtr, ytr, Xte, yte = B.load_mnist()
    anchors = json.loads((ROOT / "external_mnistfc_anchor_baseline.json").read_text())["anchors"]
    by = {a["anchor_idx"]: a for a in anchors}

    records = []
    for idx in want:
        a = by[idx]; c, t = a["true_class"], a["target_class"]
        x0 = Xte[idx]
        Xc = Xtr[ytr == c]
        nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:B.KNB]]
        _, _, Vt = np.linalg.svd(nb - x0[None, :], full_matrices=False)
        V = Vt[:B.D]; sig = (nb - x0[None, :]) @ V.T
        sig = sig.std(0)
        m4 = B.compose_pca_two(V, x0, f"rerun_a{idx}")
        vp = ROOT / "_rerun.vnnlib"
        B.emit_bounded(-1.0 * sig, 1.0 * sig, c, t, vp)   # alpha = 1 sigma
        t0 = time.time()
        st, dt, wit = B.solve(m4, vp)
        rec = {"anchor_idx": idx, "true_class": c, "target_class": t,
               "region": "R4_local_pca_bounded", "alpha": 1.0,
               "timeout_budget_s": args.timeout, "status": st, "runtime_s": dt}
        if st == "sat" and wit and len(wit) >= B.D:
            z = np.array([wit[j] for j in range(B.D)])
            xr = x0 + V.T @ z
            rec["witness_z"] = z.tolist()
            rec["frac_in_unit"] = float(((xr >= -1e-6) & (xr <= 1 + 1e-6)).mean())
        records.append(rec)
        print(f"  idx{idx} c{c}->t{t} R4b@1s (budget {args.timeout}s): {st} ({dt:.0f}s)", flush=True)
        vp.unlink(missing_ok=True)
        (ROOT / "external_mnistfc_bounded_rerun_results.json").write_text(
            json.dumps({"provenance": {"script": "external_mnistfc_bounded_rerun.py",
                                       "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                       "timeout_budget_s": args.timeout,
                                       "anchors": want,
                                       "selection": "R1@0.05 SAT and R4+bounds@1sigma timed out at 120s"},
                        "results": records}, indent=2) + "\n")
    print("DONE rerun.")


if __name__ == "__main__":
    main()
