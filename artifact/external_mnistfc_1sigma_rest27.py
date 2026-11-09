#!/usr/bin/env python3
"""Directly verify R4 bounded local-PCA at 1 sigma (valid pixels enforced) on
the 27 held-out anchors that the matrix run did NOT solve directly at 1 sigma
(every non-broad-SAT anchor). Reuses the matrix module's own pca_basis / cli /
compose / emit so the verdicts are bit-for-bit comparable to the directly-solved
23 broad-SAT anchors. Each of these 27 is already UNSAT at 2 sigma, so the
1 sigma box (a subset) must be UNSAT; this run confirms it by direct solve and
makes the 50/50 denominator entirely directly-decided.

Output: external_mnistfc_1sigma_rest27_results.json (incremental)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

import external_mnistfc_matrix as M  # reuses pca_basis, cli, ROOT, B

B = M.B
ROOT = M.ROOT
OUT = ROOT / "external_mnistfc_1sigma_rest27_results.json"


def main():
    res = json.load(open(ROOT / "external_mnistfc_matrix_results.json"))["results"]
    R1_05 = {r["anchor_idx"]: r for r in res if r["region"] == "R1_pixel_ball" and r["eps"] == 0.05}
    # c,t per anchor from any record carrying them (use R4_bounded@2sigma)
    ct = {r["anchor_idx"]: (r["true_class"], r["target_class"])
          for r in res if r["region"] == "R4_bounded" and r["alpha"] == 2.0}
    anchors = sorted(R1_05)
    rest = [a for a in anchors if R1_05[a]["status"] != "sat"]
    print(f"rest (non-broad-SAT) anchors: {len(rest)} -> {rest}", flush=True)

    Xtr, ytr, Xte, yte = B.load_mnist()
    records = []
    for i in rest:
        c, t = ct[i]
        x0 = Xte[i]
        V, sig = M.pca_basis(x0, Xtr, ytr, c)
        m = B.compose_pca_two(V, x0, f"rest27_a{i}")
        vp = ROOT / "_rest27_r4.vnnlib"
        B.emit_bounded(-1.0 * sig, 1.0 * sig, c, t, vp)   # alpha = 1 sigma
        t0 = time.time()
        st, dt, wit = M.cli(m, vp, timeout=600, want_witness=True)
        vp.unlink(missing_ok=True)
        rec = {"anchor_idx": i, "true_class": c, "target_class": t,
               "region": "R4_bounded", "alpha": 1.0, "status": st,
               "runtime_s": dt, "source": "rest27_direct"}
        if st == "sat" and wit and len(wit) >= B.D:
            rec["witness_z"] = [wit[j] for j in range(B.D)]
        records.append(rec)
        json.dump({"provenance": {"script": "external_mnistfc_1sigma_rest27.py",
                                  "alpha": 1.0, "valid_pixels": True},
                   "results": records}, open(OUT, "w"), indent=2)
        print(f"  R4b@1s idx{i} c{c}->t{t}: {st} ({dt:.0f}s)", flush=True)
    nun = sum(r["status"] == "unsat" for r in records)
    print(f"DONE rest27: {nun}/{len(records)} UNSAT", flush=True)


if __name__ == "__main__":
    main()
