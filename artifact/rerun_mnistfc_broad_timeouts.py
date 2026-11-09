#!/usr/bin/env python3
"""Timeout hygiene for the mnist_fc broad pixel-ball (R1) queries: rerun every
R1 query that TIMED OUT in the matrix run with a 10x budget (1200 s), in parallel.
Resolves how many broad-ball timeouts are actually SAT (additional broad witnesses)
vs UNSAT (robust) vs still-unresolved. Timeout is never counted as SAT/UNSAT.

Same base net (B.ONNX), same property, same clipping. Parallel for jarvis.
Output: external_mnistfc_broad_timeout_rerun_results.json
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import external_mnistfc_matrix as M
import verify_external_mnistfc_bounded_transfer as B

ROOT = M.ROOT
OUT = ROOT / "external_mnistfc_broad_timeout_rerun_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1200)   # 10x the 120 s matrix budget
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--epslist", default="0.05,0.03")
    args = ap.parse_args()
    epslist = [float(e) for e in args.epslist.split(",")]

    Xtr, ytr, Xte, yte = B.load_mnist()
    manifest = json.load(open(ROOT / "external_mnistfc_matrix_manifest.json"))
    anchors = {a["anchor_idx"]: a for a in manifest["anchors"]}
    mres = json.load(open(ROOT / "external_mnistfc_matrix_results.json"))["results"]

    tasks = []
    for eps in epslist:
        for x in mres:
            if (x["region"] == "R1_pixel_ball" and x["eps"] == eps and x["status"] == "timeout"):
                tasks.append((x["anchor_idx"], eps))
    print(f"reruning {len(tasks)} timed-out broad queries at {args.timeout}s, {args.workers} workers", flush=True)

    def work(task):
        i, eps = task
        a = anchors[i]; c, t = a["true_class"], a["target_class"]
        x0 = Xte[i]
        vp = ROOT / f"_r1rerun_{i}_{eps}.vnnlib"
        M.emit_pixel_box(np.clip(x0 - eps, 0, 1), np.clip(x0 + eps, 0, 1), c, t, vp)
        st, dt, _ = M.cli(B.ONNX, vp, timeout=args.timeout)
        vp.unlink(missing_ok=True)
        return {"anchor_idx": i, "eps": eps, "region": "R1_pixel_ball_rerun",
                "timeout_budget_s": args.timeout, "status": st, "runtime_s": dt}

    records = []
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, tk): tk for tk in tasks}
        for fut in as_completed(futs):
            records.append(fut.result()); done += 1
            (ROOT / OUT).write_text(json.dumps(
                {"provenance": {"script": "rerun_mnistfc_broad_timeouts.py", "timestamp_utc":
                                datetime.now(timezone.utc).isoformat(), "timeout_budget_s": args.timeout,
                                "note": "10x rerun of matrix R1 timeouts; timeout != SAT/UNSAT"},
                 "results": records}, indent=2) + "\n")
            print(f"  {done}/{len(tasks)}: a{fut.result()['anchor_idx']}@{fut.result()['eps']} "
                  f"-> {records[-1]['status']} ({records[-1]['runtime_s']:.0f}s)", flush=True)
    from collections import Counter
    print("DONE tally:", dict(Counter(r["status"] for r in records)), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
