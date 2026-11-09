#!/usr/bin/env python3
"""Task A3: run Marabou on the calibration slice and record verdicts against
the known VNN-COMP expected labels.

This is a solver/property-plumbing calibration: do our Marabou invocation and
VNN-LIB handling reproduce the expected SAT/UNSAT labels on external instances
with known ground truth? It is NOT a semantic-realism check.

Only instances whose ONNX and VNNLIB assets are present under
external/vnncomp/ are run; the rest are recorded as not-run (so they never
inflate or deflate the agreement rate). Uses the bundled Marabou 2.0 CLI
binary, which natively reads ONNX + VNNLIB.

Outputs:
  vnnlib_calibration_results.json
  vnnlib_calibration_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLICE_CSV = ROOT / "vnnlib_calibration_slice.csv"
EXT = ROOT / "external/vnncomp"
MARABOU = ROOT / "prophecy_env/bin/Marabou"
OUT_JSON = ROOT / "vnnlib_calibration_results.json"
OUT_CSV = ROOT / "vnnlib_calibration_results.csv"


def marabou_version():
    try:
        out = subprocess.run([str(MARABOU), "--version"], capture_output=True,
                             text=True, timeout=30).stdout
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def run_one(onnx: Path, vnnlib: Path, timeout: int, snc: bool, workers: int):
    """Return (result, runtime_s, witness_present). result in
    {sat, unsat, timeout, error}."""
    cmd = [str(MARABOU), str(onnx), str(vnnlib),
           f"--timeout={timeout}", "--verbosity=0"]
    if snc:
        cmd += ["--snc", f"--num-workers={workers}"]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return "timeout", round(time.time() - t0, 2), False
    except Exception as e:  # noqa: BLE001
        return f"error: {str(e)[:80]}", round(time.time() - t0, 2), False
    dt = round(time.time() - t0, 2)
    out = (proc.stdout + "\n" + proc.stderr).lower()
    lines = [l.strip() for l in out.splitlines()]
    witness = "input assignment" in out
    if "unsat" in lines:
        return "unsat", dt, witness
    if "sat" in lines:
        return "sat", dt, witness
    if "timeout" in out:
        return "timeout", dt, witness
    # Fall back to substring scan (some builds print 'SAT'/'UNSAT' inline).
    if "unsat" in out:
        return "unsat", dt, witness
    if "sat" in out:
        return "sat", dt, witness
    return "error", dt, witness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=60,
                    help="per-query timeout in seconds (default 60)")
    ap.add_argument("--benchmarks", default="all",
                    help="comma list of bench ids, or 'all' (default)")
    ap.add_argument("--snc", action="store_true",
                    help="use split-and-conquer (recommended for acasxu UNSAT)")
    ap.add_argument("--workers", type=int, default=4,
                    help="SnC worker threads (default 4)")
    args = ap.parse_args()

    rows = list(csv.DictReader(SLICE_CSV.open()))
    want = (None if args.benchmarks.strip().lower() == "all"
            else set(args.benchmarks.split(",")))

    records = []
    ran = skipped = 0
    for r in rows:
        if want is not None and r["bench"] not in want:
            continue
        onnx = EXT / r["bench"] / r["onnx"]
        vnnlib = EXT / r["bench"] / r["vnnlib"]
        if not (onnx.exists() and vnnlib.exists()):
            records.append({**{k: r[k] for k in
                               ("bench", "onnx", "vnnlib", "result")},
                            "expected": r["result"], "marabou_result": "not_run",
                            "runtime_s": 0.0, "timeout_s": args.timeout,
                            "witness_returned": False,
                            "detail": "assets missing"})
            skipped += 1
            continue
        res, dt, wit = run_one(onnx, vnnlib, args.timeout, args.snc, args.workers)
        rec = {"bench": r["bench"], "onnx": r["onnx"], "vnnlib": r["vnnlib"],
               "expected": r["result"], "marabou_result": res,
               "runtime_s": dt, "timeout_s": args.timeout,
               "witness_returned": bool(wit),
               "match": (res == r["result"])}
        records.append(rec)
        ran += 1
        flag = "OK " if rec["match"] else ("-- " if res in
                                           ("timeout", "error", "not_run")
                                           else "!! ")
        print(f"  {flag}{r['bench']:14s} {Path(r['vnnlib']).name:34s} "
              f"exp={r['result']:5s} got={res:7s} {dt:6.2f}s", flush=True)

    out = {
        "provenance": {
            "script": "run_vnnlib_calibration_slice.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "marabou": marabou_version(),
            "solver_binary": str(MARABOU.relative_to(ROOT)),
            "timeout_s": args.timeout,
            "split_and_conquer": bool(args.snc),
            "snc_workers": args.workers if args.snc else None,
            "asset_source": "VNN-COMP 2023 benchmark mirror "
                            "(github.com/ChristopherBrix/vnncomp2023_benchmarks)",
            "purpose": "solver/property-plumbing calibration on known SAT/UNSAT "
                       "labels; NOT a semantic-realism oracle",
        },
        "n_ran": ran, "n_skipped_missing_assets": skipped,
        "results": records,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "onnx", "vnnlib", "expected",
                                          "marabou_result", "match",
                                          "runtime_s", "timeout_s",
                                          "witness_returned"])
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in w.fieldnames})
    print(f"\nRan {ran}, skipped {skipped} (missing assets). "
          f"Wrote {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
