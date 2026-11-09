#!/usr/bin/env python3
"""Task A4: summarize agreement between Marabou and the VNN-COMP expected
labels on the calibration slice.

Input:
  vnnlib_calibration_results.json
Output:
  vnnlib_calibration_summary.json

Two agreement rates are reported: one over completed queries only (the
plumbing-correctness signal) and one that counts timeouts/errors as
non-matches (the conservative whole-slice rate).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_JSON = ROOT / "vnnlib_calibration_results.json"
OUT_JSON = ROOT / "vnnlib_calibration_summary.json"

COMPLETED = {"sat", "unsat"}


def tally(records):
    n_total = len(records)
    exp_sat = sum(1 for r in records if r["expected"] == "sat")
    exp_unsat = sum(1 for r in records if r["expected"] == "unsat")
    matched_sat = sum(1 for r in records
                      if r["expected"] == "sat" and r["marabou_result"] == "sat")
    matched_unsat = sum(1 for r in records
                        if r["expected"] == "unsat"
                        and r["marabou_result"] == "unsat")
    timeouts = sum(1 for r in records if r["marabou_result"] == "timeout")
    errors = sum(1 for r in records
                 if str(r["marabou_result"]).startswith("error"))
    not_run = sum(1 for r in records if r["marabou_result"] == "not_run")
    completed = [r for r in records if r["marabou_result"] in COMPLETED]
    matched = matched_sat + matched_unsat
    n_completed = len(completed)
    return {
        "n_total": n_total,
        "expected_sat": exp_sat,
        "expected_unsat": exp_unsat,
        "completed": n_completed,
        "matched_sat": matched_sat,
        "matched_unsat": matched_unsat,
        "matched_total": matched,
        "timeouts": timeouts,
        "errors": errors,
        "not_run_missing_assets": not_run,
        "agreement_excluding_timeouts_errors":
            (round(matched / n_completed, 4) if n_completed else None),
        "agreement_including_timeouts_errors_as_nonmatch":
            (round(matched / (n_total - not_run), 4)
             if (n_total - not_run) else None),
    }


def main():
    data = json.loads(IN_JSON.read_text())
    records = data["results"]
    # Whole-slice (over instances that were actually attempted, i.e. not
    # missing assets) plus per-benchmark.
    attempted = [r for r in records if r["marabou_result"] != "not_run"]
    overall = tally(records)
    by_bench = {}
    grp = defaultdict(list)
    for r in records:
        grp[r["bench"]].append(r)
    for b, recs in sorted(grp.items()):
        by_bench[b] = tally(recs)

    summary = {
        "provenance": {
            **data.get("provenance", {}),
            "summarizer": "summarize_vnnlib_calibration.py",
        },
        "overall": overall,
        "per_benchmark": by_bench,
        "interpretation": (
            "Solver/property-plumbing calibration on external known-label "
            "instances. High agreement on completed queries indicates the "
            "ONNX+VNNLIB handling reproduces accepted SAT/UNSAT ground truth; "
            "it does not validate semantic realism of any TinyTaxiNet witness."),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    o = overall
    print(f"Attempted {len(attempted)} instances "
          f"({o['expected_sat']} exp-SAT, {o['expected_unsat']} exp-UNSAT)")
    print(f"  completed: {o['completed']}  "
          f"matched: {o['matched_total']} "
          f"(SAT {o['matched_sat']}, UNSAT {o['matched_unsat']})  "
          f"timeouts: {o['timeouts']}  errors: {o['errors']}")
    print(f"  agreement (completed only): "
          f"{o['agreement_excluding_timeouts_errors']}")
    print(f"  agreement (timeouts/errors as nonmatch): "
          f"{o['agreement_including_timeouts_errors_as_nonmatch']}")
    for b, s in by_bench.items():
        if s["n_total"] - s["not_run_missing_assets"] > 0:
            print(f"  {b:16s} completed {s['completed']}/{s['n_total']}  "
                  f"agree(compl)={s['agreement_excluding_timeouts_errors']}")
    print(f"Wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
