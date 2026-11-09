"""
TinyTaxiNet instantiation of the region_audit module.

Demonstrates that the network-agnostic region constructors and witness audit
of region_audit.py reproduce representative recorded verdicts from the
TinyTaxiNet study. The benchmark-specific code below is exactly what an
instantiation owes the module: dataset loading, subset predicates, the
neighbor rule, and the network reader. Every region construction, math
filter, query, and audit signal comes from the module.

Smoke checks (expected verdicts from the released results):
  1. P1a (CTE >= 10)  under R1(C_centered)            -> sat
  2. P1d (HE <= -90)  under R2 = R1 + mean equality   -> unsat
  3. P1a (CTE >= 10)  under R2 intersect TV <= 10     -> sat
  4. P3a (CTE >= 10)  under R5(label box, idx 921)    -> sat   (violated anchor)
  5. P3 all four subs under R5(label box, idx 20)     -> unsat (verified anchor)
  6. Witness from (4) scored by the three audit signals vs the real baseline.

Output: module_smoke_taxinet.json
"""
import json
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou

import region_audit as ra

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_PATH = "plausibility_classifier.npz"
OUTPUT_JSON = "module_smoke_taxinet.json"

SHAPE = (8, 16)
CTE_OUT, HE_OUT = 0, 1


def make_net():
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in np.asarray(net.inputVars[0]).flatten()]
    ovars = [int(v) for v in np.asarray(net.outputVars[0]).flatten()]
    return net, ivars, ovars


def label_neighbors(images, labels, idx, d_cte=0.5, d_he=5.0):
    """TaxiNet neighbor rule: same-state label bin, anchor included."""
    cte, he = labels[:, 0], labels[:, 1]
    mask = (np.abs(cte - cte[idx]) <= d_cte) & (np.abs(he - he[idx]) <= d_he)
    return images[mask]


def main():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    centered = images[np.abs(labels[:, 0]) < 1.0]

    checks = []

    def check(name, expected, verdict):
        ok = verdict["status"] == expected
        checks.append({"name": name, "expected": expected,
                       "got": verdict["status"], "ok": ok,
                       "time_seconds": verdict["time_seconds"],
                       "region": verdict["region"]})
        print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: "
              f"expected {expected}, got {verdict['status']} "
              f"({verdict['time_seconds']:.2f}s)", flush=True)
        return verdict

    # 1. P1a under R1(C_centered): recorded sat
    r1 = ra.r1_subset_box(centered, subset_name="centered")
    check("P1a@R1(centered)", "sat",
          ra.run_query(make_net, r1, ra.output_at_least(CTE_OUT, 10.0)))

    # 2. P1d under R2 = R1 + mean equality: recorded unsat
    r2 = ra.intersect(r1, ra.mass_equality(64.0), tag="R2")
    check("P1d@R2(centered)", "unsat",
          ra.run_query(make_net, r2, ra.output_at_most(HE_OUT, -90.0)))

    # 3. P1a under R2 intersect TV <= 10: recorded sat
    r6 = ra.intersect(r2, ra.tv_bound(SHAPE, 10.0), tag="R6")
    check("P1a@R6(TV<=10)", "sat",
          ra.run_query(make_net, r6, ra.output_at_least(CTE_OUT, 10.0)))

    # 4./5. P3 under R5 label boxes: idx 921 violated (sub a sat),
    #       idx 20 verified (all four subs unsat)
    sat_witness = None
    r5_bad = ra.intersect(
        ra.r5_neighbor_box(label_neighbors(images, labels, 921)),
        ra.mass_equality(64.0))
    v = check("P3a@R5(idx921)", "sat",
              ra.run_query(make_net, r5_bad, ra.output_at_least(CTE_OUT, 10.0)))
    if v["status"] == "sat":
        sat_witness = np.array(v["witness"])

    r5_good = ra.intersect(
        ra.r5_neighbor_box(label_neighbors(images, labels, 20)),
        ra.mass_equality(64.0))
    for name, post in [("a", ra.output_at_least(CTE_OUT, 10.0)),
                       ("b", ra.output_at_most(CTE_OUT, -10.0)),
                       ("c", ra.output_at_least(HE_OUT, 90.0)),
                       ("d", ra.output_at_most(HE_OUT, -90.0))]:
        check(f"P3{name}@R5(idx20)", "unsat",
              ra.run_query(make_net, r5_good, post))

    # 6. Audit the recovered witness against the centered real baseline
    audit_block = None
    if sat_witness is not None:
        clf = ra.load_relu_classifier(CLF_PATH)
        audit = ra.WitnessAudit(centered, SHAPE, classifier=clf,
                                pair_quantile=99)
        audit_block = {
            "witness_scores": audit.score(sat_witness),
            "real_baseline": audit.baseline(centered[::20]),
        }
        print(f"  witness audit: {audit_block['witness_scores']}")
        print(f"  real baseline: {audit_block['real_baseline']}")

    n_ok = sum(c["ok"] for c in checks)
    out = {
        "provenance": {
            "script": "taxinet_instantiation.py",
            "module": "region_audit.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "checks": checks,
        "summary": {"passed": n_ok, "total": len(checks)},
        "witness_audit": audit_block,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{n_ok}/{len(checks)} verdict reproductions passed. "
          f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
