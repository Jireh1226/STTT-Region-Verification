#!/usr/bin/env python3
"""External region-transfer verification on VNN-COMP mnist_fc (mnist-net_256x2),
a 784-256-256-10 ReLU image classifier we did not train.

Tests whether the region-tagged workflow produces region-dependent verdicts on
an external network. The intended path is region_audit.run_query, but maraboupy
import hangs on this machine, so we solve through the Marabou 2.0 CLI on the
external ONNX with generated VNN-LIB box queries (the proven calibration path)
and use region_audit's numpy helpers for audit calibration.

Only the pixel-box regions (R1, R2, R5) are expressible in the pure VNN-LIB the
CLI parses without auxiliary variables. R3/R4 (decoder/PCA composition) and R6
(TV, needs slack variables) require maraboupy or ONNX surgery and are recorded
as not-run in the manifest. Exact status comparison only (no substring match,
since "unsat" contains "sat").

Outputs: external_mnistfc_region_transfer_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
ONNX = ROOT / "external/vnncomp/mnist_fc/onnx/mnist-net_256x2.onnx"
MARABOU = ROOT / "prophecy_env/bin/Marabou"
OUT = ROOT / "external_mnistfc_region_transfer_results.json"
EPS = 0.05
K_NB = 100
TIMEOUT = 120


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    Xtr = m["x_train"].reshape(-1, 784).astype(np.float64) / 255.0
    Xte = m["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
    return Xtr, m["y_train"], Xte, m["y_test"]


def make_session():
    sess = ort.InferenceSession(str(ONNX))
    name = sess.get_inputs()[0].name
    return sess, name


def fwd(sess, name, x):
    return np.asarray(sess.run(None, {name: x.reshape(1, 784, 1).astype(np.float32)})[0]).flatten()


def select_anchors(sess, name, Xte, yte, per_class):
    anchors = []
    for c in range(10):
        n = 0
        for i in np.where(yte == c)[0]:
            if int(np.argmax(fwd(sess, name, Xte[i]))) == c:
                anchors.append((int(i), c))
                n += 1
                if n >= per_class:
                    break
    return anchors


def emit_vnnlib(pmin, pmax, c, target, path, mass=None):
    L = [f"(declare-const X_{i} Real)" for i in range(784)]
    L += [f"(declare-const Y_{j} Real)" for j in range(10)]
    for i in range(784):
        L.append(f"(assert (>= X_{i} {pmin[i]:.6f}))")
        L.append(f"(assert (<= X_{i} {pmax[i]:.6f}))")
    if mass is not None:
        lo, hi = mass
        s = " ".join(f"X_{i}" for i in range(784))
        L.append(f"(assert (>= (+ {s}) {lo:.4f}))")
        L.append(f"(assert (<= (+ {s}) {hi:.4f}))")
    L.append(f"(assert (>= Y_{target} Y_{c}))")   # single-target confusion
    Path(path).write_text("\n".join(L) + "\n")


def solve(vnnlib):
    t0 = time.time()
    try:
        p = subprocess.run([str(MARABOU), str(ONNX), str(vnnlib),
                            f"--timeout={TIMEOUT}", "--snc", "--num-workers=4",
                            "--verbosity=0"],
                           capture_output=True, text=True, timeout=TIMEOUT + 30)
    except subprocess.TimeoutExpired:
        return "timeout", round(time.time() - t0, 1), None
    out = p.stdout + "\n" + p.stderr
    toks = out.lower().split()
    dt = round(time.time() - t0, 1)
    if "unsat" in toks:
        return "unsat", dt, None
    if "sat" in toks:
        wit = {}
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("x") and "=" in s and "_" not in s:
                try:
                    k, v = s.split("=")
                    wit[int(k.strip()[1:])] = float(v)
                except ValueError:
                    pass
        x = ([wit[i] for i in range(784)] if len(wit) >= 784 else None)
        return "sat", dt, x
    if "timeout" in out.lower():
        return "timeout", dt, None
    return "error", dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1,
                    help="anchors per class (manifest target 2; bounded for compute)")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_mnist()
    sess, name = make_session()
    anchors = select_anchors(sess, name, Xte, yte, args.per_class)

    # class mass-band stats from train (for R2)
    mass = {c: (float(Xtr[ytr == c].sum(1).mean()), float(Xtr[ytr == c].sum(1).std()))
            for c in range(10)}

    records = []
    for (idx, c) in anchors:
        x0 = Xte[idx]
        logits = fwd(sess, name, x0)
        target = int(np.argsort(logits)[-2])
        Xc = Xtr[ytr == c]
        nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:K_NB]]
        mu, sd = mass[c]
        regions = {
            "R1_linf_ball": (np.clip(x0 - EPS, 0, 1), np.clip(x0 + EPS, 0, 1), None),
            "R2_ball_mass_band": (np.clip(x0 - EPS, 0, 1), np.clip(x0 + EPS, 0, 1),
                                  (mu - 2 * sd, mu + 2 * sd)),
            "R5_neighbor_box": (np.clip(nb.min(0), 0, 1), np.clip(nb.max(0), 0, 1), None),
        }
        for rname, (pmin, pmax, m) in regions.items():
            vp = ROOT / f"_ext_{rname}.vnnlib"
            emit_vnnlib(pmin, pmax, c, target, vp, mass=m)
            st, dt, wit = solve(vp)
            rec = {"anchor_idx": idx, "true_class": c, "target_class": target,
                   "region": rname, "eps": EPS, "property": "P4ext_single_target",
                   "status": st, "runtime_s": dt,
                   "median_half_width": float(np.median((pmax - pmin) / 2))}
            if st == "sat" and wit is not None:
                rec["witness"] = wit
                rec["witness_argmax"] = int(np.argmax(fwd(sess, name, np.array(wit))))
            records.append(rec)
            vp.unlink(missing_ok=True)
            print(f"  idx{idx} cls{c}->t{target} {rname:18s} {st:8s} {dt:5.0f}s",
                  flush=True)

    from collections import Counter
    counts = Counter((r["region"], r["status"]) for r in records)
    out = {
        "provenance": {
            "script": "verify_external_mnistfc_region_transfer.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "external_network": "VNN-COMP mnist_fc mnist-net_256x2 (not trained by us)",
            "solver": "Marabou 2.0 CLI on external ONNX, SnC, 4 workers, 120s",
            "regions_run": ["R1_linf_ball", "R2_ball_mass_band", "R5_neighbor_box"],
            "regions_not_run": {"R3_classAE_R4_localPCA": "need maraboupy/ONNX surgery (unavailable)",
                                "R6_TV": "needs aux slack vars, not pure VNN-LIB"},
            "anchors": len(anchors), "per_class": args.per_class, "eps": EPS,
        },
        "status_counts_by_region": {f"{r}:{s}": n for (r, s), n in counts.items()},
        "results": records,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nstatus counts: {dict(out['status_counts_by_region'])}")
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
