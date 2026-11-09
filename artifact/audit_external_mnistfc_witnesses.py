#!/usr/bin/env python3
"""Audit SAT witnesses from the external mnist_fc region-transfer run against
held-out real MNIST data, and report real-data baselines.

Uses region_audit.grid_pairs / calibrate_per_pair for the per-pair Lipschitz
signal. Held-out test images are the real reference for the audit; train images
are reported separately. Forward eval (model accuracy/margin on anchors) uses
onnxruntime on the external ONNX.

Inputs:  external_mnistfc_region_transfer_results.json
Outputs: external_mnistfc_witness_audit.json
         external_mnistfc_real_baselines.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

import region_audit as ra

ROOT = Path(__file__).resolve().parent
ONNX = ROOT / "external/vnncomp/mnist_fc/onnx/mnist-net_256x2.onnx"
RESULTS = ROOT / "external_mnistfc_region_transfer_results.json"
SHAPE = (28, 28)


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    return (m["x_train"].reshape(-1, 784) / 255.0, m["y_train"],
            m["x_test"].reshape(-1, 784) / 255.0, m["y_test"])


def tv(x):
    img = x.reshape(28, 28)
    return float(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())


def nn_l2(x, ref):
    return float(np.min(np.linalg.norm(ref - x[None, :], axis=1)))


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    pairs, taus = ra.calibrate_per_pair(Xtr, SHAPE, q=99)

    def per_pair_frac(x):
        return float(np.mean([abs(x[a] - x[b]) > t for (a, b), t in zip(pairs, taus)]))

    # Real baselines on held-out test (sampled for the L2 self-NN to bound cost)
    samp = Xte[::20]
    nn = []
    for i in range(len(samp)):
        d = np.linalg.norm(samp - samp[i][None, :], axis=1)
        d[i] = np.inf
        nn.append(d.min())
    baselines = {
        "test_self_nn_L2_median": float(np.median(nn)),
        "test_self_nn_L2_p95": float(np.percentile(nn, 95)),
        "test_per_pair_violation_frac_mean": float(np.mean([per_pair_frac(x) for x in samp])),
        "test_tv_median": float(np.median([tv(x) for x in samp])),
        "note": "held-out test reference (sampled); train used only for per-pair calibration",
    }

    # Model accuracy / margin on the selected anchors
    res = json.loads(RESULTS.read_text())
    sess = ort.InferenceSession(str(ONNX))
    iname = sess.get_inputs()[0].name

    def fwd(x):
        return np.asarray(sess.run(None, {iname: x.reshape(1, 784, 1).astype(np.float32)})[0]).flatten()

    anchors = sorted({(r["anchor_idx"], r["true_class"]) for r in res["results"]})
    margins = []
    correct = 0
    for idx, c in anchors:
        lg = fwd(Xte[idx])
        correct += int(np.argmax(lg) == c)
        top2 = np.sort(lg)[-2:]
        margins.append(float(top2[1] - top2[0]))
    baselines["anchor_model_accuracy"] = correct / max(1, len(anchors))
    baselines["anchor_margin_median"] = float(np.median(margins)) if margins else None

    # Audit each SAT witness
    witness_audits = []
    for r in res["results"]:
        if r["status"] != "sat" or "witness" not in r:
            continue
        x = np.asarray(r["witness"], dtype=np.float64)
        c = r["true_class"]
        Xte_c, Xtr_c = Xte[yte == c], Xtr[ytr == c]
        witness_audits.append({
            "anchor_idx": r["anchor_idx"], "true_class": c,
            "target_class": r["target_class"], "region": r["region"],
            "witness_argmax": r.get("witness_argmax"),
            "nearest_test_sameclass_L2": round(nn_l2(x, Xte_c), 3),
            "nearest_test_allclass_L2": round(nn_l2(x, Xte), 3),
            "nearest_train_sameclass_L2": round(nn_l2(x, Xtr_c), 3),
            "per_pair_violation_frac": round(per_pair_frac(x), 3),
            "tv": round(tv(x), 2),
        })

    (ROOT / "external_mnistfc_real_baselines.json").write_text(
        json.dumps({"provenance": {"script": "audit_external_mnistfc_witnesses.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                    "baselines": baselines}, indent=2) + "\n")
    (ROOT / "external_mnistfc_witness_audit.json").write_text(
        json.dumps({"provenance": {"script": "audit_external_mnistfc_witnesses.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                   "n_sat_witnesses": len(witness_audits)},
                    "real_baselines": baselines,
                    "witness_audits": witness_audits}, indent=2) + "\n")

    print(f"baselines: test self-NN L2 median {baselines['test_self_nn_L2_median']:.2f}, "
          f"per-pair {baselines['test_per_pair_violation_frac_mean']:.3f}, "
          f"tv median {baselines['test_tv_median']:.1f}")
    print(f"anchor accuracy {baselines['anchor_model_accuracy']:.2f}, "
          f"margin median {baselines['anchor_margin_median']}")
    for w in witness_audits:
        print(f"  {w['region']:18s} cls{w['true_class']}->t{w['target_class']} "
              f"argmax={w['witness_argmax']} nnTest(same)={w['nearest_test_sameclass_L2']} "
              f"ppv={w['per_pair_violation_frac']} tv={w['tv']}")
    print(f"Wrote external_mnistfc_witness_audit.json, external_mnistfc_real_baselines.json")


if __name__ == "__main__":
    main()
