#!/usr/bin/env python3
"""Witness audit for the strong external mnist_fc region-transfer run.

Audits every SAT witness from R1, R4 (local-PCA), and R3 (class-AE) against
held-out real MNIST data. R3 witnesses are latent z; the pixel image is
reconstructed by running the class decoder. Reports held-out real baselines.

Inputs:  external_mnistfc_r1_r2_results.json, external_mnistfc_r4_results.json,
         external_mnistfc_r3_results.json, mnist_class_aes.npz
Outputs: external_mnistfc_strong_transfer_witness_audit.json,
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
AE = ROOT / "mnist_class_aes.npz"
SHAPE = (28, 28)


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    return (m["x_train"].reshape(-1, 784) / 255.0, m["y_train"],
            m["x_test"].reshape(-1, 784) / 255.0, m["y_test"])


def tv(x):
    img = x.reshape(28, 28)
    return float(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())


def decode_r3(z, ae, c):
    h = np.maximum(ae[f"dec1_W_class{c}"] @ z + ae[f"dec1_b_class{c}"], 0.0)
    return ae[f"dec2_W_class{c}"] @ h + ae[f"dec2_b_class{c}"]


def reconstruct_r4(z, x0, train_images, train_labels, c):
    Xc = train_images[train_labels == c]
    nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:100]]
    deviations = nb - x0[None, :]
    _, _, vt = np.linalg.svd(deviations, full_matrices=False)
    basis = vt[:8]
    return x0 + basis.T @ z


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    ae = np.load(AE)
    pairs, taus = ra.calibrate_per_pair(Xtr, SHAPE, q=99)
    sess = ort.InferenceSession(str(ONNX)); iname = sess.get_inputs()[0].name

    def fwd(x):
        return np.asarray(sess.run(None, {iname: x.reshape(1, 784, 1).astype(np.float32)})[0]).flatten()

    def ppv(x):
        return float(np.mean([abs(x[a] - x[b]) > t for (a, b), t in zip(pairs, taus)]))

    # held-out real baselines
    samp = Xte[::20]
    nn = []
    for i in range(len(samp)):
        d = np.linalg.norm(samp - samp[i][None, :], axis=1); d[i] = np.inf
        nn.append(d.min())
    baselines = {
        "test_self_nn_L2_median": float(np.median(nn)),
        "test_per_pair_violation_frac_mean": float(np.mean([ppv(x) for x in samp])),
        "test_tv_median": float(np.median([tv(x) for x in samp])),
    }

    def audit_pixels(x, c, region, extra):
        Xtc, Xec = Xtr[ytr == c], Xte[yte == c]
        lg = fwd(x)
        return {**extra, "region": region, "true_class": c,
                "pred_class": int(np.argmax(lg)),
                "in_unit_box": bool((x >= -1e-6).all() and (x <= 1 + 1e-6).all()),
                "nearest_train_sameclass_L2": round(float(np.min(np.linalg.norm(Xtc - x[None, :], axis=1))), 3),
                "nearest_test_sameclass_L2": round(float(np.min(np.linalg.norm(Xec - x[None, :], axis=1))), 3),
                "nearest_test_allclass_L2": round(float(np.min(np.linalg.norm(Xte - x[None, :], axis=1))), 3),
                "per_pair_violation_frac": round(ppv(x), 3),
                "tv": round(tv(x), 2)}

    audits = []
    for fn, region in [("external_mnistfc_r1_r2_results.json", "R1_linf_ball"),
                       ("external_mnistfc_r4_results.json", "R4_local_pca")]:
        p = ROOT / fn
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["results"]:
            if r["status"] == "sat" and region == "R4_local_pca" and "witness_z" in r:
                x0 = Xte[r["anchor_idx"]]
                x = reconstruct_r4(np.asarray(r["witness_z"], dtype=np.float64),
                                   x0, Xtr, ytr, r["true_class"])
            elif r["status"] == "sat" and "witness_pixels" in r:
                x = np.asarray(r["witness_pixels"], dtype=np.float64)
            else:
                continue
            if r["status"] == "sat":
                extra = {"anchor_idx": r["anchor_idx"], "target_class": r["target_class"],
                         "eps": r.get("eps"), "alpha": r.get("alpha")}
                audits.append(audit_pixels(x, r["true_class"], region, extra))
    p3 = ROOT / "external_mnistfc_r3_results.json"
    if p3.exists():
        for r in json.loads(p3.read_text())["results"]:
            if r["status"] == "sat" and "witness_z" in r:
                x = decode_r3(np.asarray(r["witness_z"]), ae, r["true_class"])
                extra = {"anchor_idx": r["anchor_idx"], "target_class": r["target_class"],
                         "latent": True}
                audits.append(audit_pixels(x, r["true_class"], "R3_class_ae", extra))

    (ROOT / "external_mnistfc_real_baselines.json").write_text(
        json.dumps({"provenance": {"script": "audit_external_mnistfc_strong_transfer.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                    "baselines": baselines}, indent=2) + "\n")
    (ROOT / "external_mnistfc_strong_transfer_witness_audit.json").write_text(
        json.dumps({"provenance": {"script": "audit_external_mnistfc_strong_transfer.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                   "n_witnesses": len(audits)},
                    "real_baselines": baselines, "witness_audits": audits}, indent=2) + "\n")

    # region-level summary
    print(f"baselines: self-NN L2 {baselines['test_self_nn_L2_median']:.2f}, "
          f"per-pair {baselines['test_per_pair_violation_frac_mean']:.3f}, tv {baselines['test_tv_median']:.0f}")
    from collections import defaultdict
    byreg = defaultdict(list)
    for a in audits:
        byreg[a["region"]].append(a)
    for reg, aa in sorted(byreg.items()):
        nnt = np.median([a["nearest_test_sameclass_L2"] for a in aa])
        pp = np.median([a["per_pair_violation_frac"] for a in aa])
        print(f"  {reg:14s} n={len(aa):3d}  nnTest(same) med={nnt:.2f}  per-pair med={pp:.3f}")
    print("Wrote external_mnistfc_strong_transfer_witness_audit.json, external_mnistfc_real_baselines.json")


if __name__ == "__main__":
    main()
