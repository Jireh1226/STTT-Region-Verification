#!/usr/bin/env python3
"""Correctness checks for the external bounded-transfer experiment.

(3) Output indexing: prove the composed two-output net has ONNX Runtime outputs
    [pixels, logits] but Marabou CLI variable order Y_0..9 = logits and
    Y_10..793 = reconstructed pixels. The Marabou cross-check uses a
    confusion-only query on Y_0..9 and must reproduce the unbounded R4 verdict.
(2) Monotonic sanity: every reported R4 SAT witness z* must actually produce
    class confusion when evaluated, and must remain feasible at larger alpha
    (|z*_j| <= alpha*sigma_j for alpha >= the alpha it was found at). Replays
    each witness through the network with onnxruntime.

Outputs: external_mnistfc_output_indexing_validation.json
         external_mnistfc_monotonic_sanity.json
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

import verify_external_mnistfc_bounded_transfer as B

ROOT = Path(__file__).resolve().parent
MARABOU = ROOT / "prophecy_env/bin/Marabou"


def cli(model, vnnlib, timeout=120):
    try:
        p = subprocess.run([str(MARABOU), str(model), str(vnnlib),
                            f"--timeout={timeout}", "--snc", "--num-workers=4",
                            "--verbosity=0"], capture_output=True, text=True,
                           timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return "timeout"
    toks = (p.stdout + p.stderr).lower().split()
    return "unsat" if "unsat" in toks else ("sat" if "sat" in toks else "to/err")


def main():
    Xtr, ytr, Xte, yte = B.load_mnist()
    anchors = json.loads((ROOT / "external_mnistfc_anchor_baseline.json").read_text())["anchors"]
    s0 = ort.InferenceSession(str(B.ONNX)); n0 = s0.get_inputs()[0].name

    def fwd0(x):
        return np.asarray(s0.run(None, {n0: x.reshape(1, 784, 1).astype(np.float32)})[0]).flatten()

    # ---- (3) output indexing ----
    a = anchors[0]; idx, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
    x0 = Xte[idx]
    Xc = Xtr[ytr == c]
    nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:B.KNB]]
    _, _, Vt = np.linalg.svd(nb - x0[None, :], full_matrices=False)
    V = Vt[:B.D]; sig = ((nb - x0[None, :]) @ V.T).std(0)
    m4 = B.compose_pca_two(V, x0, "validate")
    sc = ort.InferenceSession(str(m4))
    outs = sc.run(None, {sc.get_inputs()[0].name: np.zeros((1, B.D), np.float32)})
    out0, out1 = np.asarray(outs[0]).flatten(), np.asarray(outs[1]).flatten()
    lg0 = fwd0(x0)
    indexing = {
        "composed_output_names": [o.name for o in sc.get_outputs()],
        "composed_output_shapes": [list(o.shape) for o in sc.get_outputs()],
        "first_output_is_pixels_z0_matches_x0": bool(np.allclose(out0, x0, atol=1e-4)),
        "first_output_len": int(len(out0)),
        "second_output_is_logits_z0_matches_mnistfc": bool(np.allclose(out1, lg0, atol=1e-3)),
        "second_output_len": int(len(out1)),
        "onnxruntime_output_order": "output 0 = pixels, output 1 = logits",
        "marabou_pixel_indices": "Y_10..793",
        "marabou_logit_indices": "Y_0..9",
    }
    # Marabou cross-check: confusion-only on Y_0..9 at alpha=1 must equal the
    # unbounded R4@1sigma verdict for this anchor (logits-only net).
    r4u = {(r["anchor_idx"], r["alpha"]): r["status"]
           for r in json.loads((ROOT / "external_mnistfc_r4_results.json").read_text())["results"]}
    vp = ROOT / "_val.vnnlib"
    L = [f"(declare-const X_{j} Real)" for j in range(B.D)] + \
        [f"(declare-const Y_{k} Real)" for k in range(794)]
    for j in range(B.D):
        L += [f"(assert (>= X_{j} {-1.0*sig[j]:.6f}))", f"(assert (<= X_{j} {1.0*sig[j]:.6f}))"]
    L.append(f"(assert (>= Y_{t} Y_{c}))")
    vp.write_text("\n".join(L) + "\n")
    twoout_verdict = cli(m4, vp)
    vp.unlink(missing_ok=True)
    indexing["marabou_confusion_at_Y0to9_alpha1"] = twoout_verdict
    indexing["unbounded_R4_logits_only_alpha1"] = r4u.get((idx, 1.0))
    indexing["cross_check_matches"] = bool(twoout_verdict == r4u.get((idx, 1.0)))
    (ROOT / "external_mnistfc_output_indexing_validation.json").write_text(
        json.dumps({"provenance": {"script": "external_mnistfc_validation.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                   "anchor_idx": idx}, "indexing": indexing}, indent=2) + "\n")

    # ---- (2) monotonic sanity: every R4 SAT witness must produce confusion ----
    r4 = json.loads((ROOT / "external_mnistfc_r4_results.json").read_text())["results"]
    checks = []
    ok = 0
    for r in r4:
        if r["status"] != "sat" or "witness_z" not in r:
            continue
        ai = r["anchor_idx"]; cc = r["true_class"]; tt = r["target_class"]; al = r["alpha"]
        z = np.asarray(r["witness_z"])
        x0i = Xte[ai]
        Xci = Xtr[ytr == cc]
        nbi = Xci[np.argsort(np.linalg.norm(Xci - x0i[None, :], axis=1))[:B.KNB]]
        _, _, Vti = np.linalg.svd(nbi - x0i[None, :], full_matrices=False)
        Vi = Vti[:B.D]; sigi = ((nbi - x0i[None, :]) @ Vi.T).std(0)
        x = x0i + Vi.T @ z
        lg = fwd0(x)
        confusion = bool(lg[tt] >= lg[cc])           # property actually holds?
        in_box = bool(np.all(np.abs(z) <= al * sigi + 1e-5))
        feasible_larger = {f"alpha_{int(la)}": bool(np.all(np.abs(z) <= la * sigi + 1e-5))
                           for la in [1, 2, 3] if la >= al}
        checks.append({"anchor_idx": ai, "alpha": al,
                       "witness_produces_confusion": confusion,
                       "z_in_own_alpha_box": in_box,
                       "feasible_at_larger_alpha": feasible_larger})
        ok += int(confusion and in_box)
    (ROOT / "external_mnistfc_monotonic_sanity.json").write_text(
        json.dumps({"provenance": {"script": "external_mnistfc_validation.py",
                                   "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                    "n_sat_witnesses_checked": len(checks),
                    "n_confusion_and_in_box": ok,
                    "all_witnesses_valid": bool(ok == len(checks)),
                    "checks": checks}, indent=2) + "\n")

    print("INDEXING:", json.dumps({k: indexing[k] for k in
          ["first_output_is_pixels_z0_matches_x0", "second_output_is_logits_z0_matches_mnistfc",
           "marabou_confusion_at_Y0to9_alpha1", "unbounded_R4_logits_only_alpha1",
           "cross_check_matches"]}))
    print(f"MONOTONIC: {ok}/{len(checks)} SAT witnesses produce confusion AND lie in their alpha box; "
          f"all_valid={ok == len(checks)}")
    print("Wrote external_mnistfc_output_indexing_validation.json, external_mnistfc_monotonic_sanity.json")


if __name__ == "__main__":
    main()
