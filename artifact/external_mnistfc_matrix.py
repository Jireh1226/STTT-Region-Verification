#!/usr/bin/env python3
"""Comprehensive external bounded-transfer matrix on VNN-COMP mnist_fc.

Runs the controlled radius/scale matrix with validation checks for output
indexing and SAT-witness replay:

  anchors : 50 deterministic held-out test anchors (5 per class, correctly
            classified, ascending index). Train-only region calibration.
  regions : R1 broad pixel ball (eps in {0.05, 0.03}); R4 local PCA with
            VALID PIXELS enforced (alpha in {1,2,3} sigma), composed two-output
            net (logits Y_0..9, pixels Y_10..793) with 0<=x_i<=1 on pixels.
  property: targeted class confusion logit_t >= logit_c (t = top non-true logit).

Run order puts the headline first: R1@0.05 on all 50, then R4+bounds@1sigma on
the broad-SAT subset (eps=0.05 target), then eps=0.03, then alpha=2/3.

Auto-validation (always written; the run aborts if indexing fails):
  - output-indexing: self-contained Marabou cross-check (confusion@Y_0..9 must
    match the onnxruntime-evaluated verdict on a tiny z-box; @Y_784+ should differ).
  - per SAT witness: z-box feasibility, pixel-bound satisfaction (raw recon in
    [0,1]), and confusion replay through the saved composed net.

Outputs (incremental):
  external_mnistfc_matrix_results.json
  external_mnistfc_matrix_validation.json
  external_mnistfc_matrix_manifest.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

import verify_external_mnistfc_bounded_transfer as B

ROOT = Path(__file__).resolve().parent
MARABOU = ROOT / "prophecy_env/bin/Marabou"
PER_CLASS = 5
EPS_LIST = [0.05, 0.03]
ALPHAS = [1.0, 2.0, 3.0]
TIMEOUT = 600        # R4 bounded queries (these resolve in 30-400s)
R1_TIMEOUT = 120     # broad R1 pixel-ball: SAT cases resolve in <=45s; the hard
                     # ones do not resolve even at 600s, so a longer budget
                     # only wastes time (verified empirically).

RES = ROOT / "external_mnistfc_matrix_results.json"
VAL = ROOT / "external_mnistfc_matrix_validation.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def cli(model, vnnlib, timeout=TIMEOUT, want_witness=False):
    t0 = time.time()
    try:
        p = subprocess.run([str(MARABOU), str(model), str(vnnlib),
                            f"--timeout={timeout}", "--snc", "--num-workers=4",
                            "--verbosity=0"], capture_output=True, text=True,
                           timeout=timeout + 40)
    except subprocess.TimeoutExpired:
        return "timeout", round(time.time() - t0, 1), None
    out = p.stdout + "\n" + p.stderr
    toks = out.lower().split()
    dt = round(time.time() - t0, 1)
    status = "unsat" if "unsat" in toks else ("sat" if "sat" in toks
                                              else ("timeout" if "timeout" in out.lower() else "error"))
    wit = None
    if status == "sat" and want_witness:
        wit = {}
        for ln in out.splitlines():
            s = ln.strip()
            if s.startswith("x") and "=" in s and "_" not in s:
                try:
                    k, v = s.split("="); wit[int(k.strip()[1:])] = float(v)
                except ValueError:
                    pass
    return status, dt, wit


def emit_pixel_box(pmin, pmax, c, t, path):
    """R1 broad pixel ball on the BASE net (single output, logits Y_0..9)."""
    L = [f"(declare-const X_{i} Real)" for i in range(784)]
    L += [f"(declare-const Y_{k} Real)" for k in range(10)]
    for i in range(784):
        L.append(f"(assert (>= X_{i} {pmin[i]:.6f}))")
        L.append(f"(assert (<= X_{i} {pmax[i]:.6f}))")
    L.append(f"(assert (>= Y_{t} Y_{c}))")
    Path(path).write_text("\n".join(L) + "\n")


def pca_basis(x0, Xtr, ytr, c):
    Xc = Xtr[ytr == c]
    nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:B.KNB]]
    Dd = nb - x0[None, :]
    _, _, Vt = np.linalg.svd(Dd, full_matrices=False)
    V = Vt[:B.D]
    sig = (Dd @ V.T).std(0)
    return V, sig


def compose_pca_single(V, x0, tag):
    """Logits-only composed net z -> x0 + V^T z -> mnist_fc (single output,
    logits unambiguously at Y_0..9). Ground truth for the indexing check."""
    from onnx import helper, numpy_helper, compose, TensorProto
    inits = [numpy_helper.from_array(V.astype(np.float32), "Wpca"),
             numpy_helper.from_array(x0.astype(np.float32), "x0c"),
             numpy_helper.from_array(np.array([1, 784, 1], np.int64), "shp")]
    zin = helper.make_tensor_value_info("zin", TensorProto.FLOAT, [1, V.shape[0]])
    xdec = helper.make_tensor_value_info("xdec", TensorProto.FLOAT, [1, 784, 1])
    nodes = [helper.make_node("MatMul", ["zin", "Wpca"], ["mm"]),
             helper.make_node("Add", ["mm", "x0c"], ["xflat"]),
             helper.make_node("Reshape", ["xflat", "shp"], ["xdec"])]
    import onnx
    dec = helper.make_model(helper.make_graph(nodes, "dec", [zin], [xdec], inits),
                            opset_imports=[helper.make_opsetid("", B.OPSET)])
    dec.ir_version = B.IRV
    merged = compose.merge_models(dec, B.BASE, io_map=[("xdec", B.IN_NAME)])
    p = ROOT / "external_mnistfc_bounded_composed_models" / f"single_{tag}.onnx"
    onnx.save(merged, str(p))
    return p


def auto_indexing_validation(anchors, Xtr, ytr, Xte):
    """Decisive, self-contained: a single-output logits-only net gives the
    ground-truth confusion verdict over the same alpha=1 region; the two-output
    net's confusion@Y_0..9 must match it, and @Y_784+ must not. Confirms logits
    are at Y_0..9 on the two-output net. Aborts the run on failure."""
    a = anchors[0]; idx, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
    x0 = Xte[idx]; V, sig = pca_basis(x0, Xtr, ytr, c)
    m_single = compose_pca_single(V, x0, "idxval")
    m_two = B.compose_pca_two(V, x0, "matrix_idxval")
    sc = ort.InferenceSession(str(m_two))
    o = sc.run(None, {sc.get_inputs()[0].name: np.zeros((1, B.D), np.float32)})
    ort_pixels_first = bool(np.allclose(np.asarray(o[0]).flatten(), x0, atol=1e-4))

    def conf(model, ti, ci, ndecl, path):
        L = [f"(declare-const X_{j} Real)" for j in range(B.D)] + \
            [f"(declare-const Y_{k} Real)" for k in range(ndecl)]
        for j in range(B.D):
            L += [f"(assert (>= X_{j} {-1.0*sig[j]:.6f}))", f"(assert (<= X_{j} {1.0*sig[j]:.6f}))"]
        L.append(f"(assert (>= Y_{ti} Y_{ci}))")
        Path(path).write_text("\n".join(L) + "\n")
        return cli(model, path, timeout=300)[0]

    truth = conf(m_single, t, c, 10, ROOT / "_idx_truth.vnnlib")      # logits-only net
    v_lo = conf(m_two, t, c, 794, ROOT / "_idx_lo.vnnlib")            # two-out @Y_0..9
    v_hi = conf(m_two, 784 + t, 784 + c, 794, ROOT / "_idx_hi.vnnlib")  # two-out @Y_784+
    for f in ("_idx_truth.vnnlib", "_idx_lo.vnnlib", "_idx_hi.vnnlib"):
        (ROOT / f).unlink(missing_ok=True)
    logits_at = ("Y_0..9" if v_lo == truth and v_hi != truth
                 else ("Y_784..793" if v_hi == truth and v_lo != truth else "AMBIGUOUS"))
    return {"timestamp_utc": now(), "anchor_idx": idx,
            "onnxruntime_output0_is_pixels": ort_pixels_first,
            "ground_truth_logits_only_net": truth,
            "two_output_confusion_at_Y0to9": v_lo,
            "two_output_confusion_at_Y784plus": v_hi,
            "logits_at": logits_at, "expected": "Y_0..9",
            "PASS": bool(logits_at == "Y_0..9")}


def witness_checks(z, x0, V, sig, alpha, c, t, saved_net):
    """Per-witness: z-box feasibility, pixel-bound satisfaction (raw, unclipped),
    confusion replay through the saved composed net (onnxruntime)."""
    z = np.asarray(z)
    xr = x0 + V.T @ z
    sc = ort.InferenceSession(str(saved_net))
    lg = np.asarray(sc.run(None, {sc.get_inputs()[0].name: z.reshape(1, -1).astype(np.float32)})[0]).flatten()
    # composed net outputs [pixels(784), logits(10)] in onnxruntime order
    logits = lg[-10:] if len(lg) >= 794 else lg
    return {
        "z_in_alpha_box": bool(np.all(np.abs(z) <= alpha * sig + 1e-4)),
        "recon_min": round(float(xr.min()), 4), "recon_max": round(float(xr.max()), 4),
        "pixels_in_unit_cube": bool((xr >= -1e-4).all() and (xr <= 1 + 1e-4).all()),
        "confusion_replayed": bool(logits[t] >= logits[c] - 1e-3),
    }


def save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    Xtr, ytr, Xte, yte = B.load_mnist()
    s0 = ort.InferenceSession(str(B.ONNX)); n0 = s0.get_inputs()[0].name

    def fwd(x):
        return np.asarray(s0.run(None, {n0: x.reshape(1, 784, 1).astype(np.float32)})[0]).flatten()

    # 50 deterministic anchors
    anchors = []
    for c in range(10):
        k = 0
        for i in np.where(yte == c)[0]:
            lg = fwd(Xte[i])
            if int(np.argmax(lg)) == c:
                anchors.append({"anchor_idx": int(i), "true_class": int(c),
                                "target_class": int(np.argsort(lg)[-2])})
                k += 1
                if k >= PER_CLASS:
                    break
    save({"experiment": "external_mnistfc_matrix", "timestamp_utc": now(),
          "network": "VNN-COMP mnist_fc mnist-net_256x2 (not trained by us)",
          "per_class": PER_CLASS, "n_anchors": len(anchors),
          "epsilons": EPS_LIST, "alphas": ALPHAS, "timeout_s": TIMEOUT,
          "selection": "5 per class, correctly classified, ascending test index; train-only calibration",
          "anchors": anchors}, ROOT / "external_mnistfc_matrix_manifest.json")

    # auto-validation: indexing (abort on failure)
    val = {"output_indexing": auto_indexing_validation(anchors, Xtr, ytr, Xte),
           "witness_checks": []}
    save({"provenance": {"script": "external_mnistfc_matrix.py", "timestamp_utc": now()},
          "validation": val}, VAL)
    print("INDEXING VALIDATION:", json.dumps(val["output_indexing"]), flush=True)
    if not val["output_indexing"]["PASS"]:
        print("ABORT: output-indexing validation failed.", flush=True)
        sys.exit(2)

    records = []

    def run_and_log(rec):
        records.append(rec)
        save({"provenance": {"script": "external_mnistfc_matrix.py", "timestamp_utc": now(),
                             "n_anchors": len(anchors)}, "results": records}, RES)

    # cache PCA + composed nets per anchor
    cache = {}
    def get_pca(a):
        i = a["anchor_idx"]
        if i not in cache:
            x0 = Xte[i]; V, sig = pca_basis(x0, Xtr, ytr, a["true_class"])
            m = B.compose_pca_two(V, x0, f"mtx_{i}")
            cache[i] = (x0, V, sig, m)
        return cache[i]

    def r4_bounded(a, alpha):
        i, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
        x0, V, sig, m = get_pca(a)
        vp = ROOT / "_mtx_r4.vnnlib"
        B.emit_bounded(-alpha * sig, alpha * sig, c, t, vp)
        st, dt, wit = cli(m, vp, want_witness=True)
        vp.unlink(missing_ok=True)
        rec = {"anchor_idx": i, "true_class": c, "target_class": t,
               "region": "R4_bounded", "alpha": alpha, "status": st, "runtime_s": dt}
        if st == "sat" and wit and len(wit) >= B.D:
            z = [wit[j] for j in range(B.D)]
            rec["witness_z"] = z
            wc = witness_checks(z, x0, V, sig, alpha, c, t, m)
            rec["witness_checks"] = wc
            val["witness_checks"].append({"anchor_idx": i, "region": "R4_bounded",
                                          "alpha": alpha, **wc})
            save({"provenance": {"script": "external_mnistfc_matrix.py"}, "validation": val}, VAL)
        return rec

    def r1(a, eps):
        i, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i]
        vp = ROOT / "_mtx_r1.vnnlib"
        emit_pixel_box(np.clip(x0 - eps, 0, 1), np.clip(x0 + eps, 0, 1), c, t, vp)
        st, dt, _ = cli(B.ONNX, vp, timeout=R1_TIMEOUT)
        vp.unlink(missing_ok=True)
        return {"anchor_idx": i, "true_class": c, "target_class": t,
                "region": "R1_pixel_ball", "eps": eps, "status": st, "runtime_s": dt}

    # ---- Phase 1 (headline, eps=0.05): R1@0.05 on all, then R4b@1sigma on R1-SAT ----
    for a in anchors:
        rec = r1(a, 0.05); run_and_log(rec)
        print(f"  R1@0.05 idx{a['anchor_idx']}: {rec['status']} ({rec['runtime_s']:.0f}s)", flush=True)
    r1sat_05 = [a for a in anchors
                if any(r["anchor_idx"] == a["anchor_idx"] and r["region"] == "R1_pixel_ball"
                       and r["eps"] == 0.05 and r["status"] == "sat" for r in records)]
    print(f"== eps=0.05 broad-SAT anchors: {len(r1sat_05)} ==", flush=True)
    for a in r1sat_05:
        rec = r4_bounded(a, 1.0); run_and_log(rec)
        print(f"  R4b@1s idx{a['anchor_idx']}: {rec['status']} ({rec['runtime_s']:.0f}s)", flush=True)

    # ---- Phase 2 (eps=0.03): R1@0.03 on all, then R4b@1sigma on its R1-SAT ----
    for a in anchors:
        rec = r1(a, 0.03); run_and_log(rec)
        print(f"  R1@0.03 idx{a['anchor_idx']}: {rec['status']} ({rec['runtime_s']:.0f}s)", flush=True)
    r1sat_03 = [a for a in anchors
                if any(r["anchor_idx"] == a["anchor_idx"] and r["region"] == "R1_pixel_ball"
                       and r["eps"] == 0.03 and r["status"] == "sat" for r in records)]
    done_1s = {r["anchor_idx"] for r in records if r["region"] == "R4_bounded" and r["alpha"] == 1.0}
    for a in r1sat_03:
        if a["anchor_idx"] in done_1s:
            continue
        rec = r4_bounded(a, 1.0); run_and_log(rec)
        print(f"  R4b@1s(03) idx{a['anchor_idx']}: {rec['status']} ({rec['runtime_s']:.0f}s)", flush=True)

    # ---- Phase 3 (alpha matrix 2,3 sigma on all anchors) ----
    for alpha in [2.0, 3.0]:
        for a in anchors:
            rec = r4_bounded(a, alpha); run_and_log(rec)
            print(f"  R4b@{alpha}s idx{a['anchor_idx']}: {rec['status']} ({rec['runtime_s']:.0f}s)", flush=True)

    print("DONE matrix.", flush=True)


if __name__ == "__main__":
    main()
