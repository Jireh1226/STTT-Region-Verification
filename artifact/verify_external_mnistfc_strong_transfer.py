#!/usr/bin/env python3
"""Strong external-transfer verification on VNN-COMP mnist_fc (mnist-net_256x2).

Runs the region-tagged workflow on an external image classifier we did not
train, including genuinely data-conditioned NON-box regions encoded as
composed ONNX networks verified with the Marabou CLI (maraboupy hangs on this
filesystem, so composition + CLI is used instead):

  R1  : pixel-space L_inf ball (eps), VNN-LIB box over the 784 pixels.
  R4  : local-PCA region. Composed net z -> x0 + V_d^T z -> mnist_fc; verify
        the z-box |z_j| <= alpha*sigma_j (alpha in {1,2,3}).
  R3  : class-AE decoder-box. Composed net z -> D_k(z) -> mnist_fc; verify the
        per-class train latent box [latent_lo, latent_hi].

Property: targeted class confusion logit_target >= logit_true (SAT = witness).
Exact status parsing only. Results saved incrementally.

Outputs: external_mnistfc_r1_r2_results.json, external_mnistfc_r4_results.json,
         external_mnistfc_r3_results.json, external_mnistfc_anchor_baseline.json,
         composed models under external_mnistfc_r4_composed_models/.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import compose, helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parent
ONNX = ROOT / "external/vnncomp/mnist_fc/onnx/mnist-net_256x2.onnx"
MARABOU = ROOT / "prophecy_env/bin/Marabou"
AE = ROOT / "mnist_class_aes.npz"
CMODELS = ROOT / "external_mnistfc_r4_composed_models"
CMODELS.mkdir(exist_ok=True)
PER_CLASS = 2
EPS_LIST = [0.05, 0.03]
ALPHAS = [1.0, 2.0, 3.0]
D = 8
KNB = 100
TIMEOUT = 120


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    return (m["x_train"].reshape(-1, 784) / 255.0, m["y_train"],
            m["x_test"].reshape(-1, 784) / 255.0, m["y_test"])


def session(path):
    s = ort.InferenceSession(str(path))
    return s, s.get_inputs()[0].name


def fwd(s, n, x, shape=(1, 784, 1)):
    return np.asarray(s.run(None, {n: x.reshape(shape).astype(np.float32)})[0]).flatten()


BASE = onnx.load(str(ONNX))
IN_NAME = BASE.graph.input[0].name
OPSET = BASE.opset_import[0].version
IRV = BASE.ir_version


def _reshape_node(inp, out):
    return helper.make_node("Reshape", [inp, "shp784"], [out])


def compose_affine(V, x0, tag):
    """z[1,d] -> x = z@V + x0 -> [1,784,1] -> mnist_fc. V:(d,784), x0:(784,)."""
    d = V.shape[0]
    inits = [numpy_helper.from_array(V.astype(np.float32), "Wpca"),
             numpy_helper.from_array(x0.astype(np.float32), "x0c"),
             numpy_helper.from_array(np.array([1, 784, 1], np.int64), "shp784")]
    zin = helper.make_tensor_value_info("zin", TensorProto.FLOAT, [1, d])
    xout = helper.make_tensor_value_info("xdec", TensorProto.FLOAT, [1, 784, 1])
    nodes = [helper.make_node("MatMul", ["zin", "Wpca"], ["mm"]),
             helper.make_node("Add", ["mm", "x0c"], ["xflat"]),
             _reshape_node("xflat", "xdec")]
    dec = helper.make_model(helper.make_graph(nodes, "dec", [zin], [xout], inits),
                            opset_imports=[helper.make_opsetid("", OPSET)])
    dec.ir_version = IRV
    comp = compose.merge_models(dec, BASE, io_map=[("xdec", IN_NAME)])
    p = CMODELS / f"r4_{tag}.onnx"
    onnx.save(comp, str(p))
    return p


def compose_decoder(dec1_W, dec1_b, dec2_W, dec2_b, tag):
    """z[1,8] -> ReLU(z@dec1_W.T + dec1_b) -> h@dec2_W.T + dec2_b -> [1,784,1] -> mnist_fc.
    dec1_W:(32,8) dec2_W:(784,32)."""
    inits = [numpy_helper.from_array(dec1_W.T.astype(np.float32), "D1"),
             numpy_helper.from_array(dec1_b.astype(np.float32), "b1"),
             numpy_helper.from_array(dec2_W.T.astype(np.float32), "D2"),
             numpy_helper.from_array(dec2_b.astype(np.float32), "b2"),
             numpy_helper.from_array(np.array([1, 784, 1], np.int64), "shp784")]
    zin = helper.make_tensor_value_info("zin", TensorProto.FLOAT, [1, 8])
    xout = helper.make_tensor_value_info("xdec", TensorProto.FLOAT, [1, 784, 1])
    nodes = [helper.make_node("MatMul", ["zin", "D1"], ["m1"]),
             helper.make_node("Add", ["m1", "b1"], ["a1"]),
             helper.make_node("Relu", ["a1"], ["h"]),
             helper.make_node("MatMul", ["h", "D2"], ["m2"]),
             helper.make_node("Add", ["m2", "b2"], ["xflat"]),
             _reshape_node("xflat", "xdec")]
    dec = helper.make_model(helper.make_graph(nodes, "dec", [zin], [xout], inits),
                            opset_imports=[helper.make_opsetid("", OPSET)])
    dec.ir_version = IRV
    comp = compose.merge_models(dec, BASE, io_map=[("xdec", IN_NAME)])
    p = CMODELS / f"r3_{tag}.onnx"
    onnx.save(comp, str(p))
    return p


def emit_box(lo, hi, c, t, path, ndim):
    L = [f"(declare-const X_{i} Real)" for i in range(ndim)]
    L += [f"(declare-const Y_{k} Real)" for k in range(10)]
    for i in range(ndim):
        L.append(f"(assert (>= X_{i} {lo[i]:.6f}))")
        L.append(f"(assert (<= X_{i} {hi[i]:.6f}))")
    L.append(f"(assert (>= Y_{t} Y_{c}))")
    Path(path).write_text("\n".join(L) + "\n")


def solve(model, vnnlib):
    t0 = time.time()
    try:
        p = subprocess.run([str(MARABOU), str(model), str(vnnlib),
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
        for ln in out.splitlines():
            s = ln.strip()
            if s.startswith("x") and "=" in s and "_" not in s:
                try:
                    k, v = s.split("="); wit[int(k.strip()[1:])] = float(v)
                except ValueError:
                    pass
        return "sat", dt, wit
    if "timeout" in out.lower():
        return "timeout", dt, None
    return "error", dt, None


def save(obj, name):
    (ROOT / name).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    s0, n0 = session(ONNX)
    aedat = np.load(AE)

    anchors = []
    for c in range(10):
        k = 0
        for i in np.where(yte == c)[0]:
            if int(np.argmax(fwd(s0, n0, Xte[i]))) == c:
                anchors.append((int(i), c)); k += 1
                if k >= PER_CLASS:
                    break

    prov = {"script": "verify_external_mnistfc_strong_transfer.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "external_network": "VNN-COMP mnist_fc mnist-net_256x2 (not trained by us)",
            "solver": "Marabou 2.0 CLI on base/composed ONNX, SnC 4 workers, 120s",
            "anchors": len(anchors)}

    # baselines
    base = []
    for i, c in anchors:
        lg = fwd(s0, n0, Xte[i]); t = int(np.argsort(lg)[-2])
        Xtc, Xec = Xtr[ytr == c], Xte[yte == c]
        base.append({"anchor_idx": i, "true_class": c, "target_class": t,
                     "correct": bool(int(np.argmax(lg)) == c),
                     "margin": float(np.sort(lg)[-1] - np.sort(lg)[-2]),
                     "nearest_train_sameclass_L2": float(np.min(np.linalg.norm(Xtc - Xte[i][None, :], axis=1))),
                     "nearest_test_sameclass_L2": float(np.min(np.linalg.norm(Xec[Xec.sum(1) != Xte[i].sum()] - Xte[i][None, :], axis=1)) if len(Xec) > 1 else 0.0)})
    save({"provenance": prov, "anchors": base}, "external_mnistfc_anchor_baseline.json")
    tgt = {i: b["target_class"] for (i, c), b in zip(anchors, base)}

    r1, r4, r3 = [], [], []
    for ai, (i, c) in enumerate(anchors):
        x0 = Xte[i]; t = tgt[i]
        # ---- R1 pixel ball ----
        for eps in EPS_LIST:
            vp = ROOT / "_r1.vnnlib"
            emit_box(np.clip(x0 - eps, 0, 1), np.clip(x0 + eps, 0, 1), c, t, vp, 784)
            st, dt, wit = solve(ONNX, vp)
            rec = {"anchor_idx": i, "true_class": c, "target_class": t,
                   "region": "R1_linf_ball", "eps": eps, "status": st, "runtime_s": dt}
            if st == "sat" and wit and len(wit) >= 784:
                rec["witness_pixels"] = [wit[j] for j in range(784)]
            r1.append(rec)
            print(f"  a{ai} idx{i} c{c}->t{t} R1 eps{eps}: {st} ({dt:.0f}s)", flush=True)
            vp.unlink(missing_ok=True)
        # ---- R4 local PCA composed ----
        Xc = Xtr[ytr == c]
        nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:KNB]]
        Dd = nb - x0[None, :]
        _, S, Vt = np.linalg.svd(Dd, full_matrices=False)
        V = Vt[:D]; sig = (Dd @ V.T).std(0)
        var_expl = float((S[:D] ** 2).sum() / (S ** 2).sum())
        model = compose_affine(V, x0, f"a{ai}")
        for alpha in ALPHAS:
            vp = ROOT / "_r4.vnnlib"
            zb = alpha * sig
            emit_box(-zb, zb, c, t, vp, D)
            st, dt, wit = solve(model, vp)
            rec = {"anchor_idx": i, "true_class": c, "target_class": t,
                   "region": "R4_local_pca", "alpha": alpha, "d": D,
                   "var_explained": round(var_expl, 3), "status": st, "runtime_s": dt}
            if st == "sat" and wit and len(wit) >= D:
                z = np.array([wit[j] for j in range(D)])
                rec["witness_z"] = z.tolist()
                rec["witness_pixels"] = (x0 + V.T @ z).tolist()
            r4.append(rec)
            print(f"  a{ai} idx{i} c{c}->t{t} R4 a{alpha}: {st} ({dt:.0f}s)", flush=True)
            vp.unlink(missing_ok=True)
        # ---- R3 class-AE composed ----
        d1W, d1b = aedat[f"dec1_W_class{c}"], aedat[f"dec1_b_class{c}"]
        d2W, d2b = aedat[f"dec2_W_class{c}"], aedat[f"dec2_b_class{c}"]
        llo, lhi = aedat[f"latent_lo_class{c}"], aedat[f"latent_hi_class{c}"]
        dmodel = compose_decoder(d1W, d1b, d2W, d2b, f"c{c}")
        vp = ROOT / "_r3.vnnlib"
        emit_box(llo, lhi, c, t, vp, 8)
        st, dt, wit = solve(dmodel, vp)
        rec = {"anchor_idx": i, "true_class": c, "target_class": t,
               "region": "R3_class_ae", "status": st, "runtime_s": dt}
        if st == "sat" and wit and len(wit) >= 8:
            rec["witness_z"] = [wit[j] for j in range(8)]
        r3.append(rec)
        print(f"  a{ai} idx{i} c{c}->t{t} R3: {st} ({dt:.0f}s)", flush=True)
        vp.unlink(missing_ok=True)
        # incremental save
        save({"provenance": prov, "results": r1}, "external_mnistfc_r1_r2_results.json")
        save({"provenance": prov, "results": r4}, "external_mnistfc_r4_results.json")
        save({"provenance": prov, "results": r3}, "external_mnistfc_r3_results.json")

    print("\nDONE. Wrote R1/R4/R3 result JSONs.")


if __name__ == "__main__":
    main()
