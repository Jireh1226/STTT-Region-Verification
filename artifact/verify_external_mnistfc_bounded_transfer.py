#!/usr/bin/env python3
"""Bounded external-transfer verification on VNN-COMP mnist_fc.

Tests whether the R4 (local-PCA) and R3 (class-AE) SAT witnesses survive valid
pixel constraints, or whether they are artifacts of the composed region leaving
the image cube [0,1]. The composed ONNX graph exposes two outputs, pixels and
logits, but the Marabou CLI maps the merged network as Y_0..9=logits and
Y_10..793=reconstructed pixels. This index map is validated separately in
external_mnistfc_output_indexing_validation.json. VNN-LIB imposes the z/latent
box on inputs, 0<=Y_i<=1 for i in 10..793, and Y_target>=Y_true on Y_0..9.

Same 20 anchors / targets / train-only calibration as the strong-transfer run
(loaded from external_mnistfc_anchor_baseline.json). Reconstructed pixels are
NOT clipped; witness validity is reported. Exact status parsing.

Output: external_mnistfc_bounded_transfer_results.json (incremental)
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
CMODELS = ROOT / "external_mnistfc_bounded_composed_models"
CMODELS.mkdir(exist_ok=True)
ALPHAS = [1.0, 2.0, 3.0]
D = 8
KNB = 100
TIMEOUT = 120

BASE = onnx.load(str(ONNX))
IN_NAME = BASE.graph.input[0].name
OUT_NAME = BASE.graph.output[0].name
OPSET = BASE.opset_import[0].version
IRV = BASE.ir_version


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    return (m["x_train"].reshape(-1, 784) / 255.0, m["y_train"],
            m["x_test"].reshape(-1, 784) / 255.0, m["y_test"])


def _two_output(dec_nodes, dec_inits, dec_in, tag, kind):
    """Decoder graph exposes [xdec(1,784,1) for io_map, xflat(1,784)]; merge with
    mnist_fc so the combined model outputs [xflat(pixels), logits]."""
    xdec = helper.make_tensor_value_info("xdec", TensorProto.FLOAT, [1, 784, 1])
    xflat = helper.make_tensor_value_info("xflat", TensorProto.FLOAT, [1, 784])
    dec = helper.make_model(
        helper.make_graph(dec_nodes, "dec", [dec_in], [xdec, xflat], dec_inits),
        opset_imports=[helper.make_opsetid("", OPSET)])
    dec.ir_version = IRV
    merged = compose.merge_models(dec, BASE, io_map=[("xdec", IN_NAME)])
    p = CMODELS / f"{kind}_{tag}.onnx"
    onnx.save(merged, str(p))
    return p


def compose_pca_two(V, x0, tag):
    inits = [numpy_helper.from_array(V.astype(np.float32), "Wpca"),
             numpy_helper.from_array(x0.astype(np.float32), "x0c"),
             numpy_helper.from_array(np.array([1, 784, 1], np.int64), "shp")]
    zin = helper.make_tensor_value_info("zin", TensorProto.FLOAT, [1, V.shape[0]])
    nodes = [helper.make_node("MatMul", ["zin", "Wpca"], ["mm"]),
             helper.make_node("Add", ["mm", "x0c"], ["xflat"]),
             helper.make_node("Reshape", ["xflat", "shp"], ["xdec"])]
    return _two_output(nodes, inits, zin, tag, "r4b")


def compose_dec_two(d1W, d1b, d2W, d2b, tag):
    inits = [numpy_helper.from_array(d1W.T.astype(np.float32), "D1"),
             numpy_helper.from_array(d1b.astype(np.float32), "b1"),
             numpy_helper.from_array(d2W.T.astype(np.float32), "D2"),
             numpy_helper.from_array(d2b.astype(np.float32), "b2"),
             numpy_helper.from_array(np.array([1, 784, 1], np.int64), "shp")]
    zin = helper.make_tensor_value_info("zin", TensorProto.FLOAT, [1, 8])
    nodes = [helper.make_node("MatMul", ["zin", "D1"], ["m1"]),
             helper.make_node("Add", ["m1", "b1"], ["a1"]),
             helper.make_node("Relu", ["a1"], ["h"]),
             helper.make_node("MatMul", ["h", "D2"], ["m2"]),
             helper.make_node("Add", ["m2", "b2"], ["xflat"]),
             helper.make_node("Reshape", ["xflat", "shp"], ["xdec"])]
    return _two_output(nodes, inits, zin, tag, "r3b")


def emit_bounded(lo, hi, c, t, path):
    """z/latent box on X_0..X_{d-1}. Marabou maps the composed two-output net as
    Y_0..9 = logits, Y_10..793 = reconstructed pixels (verified empirically; the
    graph output order from onnxruntime is the reverse, but Marabou orders the
    mnist_fc logits first). So: pixels Y_10..793 in [0,1]; confusion Y_t>=Y_c on
    logits Y_0..9."""
    d = len(lo)
    L = [f"(declare-const X_{i} Real)" for i in range(d)]
    L += [f"(declare-const Y_{k} Real)" for k in range(794)]
    for i in range(d):
        L.append(f"(assert (>= X_{i} {lo[i]:.6f}))")
        L.append(f"(assert (<= X_{i} {hi[i]:.6f}))")
    for k in range(10, 794):
        L.append(f"(assert (>= Y_{k} 0.0))")
        L.append(f"(assert (<= Y_{k} 1.0))")
    L.append(f"(assert (>= Y_{t} Y_{c}))")
    Path(path).write_text("\n".join(L) + "\n")


def solve(model, vnnlib):
    t0 = time.time()
    try:
        p = subprocess.run([str(MARABOU), str(model), str(vnnlib),
                            f"--timeout={TIMEOUT}", "--snc", "--num-workers=4",
                            "--verbosity=0"],
                           capture_output=True, text=True, timeout=TIMEOUT + 40)
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


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    ae = np.load(AE)
    anchors = json.loads((ROOT / "external_mnistfc_anchor_baseline.json").read_text())["anchors"]

    prov = {"script": "verify_external_mnistfc_bounded_transfer.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "external_network": "VNN-COMP mnist_fc mnist-net_256x2 (not trained by us)",
            "solver": "Marabou 2.0 CLI on two-output composed ONNX, SnC 4 workers, 120s",
            "output_map": "Y_0..9 = logits, Y_10..793 = pixels",
            "pixel_bounds": "0 <= Y_i <= 1 for i in 10..793",
            "anchors": len(anchors)}

    records = []
    for ai, a in enumerate(anchors):
        i, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i]
        # R4 PCA basis (same calibration as strong run)
        Xc = Xtr[ytr == c]
        nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:KNB]]
        Dd = nb - x0[None, :]
        _, _, Vt = np.linalg.svd(Dd, full_matrices=False)
        V = Vt[:D]; sig = (Dd @ V.T).std(0)
        m4 = compose_pca_two(V, x0, f"a{ai}")
        for alpha in ALPHAS:
            vp = ROOT / "_r4b.vnnlib"
            emit_bounded(-alpha * sig, alpha * sig, c, t, vp)
            st, dt, wit = solve(m4, vp)
            rec = {"anchor_idx": i, "true_class": c, "target_class": t,
                   "region": "R4_local_pca_bounded", "alpha": alpha, "status": st,
                   "runtime_s": dt}
            if st == "sat" and wit and len(wit) >= D:
                z = np.array([wit[j] for j in range(D)])
                xr = x0 + V.T @ z   # raw, not clipped
                rec["witness_z"] = z.tolist()
                rec["recon_min"] = float(xr.min()); rec["recon_max"] = float(xr.max())
                rec["frac_in_unit"] = float(((xr >= -1e-6) & (xr <= 1 + 1e-6)).mean())
            records.append(rec)
            print(f"  a{ai} idx{i} c{c}->t{t} R4b a{alpha}: {st} ({dt:.0f}s)", flush=True)
            vp.unlink(missing_ok=True)
        # R3 bounded
        d1W, d1b = ae[f"dec1_W_class{c}"], ae[f"dec1_b_class{c}"]
        d2W, d2b = ae[f"dec2_W_class{c}"], ae[f"dec2_b_class{c}"]
        llo, lhi = ae[f"latent_lo_class{c}"], ae[f"latent_hi_class{c}"]
        m3 = compose_dec_two(d1W, d1b, d2W, d2b, f"c{c}")
        vp = ROOT / "_r3b.vnnlib"
        emit_bounded(llo, lhi, c, t, vp)
        st, dt, wit = solve(m3, vp)
        rec = {"anchor_idx": i, "true_class": c, "target_class": t,
               "region": "R3_class_ae_bounded", "status": st, "runtime_s": dt}
        if st == "sat" and wit and len(wit) >= 8:
            z = np.array([wit[j] for j in range(8)])
            h = np.maximum(d1W @ z + d1b, 0.0)
            xr = d2W @ h + d2b   # raw decoded, not clipped
            rec["witness_z"] = z.tolist()
            rec["recon_min"] = float(xr.min()); rec["recon_max"] = float(xr.max())
            rec["frac_in_unit"] = float(((xr >= -1e-6) & (xr <= 1 + 1e-6)).mean())
        records.append(rec)
        print(f"  a{ai} idx{i} c{c}->t{t} R3b: {st} ({dt:.0f}s)", flush=True)
        vp.unlink(missing_ok=True)
        (ROOT / "external_mnistfc_bounded_transfer_results.json").write_text(
            json.dumps({"provenance": prov, "results": records}, indent=2) + "\n")

    print("\nDONE bounded run.")


if __name__ == "__main__":
    main()
