#!/usr/bin/env python3
"""Part B: region-tagged verification of the controlled models.

For each model we test the P4-analog sign property on the left-offset region
(true CTE > BUG_CTE), where the correct prediction sign is positive (y0 > 0).
The wrong-sign witness query asserts y0 <= 0; SAT means a wrong-sign input
exists in the region. We run it under:

  R1 : broad per-pixel box over rendered left-offset images (loose -- admits
       non-renderable pixel combinations such as the checkerboard corner).
  R5 : per-pixel box over same-state neighbours of a left-offset anchor
       (tight, data-conditioned).

Verification uses the bundled Marabou 2.0 CLI on the exported .nnet models with
generated VNN-LIB box queries (the same solver path validated by the external
ACAS Xu calibration). SAT witnesses are parsed and saved for the audit step.
Pure numpy + subprocess; no maraboupy import.

Output: controlled_verify_raw.json
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import controlled_renderer as R
from train_controlled_models import BUG_CTE

ROOT = Path(__file__).resolve().parent
MARABOU = ROOT / "prophecy_env/bin/Marabou"
MODELS = ["M_good", "M_bug", "M_artifact"]
TIMEOUT = 60
N_PIX = 128
# Require a wrong-sign MARGIN, not just y0<=0: boundary witnesses (y0=0) are
# solver-arbitrary (cf. the TinyTaxiNet P4 threshold sweep). A SAT here is a
# genuine strong wrong-sign prediction of at least this magnitude.
WRONG_SIGN_MARGIN = -1.0
WIT_RE = re.compile(r"x(\d+)\s*=\s*([-\d.eE+]+)")


def load_nnet(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    nlayers = int(lines[i].split(",")[0]); i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]; i += 1
    i += 5
    Ws, bs = [], []
    for L in range(nlayers):
        rows, cols = sizes[L + 1], sizes[L]
        W = np.array([[float(v) for v in lines[i + r].strip().rstrip(",").split(",")][:cols]
                      for r in range(rows)]); i += rows
        b = np.array([float(lines[i + r].strip().rstrip(",")) for r in range(rows)]); i += rows
        Ws.append(W); bs.append(b)
    return Ws, bs


def forward(Ws, bs, x):
    a = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W.T + b
        if k < len(Ws) - 1:
            a = np.maximum(a, 0.0)
    return a


def write_box_vnnlib(pmin, pmax, path: Path, n_in=N_PIX):
    out = [f"(declare-const X_{i} Real)" for i in range(n_in)]
    out.append("(declare-const Y_0 Real)")
    for i in range(n_in):
        out.append(f"(assert (>= X_{i} {pmin[i]:.6f}))")
        out.append(f"(assert (<= X_{i} {pmax[i]:.6f}))")
    out.append(f"(assert (<= Y_0 {WRONG_SIGN_MARGIN:.4f}))")   # strong wrong-sign
    path.write_text("\n".join(out) + "\n")


def write_nnet(path: Path, Ws, bs):
    sizes = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
    n_in = sizes[0]
    lines = ["// composed controlled net (local-PCA decoder folded in)"]
    lines.append(f"{len(Ws)},{n_in},{sizes[-1]},{max(sizes)},")
    lines.append(",".join(str(s) for s in sizes) + ",")
    lines.append("0,")
    lines.append(",".join(["0.0"] * n_in) + ",")
    lines.append(",".join(["1.0"] * n_in) + ",")
    lines.append(",".join(["0.0"] * (n_in + 1)) + ",")
    lines.append(",".join(["1.0"] * (n_in + 1)) + ",")
    for W, b in zip(Ws, bs):
        for r in range(W.shape[0]):
            lines.append(",".join(f"{v:.8e}" for v in W[r]) + ",")
        for r in range(W.shape[0]):
            lines.append(f"{b[r]:.8e},")
    path.write_text("\n".join(lines) + "\n")


def compose_local_pca(Ws, bs, x0, V, sigmas, alpha):
    """Fold the affine local-PCA decoder x = x0 + V^T z into the model's first
    ReLU layer, giving a net whose input is the d-dim latent z. The witness in
    z maps back to x = x0 + V^T z, which lies on the local affine manifold by
    construction (so it is renderable up to the PCA residual)."""
    W1, b1 = Ws[0], bs[0]                 # (16,128),(16,)
    W1p = W1 @ V.T                        # (16,d)
    b1p = W1 @ x0 + b1                    # (16,)
    cWs = [W1p] + [w.copy() for w in Ws[1:]]
    cbs = [b1p] + [b.copy() for b in bs[1:]]
    zb = alpha * sigmas                   # |z_j| <= alpha*sigma_j
    return cWs, cbs, -zb, zb


def run_cli(nnet: Path, vnnlib: Path, timeout: int, n_in=N_PIX):
    try:
        proc = subprocess.run(
            [str(MARABOU), str(nnet), str(vnnlib),
             f"--timeout={timeout}", "--verbosity=0"],
            capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return "timeout", None
    out = proc.stdout + "\n" + proc.stderr
    low = out.lower()
    if "unsat" in low:
        return "unsat", None
    if "sat" in low:
        wit = {int(m.group(1)): float(m.group(2)) for m in WIT_RE.finditer(out)}
        if len(wit) >= n_in:
            return "sat", np.array([wit[i] for i in range(n_in)])
        return "sat", None
    return "error", None


def local_pca(x0, Xnb, d=8):
    D = Xnb - x0[None, :]
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    V = Vt[:d]
    sigmas = (D @ V.T).std(axis=0)
    return V, sigmas


def build_regions(rng):
    Xleft, stleft = R.sample_states(500, rng, cte_range=(BUG_CTE, 6.0))
    a_idx = int(np.argmin(np.abs(stleft["cte"] - 4.0)))
    x0 = Xleft[a_idx]
    nb_rng = np.random.default_rng(123)
    Xnb, _ = R.sample_states(120, nb_rng,
                             cte_range=(stleft["cte"][a_idx] - 0.5,
                                        stleft["cte"][a_idx] + 0.5))
    V, sigmas = local_pca(x0, Xnb, d=8)
    box_regions = {
        "R1_broad_box": (np.maximum(0.0, Xleft.min(0)), np.minimum(1.0, Xleft.max(0))),
        "R5_neighbor_box": (np.maximum(0.0, Xnb.min(0)), np.minimum(1.0, Xnb.max(0))),
    }
    pca = {"x0": x0, "V": V, "sigmas": sigmas, "alpha": 1.0}
    return box_regions, pca


def main():
    rng = np.random.default_rng(2024)
    box_regions, pca = build_regions(rng)
    vfiles = {}
    for rname, (pmin, pmax) in box_regions.items():
        p = ROOT / f"controlled_{rname}.vnnlib"
        write_box_vnnlib(pmin, pmax, p)
        vfiles[rname] = p

    nets = {m: load_nnet(ROOT / f"controlled_{m}.nnet") for m in MODELS}

    out = {
        "provenance": {
            "script": "verify_controlled_oracle.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "solver": "Marabou 2.0 CLI on exported .nnet with generated VNN-LIB",
            "property": "P4-analog: correct sign y0>0 on left-offset region "
                        f"(true CTE>{BUG_CTE}); witness query asserts "
                        f"y0<={WRONG_SIGN_MARGIN} (strong wrong-sign)",
            "bug_cte": BUG_CTE,
            "wrong_sign_margin": WRONG_SIGN_MARGIN,
            "timeout_s": TIMEOUT,
        },
        "verdicts": [],
    }
    out["provenance"]["local_pca_alpha"] = pca["alpha"]
    for m in MODELS:
        Ws, bs = nets[m]
        # Box regions R1, R5 (per-pixel boxes; admit off-manifold combinations).
        for rname, vf in vfiles.items():
            status, x = run_cli(ROOT / f"controlled_{m}.nnet", vf, TIMEOUT)
            rec = {"model": m, "region": rname, "status": status}
            if status == "sat" and x is not None:
                rec["witness"] = [float(v) for v in x]
                rec["pred_cte"] = float(forward(Ws, bs, x)[0])
            out["verdicts"].append(rec)
            extra = f" pred_cte={rec['pred_cte']:+.2f}" if "pred_cte" in rec else ""
            print(f"  {m:11s} {rname:16s} -> {status:7s}{extra}", flush=True)

        # R4 local-PCA (manifold-constrained): verify over the latent z; the
        # witness maps back to x = x0 + V^T z, on the local affine manifold.
        cWs, cbs, zlo, zhi = compose_local_pca(Ws, bs, pca["x0"], pca["V"],
                                               pca["sigmas"], pca["alpha"])
        comp_nnet = ROOT / f"controlled_{m}_R4composed.nnet"
        write_nnet(comp_nnet, cWs, cbs)
        comp_vnnlib = ROOT / f"controlled_{m}_R4.vnnlib"
        write_box_vnnlib(zlo, zhi, comp_vnnlib, n_in=len(zlo))
        status, z = run_cli(comp_nnet, comp_vnnlib, TIMEOUT, n_in=len(zlo))
        rec = {"model": m, "region": "R4_local_pca", "status": status}
        if status == "sat" and z is not None:
            xw = np.clip(pca["x0"] + pca["V"].T @ z, 0.0, 1.0)
            rec["witness"] = [float(v) for v in xw]
            rec["latent"] = [float(v) for v in z]
            rec["pred_cte"] = float(forward(Ws, bs, xw)[0])
        out["verdicts"].append(rec)
        extra = f" pred_cte={rec['pred_cte']:+.2f}" if "pred_cte" in rec else ""
        print(f"  {m:11s} {'R4_local_pca':16s} -> {status:7s}{extra}", flush=True)

    (ROOT / "controlled_verify_raw.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Wrote controlled_verify_raw.json")


if __name__ == "__main__":
    main()
