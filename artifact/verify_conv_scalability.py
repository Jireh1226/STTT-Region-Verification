#!/usr/bin/env python3
"""Conv-architecture scalability probe of the region-tagged Marabou workflow.

Runs the P3 property (|CTE|>=10 as two sub-queries CTE>=10 and CTE<=-10) on the
trained small CNN (conv_taxinet.onnx) under:
  R1 : per-anchor pixel ball [x0-eps, x0+eps] cap [0,1]
  R4 : local-PCA 8D region x = x0 + V^T z, |z_j|<=alpha*sigma_j, cap [0,1]
on a few centered anchors, with honest timeout reporting. Measures whether the
workflow scales to a conv front end or hits a solver-time boundary. Exact status
parsing. Output: verification_conv_scalability.json.
"""
import json, time
import numpy as np
import h5py
from maraboupy import Marabou

ONNX = "conv_taxinet.onnx"
DATA = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
EPS = 0.05
ALPHA = 1.0
D = 8
N_ANCHORS = 8
TIMEOUT = 300
OPT = Marabou.createOptions(timeoutInSeconds=TIMEOUT, verbosity=0)


def status_of(code):
    s = str(code).strip().lower()
    return "sat" if s == "sat" else ("unsat" if s == "unsat" else "timeout" if "time" in s else "error")


def load():
    with h5py.File(DATA, "r") as f:
        return f["y_train"][:].reshape(-1, 128).astype(np.float64), f["X_train"][:].astype(np.float64)


def base_net():
    net = Marabou.read_onnx(ONNX)
    iv = [int(v) for v in net.inputVars[0].flatten()]
    ov = [int(v) for v in net.outputVars[0].flatten()]   # [CTE, HE]
    return net, iv, ov


def p3_query(region, x0, V=None, sig=None):
    """region in {'R1','R4'}. Returns list of (subquery, status, time)."""
    out = []
    for name, direc, bound in [("CTE_upper", ">=", 10.0), ("CTE_lower", "<=", -10.0)]:
        net, iv, ov = base_net()
        if region == "R1":
            for i in range(128):
                net.setLowerBound(iv[i], float(max(0.0, x0[i] - EPS)))
                net.setUpperBound(iv[i], float(min(1.0, x0[i] + EPS)))
        else:  # R4 local PCA
            z = [int(net.getNewVariable()) for _ in range(D)]
            for j in range(D):
                net.setLowerBound(z[j], float(-ALPHA * sig[j])); net.setUpperBound(z[j], float(ALPHA * sig[j]))
            for i in range(128):
                col = V[:, i]; nz = np.where(np.abs(col) > 1e-12)[0]
                net.addEquality([iv[i]] + [z[j] for j in nz], [1.0] + [float(-col[j]) for j in nz], float(x0[i]))
                net.setLowerBound(iv[i], 0.0); net.setUpperBound(iv[i], 1.0)
        if direc == ">=":
            net.setLowerBound(ov[0], bound)
        else:
            net.setUpperBound(ov[0], bound)
        t0 = time.time()
        code, _, _ = net.solve(verbose=False, options=OPT)
        out.append({"subquery": name, "status": status_of(code), "time_s": round(time.time() - t0, 1)})
    return out


def main():
    images, labels = load()
    cte, he = labels[:, 0], labels[:, 1]
    cen = np.where(np.abs(cte) < 1.0)[0]
    anchors = cen[np.linspace(0, len(cen) - 1, N_ANCHORS, dtype=int)]
    results = []
    for idx in anchors:
        x0 = images[int(idx)]
        m = (np.abs(cte - cte[idx]) <= 0.5) & (np.abs(he - he[idx]) <= 5.0); m[idx] = False
        nb = images[m] - x0
        if len(nb) < D + 1:
            continue
        _, S, Vt = np.linalg.svd(nb, full_matrices=False); V = Vt[:D]; sig = (nb @ V.T).std(0)
        r1 = p3_query("R1", x0)
        r4 = p3_query("R4", x0, V, sig)
        results.append({"anchor_idx": int(idx), "R1": r1, "R4": r4})
        print(f"anchor {idx}: R1 {[q['status'] for q in r1]} {[q['time_s'] for q in r1]}  "
              f"R4 {[q['status'] for q in r4]} {[q['time_s'] for q in r4]}", flush=True)
        json.dump({"provenance": {"script": "verify_conv_scalability.py", "onnx": ONNX,
                                  "eps": EPS, "alpha": ALPHA, "d": D, "timeout_s": TIMEOUT},
                   "results": results}, open("verification_conv_scalability.json", "w"), indent=2)
    # tally
    from collections import Counter
    def tally(region):
        c = Counter()
        for r in results:
            for q in r[region]:
                c[q["status"]] += 1
        return dict(c)
    print("R1 tally:", tally("R1"), "| R4 tally:", tally("R4"))


if __name__ == "__main__":
    main()
