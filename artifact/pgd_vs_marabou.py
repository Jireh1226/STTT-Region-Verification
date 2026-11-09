#!/usr/bin/env python3
"""PGD-in-subspace vs Marabou-in-subspace on the TinyTaxiNet P3 regions.

Same basis, same z-box bounds, same anchors, same alpha as the dimension sweep
(verify_p3_dimension_sweep.py). For each region we run projected-gradient-descent
(sign-PGD, multiple restarts) inside the z-box to maximize each P3 violation
objective (CTE>=10, CTE<=-10, HE>=90, HE<=-90), scoring a witness only if the
reconstruction stays in the pixel cube [0,1] (valid-pixel feasibility). We then
join against the Marabou verdicts from the sweep to report:
  both (SAT & PGD-witness), solver-only-SAT (SAT but PGD missed),
  solver-only-UNSAT (UNSAT certificate PGD cannot provide),
  attack-only (PGD-witness but Marabou UNSAT -- must be 0 for soundness),
  wall-clock.

Output: pgd_results.json  (PGD verdicts + timing; join done by pgd_vs_marabou_join).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET = ROOT / "VerifyGAN/models/TinyTaxiNet.nnet"
CTE_BOUND, HE_BOUND = 10.0, 90.0


def load_nnet(path):
    lines = path.read_text().splitlines(); i = 0
    while lines[i].startswith("//"):
        i += 1
    nl = int(lines[i].split(",")[0]); i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]; i += 1
    i += 5
    Ws, bs = [], []
    for L in range(nl):
        r, c = sizes[L + 1], sizes[L]; W = np.zeros((r, c))
        for j in range(r):
            W[j] = [float(x) for x in lines[i].strip().rstrip(",").split(",")][:c]; i += 1
        b = np.zeros(r)
        for j in range(r):
            b[j] = float(lines[i].strip().rstrip(",")); i += 1
        Ws.append(W); bs.append(b)
    return Ws, bs


def fwd(x, Ws, bs):
    h = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if k < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def grad_out(x, Ws, bs, oidx):
    """d out[oidx] / d x for the ReLU MLP (linear output layer)."""
    pre = []; h = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = h @ W.T + b; pre.append(z)
        h = np.maximum(z, 0.0) if k < len(Ws) - 1 else z
    g = np.zeros(Ws[-1].shape[0]); g[oidx] = 1.0
    for k in reversed(range(len(Ws))):
        g = g @ Ws[k]                          # grad wrt this layer's input
        if k > 0:
            g = g * (pre[k - 1] > 0)           # ReLU derivative
    return g


def random_basis(seed, idx, d, n=128):
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(idx) * 97 + d)
    cols = []
    for _ in range(d):
        v = rng.normal(size=n); v = v - v.mean()
        for q in cols:
            v = v - np.dot(v, q) * q
        cols.append(v / np.linalg.norm(v))
    return np.vstack(cols)


OBJ = [(0, +1, CTE_BOUND), (0, -1, CTE_BOUND), (1, +1, HE_BOUND), (1, -1, HE_BOUND)]


def pgd_region(x0, V, sigma, alpha, Ws, bs, restarts=30, iters=50):
    """Cheap subspace attack: multi-restart corner-jump ascent (Frank-Wolfe-style
    on the piecewise-linear net over the z-box). Returns (found, best_margins) where
    best_margins is the max attained value of each objective minus its bound.
    A witness counts only if the reconstruction stays in the pixel cube [0,1]."""
    box = alpha * sigma
    rng = np.random.default_rng(12345)
    best_margin = {}
    found = False
    for oidx, sgn, bound in OBJ:
        bm = -1e9
        for r in range(restarts):
            z = box * np.sign(rng.uniform(-1, 1, len(box))) if r else np.zeros(len(box))
            for _ in range(iters):
                x = x0 + V.T @ z
                g = V @ (grad_out(x, Ws, bs, oidx) * sgn)   # grad of sgn*out wrt z
                znew = box * np.sign(g)                       # LP optimum in current linear region
                if np.array_equal(znew, z):
                    break
                z = znew
            x = x0 + V.T @ z
            if x.min() >= -1e-6 and x.max() <= 1 + 1e-6:      # valid-pixel feasibility
                val = sgn * fwd(x, Ws, bs)[oidx]
                bm = max(bm, val - bound)
                if val >= bound - 1e-6:
                    found = True
        best_margin[f"{'+-'[sgn<0]}{'CTE' if oidx==0 else 'HE'}"] = round(float(bm), 3)
    return found, best_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="2,4,8,16,32,64")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)   # deterministic linspace subsample (e.g. 200)
    ap.add_argument("--start", type=int, default=0)    # chunk slice over centered indices
    ap.add_argument("--count", type=int, default=-1)
    ap.add_argument("--output", default="pgd_results.json")
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    dmax = max(dims)

    with h5py.File(DATA, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    Ws, bs = load_nnet(NNET)
    cte, he = labels[:, 0], labels[:, 1]
    cen = np.where(np.abs(cte) < 1.0)[0]
    if args.limit > 0:
        cen = cen[np.linspace(0, len(cen) - 1, args.limit, dtype=int)]
    if args.count >= 0:
        cen = cen[args.start:args.start + args.count]

    records = []
    t0 = time.time()
    for k, idx in enumerate(cen):
        x0 = images[int(idx)]
        mask = (np.abs(cte - cte[idx]) <= 0.5) & (np.abs(he - he[idx]) <= 5.0); mask[idx] = False
        nb = images[mask]
        if len(nb) < dmax + 1:
            continue
        Dm = nb - x0
        _, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        Vfull = Vt[:dmax]; sig = (Dm @ Vfull.T).std(0)
        for d in dims:
            ts = time.time()
            w, bm = pgd_region(x0, Vfull[:d], sig[:d], args.alpha, Ws, bs)
            records.append({"anchor_idx": int(idx), "d": d, "basis": "pca", "seed": None,
                            "pgd_witness": bool(w), "best_margins": bm,
                            "pgd_time_s": round(time.time() - ts, 4)})
            for seed in seeds:
                Vr = random_basis(seed, idx, d)
                ts = time.time()
                w, bm = pgd_region(x0, Vr, sig[:d], args.alpha, Ws, bs)
                records.append({"anchor_idx": int(idx), "d": d, "basis": "random", "seed": seed,
                                "pgd_witness": bool(w), "best_margins": bm,
                                "pgd_time_s": round(time.time() - ts, 4)})
        if (k + 1) % 50 == 0 or k + 1 == len(cen):
            Path(args.output).write_text(json.dumps(
                {"provenance": {"script": "pgd_vs_marabou.py", "dims": dims, "seeds": seeds,
                                "alpha": args.alpha, "n_anchors": len(cen)}, "results": records}, indent=2) + "\n")
            print(f"  {k+1}/{len(cen)} anchors ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE PGD {len(records)} records ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
