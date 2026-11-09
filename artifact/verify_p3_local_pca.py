"""
P3 verification under per-image local-PCA regions.

Construction:
  For each sampled centered image x0:
    1. Find same-state neighbors:  |dCTE| <= 0.5, |dHE| <= 5.
    2. Compute D = neighbors - x0  (deviation matrix).
    3. SVD: D = U S V^T. Keep top d=8 directions (rows of V_top).
    4. Per-direction projection std sigma_j = std of D @ V_top[j].
    5. Region: x = x0 + V_top^T @ z  with  |z_j| <= ALPHA * sigma_j,
       intersected with  0 <= x_i <= 1  and  sum(x_i) = 64.

We encode this in Marabou by treating x as the input and adding
128 - d orthogonal-complement equalities  V_null @ x = V_null @ x0
(forces x to lie in the d-dim affine subspace through x0), plus the
per-pixel envelope box implied by the |z_j| <= ALPHA*sigma_j bounds,
plus the mean=0.5 equality.

Verification: for each of 20 images, run the four P3 safety sub-queries
(|CTE| < 10, |HE| < 90).  An image is verified if all four are UNSAT.

We sweep ALPHA in {1, 2, 3} so the report can quote a reachable range
of the local-PCA region (1-sigma, 2-sigma, 3-sigma along each of the 8
PCA directions) and show how the verification rate scales.

Sample size and alpha list are configurable (--num-images, --alphas) so the
same diagnostic can be run on the full set of 893 centered images instead of
the 20-anchor diagnostic sample. The property, thresholds, network,
preprocessing mean invariant, clipping, neighbor rule, PCA dimension, and
solver semantics (300 s per sub-query) are unchanged; only the anchor count,
alpha list, and reporting granularity differ. The selection rule is identical
across sizes: centered = ascending dataset indices with |CTE| < 1.0, and the
sample is centered[ round(linspace(0, n_centered-1, num_images)) ]; num_images
"all" uses every centered image in order (linspace with N = n_centered is the
identity, so the 893-image run is the limiting case of the same rule). Running
with no arguments reproduces the original 20-anchor run and output file.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_local_pca.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
NUM_IMAGES = 20

D = 8                    # number of PCA directions
CTE_BIN_HALF = 0.5       # m
HE_BIN_HALF = 5.0        # deg
ALPHA_VALUES = [1.0, 2.0, 3.0]
TIMEOUT_S = 300


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64), \
               f["X_train"][:].astype(np.float64)


def centered_indices(labels):
    """Ascending dataset indices of centered images (|CTE| < threshold)."""
    cte = labels[:, 0]
    return np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]


def select_centered_sample(labels, num_images=NUM_IMAGES):
    """Deterministic anchor selection, identical rule at every size.

    num_images == "all" selects every centered image in ascending order;
    np.linspace(0, n-1, n) is the identity, so the full run is the limiting
    case of the same rule used for the 20-anchor sample.
    """
    centered_idx = centered_indices(labels)
    if num_images == "all" or int(num_images) >= len(centered_idx):
        num_images = len(centered_idx)
    return centered_idx[np.linspace(0, len(centered_idx) - 1, int(num_images), dtype=int)]


def build_local_pca(images, labels, idx, d=D):
    """Return (x0, V_top, sigma_proj, V_null, n_neighbors, expl_var)."""
    cte = labels[:, 0]
    he = labels[:, 1]
    x0 = images[int(idx)]
    cte_q, he_q = cte[int(idx)], he[int(idx)]
    mask = (np.abs(cte - cte_q) <= CTE_BIN_HALF) & \
           (np.abs(he - he_q) <= HE_BIN_HALF)
    mask[int(idx)] = False
    neighbors = images[mask]
    m = len(neighbors)
    if m < d + 1:
        return None
    D_mat = neighbors - x0
    # SVD: D_mat (m, 128) = U (m, k) diag(S) V^T (k, 128), k = min(m, 128)
    _, S, Vt = np.linalg.svd(D_mat, full_matrices=False)
    V_top = Vt[:d]                 # (d, 128)  unit rows in pixel space
    V_null = Vt[d:]                # (k - d, 128) orthogonal complement (within row-space of D_mat)

    proj = D_mat @ V_top.T         # (m, d)
    sigma_proj = proj.std(axis=0)  # (d,)

    expl_var = float((S[:d] ** 2).sum() / (S ** 2).sum())
    return {
        "x0": x0,
        "V_top": V_top,
        "sigma_proj": sigma_proj,
        "V_null": V_null,
        "n_neighbors": int(m),
        "explained_variance_ratio_top_d": expl_var,
        "singular_values": S.tolist(),
    }


def feasibility(x0, V_top, alphas):
    """Per-pixel envelope of x = x0 + V_top^T z, |z_j| <= alpha_j."""
    half_widths = np.sum(np.abs(V_top.T) * alphas[None, :], axis=1)
    pixel_lo = np.maximum(PIXEL_MIN, x0 - half_widths)
    pixel_hi = np.minimum(PIXEL_MAX, x0 + half_widths)
    return pixel_lo, pixel_hi


def solve_one(x0, V_top, V_null, alphas, out_var, direc, bound):
    pixel_lo, pixel_hi = feasibility(x0, V_top, alphas)
    if np.any(pixel_lo > pixel_hi):
        return {"status": "region_empty_after_clip"}

    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]

    # Per-pixel envelope (axis-aligned bounding box of the d-dim region after clipping)
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_lo[i]))
        net.setUpperBound(input_vars[i], float(pixel_hi[i]))

    # Auxiliary latent variables z_1, ..., z_d  with bounds  |z_j| <= alpha_j.
    # Note: V_top has shape (d, 128); rows are unit-norm pixel-space directions.
    # We define   x_i = x0_i + sum_j V_top[j, i] * z_j   via 128 sparse equalities.
    d = V_top.shape[0]
    z_vars = [int(net.getNewVariable()) for _ in range(d)]
    for j in range(d):
        net.setLowerBound(z_vars[j], float(-alphas[j]))
        net.setUpperBound(z_vars[j], float(alphas[j]))

    # Sparse equality per pixel:  x_i  -  sum_j V_top[j, i] z_j  =  x0_i
    for i in range(NUM_PIXELS):
        col = V_top[:, i]
        nz = np.where(np.abs(col) > 1e-12)[0]
        vars_eq = [input_vars[i]] + [z_vars[j] for j in nz]
        coeffs_eq = [1.0] + [float(-col[j]) for j in nz]
        net.addEquality(vars_eq, coeffs_eq, float(x0[i]))

    # Mean constraint  sum(x) = 64
    net.addEquality(list(input_vars), [1.0] * NUM_PIXELS, SUM_TARGET)

    # Negated postcondition
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()

    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        cx = np.array([vals[v] for v in input_vars])
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE": round(float(vals[193]), 6),
            "input_mean": round(float(cx.mean()), 6),
            "input_min": round(float(cx.min()), 6),
            "input_max": round(float(cx.max()), 6),
            "linf_from_x0": round(float(np.max(np.abs(cx - x0))), 6),
        }
    return out


def _dist(times):
    """Runtime distribution summary over a list of per-anchor solve seconds."""
    if not times:
        return {"n": 0}
    s = sorted(times)
    return {
        "n": len(s),
        "min": round(s[0], 4),
        "median": round(statistics.median(s), 4),
        "mean": round(statistics.fmean(s), 4),
        "max": round(s[-1], 4),
        "total": round(sum(s), 4),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3 verification under per-image local-PCA regions (configurable size/alpha).")
    ap.add_argument("--num-images", default=str(NUM_IMAGES),
                    help='number of deterministic centered anchors, or "all" for every centered image (default: %(default)s)')
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHA_VALUES),
                    help="comma-separated alpha (sigma) multipliers (default: %(default)s)")
    ap.add_argument("--output", default=OUTPUT_JSON,
                    help="output JSON path (default: %(default)s)")
    ap.add_argument("--manifest", default=None,
                    help="if set and the sample is a strict subset, write the selected dataset indices here")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alphas_list = [float(a) for a in args.alphas.split(",") if a.strip()]
    num_images_arg = args.num_images.strip()

    images, labels = load_data()
    centered_all = centered_indices(labels)
    n_centered = int(len(centered_all))
    sample = select_centered_sample(labels, num_images_arg)
    sample_size = int(len(sample))
    is_subset = sample_size < n_centered

    sub_queries = [
        ("a", 192, ">=",  P3_CTE_BOUND),
        ("b", 192, "<=", -P3_CTE_BOUND),
        ("c", 193, ">=",  P3_HE_BOUND),
        ("d", 193, "<=", -P3_HE_BOUND),
    ]

    selection_rule = (
        f"centered = ascending dataset indices with |CTE| < {P3_CENTERED_THRESHOLD} "
        f"(n_centered={n_centered}); sample = centered[round(linspace(0, n_centered-1, "
        f"num_images))]; num_images='all' selects every centered image in order. "
        "Identical rule at every size; no post-hoc anchor selection."
    )

    manifest_path = None
    if is_subset and args.manifest:
        manifest_path = args.manifest
        Path(manifest_path).write_text(json.dumps({
            "selection_rule": selection_rule,
            "n_centered": n_centered,
            "num_images_requested": num_images_arg,
            "sample_size": sample_size,
            "selected_dataset_indices": [int(i) for i in sample],
        }, indent=2))

    out = {
        "script": "verify_p3_local_pca.py",
        "provenance": {
            "command": "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "nnet_path": NNET_PATH,
            "data_path": DATA_PATH,
            "timeout_seconds_per_subquery": TIMEOUT_S,
            "n_centered_total": n_centered,
            "num_images_requested": num_images_arg,
            "sample_size": sample_size,
            "is_subset": is_subset,
            "manifest": manifest_path,
            "selection_rule": selection_rule,
            "unchanged": ("property P3, thresholds (|CTE|<10, |HE|<90), nnet, mean=0.5 "
                          "invariant, 0<=x<=1 clip, neighbor bins, PCA dimension d=8, "
                          "Marabou solver semantics"),
        },
        "d": D,
        "alpha_values": alphas_list,
        "neighbor_bins": {"cte_half_width": CTE_BIN_HALF,
                          "he_half_width": HE_BIN_HALF},
        "thresholds": {"p3_cte_bound": P3_CTE_BOUND, "p3_he_bound": P3_HE_BOUND,
                       "centered_threshold": P3_CENTERED_THRESHOLD,
                       "mean_target": SUM_TARGET / NUM_PIXELS},
        "results_by_alpha": {},
        "per_image_pca_meta": [],
    }

    # Precompute PCA per image once (shared across all alphas)
    pcas = []
    n_empty_neighbor = 0
    print(f"Building local PCA (d={D}) for {sample_size} centered anchors "
          f"(of {n_centered} centered; subset={is_subset})...\n", flush=True)
    for k, idx in enumerate(sample):
        info = build_local_pca(images, labels, idx)
        if info is None:
            n_empty_neighbor += 1
            pcas.append(None)
            if sample_size <= 60:
                print(f"  img{k+1:>2} idx={idx} -- not enough neighbors")
            continue
        out["per_image_pca_meta"].append({
            "image_idx": int(idx),
            "n_neighbors": info["n_neighbors"],
            "explained_variance_ratio_top_d": round(info["explained_variance_ratio_top_d"], 4),
            "sigma_proj": [round(float(s), 5) for s in info["sigma_proj"]],
            "top_singular_values": [round(float(s), 4) for s in info["singular_values"][:D]],
        })
        pcas.append(info)
        if sample_size <= 60:
            print(f"  img{k+1:>2} idx={idx}  m={info['n_neighbors']:>3}  "
                  f"expl_var(top {D}) = {info['explained_variance_ratio_top_d']:.3f}  "
                  f"sigma_proj = [{info['sigma_proj'][0]:.4f}..{info['sigma_proj'][-1]:.4f}]", flush=True)
    print(f"  PCA built: {sample_size - n_empty_neighbor} ok, "
          f"{n_empty_neighbor} empty-neighbor.\n", flush=True)

    for alpha in alphas_list:
        print(f"=== ALPHA = {alpha} sigma  (n={sample_size}) ===", flush=True)
        per_image = {}
        verified = sat = timeout = empty_neighbor = region_empty = other = 0
        anchor_times = []
        t_alpha = time.time()
        for k, idx in enumerate(sample):
            info = pcas[k]
            if info is None:
                empty_neighbor += 1
                per_image[f"image_{k+1}"] = {"dataset_index": int(idx),
                                             "category": "empty_neighbor",
                                             "verified": False}
                continue
            alphas = alpha * info["sigma_proj"]
            sub_results = {}
            all_unsat = True
            for tag, var, direc, b in sub_queries:
                r = solve_one(info["x0"], info["V_top"], info["V_null"],
                              alphas, var, direc, b)
                sub_results[tag] = r
                if r["status"] != "unsat":
                    all_unsat = False
            statuses = [sub_results[t]["status"] for t, *_ in sub_queries]
            # Verification decision is unchanged: verified iff all four UNSAT.
            # Finer reporting categories are derived from the same sub-results.
            if "region_empty_after_clip" in statuses:
                category = "region_empty"
                region_empty += 1
            elif "sat" in statuses:
                category = "sat"
                sat += 1
            elif "timeout" in statuses:
                category = "timeout"
                timeout += 1
            elif all_unsat:
                category = "verified"
                verified += 1
            else:
                category = "other"
                other += 1
            anchor_solve_s = sum(sub_results[t].get("time_seconds", 0.0) for t, *_ in sub_queries)
            if category not in ("empty_neighbor", "region_empty"):
                anchor_times.append(anchor_solve_s)
            per_image[f"image_{k+1}"] = {
                "dataset_index": int(idx),
                "n_neighbors": info["n_neighbors"],
                "alphas_first_last": [round(float(alphas[0]), 5),
                                       round(float(alphas[-1]), 5)],
                "category": category,
                "verified": bool(all_unsat),
                "anchor_solve_seconds": round(anchor_solve_s, 4),
                "sub_queries": sub_results,
            }
            if sample_size <= 60:
                short = " ".join(s[:3] for s in statuses)
                print(f"  img{k+1:>2} idx={idx} {category:<13} [{short}]", flush=True)
            elif (k + 1) % 100 == 0:
                print(f"  ...{k+1}/{sample_size} done "
                      f"(verified={verified} sat={sat} timeout={timeout})", flush=True)
        elapsed = round(time.time() - t_alpha, 2)
        out["results_by_alpha"][f"alpha_{alpha}"] = {
            "sample_size": sample_size,
            "attempted": sample_size,
            "verified": verified,
            "sat": sat,
            "timeout": timeout,
            "empty_neighbor": empty_neighbor,
            "region_empty": region_empty,
            "other": other,
            "violated_legacy": sat + timeout + region_empty + other,
            "elapsed_seconds": elapsed,
            "runtime_distribution_seconds": _dist(anchor_times),
            "selection_rule": selection_rule,
            "per_image": per_image,
        }
        print(f"  => {verified}/{sample_size} verified, {sat} sat, {timeout} timeout, "
              f"{empty_neighbor} empty-neighbor, {region_empty} region-empty  ({elapsed:.1f}s)\n",
              flush=True)
        # incremental save so a long run is never lost
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
