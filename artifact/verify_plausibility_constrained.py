"""
Plausibility-Constrained Verification of TinyTaxiNet Properties.

This script implements Tier 1 plausibility constraints compatible with Marabou 2.0's
linear + piecewise-linear theory, and applies them to existing SAT queries to
determine which failures are genuine vs box-abstraction artifacts.

Three constraints are added simultaneously:
  1. PCA-box constraint: input lies within c*sqrt(lambda_j) of training mean
     along each of k dominant principal directions
  2. Total variation bound: sum of adjacent-pixel absolute differences bounded
     at the p-th percentile of training TV scores
  3. Per-row mean bands: each of 8 rows has its mean within training percentile
     bounds

Applied to: P1a, P1b (safety; C_centered), P4 C_left/C_right (directional),
           P5 (deadzone; C_off_center)

Each query is run four ways:
  (a) Unconstrained (baseline)
  (b) + Mean=0.5 only (original follow-up)
  (c) + All Tier 1 plausibility constraints
"""

import json
import time
import numpy as np
import h5py
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_plausibility_constrained.json"

NUM_PIXELS = 128
H, W = 8, 16                        # 8 rows, 16 cols
PCA_K = 128                         # bound ALL principal directions (not just top k)
PCA_SIGMA = 2.0                     # +/- 2 sigma per direction (tighter)
TV_PERCENTILE = 95                  # threshold = this percentile of training TV (tighter)
ROW_MEAN_PERCENTILE = 95            # per-row mean band percentile (tighter)


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def grid_neighbors():
    """Return (flat_idx_a, flat_idx_b) for each adjacent-pixel edge in an 8x16 grid."""
    edges = []
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            if c + 1 < W:          # horizontal neighbor
                edges.append((idx, idx + 1))
            if r + 1 < H:          # vertical neighbor
                edges.append((idx, idx + W))
    return edges


def fit_plausibility_model(images):
    """Precompute PCA components, TV threshold, per-row mean bounds."""
    mu = images.mean(axis=0)                            # (128,)
    centered = images - mu
    # SVD: U (N, k) @ diag(s) @ Vt (k, 128). Columns of Vt are principal components.
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (s ** 2) / (len(images) - 1)          # variance along each axis
    components = Vt                                     # (128, 128); first k rows = top-k

    # TV threshold on training set
    edges = grid_neighbors()
    tv_scores = np.array([
        sum(abs(img[a] - img[b]) for a, b in edges) for img in images
    ])
    tv_threshold = np.percentile(tv_scores, TV_PERCENTILE)

    # Per-row mean bounds
    row_means = images.reshape(-1, H, W).mean(axis=2)   # (N, 8)
    # We want symmetric bounds: median +/- delta, where delta covers the
    # (ROW_MEAN_PERCENTILE)-th percentile of absolute deviations
    row_mean_medians = np.median(row_means, axis=0)     # (8,)
    row_mean_dev = np.abs(row_means - row_mean_medians)
    row_mean_bands = np.percentile(row_mean_dev, ROW_MEAN_PERCENTILE, axis=0)  # (8,)

    return {
        'mu': mu,
        'pca_components': components,                    # (128, 128)
        'pca_eigenvalues': eigenvalues,                  # (128,)
        'edges': edges,
        'tv_threshold': float(tv_threshold),
        'row_mean_medians': row_mean_medians,
        'row_mean_bands': row_mean_bands,
    }


def add_pca_box(net, input_vars, model, k=PCA_K, sigma=PCA_SIGMA):
    """Add -sigma*sqrt(lambda_j) <= e_j^T (x - mu) <= sigma*sqrt(lambda_j) for j=0..k-1.

    With k = NUM_PIXELS, this bounds ALL principal directions — not just the top ones.
    Low-variance directions get very tight bounds (since sigma*sqrt(lambda_j) is small
    when lambda_j is small), which forces the counterexample to lie near the training
    data's affine subspace rather than drifting into noise directions where real images
    don't vary.
    """
    mu = model['mu']
    components = model['pca_components']
    eigenvalues = model['pca_eigenvalues']

    # Floor the bound at a small value to avoid infeasibility from numerical noise
    # in directions with essentially zero variance.
    MIN_BOUND = 1e-4

    vars_list = [int(v) for v in input_vars]
    for j in range(k):
        e_j = components[j]                              # (128,)
        raw_bound = sigma * np.sqrt(max(eigenvalues[j], 0.0))
        bound = max(raw_bound, MIN_BOUND)
        # e_j^T x <= e_j^T mu + bound
        net.addInequality(vars_list, e_j.tolist(), float(e_j @ mu) + bound)
        # e_j^T x >= e_j^T mu - bound  =>  -e_j^T x <= -e_j^T mu + bound
        net.addInequality(vars_list, (-e_j).tolist(), -float(e_j @ mu) + bound)


def add_tv_bound(net, input_vars, model):
    """Add sum_{(a,b) in edges} |x_a - x_b| <= tv_threshold using abs constraints."""
    edges = model['edges']
    tv_threshold = model['tv_threshold']

    # For each edge, introduce raw_ab = x_a - x_b and abs_ab = |raw_ab|
    abs_vars = []
    for a, b in edges:
        raw_var = net.getNewVariable()
        abs_var = net.getNewVariable()
        # raw_var - x_a + x_b = 0
        net.addEquality([raw_var, int(input_vars[a]), int(input_vars[b])],
                        [1.0, -1.0, 1.0], 0.0)
        net.addAbsConstraint(raw_var, abs_var)
        abs_vars.append(abs_var)

    # Sum of abs_vars <= tv_threshold
    net.addInequality(abs_vars, [1.0] * len(abs_vars), float(tv_threshold))


def add_row_mean_bands(net, input_vars, model):
    """For each row r: |mean of row r pixels - row_mean_medians[r]| <= row_mean_bands[r]."""
    row_mean_medians = model['row_mean_medians']
    row_mean_bands = model['row_mean_bands']

    for r in range(H):
        row_pixels = [int(input_vars[r * W + c]) for c in range(W)]
        # sum of row pixels / W in [median - band, median + band]
        # <=> sum in [W*(median - band), W*(median + band)]
        lo = W * (row_mean_medians[r] - row_mean_bands[r])
        hi = W * (row_mean_medians[r] + row_mean_bands[r])
        # sum <= hi
        net.addInequality(row_pixels, [1.0] * W, float(hi))
        # sum >= lo  =>  -sum <= -lo
        net.addInequality(row_pixels, [-1.0] * W, float(-lo))


def add_mean_constraint(net, input_vars):
    """sum(x_i) = 64   (equivalent to mean=0.5)."""
    net.addEquality([int(v) for v in input_vars], [1.0] * NUM_PIXELS, 64.0)


def setup_base_net(pixel_mins, pixel_maxs):
    """Load network and apply per-pixel input box bounds."""
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    for i in range(NUM_PIXELS):
        net.setLowerBound(int(input_vars[i]), float(pixel_mins[i]))
        net.setUpperBound(int(input_vars[i]), float(pixel_maxs[i]))
    return net, input_vars


def solve_with_constraints(net, input_vars, output_constraint_fn, mode, model, label):
    """Solve a query and return status + optional witness info."""
    if mode in ('mean', 'plausibility'):
        add_mean_constraint(net, input_vars)
    if mode == 'plausibility':
        add_pca_box(net, input_vars, model)
        add_tv_bound(net, input_vars, model)
        add_row_mean_bands(net, input_vars, model)
    output_constraint_fn(net)

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=300)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=options)
    elapsed = time.time() - t0
    status = status.strip().lower()

    result = {
        'mode': mode,
        'status': status,
        'time_seconds': round(elapsed, 4),
    }
    if status == 'sat':
        cx = np.array([vals[int(input_vars[i])] for i in range(NUM_PIXELS)],
                      dtype=np.float64)
        result['counterexample'] = {
            'CTE': round(float(vals[192]), 6),
            'HE': round(float(vals[193]), 6),
            'input_mean': round(float(cx.mean()), 6),
        }
    return result


def compute_box(images):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def main():
    print("Loading data...")
    images, labels = load_data()
    print(f"Loaded {len(images)} images")

    print("Fitting plausibility model (PCA, TV, row-mean)...")
    cte_labels = labels[:, 0]
    model = fit_plausibility_model(images)
    print(f"  PCA k={PCA_K}, top eigenvalues: {model['pca_eigenvalues'][:PCA_K].round(4).tolist()}")
    print(f"  TV threshold ({TV_PERCENTILE}th percentile): {model['tv_threshold']:.4f}")
    print(f"  Row-mean medians: {model['row_mean_medians'].round(4).tolist()}")
    print(f"  Row-mean bands: {model['row_mean_bands'].round(4).tolist()}")

    centered = images[np.abs(cte_labels) < 1.0]
    left = images[cte_labels > 2.0]
    right = images[cte_labels < -2.0]
    off_center = images[np.abs(cte_labels) > 1.0]

    queries = []

    # P1a: CTE >= 10 over C_centered
    pixel_mins, pixel_maxs = compute_box(centered)
    queries.append(('P1a_CTE_upper', centered, pixel_mins, pixel_maxs,
                    lambda net: net.setLowerBound(192, 10.0)))

    # P1b: CTE <= -10 over C_centered
    queries.append(('P1b_CTE_lower', centered, pixel_mins, pixel_maxs,
                    lambda net: net.setUpperBound(192, -10.0)))

    # P4 C_left: CTE <= 0 (violation of CTE > 0 postcondition)
    left_mins, left_maxs = compute_box(left)
    queries.append(('P4_C_left', left, left_mins, left_maxs,
                    lambda net: net.setUpperBound(192, 0.0)))

    # P4 C_right: CTE >= 0 (violation of CTE < 0)
    right_mins, right_maxs = compute_box(right)
    queries.append(('P4_C_right', right, right_mins, right_maxs,
                    lambda net: net.setLowerBound(192, 0.0)))

    # P5: deadzone -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01
    oc_mins, oc_maxs = compute_box(off_center)
    def p5_constraint(net):
        net.setLowerBound(192, -0.01); net.setUpperBound(192, 0.01)
        net.setLowerBound(193, -0.01); net.setUpperBound(193, 0.01)
    queries.append(('P5_deadzone', off_center, oc_mins, oc_maxs, p5_constraint))

    # Also fit the plausibility model on subset-specific data for P4/P5 later if needed
    # For now, use the full-dataset model (more principled: single plausibility specification)

    all_results = {'config': {
        'pca_k': PCA_K,
        'pca_sigma': PCA_SIGMA,
        'tv_percentile': TV_PERCENTILE,
        'row_mean_percentile': ROW_MEAN_PERCENTILE,
        'tv_threshold': model['tv_threshold'],
        'row_mean_medians': model['row_mean_medians'].tolist(),
        'row_mean_bands': model['row_mean_bands'].tolist(),
        'pca_top_eigenvalues': model['pca_eigenvalues'][:PCA_K].tolist(),
    }, 'queries': {}}

    for name, subset, pmins, pmaxs, constraint_fn in queries:
        print(f"\n=== {name} ===")
        all_results['queries'][name] = {}
        for mode in ['unconstrained', 'mean', 'plausibility']:
            print(f"  [{mode}]", end=' ', flush=True)
            net, input_vars = setup_base_net(pmins, pmaxs)
            r = solve_with_constraints(net, input_vars, constraint_fn, mode, model, name)
            print(f"{r['status']} ({r['time_seconds']:.2f}s)", end='')
            if r['status'] == 'sat':
                cx = r['counterexample']
                print(f"   CTE={cx['CTE']:.4f}, HE={cx['HE']:.4f}, mean={cx['input_mean']:.4f}", end='')
            print()
            all_results['queries'][name][mode] = r

    print(f"\nSaving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Query':<20} {'Unconstrained':<15} {'+ Mean=0.5':<15} {'+ Full Plaus.':<15}")
    print("-" * 72)
    for name in all_results['queries']:
        r = all_results['queries'][name]
        print(f"{name:<20} {r['unconstrained']['status']:<15} {r['mean']['status']:<15} {r['plausibility']['status']:<15}")


if __name__ == '__main__':
    main()
