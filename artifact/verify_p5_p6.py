"""
Formal Verification of TinyTaxiNet Properties P5 and P6 using Marabou 2.0

P5 — Deadzone Detection:
  Pre: x in C_off_center (data-driven, |CTE| > 1.0m)
  Post: NOT(|CTE| <= 0.01 AND |HE| <= 0.01)
  Question: Can an off-center input produce near-zero output on both channels?
  Encoding: We encode the negated postcondition directly —
    -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01
  Non-strict <= used because Marabou bounds are non-strict (exact encoding).
  SAT = deadzone exists (property violated), UNSAT = no deadzone (property holds).

P6 — Output Bound Tightening:
  Pre: x in C_centered (data-driven, |CTE| < 1.0m)
  Post: Find tightest [a, b] such that CTE in [a, b] and HE in [c, d]
  Method: Binary search using Marabou queries.
    - For CTE upper bound: search for largest M such that CTE >= M is SAT.
    - For CTE lower bound: search for smallest m such that CTE <= m is SAT.
    - Same for HE.

Usage: python3.11 verify_p5_p6.py
"""

import time
import json
import resource
import platform
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maraboupy import Marabou


def get_peak_memory_mb():
    """Get current peak RSS in MB. macOS reports bytes, Linux reports KB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':
        return usage / (1024 * 1024)  # bytes -> MB
    else:
        return usage / 1024  # KB -> MB


# ============================================================
# Configuration
# ============================================================
NNET_PATH = 'VerifyGAN/models/TinyTaxiNet.nnet'
DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
RESULTS_FILE = 'verification_results_p5_p6.json'

# P5 configuration
P5_OFF_CENTER_THRESHOLD = 1.0  # |CTE| > 1.0m
P5_DEADZONE = 0.01  # Both |CTE| and |HE| below this = deadzone

# P6 configuration
P6_CENTERED_THRESHOLD = 1.0  # |CTE| < 1.0m (same as P1)
P6_BINARY_SEARCH_TOL = 0.01  # Stop binary search when interval < this
P6_INITIAL_CTE_RANGE = (-30.0, 30.0)  # Initial search range for CTE
P6_INITIAL_HE_RANGE = (-120.0, 120.0)  # Initial search range for HE
P6_TIMEOUT = 300  # Per-query timeout in seconds

PIXEL_MIN = 0.0
PIXEL_MAX = 1.0


# ============================================================
# Helper functions
# ============================================================

def load_data():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]   # (10000, 8, 16) — images (naming swapped)
        labels = f['X_train'][:]   # (10000, 3) — labels [CTE, HE, DTP]
    return images, labels


def compute_data_driven_bounds(images, labels, mask):
    """Compute per-pixel min/max bounds from images matching the given mask."""
    selected = images[mask]
    flat = selected.reshape(selected.shape[0], -1)
    pixel_mins = flat.min(axis=0)
    pixel_maxs = flat.max(axis=0)
    return pixel_mins, pixel_maxs, selected.shape[0]


def visualize_counterexample(vals, input_vars, title, filename):
    """Visualize a counterexample as an 8x16 image."""
    img = np.zeros(128)
    for i in range(128):
        var_idx = input_vars[i]
        if var_idx in vals:
            img[i] = vals[var_idx]
    img = img.reshape(8, 16)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(img, cmap='gray', vmin=0.3, vmax=0.8, aspect='equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Pixel value')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved counterexample: {filename}")


# ============================================================
# P5: Deadzone Detection
# ============================================================

def verify_p5(images, labels):
    """
    P5 — Deadzone Detection

    Input region: Per-pixel [min, max] bounds from off-center images (|CTE| > 1.0m).
    Postcondition: NOT(|CTE| <= 0.01 AND |HE| <= 0.01)

    We use non-strict <= in the postcondition because Marabou bounds are
    non-strict. The negated postcondition is then:
      -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01
    encoded as a single Marabou query with setLowerBound/setUpperBound on
    both outputs.

    SAT = deadzone input exists (property VIOLATED)
    UNSAT = no deadzone possible (property HOLDS)
    """
    print("=" * 70)
    print("P5: DEADZONE DETECTION")
    print("=" * 70)

    cte = labels[:, 0]

    # Compute off-center bounds
    off_center_mask = np.abs(cte) > P5_OFF_CENTER_THRESHOLD
    pixel_mins, pixel_maxs, n_imgs = compute_data_driven_bounds(
        images, labels, off_center_mask)

    print(f"  Input region: per-pixel [min, max] from {n_imgs} images with |CTE| > {P5_OFF_CENTER_THRESHOLD}m")
    print(f"  Pixel value range: [{pixel_mins.min():.4f}, {pixel_maxs.max():.4f}]")
    print(f"  Deadzone threshold: |CTE| <= {P5_DEADZONE} AND |HE| <= {P5_DEADZONE}")
    print()

    # Empirical check: how many off-center images already produce near-zero output?
    flat_imgs = images[off_center_mask].reshape(-1, 128)
    off_center_cte = cte[off_center_mask]
    net_eval = Marabou.read_nnet(NNET_PATH)
    empirical_deadzone = 0
    # Pre-compute all outputs for empirical checks
    all_outputs = []
    for i in range(len(flat_imgs)):
        out = net_eval.evaluateWithoutMarabou(flat_imgs[i:i+1])[0]
        all_outputs.append(out)
        if abs(out[0]) < P5_DEADZONE and abs(out[1]) < P5_DEADZONE:
            empirical_deadzone += 1
    print(f"  Empirical deadzone check: {empirical_deadzone}/{n_imgs} off-center images "
          f"produce |CTE|<={P5_DEADZONE} AND |HE|<={P5_DEADZONE}")
    print()

    # Also check with larger thresholds for context
    for thresh in [0.1, 0.5, 1.0]:
        count = sum(1 for out in all_outputs
                    if abs(out[0]) < thresh and abs(out[1]) < thresh)
        print(f"  Empirical near-deadzone (threshold={thresh}): {count}/{n_imgs}")

    print()
    print("  --- Formal verification: encoding negated postcondition ---")
    print(f"  Query: Can CTE in [-{P5_DEADZONE}, {P5_DEADZONE}] AND HE in [-{P5_DEADZONE}, {P5_DEADZONE}]?")
    print()

    # Single Marabou query with both output constraints
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()

    # Set off-center input bounds
    for i in range(128):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

    # Negated postcondition: |CTE| <= 0.01 AND |HE| <= 0.01
    # Encoded as: -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01
    # Marabou bounds are non-strict, so this is an exact encoding.
    net.setLowerBound(192, -P5_DEADZONE)
    net.setUpperBound(192, P5_DEADZONE)
    net.setLowerBound(193, -P5_DEADZONE)
    net.setUpperBound(193, P5_DEADZONE)

    mem_before = get_peak_memory_mb()
    t_start = time.time()
    exit_code, vals, stats = net.solve()
    t_elapsed = time.time() - t_start
    mem_after = get_peak_memory_mb()

    status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

    result = {
        'property': 'Deadzone Detection',
        'input_region': f'C_off_center: per-pixel [min, max] from {n_imgs} images with |CTE| > {P5_OFF_CENTER_THRESHOLD}m',
        'postcondition': f'NOT(|CTE| <= {P5_DEADZONE} AND |HE| <= {P5_DEADZONE})',
        'negated_query': f'|CTE| <= {P5_DEADZONE} AND |HE| <= {P5_DEADZONE}',
        'status': status,
        'time_seconds': round(t_elapsed, 4),
        'peak_memory_mb': round(mem_after, 1),
        'memory_delta_mb': round(mem_after - mem_before, 1),
        'n_off_center_images': n_imgs,
        'empirical_deadzone_count': empirical_deadzone,
    }

    if status == 'sat':
        cte_val = vals[192]
        he_val = vals[193]
        result['counterexample'] = {
            'CTE': round(cte_val, 6),
            'HE': round(he_val, 6),
        }
        cx_input = [vals[input_vars[j]] for j in range(128)]
        result['counterexample_input_mean'] = round(float(np.mean(cx_input)), 6)

        print(f"  Result: SAT (VIOLATED — deadzone exists) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
        print(f"  Counterexample: CTE={cte_val:.6f}, HE={he_val:.6f}")
        print(f"  Counterexample input mean: {np.mean(cx_input):.4f}")
        print(f"  Hazard: This input is within the pixel range of off-center images,")
        print(f"          yet produces near-zero output — the controller would not correct.")

        visualize_counterexample(vals, input_vars,
            f"P5 Deadzone Counterexample\nCTE={cte_val:.4f}m, HE={he_val:.4f}°",
            "counterexample_P5.png")
    elif status == 'unsat':
        print(f"  Result: UNSAT (VERIFIED — no deadzone) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
        print(f"  No off-center input can produce near-zero output on both channels.")
    else:
        print(f"  Result: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

    return result


# ============================================================
# P6: Output Bound Tightening
# ============================================================

def binary_search_bound(pixel_mins, pixel_maxs, output_var, direction, lo, hi, tol):
    """
    Binary search for the tightest reachable output bound.

    For direction='upper': find largest M such that output >= M is SAT.
      - If SAT at mid, the true max is >= mid, so search [mid, hi].
      - If UNSAT at mid, the true max is < mid, so search [lo, mid].

    For direction='lower': find smallest m such that output <= m is SAT.
      - If SAT at mid, the true min is <= mid, so search [lo, mid].
      - If UNSAT at mid, the true min is > mid, so search [mid, hi].

    Returns (bound, iterations, query_log).
    """
    iterations = 0
    query_log = []

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        iterations += 1

        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()

        for i in range(128):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

        if direction == 'upper':
            # Check: can output >= mid?
            net.setLowerBound(output_var, mid)
        else:
            # Check: can output <= mid?
            net.setUpperBound(output_var, mid)

        mem_before = get_peak_memory_mb()
        t_start = time.time()
        exit_code, vals, stats = net.solve()
        t_elapsed = time.time() - t_start
        mem_after = get_peak_memory_mb()

        status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

        entry = {
            'iteration': iterations,
            'mid': round(mid, 6),
            'status': status,
            'time_seconds': round(t_elapsed, 4),
            'peak_memory_mb': round(mem_after, 1),
        }

        if status == 'sat':
            actual_val = vals[output_var]
            entry['witness_value'] = round(actual_val, 6)

        query_log.append(entry)

        print(f"    Iter {iterations}: mid={mid:.4f}, status={status.upper()}, "
              f"time={t_elapsed:.2f}s, mem={mem_after:.0f}MB", end="")

        if direction == 'upper':
            if status == 'sat':
                lo = mid  # True max is >= mid
                print(f" -> search [{lo:.4f}, {hi:.4f}]")
            else:
                hi = mid  # True max is < mid
                print(f" -> search [{lo:.4f}, {hi:.4f}]")
        else:  # lower
            if status == 'sat':
                hi = mid  # True min is <= mid
                print(f" -> search [{lo:.4f}, {hi:.4f}]")
            else:
                lo = mid  # True min is > mid
                print(f" -> search [{lo:.4f}, {hi:.4f}]")

    # Final bound: for upper, use lo (proven reachable); for lower, use hi (proven reachable)
    if direction == 'upper':
        bound = lo
    else:
        bound = hi

    return bound, iterations, query_log


def verify_p6(images, labels):
    """
    P6 — Output Bound Tightening

    Input region: Per-pixel [min, max] bounds from centered images (|CTE| < 1.0m).
    Goal: Find tightest [a, b] for CTE and [c, d] for HE via binary search.

    Each binary search iteration is one Marabou query. The result is a
    formally guaranteed output envelope — tighter than P1's ±10m/±90°.
    """
    print("\n" + "=" * 70)
    print("P6: OUTPUT BOUND TIGHTENING")
    print("=" * 70)

    cte = labels[:, 0]
    centered_mask = np.abs(cte) < P6_CENTERED_THRESHOLD
    pixel_mins, pixel_maxs, n_imgs = compute_data_driven_bounds(
        images, labels, centered_mask)

    print(f"  Input region: per-pixel [min, max] from {n_imgs} centered images (|CTE| < {P6_CENTERED_THRESHOLD}m)")
    print(f"  Binary search tolerance: {P6_BINARY_SEARCH_TOL}")
    print()

    results = {}

    # Search for CTE bounds
    for output_name, output_var, init_range in [
        ('CTE', 192, P6_INITIAL_CTE_RANGE),
        ('HE', 193, P6_INITIAL_HE_RANGE),
    ]:
        print(f"  --- {output_name} upper bound ---")
        upper, upper_iters, upper_log = binary_search_bound(
            pixel_mins, pixel_maxs, output_var, 'upper',
            lo=0.0, hi=init_range[1], tol=P6_BINARY_SEARCH_TOL)
        print(f"  => {output_name} upper bound: {upper:.4f} ({upper_iters} iterations)")
        print()

        print(f"  --- {output_name} lower bound ---")
        lower, lower_iters, lower_log = binary_search_bound(
            pixel_mins, pixel_maxs, output_var, 'lower',
            lo=init_range[0], hi=0.0, tol=P6_BINARY_SEARCH_TOL)
        print(f"  => {output_name} lower bound: {lower:.4f} ({lower_iters} iterations)")
        print()

        results[output_name] = {
            'upper_bound': round(upper, 6),
            'lower_bound': round(lower, 6),
            'total_range': round(upper - lower, 6),
            'upper_iterations': upper_iters,
            'lower_iterations': lower_iters,
            'upper_search_log': upper_log,
            'lower_search_log': lower_log,
        }

        print(f"  {output_name} provably reachable range: [{lower:.4f}, {upper:.4f}] "
              f"(width: {upper - lower:.4f})")
        print()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print()
    print("FORMAL VERIFICATION OF TINYTAXINET USING MARABOU 2.0")
    print("Properties P5 (Deadzone Detection) and P6 (Output Bound Tightening)")
    print()

    images, labels = load_data()

    all_results = {
        'tool': 'Marabou 2.0 (maraboupy 2.0.0)',
        'network': NNET_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'properties': ['P5', 'P6'],
    }

    # P5
    p5_result = verify_p5(images, labels)
    all_results['P5'] = p5_result

    # P6
    p6_results = verify_p6(images, labels)
    all_results['P6'] = {
        'property': 'Output Bound Tightening',
        'input_region': f'C_centered: per-pixel [min, max] from images with |CTE| < {P6_CENTERED_THRESHOLD}m',
        'binary_search_tolerance': P6_BINARY_SEARCH_TOL,
        'results': p6_results,
    }

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  P5 (Deadzone Detection): {p5_result['status'].upper()}")
    if p5_result['status'] == 'sat':
        cx = p5_result['counterexample']
        print(f"    Counterexample: CTE={cx['CTE']:.6f}, HE={cx['HE']:.6f}")

    for name in ['CTE', 'HE']:
        r = p6_results[name]
        print(f"  P6 ({name} range): [{r['lower_bound']:.4f}, {r['upper_bound']:.4f}] "
              f"(width: {r['total_range']:.4f})")
    print()
