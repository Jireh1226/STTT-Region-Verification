"""
Formal Verification of TinyTaxiNet Properties P1 and P2 using Marabou 2.0
Version 2: Uses data-driven input bounds derived from actual dataset statistics.

HURDLE FROM V1:
  The synthetic C_centered region (cols 7,8 bright [0.7,1.0], rest dark [0.0,0.3])
  was too loose — it included many inputs that are nothing like real runway images.
  Actual pixel values range [0.34, 0.77], not [0, 1]. P1 was SAT (violated) because
  adversarial inputs within the synthetic region don't resemble real runway images.

FIX IN V2:
  P1 now uses per-pixel min/max bounds computed from near-centered images in the
  actual dataset. This is a tighter, more realistic input region that matches
  Kadron et al.'s approach of using data-derived bounds.

Usage: python3.11 verify_p1_p2_v2.py
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
RESULTS_FILE = 'verification_results_v2.json'

# P1 bounds (from Kadron et al.)
P1_CTE_BOUND = 10.0
P1_HE_BOUND = 90.0

# P2 bounds
P2_EPSILON = 0.02
P2_CTE_BOUNDS = [1.0, 1.5]  # 1.0m matches Kadron et al., 1.5m is our relaxed variant
P2_HE_BOUND = 5.0
P2_NUM_IMAGES = 20

PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

# ============================================================
# Helper functions
# ============================================================

def load_data():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]   # (10000, 8, 16)
        labels = f['X_train'][:]   # (10000, 3)
    return images, labels


def compute_data_driven_bounds(images, labels, cte_threshold=1.0):
    """Compute per-pixel min/max bounds from near-centered images.

    This creates a tighter input region than arbitrary [0,1] bounds.
    We use images with |CTE| < cte_threshold to define what 'centered' means.
    """
    cte = labels[:, 0]
    mask = np.abs(cte) < cte_threshold
    centered_imgs = images[mask]

    # Flatten to (N, 128)
    flat = centered_imgs.reshape(centered_imgs.shape[0], -1)

    # Per-pixel min and max
    pixel_mins = flat.min(axis=0)
    pixel_maxs = flat.max(axis=0)

    return pixel_mins, pixel_maxs, centered_imgs.shape[0]


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
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Pixel value')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved counterexample: {filename}")


# ============================================================
# P1: Safety Bound (Data-Driven Bounds)
# ============================================================

def verify_p1(images, labels):
    """
    P1 - Safety Bound

    Input region: Per-pixel [min, max] bounds from centered images (|CTE| < 1.0).
    This is tighter and more realistic than the synthetic C_centered region.

    Output property: |CTE| < 10.0 AND |HE| < 90.0
    """
    print("=" * 70)
    print("P1: SAFETY BOUND VERIFICATION (Data-Driven Bounds)")
    print("=" * 70)

    pixel_mins, pixel_maxs, n_imgs = compute_data_driven_bounds(images, labels, cte_threshold=1.0)
    print(f"  Input region: per-pixel [min, max] from {n_imgs} images with |CTE| < 1.0m")
    print(f"  Pixel value range in region: [{pixel_mins.min():.4f}, {pixel_maxs.max():.4f}]")
    print(f"  Property: |CTE| < {P1_CTE_BOUND}m AND |HE| < {P1_HE_BOUND}°")
    print()

    sub_queries = [
        ("P1a", "CTE >= 10", 192, ">=", P1_CTE_BOUND),
        ("P1b", "CTE <= -10", 192, "<=", -P1_CTE_BOUND),
        ("P1c", "HE >= 90", 193, ">=", P1_HE_BOUND),
        ("P1d", "HE <= -90", 193, "<=", -P1_HE_BOUND),
    ]

    results = {}
    all_unsat = True

    for name, desc, out_var, direction, bound in sub_queries:
        print(f"  --- {name}: checking if {desc} is satisfiable ---")

        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()

        # Set data-driven input bounds
        for i in range(128):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

        # Set output constraint (negated postcondition)
        if direction == ">=":
            net.setLowerBound(out_var, bound)
        else:
            net.setUpperBound(out_var, bound)

        mem_before = get_peak_memory_mb()
        t_start = time.time()
        exit_code, vals, stats = net.solve()
        t_elapsed = time.time() - t_start
        mem_after = get_peak_memory_mb()

        status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

        results[name] = {
            'description': desc,
            'status': status,
            'time_seconds': round(t_elapsed, 4),
            'peak_memory_mb': round(mem_after, 1),
            'memory_delta_mb': round(mem_after - mem_before, 1),
        }

        if status == 'sat':
            all_unsat = False
            cte_val = vals[192]
            he_val = vals[193]
            results[name]['counterexample'] = {
                'CTE': round(cte_val, 6),
                'HE': round(he_val, 6),
            }
            print(f"    Result: SAT (VIOLATED) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
            print(f"    Counterexample: CTE={cte_val:.4f}m, HE={he_val:.4f}°")
            visualize_counterexample(vals, input_vars,
                f"{name} Counterexample\nCTE={cte_val:.2f}m, HE={he_val:.2f}°",
                f"counterexample_{name}.png")
        elif status == 'unsat':
            print(f"    Result: UNSAT (holds) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
        else:
            all_unsat = False
            print(f"    Result: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

    print()
    if all_unsat:
        print("  P1 OVERALL: UNSAT — Safety bound FORMALLY VERIFIED")
    else:
        print("  P1 OVERALL: VIOLATED (some sub-queries SAT)")
    print()

    return results, all_unsat


# ============================================================
# P1 with tighter CTE bound (experiment)
# ============================================================

def verify_p1_tighter(images, labels):
    """
    P1-tight: Same as P1 but with CTE bound = 5.0m (tighter than Kadron's 10m)
    to see how tight we can make the safety bound.
    """
    print("=" * 70)
    print("P1-TIGHT: SAFETY BOUND WITH CTE < 5.0m (Experiment)")
    print("=" * 70)

    pixel_mins, pixel_maxs, n_imgs = compute_data_driven_bounds(images, labels, cte_threshold=1.0)

    CTE_TIGHT = 5.0
    sub_queries = [
        ("P1t_a", "CTE >= 5", 192, ">=", CTE_TIGHT),
        ("P1t_b", "CTE <= -5", 192, "<=", -CTE_TIGHT),
    ]

    results = {}
    all_unsat = True

    for name, desc, out_var, direction, bound in sub_queries:
        print(f"  --- {name}: checking if {desc} is satisfiable ---")

        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()

        for i in range(128):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

        if direction == ">=":
            net.setLowerBound(out_var, bound)
        else:
            net.setUpperBound(out_var, bound)

        mem_before = get_peak_memory_mb()
        t_start = time.time()
        exit_code, vals, stats = net.solve()
        t_elapsed = time.time() - t_start
        mem_after = get_peak_memory_mb()

        status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)
        results[name] = {
            'description': desc, 'status': status,
            'time_seconds': round(t_elapsed, 4),
            'peak_memory_mb': round(mem_after, 1),
            'memory_delta_mb': round(mem_after - mem_before, 1),
        }

        if status == 'sat':
            all_unsat = False
            cte_val = vals[192]
            he_val = vals[193]
            results[name]['counterexample'] = {'CTE': round(cte_val, 6), 'HE': round(he_val, 6)}
            print(f"    Result: SAT in {t_elapsed:.2f}s, mem={mem_after:.0f}MB — CTE={cte_val:.4f}, HE={he_val:.4f}")
        elif status == 'unsat':
            print(f"    Result: UNSAT in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
        else:
            all_unsat = False
            print(f"    Result: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

    print(f"\n  P1-TIGHT: {'VERIFIED' if all_unsat else 'VIOLATED'} for |CTE| < {CTE_TIGHT}m\n")
    return results, all_unsat


# ============================================================
# P2: Local Correctness
# ============================================================

def verify_p2(images, labels, cte_bound=1.0):
    """
    P2 - Local Correctness Bound

    For each sampled training image x0 with label (CTE*, HE*):
      Input: ||x - x0||_inf <= 0.02, clipped to [0, 1]
      Output: |y0 - CTE*| < cte_bound AND |y1 - HE*| < 5.0

    Note: We use strict < in the postcondition so the negation (>=) aligns
    with Marabou's setLowerBound/setUpperBound (which are non-strict).
    """
    print("=" * 70)
    print(f"P2: LOCAL CORRECTNESS VERIFICATION (CTE bound = {cte_bound}m)")
    print("=" * 70)
    print(f"  Perturbation: L-inf epsilon = {P2_EPSILON}")
    print(f"  Output bounds: |CTE - CTE*| < {cte_bound}m, |HE - HE*| < {P2_HE_BOUND}°")
    print(f"  Number of sampled training images: {P2_NUM_IMAGES}")
    print()

    cte = labels[:, 0]
    he = labels[:, 1]

    # Select near-centered images
    near_center_mask = np.abs(cte) < 2.0
    near_center_indices = np.where(near_center_mask)[0]
    print(f"  Candidate pool: {len(near_center_indices)} images with |CTE| < 2.0")

    sample_indices = near_center_indices[
        np.linspace(0, len(near_center_indices) - 1, P2_NUM_IMAGES, dtype=int)
    ]

    results = {}
    total_verified = 0
    total_violated = 0
    first_counterexample_saved = False

    for img_num, img_idx in enumerate(sample_indices):
        x0 = images[img_idx].flatten()
        cte_true = float(cte[img_idx])
        he_true = float(he[img_idx])

        # First check: does the network even get close to the true label on this image?
        net_check = Marabou.read_nnet(NNET_PATH)
        out = net_check.evaluate([x0])
        out_vals = np.array(out).flatten()
        cte_pred = float(out_vals[0])
        he_pred = float(out_vals[1])
        cte_err = abs(cte_pred - cte_true)
        he_err = abs(he_pred - he_true)

        print(f"  Image {img_num+1}/{P2_NUM_IMAGES} (idx={img_idx}, "
              f"CTE*={cte_true:.3f}, HE*={he_true:.3f})")
        print(f"    Nominal prediction: CTE={cte_pred:.3f} (err={cte_err:.3f}), "
              f"HE={he_pred:.3f} (err={he_err:.3f})")

        # If the nominal prediction already exceeds bounds, note it
        if cte_err > cte_bound or he_err > P2_HE_BOUND:
            print(f"    NOTE: Nominal prediction already outside bounds!")

        sub_queries = [
            (f"P2_img{img_num+1}a", f"CTE > CTE*+{cte_bound}",
             192, ">=", cte_true + cte_bound),
            (f"P2_img{img_num+1}b", f"CTE < CTE*-{cte_bound}",
             192, "<=", cte_true - cte_bound),
            (f"P2_img{img_num+1}c", f"HE > HE*+{P2_HE_BOUND}",
             193, ">=", he_true + P2_HE_BOUND),
            (f"P2_img{img_num+1}d", f"HE < HE*-{P2_HE_BOUND}",
             193, "<=", he_true - P2_HE_BOUND),
        ]

        image_all_unsat = True
        image_results = {}

        for name, desc, out_var, direction, bound in sub_queries:
            net = Marabou.read_nnet(NNET_PATH)
            input_vars = net.inputVars[0].flatten()

            # L-inf ball around x0, clipped to [0, 1]
            for i in range(128):
                lo = max(PIXEL_MIN, x0[i] - P2_EPSILON)
                hi = min(PIXEL_MAX, x0[i] + P2_EPSILON)
                net.setLowerBound(input_vars[i], float(lo))
                net.setUpperBound(input_vars[i], float(hi))

            if direction == ">=":
                net.setLowerBound(out_var, bound)
            else:
                net.setUpperBound(out_var, bound)

            mem_before = get_peak_memory_mb()
            t_start = time.time()
            exit_code, vals, stats = net.solve()
            t_elapsed = time.time() - t_start
            mem_after = get_peak_memory_mb()

            status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

            image_results[name] = {
                'description': desc,
                'status': status,
                'time_seconds': round(t_elapsed, 4),
                'peak_memory_mb': round(mem_after, 1),
                'memory_delta_mb': round(mem_after - mem_before, 1),
            }

            if status == 'sat':
                image_all_unsat = False
                cte_val = vals[192]
                he_val = vals[193]
                image_results[name]['counterexample'] = {
                    'CTE': round(cte_val, 6),
                    'HE': round(he_val, 6),
                }
                print(f"    {name}: SAT in {t_elapsed:.2f}s, mem={mem_after:.0f}MB — CTE={cte_val:.4f}, HE={he_val:.4f}")

                # Save first counterexample visualization
                if not first_counterexample_saved:
                    visualize_counterexample(vals, input_vars,
                        f"P2 Counterexample (Image {img_num+1})\n"
                        f"CTE={cte_val:.2f} vs CTE*={cte_true:.2f}",
                        "counterexample_P2_first.png")
                    first_counterexample_saved = True
            elif status == 'unsat':
                print(f"    {name}: UNSAT in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
            else:
                image_all_unsat = False
                print(f"    {name}: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

        if image_all_unsat:
            total_verified += 1
            print(f"    => VERIFIED")
        else:
            total_violated += 1
            print(f"    => VIOLATED")

        results[f'image_{img_num+1}'] = {
            'dataset_index': int(img_idx),
            'CTE_true': round(cte_true, 6),
            'HE_true': round(he_true, 6),
            'CTE_predicted': round(cte_pred, 6),
            'HE_predicted': round(he_pred, 6),
            'nominal_CTE_error': round(cte_err, 6),
            'nominal_HE_error': round(he_err, 6),
            'verified': image_all_unsat,
            'sub_queries': image_results,
        }
        print()

    print(f"  P2 OVERALL: {total_verified}/{P2_NUM_IMAGES} verified, "
          f"{total_violated}/{P2_NUM_IMAGES} violated")
    print()

    return results, total_verified, total_violated


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print()
    print("FORMAL VERIFICATION OF TINYTAXINET USING MARABOU 2.0 (V2)")
    print("Properties P1 (Safety Bound) and P2 (Local Correctness)")
    print()

    images, labels = load_data()

    all_results = {
        'tool': 'Marabou 2.0 (maraboupy 2.0.0)',
        'network': NNET_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v2 — data-driven input bounds',
        'assumptions': {
            'normalization': 'Pixel values used as-is from dataset, range ~[0.34, 0.77]',
            'input_model': 'L-inf box constraints on pixel values',
            'P1_input_region': 'Per-pixel [min, max] from near-centered dataset images (|CTE| < 1.0m)',
            'P2_input_region': 'L-inf ball of radius 0.02 around each training image, clipped to [0,1]',
            'P2_image_selection': 'Near-centered training images (|CTE| < 2.0) from VerifyGAN dataset (y_train/X_train)',
            'data_note': 'HDF5 naming is swapped: y_train=images, X_train=labels',
            'preprocessing_note': 'No additional normalization applied beyond what is in the .nnet file',
        },
        'hurdles': [
            'V1 used synthetic C_centered region [0.0,0.3]/[0.7,1.0] which was too loose',
            'Actual pixel values range [0.34, 0.77], not [0, 1]',
            'V2 uses data-driven per-pixel bounds for more realistic verification',
        ],
    }

    # P1 with data-driven bounds
    p1_results, p1_holds = verify_p1(images, labels)
    all_results['P1'] = {
        'property': 'Safety Bound',
        'specification': f'|CTE| < {P1_CTE_BOUND} AND |HE| < {P1_HE_BOUND}',
        'input_region': 'data-driven per-pixel bounds (|CTE| < 1.0)',
        'holds': p1_holds,
        'sub_queries': p1_results,
    }

    # P1 with tighter CTE bound (experiment)
    p1t_results, p1t_holds = verify_p1_tighter(images, labels)
    all_results['P1_tight'] = {
        'property': 'Safety Bound (tighter)',
        'specification': '|CTE| < 5.0',
        'holds': p1t_holds,
        'sub_queries': p1t_results,
    }

    # P2 at both CTE thresholds
    p2_all = {}
    for cte_bound in P2_CTE_BOUNDS:
        p2_results, p2_verified, p2_violated = verify_p2(images, labels, cte_bound=cte_bound)
        key = f'P2_CTE_{cte_bound}m'
        p2_all[key] = {
            'property': 'Local Correctness',
            'specification': f'|CTE - CTE*| < {cte_bound} AND |HE - HE*| < {P2_HE_BOUND}',
            'epsilon': P2_EPSILON,
            'cte_bound': cte_bound,
            'images_verified': p2_verified,
            'images_violated': p2_violated,
            'total_images': P2_NUM_IMAGES,
            'results': p2_results,
        }
    all_results['P2'] = p2_all

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  P1 (Safety, |CTE|<10, |HE|<90): {'VERIFIED' if p1_holds else 'VIOLATED'}")
    print(f"  P1-tight (|CTE|<5):              {'VERIFIED' if p1t_holds else 'VIOLATED'}")
    for cte_bound in P2_CTE_BOUNDS:
        key = f'P2_CTE_{cte_bound}m'
        v = p2_all[key]['images_verified']
        print(f"  P2 (CTE<{cte_bound}m, eps=0.02):      {v}/{P2_NUM_IMAGES} verified")
    print()
