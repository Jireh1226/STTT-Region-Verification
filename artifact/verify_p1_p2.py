"""
Formal Verification of TinyTaxiNet Properties P1 and P2 using Marabou 2.0

P1 (Safety Bound): For all centered inputs, |CTE| < 10.0 and |HE| < 90.0
P2 (Local Correctness): For all inputs within L-inf epsilon=0.02 of a known
    image, |CTE - CTE*| <= 1.5 and |HE - HE*| <= 5.0

Usage: python3.11 verify_p1_p2.py
"""

import time
import json
import numpy as np
import h5py
from maraboupy import Marabou, MarabouCore

# ============================================================
# Configuration
# ============================================================
NNET_PATH = 'VerifyGAN/models/TinyTaxiNet.nnet'
DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
RESULTS_FILE = 'verification_results.json'

# P1 bounds (from Kadron et al.)
P1_CTE_BOUND = 10.0   # meters
P1_HE_BOUND = 90.0    # degrees

# P2 bounds (from Kadron et al.)
P2_EPSILON = 0.02      # L-inf input perturbation
P2_CTE_BOUND = 1.5     # meters deviation from true label
P2_HE_BOUND = 5.0      # degrees deviation from true label
P2_NUM_IMAGES = 20     # number of test images to verify

# Pixel value range (images are normalized to [0, 1])
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

# P1 input region: C_centered
# Centerline columns (c in {7, 8}) bright [0.7, 1.0], rest dark [0.0, 0.3]
# Pixel (r, c) maps to index i = 16*r + c
CENTER_COLS = {7, 8}
CENTER_BRIGHT_LO = 0.7
CENTER_BRIGHT_HI = 1.0
CENTER_DARK_LO = 0.0
CENTER_DARK_HI = 0.3

# ============================================================
# Helper functions
# ============================================================

def pixel_index(row, col):
    """Convert (row, col) in 8x16 image to flattened index."""
    return 16 * row + col


def load_data():
    """Load images and labels from HDF5 file.
    Note: naming is swapped in the file (y_train=images, X_train=labels).
    """
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]   # (10000, 8, 16)
        labels = f['X_train'][:]   # (10000, 3) - col0=CTE, col1=HE, col2=?
    return images, labels


def result_to_str(exit_code):
    """Convert Marabou exit code to human-readable string."""
    if exit_code == 'unsat':
        return 'UNSAT (property HOLDS)'
    elif exit_code == 'sat':
        return 'SAT (property VIOLATED)'
    elif exit_code == 'TIMEOUT':
        return 'TIMEOUT'
    else:
        return f'UNKNOWN ({exit_code})'


# ============================================================
# P1: Safety Bound
# ============================================================

def verify_p1():
    """
    P1 - Safety Bound (Baseline, reproducing Kadron et al.)

    Precondition: x in C_centered
      - Centerline columns (c=7,8): pixel in [0.7, 1.0]
      - All other columns: pixel in [0.0, 0.3]

    Postcondition (negated for Marabou):
      We want to check |y0| < 10 AND |y1| < 90
      Marabou checks satisfiability, so we negate the postcondition.
      We check if there EXISTS an input satisfying the precondition where
      the postcondition is VIOLATED.

      Since the postcondition is a conjunction (|y0| < 10 AND |y1| < 90),
      the negation is a disjunction: |y0| >= 10 OR |y1| >= 90.

      We split this into 4 separate queries (one per bound):
        P1a: y0 >= 10   (CTE too high)
        P1b: y0 <= -10  (CTE too low)
        P1c: y1 >= 90   (HE too high)
        P1d: y1 <= -90  (HE too low)

      If ALL 4 return UNSAT, the property holds.
      If ANY returns SAT, the property is violated.

    Assumptions documented per professor feedback:
      - Input perturbation model: L-inf box constraints on pixel values
      - Bright columns [0.7, 1.0] simulate the painted runway centerline
      - Dark columns [0.0, 0.3] simulate runway surface / surroundings
      - This is a synthetic input region, not derived from actual image statistics
      - No explicit normalization is applied (pixel values used as-is in [0,1])
    """
    print("=" * 70)
    print("P1: SAFETY BOUND VERIFICATION")
    print("=" * 70)
    print(f"  Input region: C_centered (cols 7,8 bright [{CENTER_BRIGHT_LO},{CENTER_BRIGHT_HI}], "
          f"others dark [{CENTER_DARK_LO},{CENTER_DARK_HI}])")
    print(f"  Property: |CTE| < {P1_CTE_BOUND}m AND |HE| < {P1_HE_BOUND}°")
    print()

    # Define the 4 sub-queries (negated postcondition, one per output bound)
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

        # Load a fresh network for each query
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()

        # Set input bounds: C_centered region
        for r in range(8):
            for c in range(16):
                idx = input_vars[pixel_index(r, c)]
                if c in CENTER_COLS:
                    net.setLowerBound(idx, CENTER_BRIGHT_LO)
                    net.setUpperBound(idx, CENTER_BRIGHT_HI)
                else:
                    net.setLowerBound(idx, CENTER_DARK_LO)
                    net.setUpperBound(idx, CENTER_DARK_HI)

        # Set output constraint (negated postcondition)
        if direction == ">=":
            net.setLowerBound(out_var, bound)
        else:  # "<="
            net.setUpperBound(out_var, bound)

        # Solve
        t_start = time.time()
        exit_code, vals, stats = net.solve()
        t_elapsed = time.time() - t_start

        status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

        results[name] = {
            'description': desc,
            'status': status,
            'time_seconds': round(t_elapsed, 4),
        }

        if status == 'sat':
            all_unsat = False
            # Extract counterexample
            cte_val = vals[192]
            he_val = vals[193]
            results[name]['counterexample'] = {
                'CTE': round(cte_val, 6),
                'HE': round(he_val, 6),
            }
            print(f"    Result: SAT (VIOLATED!) in {t_elapsed:.2f}s")
            print(f"    Counterexample: CTE={cte_val:.4f}m, HE={he_val:.4f}°")
        elif status == 'unsat':
            print(f"    Result: UNSAT (holds) in {t_elapsed:.2f}s")
        else:
            all_unsat = False
            print(f"    Result: {status} in {t_elapsed:.2f}s")

    print()
    if all_unsat:
        print("  P1 OVERALL: UNSAT — Safety bound FORMALLY VERIFIED")
        print(f"  For ALL centered inputs, |CTE| < {P1_CTE_BOUND}m and |HE| < {P1_HE_BOUND}°")
    else:
        print("  P1 OVERALL: VIOLATED or INCONCLUSIVE")
    print()

    return results, all_unsat


# ============================================================
# P2: Local Correctness
# ============================================================

def verify_p2():
    """
    P2 - Local Correctness Bound (Baseline, reproducing Kadron et al.)

    For each test image x0 with known label (CTE*, HE*):
      Precondition: ||x - x0||_inf <= epsilon (= 0.02)
        i.e., each pixel x_i in [x0_i - 0.02, x0_i + 0.02], clipped to [0, 1]
      Postcondition: |y0 - CTE*| <= 1.5 AND |y1 - HE*| <= 5.0

    Again we negate the postcondition and split into 4 sub-queries per image.

    Assumptions documented per professor feedback:
      - Epsilon = 0.02 represents minor sensor noise / brightness variation
      - This is an L-inf perturbation model: each pixel can independently vary by ±0.02
      - The test images come from the VerifyGAN dataset (simulator-generated)
      - Labels (CTE*, HE*) are the simulator ground truth
      - No additional normalization is applied beyond what's in the .nnet file
      - We sample 20 images near-centered (|CTE| < 2.0) to match Kadron et al.'s setup
    """
    print("=" * 70)
    print("P2: LOCAL CORRECTNESS VERIFICATION")
    print("=" * 70)
    print(f"  Perturbation: L-inf epsilon = {P2_EPSILON}")
    print(f"  Property: |CTE - CTE*| <= {P2_CTE_BOUND}m AND |HE - HE*| <= {P2_HE_BOUND}°")
    print(f"  Number of test images: {P2_NUM_IMAGES}")
    print()

    # Load data
    images, labels = load_data()
    cte = labels[:, 0]
    he = labels[:, 1]

    # Select test images: near-centered (|CTE| < 2.0), evenly spaced
    near_center_mask = np.abs(cte) < 2.0
    near_center_indices = np.where(near_center_mask)[0]
    print(f"  Found {len(near_center_indices)} images with |CTE| < 2.0")

    # Sample P2_NUM_IMAGES evenly spaced from the near-center set
    sample_indices = near_center_indices[
        np.linspace(0, len(near_center_indices) - 1, P2_NUM_IMAGES, dtype=int)
    ]

    results = {}
    total_verified = 0
    total_violated = 0

    for img_num, img_idx in enumerate(sample_indices):
        x0 = images[img_idx].flatten()  # (128,)
        cte_true = float(cte[img_idx])
        he_true = float(he[img_idx])

        print(f"  Image {img_num+1}/{P2_NUM_IMAGES} (dataset idx={img_idx}, "
              f"CTE*={cte_true:.4f}m, HE*={he_true:.4f}°)")

        # 4 sub-queries for negated postcondition
        sub_queries = [
            (f"P2_img{img_num+1}a", f"CTE > CTE*+{P2_CTE_BOUND}",
             192, ">=", cte_true + P2_CTE_BOUND),
            (f"P2_img{img_num+1}b", f"CTE < CTE*-{P2_CTE_BOUND}",
             192, "<=", cte_true - P2_CTE_BOUND),
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

            # Set input bounds: L-inf ball around x0
            for i in range(128):
                lo = max(PIXEL_MIN, x0[i] - P2_EPSILON)
                hi = min(PIXEL_MAX, x0[i] + P2_EPSILON)
                net.setLowerBound(input_vars[i], float(lo))
                net.setUpperBound(input_vars[i], float(hi))

            # Set output constraint (negated postcondition)
            if direction == ">=":
                net.setLowerBound(out_var, bound)
            else:
                net.setUpperBound(out_var, bound)

            # Solve
            t_start = time.time()
            exit_code, vals, stats = net.solve()
            t_elapsed = time.time() - t_start

            status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

            image_results[name] = {
                'description': desc,
                'status': status,
                'time_seconds': round(t_elapsed, 4),
            }

            if status == 'sat':
                image_all_unsat = False
                cte_val = vals[192]
                he_val = vals[193]
                image_results[name]['counterexample'] = {
                    'CTE': round(cte_val, 6),
                    'HE': round(he_val, 6),
                }
                print(f"    {name}: SAT (violated) in {t_elapsed:.2f}s — "
                      f"CTE={cte_val:.4f}, HE={he_val:.4f}")
            elif status == 'unsat':
                print(f"    {name}: UNSAT in {t_elapsed:.2f}s")
            else:
                image_all_unsat = False
                print(f"    {name}: {status} in {t_elapsed:.2f}s")

        if image_all_unsat:
            total_verified += 1
            print(f"    => Image {img_num+1}: VERIFIED")
        else:
            total_violated += 1
            print(f"    => Image {img_num+1}: VIOLATED or INCONCLUSIVE")

        results[f'image_{img_num+1}'] = {
            'dataset_index': int(img_idx),
            'CTE_true': round(cte_true, 6),
            'HE_true': round(he_true, 6),
            'verified': image_all_unsat,
            'sub_queries': image_results,
        }
        print()

    print(f"  P2 OVERALL: {total_verified}/{P2_NUM_IMAGES} images verified, "
          f"{total_violated}/{P2_NUM_IMAGES} violated/inconclusive")
    print()

    return results, total_verified, total_violated


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    all_results = {
        'tool': 'Marabou 2.0 (maraboupy)',
        'network': NNET_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'assumptions': {
            'normalization': 'Pixel values in [0,1], no additional normalization applied',
            'input_model': 'L-inf box constraints on pixel values',
            'P1_input_region': 'Synthetic C_centered: cols 7,8 in [0.7,1.0], others in [0.0,0.3]',
            'P2_input_region': 'L-inf ball of radius 0.02 around each test image',
            'P2_image_selection': 'Near-centered images (|CTE| < 2.0) from VerifyGAN dataset',
            'note': 'HDF5 file has swapped naming: y_train=images, X_train=labels',
        },
    }

    print()
    print("FORMAL VERIFICATION OF TINYTAXINET USING MARABOU 2.0")
    print("Properties P1 (Safety Bound) and P2 (Local Correctness)")
    print()

    # Run P1
    p1_results, p1_holds = verify_p1()
    all_results['P1'] = {
        'property': 'Safety Bound',
        'specification': f'|CTE| < {P1_CTE_BOUND} AND |HE| < {P1_HE_BOUND}',
        'input_region': 'C_centered',
        'holds': p1_holds,
        'sub_queries': p1_results,
    }

    # Run P2
    p2_results, p2_verified, p2_violated = verify_p2()
    all_results['P2'] = {
        'property': 'Local Correctness',
        'specification': f'|CTE - CTE*| <= {P2_CTE_BOUND} AND |HE - HE*| <= {P2_HE_BOUND}',
        'epsilon': P2_EPSILON,
        'images_verified': p2_verified,
        'images_violated': p2_violated,
        'total_images': P2_NUM_IMAGES,
        'results': p2_results,
    }

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  P1 (Safety Bound):      {'VERIFIED' if p1_holds else 'VIOLATED/INCONCLUSIVE'}")
    print(f"  P2 (Local Correctness): {p2_verified}/{P2_NUM_IMAGES} images verified")

    # Document any hurdles
    hurdles = []
    if not p1_holds:
        hurdles.append("P1 did not fully verify — check if input region is too broad")
    if p2_violated > 0:
        hurdles.append(f"P2 violated on {p2_violated} images — may need tighter epsilon or wider output bounds")

    if hurdles:
        print()
        print("  HURDLES / NOTES:")
        for h in hurdles:
            print(f"    - {h}")
    print()
