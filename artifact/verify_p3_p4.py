"""
Formal Verification of TinyTaxiNet Properties P3 and P4 using Marabou 2.0

P3 — Robustness Under Input Noise:
  Pre: ||x - x0||_inf <= epsilon for a centered image x0, epsilon in {0.05, 0.1}
  Post: |CTE| < 10.0 AND |HE| < 90.0
  Question: Can sensor noise push the network's output outside safety bounds?

P4 — Directional Correctness:
  Pre: x in C_left (data-driven, CTE > 2.0m) or C_right (CTE < -2.0m)
  Post: CTE > 0 for C_left, CTE < 0 for C_right
  Question: Does the network always get the sign of lateral offset correct?

Note: We use strict < in postconditions so the negation (>=) aligns with
Marabou's non-strict setLowerBound/setUpperBound API.

Usage: python3.11 verify_p3_p4.py
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
RESULTS_FILE = 'verification_results_p3_p4.json'

# P3 configuration
P3_EPSILONS = [0.05, 0.1]
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_NUM_IMAGES = 20
P3_CENTERED_THRESHOLD = 1.0  # |CTE| < 1.0m for centered images

# P4 configuration
P4_LEFT_THRESHOLD = 2.0   # CTE > 2.0m = aircraft right of CL, CL on left
P4_RIGHT_THRESHOLD = -2.0  # CTE < -2.0m = aircraft left of CL, CL on right

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
    print(f"    Saved counterexample: {filename}")


# ============================================================
# P3: Robustness Under Input Noise
# ============================================================

def verify_p3(images, labels):
    """
    P3 — Robustness Under Input Noise

    For each centered image x0 and each epsilon in {0.05, 0.1}:
      Input: ||x - x0||_inf <= epsilon, clipped to [0, 1]
      Output: |CTE| < 10.0 AND |HE| < 90.0

    Note: P3 checks whether SAFETY bounds hold under perturbation,
    unlike P2 which checks CORRECTNESS (closeness to ground truth).
    """
    print("=" * 70)
    print("P3: ROBUSTNESS UNDER INPUT NOISE")
    print("=" * 70)

    cte = labels[:, 0]

    # Select centered images (|CTE| < 1.0m)
    centered_mask = np.abs(cte) < P3_CENTERED_THRESHOLD
    centered_indices = np.where(centered_mask)[0]
    print(f"  Candidate pool: {len(centered_indices)} centered images (|CTE| < {P3_CENTERED_THRESHOLD}m)")

    # Evenly sample
    sample_indices = centered_indices[
        np.linspace(0, len(centered_indices) - 1, P3_NUM_IMAGES, dtype=int)
    ]

    all_p3_results = {}

    for epsilon in P3_EPSILONS:
        print(f"\n  --- Epsilon = {epsilon} ---")
        print(f"  Property: |CTE| < {P3_CTE_BOUND}m AND |HE| < {P3_HE_BOUND}°")
        print(f"  Images: {P3_NUM_IMAGES}")

        results = {}
        total_verified = 0
        total_violated = 0
        first_cx_saved = False

        for img_num, img_idx in enumerate(sample_indices):
            x0 = images[img_idx].flatten()
            cte_true = float(cte[img_idx])

            # Get nominal prediction
            net_check = Marabou.read_nnet(NNET_PATH)
            out = net_check.evaluate([x0])
            out_vals = np.array(out).flatten()
            cte_pred = float(out_vals[0])
            he_pred = float(out_vals[1])

            print(f"\n  Image {img_num+1}/{P3_NUM_IMAGES} (idx={img_idx}, "
                  f"true CTE={cte_true:.3f})")
            print(f"    Nominal: CTE={cte_pred:.3f}, HE={he_pred:.3f}")

            # 4 sub-queries: can CTE or HE exceed safety bounds?
            sub_queries = [
                (f"P3_eps{epsilon}_img{img_num+1}a", f"CTE >= {P3_CTE_BOUND}",
                 192, ">=", P3_CTE_BOUND),
                (f"P3_eps{epsilon}_img{img_num+1}b", f"CTE <= -{P3_CTE_BOUND}",
                 192, "<=", -P3_CTE_BOUND),
                (f"P3_eps{epsilon}_img{img_num+1}c", f"HE >= {P3_HE_BOUND}",
                 193, ">=", P3_HE_BOUND),
                (f"P3_eps{epsilon}_img{img_num+1}d", f"HE <= -{P3_HE_BOUND}",
                 193, "<=", -P3_HE_BOUND),
            ]

            image_all_unsat = True
            image_results = {}

            for name, desc, out_var, direction, bound in sub_queries:
                net = Marabou.read_nnet(NNET_PATH)
                input_vars = net.inputVars[0].flatten()

                # L-inf ball around x0, clipped to [0, 1]
                for i in range(128):
                    lo = max(PIXEL_MIN, x0[i] - epsilon)
                    hi = min(PIXEL_MAX, x0[i] + epsilon)
                    net.setLowerBound(input_vars[i], float(lo))
                    net.setUpperBound(input_vars[i], float(hi))

                # Negated postcondition
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

                    if not first_cx_saved:
                        visualize_counterexample(vals, input_vars,
                            f"P3 Counterexample (eps={epsilon}, img {img_num+1})\n"
                            f"CTE={cte_val:.2f}m, HE={he_val:.2f}°",
                            f"counterexample_P3_eps{epsilon}.png")
                        first_cx_saved = True
                elif status == 'unsat':
                    print(f"    {name}: UNSAT in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
                else:
                    image_all_unsat = False
                    print(f"    {name}: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

            if image_all_unsat:
                total_verified += 1
                print(f"    => VERIFIED (safe under eps={epsilon} perturbation)")
            else:
                total_violated += 1
                print(f"    => VIOLATED (safety can be broken)")

            results[f'image_{img_num+1}'] = {
                'dataset_index': int(img_idx),
                'CTE_true': round(cte_true, 6),
                'CTE_predicted': round(cte_pred, 6),
                'HE_predicted': round(he_pred, 6),
                'verified': image_all_unsat,
                'sub_queries': image_results,
            }

        print(f"\n  P3 (eps={epsilon}): {total_verified}/{P3_NUM_IMAGES} verified, "
              f"{total_violated}/{P3_NUM_IMAGES} violated")

        all_p3_results[f'epsilon_{epsilon}'] = {
            'property': 'Robustness Under Input Noise',
            'specification': f'|CTE| < {P3_CTE_BOUND} AND |HE| < {P3_HE_BOUND}',
            'epsilon': epsilon,
            'images_verified': total_verified,
            'images_violated': total_violated,
            'total_images': P3_NUM_IMAGES,
            'results': results,
        }

    return all_p3_results


# ============================================================
# P4: Directional Correctness
# ============================================================

def verify_p4(images, labels):
    """
    P4 — Directional Correctness

    C_left: per-pixel [min, max] from images with CTE > 2.0m
      Post: CTE > 0 (network correctly identifies positive lateral offset)
      Negation: CTE <= 0

    C_right: per-pixel [min, max] from images with CTE < -2.0m
      Post: CTE < 0 (network correctly identifies negative lateral offset)
      Negation: CTE >= 0

    Note: We use strict > 0 / < 0 in the postcondition. Since Marabou bounds
    are non-strict, the negation of CTE > 0 is CTE <= 0 (setUpperBound to 0.0),
    and the negation of CTE < 0 is CTE >= 0 (setLowerBound to 0.0).
    """
    print("\n" + "=" * 70)
    print("P4: DIRECTIONAL CORRECTNESS")
    print("=" * 70)

    cte = labels[:, 0]
    flat_images = images.reshape(-1, 128)

    results = {}

    # Empirical validation first
    from maraboupy import Marabou as Mb
    net_eval = Mb.read_nnet(NNET_PATH)

    for region_name, mask_condition, post_desc, out_var, direction, bound in [
        ('C_left',  cte > P4_LEFT_THRESHOLD,   'CTE > 0',  192, '<=', 0.0),
        ('C_right', cte < P4_RIGHT_THRESHOLD,  'CTE < 0',  192, '>=', 0.0),
    ]:
        mask = mask_condition
        region_imgs = flat_images[mask]
        region_cte = cte[mask]
        n_imgs = len(region_imgs)

        # Per-pixel bounds
        pixel_mins = region_imgs.min(axis=0)
        pixel_maxs = region_imgs.max(axis=0)

        print(f"\n  --- {region_name} ({n_imgs} images) ---")
        print(f"  CTE range in region: [{region_cte.min():.2f}, {region_cte.max():.2f}]")
        print(f"  Pixel range: [{pixel_mins.min():.4f}, {pixel_maxs.max():.4f}]")
        print(f"  Property: {post_desc}")
        print(f"  Negation: {'CTE <= 0' if direction == '<=' else 'CTE >= 0'}")

        # Empirical check: how many images in this region get the sign wrong?
        wrong_sign = 0
        for idx in np.where(mask)[0]:
            out = np.array(net_eval.evaluate([flat_images[idx]])).flatten()
            if direction == '<=' and out[0] <= 0:  # should be > 0 but isn't
                wrong_sign += 1
            elif direction == '>=' and out[0] >= 0:  # should be < 0 but isn't
                wrong_sign += 1
        print(f"  Empirical wrong-sign predictions: {wrong_sign}/{n_imgs}")

        # Verification
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()

        for i in range(128):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

        # Negated postcondition
        if direction == '<=':
            net.setUpperBound(192, float(bound))
        else:
            net.setLowerBound(192, float(bound))

        mem_before = get_peak_memory_mb()
        t_start = time.time()
        exit_code, vals, stats = net.solve()
        t_elapsed = time.time() - t_start
        mem_after = get_peak_memory_mb()

        status = exit_code.strip().lower() if isinstance(exit_code, str) else str(exit_code)

        result = {
            'region': region_name,
            'n_images': n_imgs,
            'cte_range': [float(region_cte.min()), float(region_cte.max())],
            'property': post_desc,
            'status': status,
            'time_seconds': round(t_elapsed, 4),
            'peak_memory_mb': round(mem_after, 1),
            'memory_delta_mb': round(mem_after - mem_before, 1),
            'empirical_wrong_sign': wrong_sign,
        }

        if status == 'sat':
            cte_val = vals[192]
            he_val = vals[193]
            result['counterexample'] = {
                'CTE': round(cte_val, 6),
                'HE': round(he_val, 6),
            }
            # Compute input mean for mean-centering caveat
            cx_input = [vals[input_vars[j]] for j in range(128)]
            result['counterexample_input_mean'] = round(float(np.mean(cx_input)), 6)

            print(f"  Result: SAT (VIOLATED) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
            print(f"  Counterexample: CTE={cte_val:.4f}, HE={he_val:.4f}")
            print(f"  Counterexample input mean: {np.mean(cx_input):.4f}")

            visualize_counterexample(vals, input_vars,
                f"P4 {region_name} Counterexample\nCTE={cte_val:.2f}m",
                f"counterexample_P4_{region_name}.png")
        elif status == 'unsat':
            print(f"  Result: UNSAT (VERIFIED) in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")
            print(f"  The network ALWAYS outputs {post_desc} for inputs in {region_name}")
        else:
            print(f"  Result: {status} in {t_elapsed:.2f}s, mem={mem_after:.0f}MB")

        results[region_name] = result

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print()
    print("FORMAL VERIFICATION OF TINYTAXINET USING MARABOU 2.0")
    print("Properties P3 (Robustness) and P4 (Directional Correctness)")
    print()

    images, labels = load_data()

    all_results = {
        'tool': 'Marabou 2.0 (maraboupy 2.0.0)',
        'network': NNET_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'properties': ['P3', 'P4'],
    }

    # P3
    p3_results = verify_p3(images, labels)
    all_results['P3'] = p3_results

    # P4
    p4_results = verify_p4(images, labels)
    all_results['P4'] = {
        'property': 'Directional Correctness',
        'results': p4_results,
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
    for eps in P3_EPSILONS:
        key = f'epsilon_{eps}'
        v = p3_results[key]['images_verified']
        t = p3_results[key]['total_images']
        print(f"  P3 (eps={eps}, safety): {v}/{t} verified")

    for region in ['C_left', 'C_right']:
        r = p4_results[region]
        print(f"  P4 ({region}): {r['status'].upper()}")
    print()
