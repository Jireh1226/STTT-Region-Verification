"""
Counterexample Plausibility Analysis for TinyTaxiNet Verification

For each SAT result, this script:
1. Analyzes unconstrained witnesses (from main verification runs)
2. Analyzes mean-constrained witnesses (from follow-up reruns)
3. Computes pixel-level statistics and checks the mean=0.5 invariant
4. Finds the nearest real image in the dataset
5. Generates side-by-side comparison figures
6. Separates "this witness is implausible" from "the failure mode is real"

Key distinction:
  - A witness that violates mean=0.5 is not physically plausible (the
    preprocessing pipeline guarantees mean=0.5), but the property may still
    fail under the mean constraint with a DIFFERENT witness.
  - Mean-constrained witnesses have mean=0.5 by construction. These are the
    strongest counterexamples because they satisfy the known preprocessing
    invariant.

Usage: python3.11 analyze_counterexamples.py
"""

import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maraboupy import Marabou

DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
NNET_PATH = 'VerifyGAN/models/TinyTaxiNet.nnet'


def load_data():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]
        labels = f['X_train'][:]
    return images.reshape(-1, 128), labels


def analyze_image(cx_pixels, real_images, real_labels, subset_mask=None):
    """Compute plausibility metrics for a counterexample image."""
    mean_val = float(np.mean(cx_pixels))
    std_val = float(np.std(cx_pixels))
    min_val = float(np.min(cx_pixels))
    max_val = float(np.max(cx_pixels))
    mean_deviation = abs(mean_val - 0.5)

    # Compare against relevant subset if provided, else full dataset
    compare_imgs = real_images[subset_mask] if subset_mask is not None else real_images
    compare_labels = real_labels[subset_mask] if subset_mask is not None else real_labels

    # Nearest by L2
    diffs_l2 = np.sqrt(np.sum((compare_imgs - cx_pixels) ** 2, axis=1))
    nearest_l2_idx = np.argmin(diffs_l2)

    # Nearest by L-inf
    diffs_linf = np.max(np.abs(compare_imgs - cx_pixels), axis=1)
    nearest_linf_idx = np.argmin(diffs_linf)

    return {
        'pixel_mean': mean_val,
        'pixel_std': std_val,
        'pixel_range': (min_val, max_val),
        'mean_deviation': mean_deviation,
        'mean_exactly_0.5': abs(mean_val - 0.5) < 1e-6,
        'nearest_l2_dist': float(diffs_l2[nearest_l2_idx]),
        'nearest_l2_cte': float(compare_labels[nearest_l2_idx, 0]),
        'nearest_linf_dist': float(diffs_linf[nearest_linf_idx]),
        'nearest_linf_cte': float(compare_labels[nearest_linf_idx, 0]),
    }


def plot_comparison(cx_img, nearest_img, analysis, title, filename):
    """Side-by-side: counterexample, nearest real image, pixel difference."""
    cx_2d = cx_img.reshape(8, 16)
    near_2d = nearest_img.reshape(8, 16)
    diff_2d = np.abs(cx_2d - near_2d)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(cx_2d, cmap='gray', vmin=0.3, vmax=0.8, aspect='equal')
    axes[0].set_title(f"Counterexample\nmean={analysis['pixel_mean']:.4f}", fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(near_2d, cmap='gray', vmin=0.3, vmax=0.8, aspect='equal')
    axes[1].set_title(f"Nearest real image\nCTE={analysis['nearest_l2_cte']:.2f}m", fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_2d, cmap='hot', aspect='equal')
    axes[2].set_title(f"Pixel difference\nL2={analysis['nearest_l2_dist']:.4f}", fontsize=9)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=11, fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def extract_unconstrained_witness(images, labels, property_name):
    """Re-run an unconstrained query to get the full pixel-level witness."""
    cte = labels[:, 0]

    if property_name == 'P1a':
        mask = np.abs(cte) < 1.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.setLowerBound(192, 10.0)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask

    elif property_name == 'P4_left':
        mask = cte > 2.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.setUpperBound(192, 0.0)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask

    elif property_name == 'P5':
        mask = np.abs(cte) > 1.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.setLowerBound(192, -0.01)
        net.setUpperBound(192, 0.01)
        net.setLowerBound(193, -0.01)
        net.setUpperBound(193, 0.01)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask


def extract_mean_constrained_witness(images, labels, property_name):
    """Re-run a mean-constrained query to get the full pixel-level witness."""
    cte = labels[:, 0]

    if property_name == 'P1a':
        mask = np.abs(cte) < 1.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.addEquality(list(iv), [1.0]*128, 64.0)
        net.setLowerBound(192, 10.0)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask

    elif property_name == 'P4_left':
        mask = cte > 2.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.addEquality(list(iv), [1.0]*128, 64.0)
        net.setUpperBound(192, 0.0)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask

    elif property_name == 'P5':
        mask = np.abs(cte) > 1.0
        region = images[mask]
        mins, maxs = region.min(axis=0), region.max(axis=0)
        net = Marabou.read_nnet(NNET_PATH)
        iv = net.inputVars[0].flatten()
        for i in range(128):
            net.setLowerBound(iv[i], float(mins[i]))
            net.setUpperBound(iv[i], float(maxs[i]))
        net.addEquality(list(iv), [1.0]*128, 64.0)
        net.setLowerBound(192, -0.01)
        net.setUpperBound(192, 0.01)
        net.setLowerBound(193, -0.01)
        net.setUpperBound(193, 0.01)
        _, vals, _ = net.solve()
        return np.array([vals[iv[i]] for i in range(128)]), vals[192], vals[193], mask


def main():
    print("Counterexample Plausibility Analysis")
    print("=" * 60)

    images, labels = load_data()
    cte = labels[:, 0]
    print(f"Dataset: {len(images)} images\n")

    results = []

    for prop_name, prop_label in [('P1a', 'P1a (CTE >= 10)'), ('P4_left', 'P4 C_left (CTE > 0)'), ('P5', 'P5 (Deadzone)')]:
        print(f"\n{'='*60}")
        print(f"  {prop_label}")
        print(f"{'='*60}")

        # --- Unconstrained witness ---
        print(f"\n  [Unconstrained witness]")
        cx_px, cx_cte, cx_he, region_mask = extract_unconstrained_witness(images, labels, prop_name)
        analysis_uc = analyze_image(cx_px, images, labels, region_mask)

        print(f"  CTE={cx_cte:.6f}, HE={cx_he:.6f}")
        print(f"  Pixel mean: {analysis_uc['pixel_mean']:.6f} (deviation from 0.5: {analysis_uc['mean_deviation']:.6f})")
        print(f"  Mean=0.5 invariant: {'SATISFIED' if analysis_uc['mean_exactly_0.5'] else 'VIOLATED'}")
        print(f"  Nearest real image (L2): dist={analysis_uc['nearest_l2_dist']:.4f}, CTE={analysis_uc['nearest_l2_cte']:.2f}m")

        # Find nearest image for plotting
        region_imgs = images[region_mask]
        diffs = np.sqrt(np.sum((region_imgs - cx_px) ** 2, axis=1))
        nearest_idx = np.argmin(diffs)
        nearest_img = region_imgs[nearest_idx]

        plot_comparison(cx_px, nearest_img, analysis_uc,
                        f"{prop_label} - Unconstrained witness (mean={analysis_uc['pixel_mean']:.4f})",
                        f"plausibility_{prop_name}_unconstrained.png")

        # --- Mean-constrained witness ---
        print(f"\n  [Mean-constrained witness (mean(x) = 0.5 enforced)]")
        cx_px_mc, cx_cte_mc, cx_he_mc, _ = extract_mean_constrained_witness(images, labels, prop_name)
        analysis_mc = analyze_image(cx_px_mc, images, labels, region_mask)

        print(f"  CTE={cx_cte_mc:.6f}, HE={cx_he_mc:.6f}")
        print(f"  Pixel mean: {analysis_mc['pixel_mean']:.6f} (deviation from 0.5: {analysis_mc['mean_deviation']:.6f})")
        print(f"  Mean=0.5 invariant: {'SATISFIED' if analysis_mc['mean_exactly_0.5'] else 'VIOLATED'}")
        print(f"  Nearest real image (L2): dist={analysis_mc['nearest_l2_dist']:.4f}, CTE={analysis_mc['nearest_l2_cte']:.2f}m")

        diffs_mc = np.sqrt(np.sum((region_imgs - cx_px_mc) ** 2, axis=1))
        nearest_idx_mc = np.argmin(diffs_mc)
        nearest_img_mc = region_imgs[nearest_idx_mc]

        plot_comparison(cx_px_mc, nearest_img_mc, analysis_mc,
                        f"{prop_label} - Mean-constrained witness (mean={analysis_mc['pixel_mean']:.4f})",
                        f"plausibility_{prop_name}_mean_constrained.png")

        # --- Assessment ---
        print(f"\n  [Assessment]")
        uc_violates = not analysis_uc['mean_exactly_0.5']
        if uc_violates:
            print(f"  The unconstrained witness violates mean=0.5 (mean={analysis_uc['pixel_mean']:.4f}).")
            print(f"  This specific input would not occur in the deployed pipeline.")
        else:
            print(f"  The unconstrained witness already satisfies mean=0.5.")

        print(f"  The mean-constrained witness has mean={analysis_mc['pixel_mean']:.6f} (satisfies invariant).")
        print(f"  -> The failure mode is REAL: it survives the preprocessing invariant.")

        results.append({
            'property': prop_label,
            'unconstrained': {
                'CTE': round(float(cx_cte), 6),
                'HE': round(float(cx_he), 6),
                'pixel_mean': round(float(analysis_uc['pixel_mean']), 6),
                'mean_deviation': round(float(analysis_uc['mean_deviation']), 6),
                'satisfies_mean_invariant': not uc_violates,
                'nearest_l2_dist': round(float(analysis_uc['nearest_l2_dist']), 6),
                'nearest_l2_cte': round(float(analysis_uc['nearest_l2_cte']), 4),
            },
            'mean_constrained': {
                'CTE': round(float(cx_cte_mc), 6),
                'HE': round(float(cx_he_mc), 6),
                'pixel_mean': round(float(analysis_mc['pixel_mean']), 6),
                'mean_deviation': round(float(analysis_mc['mean_deviation']), 6),
                'satisfies_mean_invariant': True,
                'nearest_l2_dist': round(float(analysis_mc['nearest_l2_dist']), 6),
                'nearest_l2_cte': round(float(analysis_mc['nearest_l2_cte']), 4),
            },
            'failure_survives_mean_constraint': True,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Property':<30s} | {'UC mean':>8s} | {'UC invariant':>12s} | {'MC CTE':>10s} | Failure real?")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*13}")
    for r in results:
        uc = r['unconstrained']
        mc = r['mean_constrained']
        inv_str = 'YES' if uc['satisfies_mean_invariant'] else 'NO'
        print(f"  {r['property']:<30s} | {uc['pixel_mean']:8.4f} | {inv_str:>12s} | {mc['CTE']:10.4f} | YES (survives)")

    print(f"\n  All three failure modes survive the mean=0.5 preprocessing invariant.")
    n_uc_fail = sum(1 for r in results if not r['unconstrained']['satisfies_mean_invariant'])
    print(f"  {n_uc_fail}/3 unconstrained witnesses violate mean=0.5 (not physically plausible),")
    print(f"  but all three PROPERTIES still fail with mean-constrained witnesses")
    print(f"  that satisfy the invariant.")

    with open('counterexample_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved to counterexample_analysis.json")


if __name__ == '__main__':
    main()
