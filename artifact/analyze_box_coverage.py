"""
Analyze how tightly the per-pixel input boxes cover the true input distribution.

Metrics computed:
  1. Per-pixel range width distribution (per-pixel box vs. per-pixel std)
  2. Volume ratio: per-pixel box vs. bounding ellipsoid (via std product)
  3. Density: fraction of box reachable by any training image (k-NN coverage)
  4. Relation to P3 epsilon ball: per-pixel box half-width vs. epsilon
"""
import numpy as np
import h5py

DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'

def load():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]
        labels = f['X_train'][:]
    return images.reshape(-1, 128), labels


def analyze_region(images, labels, mask, name):
    imgs = images[mask]
    N = len(imgs)
    print(f"\n=== {name} (N={N}) ===")

    pixel_min = imgs.min(axis=0)
    pixel_max = imgs.max(axis=0)
    pixel_mean = imgs.mean(axis=0)
    pixel_std = imgs.std(axis=0)

    box_width = pixel_max - pixel_min
    half_width = box_width / 2

    print(f"Per-pixel box half-width stats:")
    print(f"  min:    {half_width.min():.4f}")
    print(f"  max:    {half_width.max():.4f}")
    print(f"  mean:   {half_width.mean():.4f}")
    print(f"  median: {np.median(half_width):.4f}")

    print(f"Per-pixel std stats:")
    print(f"  mean std:   {pixel_std.mean():.4f}")
    print(f"  median std: {np.median(pixel_std):.4f}")

    # Ratio: box half-width vs std (Gaussian would be ~2-3 std for outliers)
    ratio = half_width / (pixel_std + 1e-9)
    print(f"Box half-width / std ratio (how many std does the box extend):")
    print(f"  min:    {ratio.min():.2f}")
    print(f"  max:    {ratio.max():.2f}")
    print(f"  mean:   {ratio.mean():.2f}")
    print(f"  median: {np.median(ratio):.2f}")

    # Box volume in log-space (for 128D, actual volume underflows)
    log_vol = np.sum(np.log(box_width + 1e-9))
    log_ellipsoid_vol = np.sum(np.log(pixel_std * 2 * np.sqrt(3) + 1e-9))  # uniform-equivalent
    print(f"log(box volume):              {log_vol:.2f}")
    print(f"log(uniform-equivalent vol):  {log_ellipsoid_vol:.2f}")
    print(f"log-ratio (box / uniform):    {log_vol - log_ellipsoid_vol:.2f}")

    # How many images are "corner-like" (close to box corners)?
    # Normalized distance to box corner: (x - center) / half_width in [-1, 1]
    # An image is corner-like if many of its pixels are near ±1
    center = (pixel_min + pixel_max) / 2
    norm = (imgs - center) / (half_width + 1e-9)
    # Per-image: fraction of pixels within 20% of box edge
    frac_at_edge = (np.abs(norm) > 0.8).mean(axis=1)
    print(f"Fraction of images with >50% pixels near box edge: {(frac_at_edge > 0.5).mean()*100:.1f}%")
    print(f"Mean per-image edge-pixel fraction: {frac_at_edge.mean()*100:.1f}%")

    # For P3 context: compare half-width to epsilon radii
    for eps in [0.02, 0.05, 0.10]:
        # Per-pixel box is much wider than ε-ball around any single image
        mean_hw = half_width.mean()
        print(f"  eps={eps}: per-pixel box half-width ({mean_hw:.4f}) / eps = {mean_hw/eps:.1f}x")

    return {
        'N': N, 'half_width_median': float(np.median(half_width)),
        'std_median': float(np.median(pixel_std)),
        'ratio_median': float(np.median(ratio)),
        'log_vol': float(log_vol),
        'log_uniform_vol': float(log_ellipsoid_vol),
        'edge_frac_mean': float(frac_at_edge.mean()),
    }


def main():
    images, labels = load()
    cte = labels[:, 0]

    centered = np.abs(cte) < 1.0
    left = cte > 2.0
    right = cte < -2.0
    off_center = np.abs(cte) > 1.0

    results = {}
    results['centered'] = analyze_region(images, labels, centered, 'C_centered (|CTE|<1.0)')
    results['left'] = analyze_region(images, labels, left, 'C_left (CTE>2.0)')
    results['right'] = analyze_region(images, labels, right, 'C_right (CTE<-2.0)')
    results['off_center'] = analyze_region(images, labels, off_center, 'C_off_center (|CTE|>1.0)')

    print("\n\n=== SUMMARY (for report) ===")
    print(f"{'Region':<20} {'N':<6} {'hw median':<12} {'box/std':<10} {'log vol ratio':<15}")
    for name, r in results.items():
        lv_ratio = r['log_vol'] - r['log_uniform_vol']
        print(f"{name:<20} {r['N']:<6} {r['half_width_median']:<12.4f} {r['ratio_median']:<10.2f} {lv_ratio:<15.2f}")


if __name__ == '__main__':
    main()
