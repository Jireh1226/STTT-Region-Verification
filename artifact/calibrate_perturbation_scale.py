"""
Empirical calibration of the L_inf perturbation scale, using only the
post-preprocessed VerifyGAN dataset (no raw images required).

We compute four families of statistics on the 10000-image dataset to anchor
the choice of epsilon = {0.02, 0.05, 0.10}:

  1. Per-pixel standard deviation within label-conditioned subsets
     (centered, left, right). Reports min/median/max across the 128 pixels.

  2. Within-bin L_inf distance: for each image, find the closest other
     image whose CTE label is within +/- 0.5 m and HE label within +/- 5 deg
     (a "same-state" neighbor), and report the L_inf distance distribution.

  3. Same-position frame variation. For images that share the same (CTE, HE)
     label to within 1e-3 (i.e. multiple renders of the same simulator state),
     compute pairwise L_inf and report the distribution.

  4. Random-pair baseline. Sample 1000 random image pairs and report the
     L_inf distribution. This is the loose upper end (different states,
     different appearance).

Result: a calibration table that lets us state things like
  "epsilon = 0.02 is below the 10th percentile of same-state nearest-neighbor
   distances, while epsilon = 0.10 is in the lower tail of random-pair
   distances."
"""
import json

import h5py
import numpy as np


DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "perturbation_scale_calibration.json"

CTE_BIN_HALF = 0.5
HE_BIN_HALF = 5.0
SAME_POS_TOL = 1e-3
RANDOM_PAIR_SEED = 42
RANDOM_PAIR_N = 1000


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def percentiles(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def per_pixel_std(images, mask, name):
    sub = images[mask]
    stds = sub.std(axis=0)
    return {
        "subset": name,
        "n_images": int(mask.sum()),
        "per_pixel_std_min": float(stds.min()),
        "per_pixel_std_median": float(np.median(stds)),
        "per_pixel_std_max": float(stds.max()),
        "per_pixel_std_mean": float(stds.mean()),
    }


def label_neighbor_linf(images, labels, mask, name, max_query=400, seed=0):
    """For up to max_query images in the masked subset, find the nearest
    other image (excluding the query itself) whose label is within
    +-CTE_BIN_HALF in CTE and +-HE_BIN_HALF in HE, and return the L_inf
    distance to that nearest same-state neighbor."""
    rng = np.random.default_rng(seed)
    cte = labels[:, 0]
    he = labels[:, 1]
    sub_idx = np.where(mask)[0]
    queries = sub_idx if len(sub_idx) <= max_query else \
        rng.choice(sub_idx, size=max_query, replace=False)

    distances = []
    for qi in queries:
        cte_q, he_q = cte[qi], he[qi]
        bin_mask = (np.abs(cte - cte_q) <= CTE_BIN_HALF) & \
                   (np.abs(he - he_q) <= HE_BIN_HALF)
        bin_mask[qi] = False
        if not bin_mask.any():
            continue
        diffs = np.max(np.abs(images[bin_mask] - images[qi]), axis=1)
        distances.append(float(diffs.min()))
    if not distances:
        return {"subset": name, "n_queries": 0}
    arr = np.array(distances)
    return {
        "subset": name,
        "n_queries": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        **percentiles(arr),
    }


def same_position_pair_linf(images, labels, max_pairs=500, seed=0):
    """Find images that share (CTE, HE) labels to within SAME_POS_TOL,
    and report the pairwise L_inf distribution. This is the closest analogue
    we have to 'frame-to-frame variation at the same simulator position'."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    # Bucket by quantized (CTE, HE) at SAME_POS_TOL resolution
    keys = np.round(labels[:, :2] / SAME_POS_TOL).astype(np.int64)
    keys = [tuple(k) for k in keys]
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    pairs_buckets = [v for v in buckets.values() if len(v) >= 2]
    if not pairs_buckets:
        return {"note": "no buckets with >=2 members at this tolerance"}

    distances = []
    while pairs_buckets and len(distances) < max_pairs:
        b = pairs_buckets[rng.integers(0, len(pairs_buckets))]
        i, j = rng.choice(b, size=2, replace=False)
        d = float(np.max(np.abs(images[i] - images[j])))
        distances.append(d)
    arr = np.array(distances)
    return {
        "n_buckets_with_pairs": len(pairs_buckets),
        "n_pairs_sampled": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        **percentiles(arr),
    }


def random_pair_linf(images, n=RANDOM_PAIR_N, seed=RANDOM_PAIR_SEED):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(images), size=n)
    b = rng.integers(0, len(images), size=n)
    distances = np.max(np.abs(images[a] - images[b]), axis=1)
    return {
        "n_pairs": int(n),
        "min": float(distances.min()),
        "max": float(distances.max()),
        "mean": float(distances.mean()),
        **percentiles(distances),
    }


def fraction_below(eps, distances):
    return float((np.array(distances) < eps).mean()) if len(distances) else None


def main():
    images, labels = load_data()
    cte = labels[:, 0]

    masks = {
        "centered":   np.abs(cte) < 1.0,
        "left":       cte > 2.0,
        "right":      cte < -2.0,
        "off_center": np.abs(cte) > 1.0,
    }

    results = {}

    print("Per-pixel std within subsets (intra-subset variation):")
    pp = []
    for name, m in masks.items():
        s = per_pixel_std(images, m, name)
        pp.append(s)
        print(f"  {name:<10} n={s['n_images']:>5}  "
              f"std min/med/max = {s['per_pixel_std_min']:.4f} / "
              f"{s['per_pixel_std_median']:.4f} / "
              f"{s['per_pixel_std_max']:.4f}")
    results["per_pixel_std"] = pp

    print("\nNearest same-state neighbor L_inf distance "
          "(|dCTE|<=0.5, |dHE|<=5):")
    nn = []
    for name in ["centered", "left", "right"]:
        s = label_neighbor_linf(images, labels, masks[name], name,
                                max_query=400, seed=0)
        nn.append(s)
        if s.get("n_queries"):
            print(f"  {name:<10} n_queries={s['n_queries']:>4}  "
                  f"p10/median/p90 = {s['p10']:.4f} / {s['p50']:.4f} / "
                  f"{s['p90']:.4f}  max={s['max']:.4f}")
    results["nearest_state_neighbor"] = nn

    print("\nSame-position pairs (CTE, HE matched to 1e-3):")
    sp = same_position_pair_linf(images, labels)
    if "note" in sp:
        print(f"  {sp['note']}")
    else:
        print(f"  buckets={sp['n_buckets_with_pairs']}, "
              f"pairs sampled={sp['n_pairs_sampled']}, "
              f"p10/median/p90 = {sp['p10']:.4f} / {sp['p50']:.4f} / "
              f"{sp['p90']:.4f}")
    results["same_position_pairs"] = sp

    print("\nRandom-pair baseline:")
    rp = random_pair_linf(images)
    print(f"  n={rp['n_pairs']}, "
          f"p10/median/p90 = {rp['p10']:.4f} / {rp['p50']:.4f} / "
          f"{rp['p90']:.4f}")
    results["random_pairs"] = rp

    # Where does each epsilon sit?
    summary = {}
    print("\n\nWhere does each epsilon sit in these distributions?")
    print(f"{'ε':>6}  {'< NN p50 (centered)':>22}  {'< same-pos p50':>16}  "
          f"{'< random-pair p50':>20}")
    centered_nn = next((x for x in nn if x.get("subset") == "centered"
                        and "p50" in x), None)
    for eps in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
        anchors = {}
        if centered_nn:
            anchors["below_centered_nn_p50"] = bool(eps < centered_nn["p50"])
            anchors["below_centered_nn_p10"] = bool(eps < centered_nn["p10"])
        if "p50" in sp:
            anchors["below_same_position_p50"] = bool(eps < sp["p50"])
        anchors["below_random_pair_p50"] = bool(eps < rp["p50"])
        anchors["below_random_pair_p10"] = bool(eps < rp["p10"])
        summary[f"eps_{eps}"] = anchors

        line = f"{eps:>6.3f}  "
        line += f"{str(anchors.get('below_centered_nn_p50','?')):>22}  "
        line += f"{str(anchors.get('below_same_position_p50','?')):>16}  "
        line += f"{str(anchors.get('below_random_pair_p50','?')):>20}"
        print(line)
    results["eps_anchoring"] = summary

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
