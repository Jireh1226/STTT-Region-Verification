"""P3 control under random zero-mean 8D subspaces.

This script matches the local-PCA P3 construction in verify_p3_local_pca.py
except that the basis directions are random rather than data-derived.  For each
anchor it:

  1. Builds the same local-PCA object only to obtain the per-direction scale
     vector sigma_j and the same anchor/neighbor metadata.
  2. Draws a deterministic random orthonormal basis in the zero-mean pixel
     subspace, so the mean(x)=0.5 equality does not add an extra unfair
     restriction beyond the random 8D choice.
  3. Reuses the same Marabou encoding and P3 postconditions as the local-PCA
     experiment.

The control tests whether P3 improvement is caused merely by reducing the
solver to an 8D affine subspace, or whether the data-conditioned PCA directions
matter.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import verify_p3_local_pca as p3


OUTPUT_JSON = "verification_p3_random_subspace.json"


def random_zero_mean_basis(anchor_idx, seed, d=p3.D, n=p3.NUM_PIXELS):
    """Return d orthonormal row vectors in the zero-mean subspace."""
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(anchor_idx))
    cols = []
    for _ in range(d):
        v = rng.normal(size=n)
        v = v - v.mean()
        for q in cols:
            v = v - np.dot(v, q) * q
        norm = np.linalg.norm(v)
        # Degeneracy is astronomically unlikely; this keeps the script total.
        while norm < 1e-12:
            v = rng.normal(size=n)
            v = v - v.mean()
            for q in cols:
                v = v - np.dot(v, q) * q
            norm = np.linalg.norm(v)
        cols.append(v / norm)
    return np.vstack(cols)


def basis_checks(V):
    """Return (orthonormality_error, zero_mean_error) of a basis with row vectors.

    orthonormality_error = max|V V^T - I|; zero_mean_error = max|row mean|.
    """
    d = V.shape[0]
    ortho_err = float(np.max(np.abs(V @ V.T - np.eye(d))))
    zeromean_err = float(np.max(np.abs(V.mean(axis=1))))
    return ortho_err, zeromean_err


def _dist(times):
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
    ap = argparse.ArgumentParser(description="P3 random-subspace control matched to local-PCA scale.")
    ap.add_argument("--num-images", default="20",
                    help='number of deterministic centered anchors, or "all" (default: %(default)s)')
    ap.add_argument("--alphas", default="1.0,2.0",
                    help="comma-separated alpha multipliers (default: %(default)s)")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated random seeds (default: %(default)s)")
    ap.add_argument("--output", default=OUTPUT_JSON,
                    help="output JSON path (default: %(default)s)")
    ap.add_argument("--manifest", default=None,
                    help="optional manifest path for strict subsets")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    alphas_list = [float(a) for a in args.alphas.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    num_images_arg = args.num_images.strip()

    images, labels = p3.load_data()
    centered_all = p3.centered_indices(labels)
    n_centered = int(len(centered_all))
    sample = p3.select_centered_sample(labels, num_images_arg)
    sample_size = int(len(sample))
    is_subset = sample_size < n_centered

    selection_rule = (
        f"centered = ascending dataset indices with |CTE| < {p3.P3_CENTERED_THRESHOLD} "
        f"(n_centered={n_centered}); sample = centered[round(linspace(0, n_centered-1, "
        f"num_images))]; num_images='all' selects every centered image in order."
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
            "seeds": seeds,
        }, indent=2))

    sub_queries = [
        ("a", 192, ">=",  p3.P3_CTE_BOUND),
        ("b", 192, "<=", -p3.P3_CTE_BOUND),
        ("c", 193, ">=",  p3.P3_HE_BOUND),
        ("d", 193, "<=", -p3.P3_HE_BOUND),
    ]

    out = {
        "script": "verify_p3_random_subspace.py",
        "provenance": {
            "command": "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "nnet_path": p3.NNET_PATH,
            "data_path": p3.DATA_PATH,
            "timeout_seconds_per_subquery": p3.TIMEOUT_S,
            "n_centered_total": n_centered,
            "num_images_requested": num_images_arg,
            "sample_size": sample_size,
            "is_subset": is_subset,
            "manifest": manifest_path,
            "selection_rule": selection_rule,
            "control": (
                "Random zero-mean orthonormal 8D basis per anchor and seed; "
                "uses the same local-PCA sigma_proj scale vector, alpha list, "
                "mean invariant, clipping, property, thresholds, and solver semantics."
            ),
        },
        "d": p3.D,
        "alpha_values": alphas_list,
        "seeds": seeds,
        "sample_dataset_indices": [int(i) for i in sample],
        "basis_validation": {
            "tolerance": 1e-9,
            "worst_orthonormality_error": 0.0,
            "worst_zero_mean_error": 0.0,
            "n_bases_checked": 0,
            "all_within_tolerance": True,
        },
        "mean_invariant_check": {
            "anchor_mean_target": p3.SUM_TARGET / p3.NUM_PIXELS,
            "worst_abs_dev_from_target": None,
        },
        "results_by_seed_alpha": {},
        "aggregate_by_alpha": {},
        "per_image_scale_meta": [],
    }

    # One-time mean-invariant check on the anchors themselves (they must have
    # mean 0.5 for the zero-mean basis to keep mean(x)=0.5 non-binding).
    _mean_dev = 0.0
    for k, idx in enumerate(sample):
        _mean_dev = max(_mean_dev, abs(float(images[int(idx)].mean()) - p3.SUM_TARGET / p3.NUM_PIXELS))
    out["mean_invariant_check"]["worst_abs_dev_from_target"] = round(_mean_dev, 10)

    print(f"Building matched scale metadata for {sample_size} centered anchors...", flush=True)
    pca_infos = []
    for k, idx in enumerate(sample):
        info = p3.build_local_pca(images, labels, idx)
        if info is None:
            pca_infos.append(None)
            continue
        pca_infos.append(info)
        out["per_image_scale_meta"].append({
            "image_idx": int(idx),
            "n_neighbors": info["n_neighbors"],
            "pca_explained_variance_ratio_top_d": round(info["explained_variance_ratio_top_d"], 4),
            "sigma_proj": [round(float(s), 5) for s in info["sigma_proj"]],
        })

    bv = out["basis_validation"]
    for seed in seeds:
        print(f"=== random seed {seed} ===", flush=True)
        for alpha in alphas_list:
            print(f"  alpha={alpha} (n={sample_size})", flush=True)
            per_image = {}
            anchor_times = []
            key = f"seed_{seed}_alpha_{alpha}"
            entry = {
                "seed": seed, "alpha": alpha, "sample_size": sample_size,
                "status": "in_progress",
                "verified": 0, "sat": 0, "timeout": 0,
                "empty_neighbor": 0, "region_empty": 0, "other": 0,
                "sat_subquery_breakdown": {"cte_upper": 0, "cte_lower": 0,
                                           "he_upper": 0, "he_lower": 0},
                "elapsed_seconds": None,
                "runtime_distribution_seconds": None,
                "per_image": per_image,
            }
            out["results_by_seed_alpha"][key] = entry
            t_alpha = time.time()
            for k, idx in enumerate(sample):
                info = pca_infos[k]
                if info is None:
                    entry["empty_neighbor"] += 1
                    per_image[f"image_{k+1}"] = {
                        "dataset_index": int(idx),
                        "category": "empty_neighbor",
                        "verified": False,
                    }
                    continue
                V_random = random_zero_mean_basis(int(idx), seed)
                ortho_err, zeromean_err = basis_checks(V_random)
                bv["n_bases_checked"] += 1
                bv["worst_orthonormality_error"] = max(bv["worst_orthonormality_error"], ortho_err)
                bv["worst_zero_mean_error"] = max(bv["worst_zero_mean_error"], zeromean_err)
                if ortho_err > bv["tolerance"] or zeromean_err > bv["tolerance"]:
                    bv["all_within_tolerance"] = False
                alphas = alpha * info["sigma_proj"]
                sub_results = {}
                all_unsat = True
                for tag, var, direc, bound in sub_queries:
                    r = p3.solve_one(info["x0"], V_random, None, alphas, var, direc, bound)
                    sub_results[tag] = r
                    if r["status"] != "unsat":
                        all_unsat = False
                statuses = [sub_results[t]["status"] for t, *_ in sub_queries]
                # sub-query -> axis/direction: a=CTE upper, b=CTE lower, c=HE upper, d=HE lower
                sat_axes = [name for tag, name in (("a", "cte_upper"), ("b", "cte_lower"),
                                                   ("c", "he_upper"), ("d", "he_lower"))
                            if sub_results[tag]["status"] == "sat"]
                for ax in sat_axes:
                    entry["sat_subquery_breakdown"][ax] += 1
                if "region_empty_after_clip" in statuses:
                    category = "region_empty"
                    entry["region_empty"] += 1
                elif "sat" in statuses:
                    category = "sat"
                    entry["sat"] += 1
                elif "timeout" in statuses:
                    category = "timeout"
                    entry["timeout"] += 1
                elif all_unsat:
                    category = "verified"
                    entry["verified"] += 1
                else:
                    category = "other"
                    entry["other"] += 1
                anchor_solve_s = sum(sub_results[t].get("time_seconds", 0.0) for t, *_ in sub_queries)
                if category not in ("empty_neighbor", "region_empty"):
                    anchor_times.append(anchor_solve_s)
                per_image[f"image_{k+1}"] = {
                    "dataset_index": int(idx),
                    "category": category,
                    "verified": bool(all_unsat),
                    "sat_subqueries": sat_axes,
                    "alpha_scale_first_last": [round(float(alphas[0]), 5), round(float(alphas[-1]), 5)],
                    "basis_ortho_err": round(ortho_err, 12),
                    "basis_zero_mean_err": round(zeromean_err, 12),
                    "anchor_solve_seconds": round(anchor_solve_s, 4),
                    "sub_queries": sub_results,
                }
                if sample_size <= 60:
                    short = " ".join(s[:3] for s in statuses)
                    print(f"    img{k+1:>2} idx={idx} {category:<13} [{short}]", flush=True)
                elif (k + 1) % 50 == 0:
                    print(f"    ...{k+1}/{sample_size} done "
                          f"(verified={entry['verified']} sat={entry['sat']} "
                          f"timeout={entry['timeout']})", flush=True)
                    entry["runtime_distribution_seconds"] = _dist(anchor_times)
                    with open(args.output, "w") as f:
                        json.dump(out, f, indent=2)
            elapsed = round(time.time() - t_alpha, 2)
            entry["status"] = "complete"
            entry["elapsed_seconds"] = elapsed
            entry["runtime_distribution_seconds"] = _dist(anchor_times)
            verified, sat, timeout = entry["verified"], entry["sat"], entry["timeout"]
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"    => {verified}/{sample_size} verified, {sat} sat, "
                  f"{timeout} timeout ({elapsed:.1f}s)", flush=True)

    for alpha in alphas_list:
        rows = [r for r in out["results_by_seed_alpha"].values() if r["alpha"] == alpha]
        out["aggregate_by_alpha"][f"alpha_{alpha}"] = {
            "n_seeds": len(rows),
            "verified_min": min(r["verified"] for r in rows),
            "verified_median": statistics.median(r["verified"] for r in rows),
            "verified_max": max(r["verified"] for r in rows),
            "sat_min": min(r["sat"] for r in rows),
            "sat_median": statistics.median(r["sat"] for r in rows),
            "sat_max": max(r["sat"] for r in rows),
            "timeout_min": min(r["timeout"] for r in rows),
            "timeout_median": statistics.median(r["timeout"] for r in rows),
            "timeout_max": max(r["timeout"] for r in rows),
        }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[validation] bases checked={bv['n_bases_checked']} "
          f"worst_ortho_err={bv['worst_orthonormality_error']:.2e} "
          f"worst_zero_mean_err={bv['worst_zero_mean_error']:.2e} "
          f"within_tol={bv['all_within_tolerance']} | "
          f"anchor mean dev from 0.5={out['mean_invariant_check']['worst_abs_dev_from_target']}",
          flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
