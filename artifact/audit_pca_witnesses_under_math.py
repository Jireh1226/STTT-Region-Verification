"""
Post-hoc cross-check: for every PCA SAT witness (already extracted in
`pca_witness_scores.json`), compute its TV, Fourier-HF-L1, and photometric
residual L_inf. Compare each to the empirical centered-subset quantiles.

This addresses: "do PCA-region SAT witnesses also satisfy the math-encoded
regions, or are they non-physical along at least one of the math constraints?"

A witness that is "math-region-feasible" (TV <= centered max, Fourier-HF-L1 <=
centered max, residual <= centered max) is one that no math constraint would
reject; the only thing flagging it is the AE/PCA support model. A witness that
violates at least one math constraint is "math-rejected" — the math regions
would catch it independent of the PCA classifier.

Output: `pca_witnesses_math_audit.json` with per-witness math-feasibility data
and a summary comparison.
"""
import json
import numpy as np
from datetime import datetime, timezone

NUM_PIXELS = 128

# Empirical math-region quantiles from the centered subset (n=893)
# These come from the calibration runs already in the three math-region scripts.
CENTERED_TV_MAX = 12.647794
CENTERED_TV_P99 = 12.184696
CENTERED_TV_MEDIAN = 9.374
CENTERED_TV_MIN = 6.033

CENTERED_FOURIER_HF_L1_MAX = 8.156757
CENTERED_FOURIER_HF_L1_P99 = 7.917473
CENTERED_FOURIER_HF_L1_MIN = 4.531346

CENTERED_PHOTO_RESIDUAL_MAX = 0.262071
CENTERED_PHOTO_RESIDUAL_P99 = 0.249737
CENTERED_PHOTO_RESIDUAL_MEDIAN = 0.207994
CENTERED_PHOTO_RESIDUAL_MIN = 0.158279


def tv(img_flat: np.ndarray) -> float:
    """Total variation of an 8x16 image (flattened to 128)."""
    img = img_flat.reshape(8, 16)
    h_diff = np.abs(np.diff(img, axis=1))
    v_diff = np.abs(np.diff(img, axis=0))
    return float(h_diff.sum() + v_diff.sum())


def fourier_hf_l1(img_flat: np.ndarray) -> float:
    """Sum of |Re| + |Im| over high-frequency Fourier coefficients (k>=4 or l>=8)."""
    img = img_flat.reshape(8, 16)
    # 2D DFT with 1/sqrt(N) normalisation to match the verification scripts
    C = np.fft.fft2(img) / np.sqrt(128.0)
    total = 0.0
    for k in range(8):
        for l in range(16):
            if k >= 4 or l >= 8:
                total += abs(C[k, l].real) + abs(C[k, l].imag)
    return float(total)


def photometric_residual_linf(img_flat: np.ndarray) -> float:
    """L_inf of the residual after projecting onto (uniform, horizontal-gradient) basis,
    using L_inf-best-fit projection (matching what Marabou would find)."""
    from scipy.optimize import linprog
    B1 = np.ones(128) / np.sqrt(128.0)
    B2_raw = np.array([(j - 7.5) for i in range(8) for j in range(16)], dtype=float)
    B2 = B2_raw / np.linalg.norm(B2_raw)
    # Solve: minimize t s.t. for each i, x_i - alpha*B1[i] - beta*B2[i] <= t AND
    #                                    -(x_i - alpha*B1[i] - beta*B2[i]) <= t.
    n = 128
    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []
    for i in range(n):
        A_ub.append([-B1[i], -B2[i], -1.0])
        b_ub.append(-img_flat[i])
        A_ub.append([B1[i], B2[i], -1.0])
        b_ub.append(img_flat[i])
    bounds = [(None, None), (None, None), (0, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return float(r.fun) if r.success else float('nan')


def main():
    print("Loading PCA witness scores...", flush=True)
    with open('pca_witness_scores.json') as f:
        scores = json.load(f)

    witnesses_data = []
    for w in scores['witnesses']:
        if w.get('status') != 'sat':
            continue
        if 'witness' not in w:
            # Witness not stored in this JSON; we re-extracted them post-hoc
            # and only kept summary stats. Skip.
            print(f"  Skipping {w.get('image_key')} sub={w.get('sub_query')} (no witness)", flush=True)
            continue
        x = np.array(w['witness'])
        tv_val = tv(x)
        fhf = fourier_hf_l1(x)
        photo = photometric_residual_linf(x)
        rec = {
            "image_key": w['image_key'],
            "dataset_index": w['dataset_index'],
            "sub_query": w['sub_query'],
            "alpha_sigma": w['alpha_sigma'],
            "tv": round(tv_val, 4),
            "tv_vs_centered_max": round(tv_val - CENTERED_TV_MAX, 4),
            "fourier_hf_l1": round(fhf, 4),
            "fourier_hf_l1_vs_centered_max": round(fhf - CENTERED_FOURIER_HF_L1_MAX, 4),
            "photo_residual_linf": round(photo, 4),
            "photo_residual_vs_centered_max": round(photo - CENTERED_PHOTO_RESIDUAL_MAX, 4),
            "rejected_by_tv": bool(tv_val > CENTERED_TV_MAX),
            "rejected_by_fourier": bool(fhf > CENTERED_FOURIER_HF_L1_MAX),
            "rejected_by_photometric": bool(photo > CENTERED_PHOTO_RESIDUAL_MAX),
            "math_feasible": bool(
                tv_val <= CENTERED_TV_MAX
                and fhf <= CENTERED_FOURIER_HF_L1_MAX
                and photo <= CENTERED_PHOTO_RESIDUAL_MAX
            ),
            "classifier_prob_real": w.get('classifier_prob_real'),
            "l2_to_nearest_real": w.get('l2_to_nearest_real'),
        }
        witnesses_data.append(rec)
        print(
            f"  {w['image_key']} sub={w['sub_query']} alpha={w['alpha_sigma']}: "
            f"TV={tv_val:.2f} (vs max {CENTERED_TV_MAX:.2f}), "
            f"FHF-L1={fhf:.2f} (vs max {CENTERED_FOURIER_HF_L1_MAX:.2f}), "
            f"photo-r={photo:.3f} (vs max {CENTERED_PHOTO_RESIDUAL_MAX:.3f}); "
            f"math-feasible={rec['math_feasible']}",
            flush=True,
        )

    # Summary
    n = len(witnesses_data)
    n_math_feasible = sum(1 for w in witnesses_data if w['math_feasible'])
    n_rej_tv = sum(1 for w in witnesses_data if w['rejected_by_tv'])
    n_rej_fourier = sum(1 for w in witnesses_data if w['rejected_by_fourier'])
    n_rej_photo = sum(1 for w in witnesses_data if w['rejected_by_photometric'])
    n_rej_any = sum(1 for w in witnesses_data
                    if w['rejected_by_tv'] or w['rejected_by_fourier'] or w['rejected_by_photometric'])

    summary = {
        "n_witnesses_audited": n,
        "n_math_feasible": n_math_feasible,
        "n_rejected_by_tv": n_rej_tv,
        "n_rejected_by_fourier": n_rej_fourier,
        "n_rejected_by_photometric": n_rej_photo,
        "n_rejected_by_at_least_one": n_rej_any,
        "centered_calibration": {
            "tv_max": CENTERED_TV_MAX,
            "fourier_hf_l1_max": CENTERED_FOURIER_HF_L1_MAX,
            "photometric_residual_linf_max": CENTERED_PHOTO_RESIDUAL_MAX,
        },
    }

    out = {
        "provenance": {
            "script": "audit_pca_witnesses_under_math.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "witnesses": witnesses_data,
    }
    with open('pca_witnesses_math_audit.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"PCA SAT witnesses audited: {n}", flush=True)
    print(f"  rejected by TV (TV > centered max {CENTERED_TV_MAX:.2f}): {n_rej_tv}", flush=True)
    print(f"  rejected by Fourier HF L1 (> {CENTERED_FOURIER_HF_L1_MAX:.2f}): {n_rej_fourier}", flush=True)
    print(f"  rejected by photometric (residual > {CENTERED_PHOTO_RESIDUAL_MAX:.3f}): {n_rej_photo}", flush=True)
    print(f"  rejected by at least one math constraint: {n_rej_any}", flush=True)
    print(f"  math-feasible (passes all three): {n_math_feasible}", flush=True)

    print("\nSaved pca_witnesses_math_audit.json", flush=True)


if __name__ == "__main__":
    main()
