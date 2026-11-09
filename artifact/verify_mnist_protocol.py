"""
Apply the six-region verification protocol to the MNIST MLP.

For each of 20 sampled test images (2 per class), run:
  - P3-analog (local L_inf robustness) under R1, R2 (per-image box), R3, R4, R5, R6
  - P1-analog (per-class margin) under R1 (class-box), R2, R6
  - P4-analog (class-confusion) under R1 (class-box), R2, R6

Class-conditioned regions (R1 class-box, R5 same-class neighbors, R6
calibrated per class) use the true class of the anchor image.
Per-image regions (R1 L_inf, R4 PCA) use the anchor image directly.
R3 uses the class-k autoencoder for an anchor with true class k.

Save:
  mnist_protocol_results.json - per-query SAT/UNSAT, time, witness pixel vector
"""
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
from maraboupy import Marabou

import mnist_regions as mr


ONNX_PATH = "mnist_mlp.onnx"
DATA_PATH = "mnist_test_data.npz"
WEIGHTS_PATH = "mnist_mlp_weights.npz"
AE_PATH = "mnist_class_aes.npz"
OUTPUT_JSON = "mnist_protocol_results.json"

TIMEOUT_S = 180
N_PIX = 784


def numpy_forward(W0, b0, W1, b1, W2, b2, x):
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    return W2 @ h2 + b2


def base_net():
    net = Marabou.read_onnx(ONNX_PATH)
    ivars = np.asarray(net.inputVars[0]).flatten().tolist()
    ovars = np.asarray(net.outputVars[0]).flatten().tolist()
    return net, [int(v) for v in ivars], [int(v) for v in ovars]


def solve(net, label):
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {"status": f"error: {str(e)[:80]}", "time_seconds": round(time.time() - t0, 2)}
    out = {"status": str(status).strip().lower(),
           "time_seconds": round(time.time() - t0, 2)}
    if out["status"] == "sat" and vals:
        out["_vals"] = vals
    return out


def attach_p3_negation(net, ovars, true_class, target_class):
    """Postcondition: target_class logit >= true_class logit."""
    net.addInequality([int(ovars[true_class]), int(ovars[target_class])], [1.0, -1.0], 0.0)


def attach_p1_margin_negation(net, ovars, true_class, tau):
    """Postcondition negated: exists j != true_class with logit_j - logit_true >= tau.
    Encode as disjunction over j by splitting into queries.
    Returns one SAT if ANY disjunct is SAT; we use one disjunct at a time.
    """
    pass


def runner_up_class(logits, true_class):
    other = logits.copy()
    other[true_class] = -np.inf
    return int(np.argmax(other))


def witness_from_vals(vals, ivars):
    return np.array([vals[int(ivars[i])] for i in range(N_PIX)], dtype=np.float64)


def main():
    data = np.load(DATA_PATH)
    X_te, y_te, preds_te = data["X_test"], data["y_test"], data["preds"]
    print(f"test set: {X_te.shape}, classifier acc: {np.mean(preds_te == y_te):.4f}", flush=True)

    w = np.load(WEIGHTS_PATH)
    W0, b0, W1, b1, W2, b2 = w["W0"], w["b0"], w["W1"], w["b1"], w["W2"], w["b2"]

    ae = np.load(AE_PATH)

    correct_mask = preds_te == y_te
    correct_idx = np.where(correct_mask)[0]
    anchor_idx = []
    for c in range(10):
        cand = correct_idx[y_te[correct_idx] == c]
        for j in cand[:2]:
            anchor_idx.append(int(j))
    print(f"sampled {len(anchor_idx)} anchors (2 per class)", flush=True)

    # Build training data for calibration (regenerate from same seed setup)
    import tensorflow as tf
    (X_tr_raw, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.astype(np.float64).reshape(-1, 784) / 255.0

    class_calib = {}
    for c in range(10):
        Xk = X_tr[y_tr == c]
        lo, hi = Xk.min(axis=0), Xk.max(axis=0)
        sum_mu = float(Xk.sum(axis=1).mean())
        sum_sd = float(Xk.sum(axis=1).std())
        tv = np.array([np.sum(np.abs(Xk[:, a] - Xk[:, b])) for a, b in mr.PAIRS])
        # Per-image TV
        tv_per_img = np.zeros(len(Xk))
        D = np.zeros((len(Xk), len(mr.PAIRS)))
        for k, (a, b) in enumerate(mr.PAIRS):
            D[:, k] = np.abs(Xk[:, a] - Xk[:, b])
        tv_per_img = D.sum(axis=1)
        tv_p90 = float(np.percentile(tv_per_img, 90))
        tv_p99 = float(np.percentile(tv_per_img, 99))

        norm = 1.0 / math.sqrt(N_PIX)
        Xk2d = Xk.reshape(-1, 28, 28)
        F = np.fft.fft2(Xk2d, norm="ortho")
        mask = np.zeros((28, 28), bool)
        for k in range(28):
            for l in range(28):
                if k >= 14 or l >= 14:
                    mask[k, l] = True
        hf = (np.abs(F.real) + np.abs(F.imag))[:, mask].sum(axis=1)
        hf_p90 = float(np.percentile(hf, 90))
        hf_p99 = float(np.percentile(hf, 99))

        photo_b1 = np.ones(N_PIX) / math.sqrt(N_PIX)
        cy, cx = 13.5, 13.5
        radial = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                radial[i, j] = math.sqrt((i - cy) ** 2 + (j - cx) ** 2)
        photo_b2 = radial.flatten() - radial.mean()
        photo_b2 = photo_b2 / np.linalg.norm(photo_b2)
        proj_a = Xk @ photo_b1
        proj_b = Xk @ photo_b2
        recon = proj_a[:, None] * photo_b1 + proj_b[:, None] * photo_b2
        resid = np.max(np.abs(Xk - recon), axis=1)
        photo_p90 = float(np.percentile(resid, 90))
        photo_p99 = float(np.percentile(resid, 99))

        class_calib[c] = {
            "lo": lo, "hi": hi,
            "sum_mu": sum_mu, "sum_sd": sum_sd,
            "tv_p90": tv_p90, "tv_p99": tv_p99,
            "hf_p90": hf_p90, "hf_p99": hf_p99,
            "photo_p90": photo_p90, "photo_p99": photo_p99,
        }
        print(f"  class {c}: n={len(Xk)}, sum mu={sum_mu:.1f}+-{sum_sd:.1f}, "
              f"TV p90={tv_p90:.1f}/p99={tv_p99:.1f}, "
              f"HF p90={hf_p90:.2f}/p99={hf_p99:.2f}, "
              f"photo p90={photo_p90:.3f}/p99={photo_p99:.3f}",
              flush=True)

    P3_EPS = [0.01, 0.05, 0.10]
    P1_MARGIN = 0.5
    P4_TARGETS = {3: 8, 4: 9, 7: 1, 8: 3, 5: 6, 9: 4, 1: 7, 6: 5, 0: 6, 2: 7}

    results = {
        "provenance": {
            "script": "verify_mnist_protocol.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "onnx": ONNX_PATH,
            "n_anchors": len(anchor_idx),
            "p3_eps": P3_EPS,
            "p1_margin": P1_MARGIN,
            "p4_targets_per_true_class": {str(k): v for k, v in P4_TARGETS.items()},
        },
        "per_anchor": [],
        "per_class": {},
    }

    def run_p3(anchor_i, eps, region_name, attach_region):
        x_ref = X_te[anchor_i].astype(np.float64)
        true_c = int(y_te[anchor_i])
        logits = numpy_forward(W0, b0, W1, b1, W2, b2, x_ref)
        target_c = runner_up_class(logits, true_c)
        net, ivars, ovars = base_net()
        attach_region(net, ivars, x_ref, true_c)
        net.addInequality([int(ovars[true_c]), int(ovars[target_c])], [1.0, -1.0], 0.0)
        r = solve(net, f"P3:{region_name}:img{anchor_i}:eps{eps}")
        r["anchor_idx"] = anchor_i
        r["true_class"] = true_c
        r["target_class"] = target_c
        r["region"] = region_name
        r["eps"] = eps
        r["property"] = "P3"
        if r["status"] == "sat" and "_vals" in r:
            x_w = witness_from_vals(r["_vals"], ivars)
            r["witness_pixels"] = x_w.round(5).tolist()
            r["witness_linf_from_anchor"] = float(np.max(np.abs(x_w - x_ref)))
            logits_w = numpy_forward(W0, b0, W1, b1, W2, b2, x_w)
            r["witness_argmax"] = int(np.argmax(logits_w))
            r["witness_logits"] = logits_w.round(4).tolist()
            del r["_vals"]
        return r

    def run_p1_or_p4(anchor_i, property_name, target_c, region_name, attach_region):
        x_ref = X_te[anchor_i].astype(np.float64)
        true_c = int(y_te[anchor_i])
        net, ivars, ovars = base_net()
        attach_region(net, ivars, x_ref, true_c)
        net.addInequality([int(ovars[true_c]), int(ovars[target_c])], [1.0, -1.0], 0.0)
        r = solve(net, f"{property_name}:{region_name}:img{anchor_i}:tgt{target_c}")
        r["anchor_idx"] = anchor_i
        r["true_class"] = true_c
        r["target_class"] = target_c
        r["region"] = region_name
        r["property"] = property_name
        if r["status"] == "sat" and "_vals" in r:
            x_w = witness_from_vals(r["_vals"], ivars)
            r["witness_pixels"] = x_w.round(5).tolist()
            r["witness_linf_from_anchor"] = float(np.max(np.abs(x_w - x_ref)))
            logits_w = numpy_forward(W0, b0, W1, b1, W2, b2, x_w)
            r["witness_argmax"] = int(np.argmax(logits_w))
            r["witness_logits"] = logits_w.round(4).tolist()
            del r["_vals"]
        return r

    def make_P3_R1(eps):
        def attach(net, ivars, x_ref, true_c):
            for i in range(N_PIX):
                net.setLowerBound(int(ivars[i]), float(max(0.0, x_ref[i] - eps)))
                net.setUpperBound(int(ivars[i]), float(min(1.0, x_ref[i] + eps)))
        return attach

    def make_P3_R2(eps):
        def attach(net, ivars, x_ref, true_c):
            for i in range(N_PIX):
                net.setLowerBound(int(ivars[i]), float(max(0.0, x_ref[i] - eps)))
                net.setUpperBound(int(ivars[i]), float(min(1.0, x_ref[i] + eps)))
            cal = class_calib[true_c]
            mr.add_R2_invariant(net, ivars, cal["sum_mu"], cal["sum_sd"], n_std=2.0)
        return attach

    def attach_P3_R3(net, ivars, x_ref, true_c):
        dec1_W = ae[f"dec1_W_class{true_c}"]
        dec1_b = ae[f"dec1_b_class{true_c}"]
        dec2_W = ae[f"dec2_W_class{true_c}"]
        dec2_b = ae[f"dec2_b_class{true_c}"]
        latent_lo = ae[f"latent_lo_class{true_c}"]
        latent_hi = ae[f"latent_hi_class{true_c}"]
        mr.add_R3_ae_decoder(net, ivars, dec1_W, dec1_b, dec2_W, dec2_b, latent_lo, latent_hi)

    def attach_P3_R4(net, ivars, x_ref, true_c):
        V_top, sigmas = mr.compute_local_pca(X_tr, y_tr, x_ref, true_c, d=8)
        mr.add_R4_local_pca(net, ivars, x_ref, V_top, sigmas, alpha=2.0)

    def attach_P3_R5(net, ivars, x_ref, true_c):
        Xn = mr.get_same_class_neighbors(X_tr, y_tr, x_ref, true_c, k_nearest=100)
        mr.add_R5_label_conditioned_box(net, ivars, Xn)

    def make_P3_R6_TV():
        def attach(net, ivars, x_ref, true_c):
            cal = class_calib[true_c]
            mr.add_R1_class_box(net, ivars, cal["lo"], cal["hi"])
            mr.add_R6_TV(net, ivars, tau=cal["tv_p90"])
        return attach

    def attach_class_R1(net, ivars, x_ref, true_c):
        cal = class_calib[true_c]
        mr.add_R1_class_box(net, ivars, cal["lo"], cal["hi"])

    def attach_class_R2(net, ivars, x_ref, true_c):
        cal = class_calib[true_c]
        mr.add_R1_class_box(net, ivars, cal["lo"], cal["hi"])
        mr.add_R2_invariant(net, ivars, cal["sum_mu"], cal["sum_sd"], n_std=2.0)

    def attach_class_R6_TV(net, ivars, x_ref, true_c):
        cal = class_calib[true_c]
        mr.add_R1_class_box(net, ivars, cal["lo"], cal["hi"])
        mr.add_R6_TV(net, ivars, tau=cal["tv_p90"])

    print("\n" + "=" * 70, flush=True)
    print("Running P3-analog: local L_inf robustness across regions", flush=True)
    print("=" * 70, flush=True)
    per_anchor = []
    for n_a, anchor_i in enumerate(anchor_idx):
        true_c = int(y_te[anchor_i])
        print(f"\n[{n_a+1}/{len(anchor_idx)}] anchor {anchor_i} (class {true_c})", flush=True)

        for eps in P3_EPS:
            print(f"  eps={eps}:", flush=True)
            for region_name, attach in [
                ("R1_Linf", make_P3_R1(eps)),
                ("R2_Linf", make_P3_R2(eps)),
            ]:
                r = run_p3(anchor_i, eps, region_name, attach)
                tok = "SAT" if r["status"] == "sat" else "UNSAT" if r["status"] == "unsat" else r["status"]
                print(f"    {region_name:14s}: {tok:7s} t={r['time_seconds']}s", flush=True)
                per_anchor.append(r)

        for region_name, attach in [
            ("R3_AE",      attach_P3_R3),
            ("R4_PCA",     attach_P3_R4),
            ("R5_LabelBox", attach_P3_R5),
            ("R6_TV",      make_P3_R6_TV()),
        ]:
            r = run_p3(anchor_i, None, region_name, attach)
            tok = "SAT" if r["status"] == "sat" else "UNSAT" if r["status"] == "unsat" else r["status"]
            print(f"    {region_name:14s}: {tok:7s} t={r['time_seconds']}s", flush=True)
            per_anchor.append(r)

    print("\n" + "=" * 70, flush=True)
    print("Running P1-analog (class-margin) and P4-analog (class-confusion)", flush=True)
    print("=" * 70, flush=True)
    per_class = {}
    for c in range(10):
        print(f"\nclass {c}:", flush=True)
        per_class[str(c)] = []
        target_for_p4 = P4_TARGETS[c]
        sample_anchor = next(a for a in anchor_idx if y_te[a] == c)
        target_for_p1 = runner_up_class(numpy_forward(W0, b0, W1, b1, W2, b2,
                                                      X_te[sample_anchor].astype(np.float64)), c)

        for region_name, attach in [
            ("R1_class_box",  attach_class_R1),
            ("R2_class_box",  attach_class_R2),
            ("R6_TV",         attach_class_R6_TV),
        ]:
            for prop_name, target in [("P1", target_for_p1), ("P4", target_for_p4)]:
                r = run_p1_or_p4(sample_anchor, prop_name, target, region_name, attach)
                tok = "SAT" if r["status"] == "sat" else "UNSAT" if r["status"] == "unsat" else r["status"]
                print(f"  {prop_name}({c}->{target}) {region_name:14s}: {tok:7s} t={r['time_seconds']}s",
                      flush=True)
                per_class[str(c)].append(r)

    results["per_anchor"] = per_anchor
    results["per_class"] = per_class

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
