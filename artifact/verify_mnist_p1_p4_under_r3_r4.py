"""
Extend the MNIST P1/P4-analog verification grid to include R3 (class-AE
decoder-box) and R4 (per-image local-PCA), to mirror the per-anchor P3-analog
that already covered R3/R4 and to complete Table mnist_p1p4 with the two
learned/local-support regions that were left to P3 in the original run.

For each of 10 classes:
  - Take the same representative anchor as verify_mnist_protocol.py: the first
    correctly-classified test image of that class.
  - Compute the runner-up class via a forward pass on x_0 (the P1-analog
    target; same logic as the original protocol).
  - Look up the P4-analog target from P4_TARGETS (identical mapping).
  - Verify both P1 and P4 under R3 (class-AE decoder-box) and R4 (local-PCA
    same-class neighbors, alpha=2sigma, d=8).
  - Total queries: 10 classes x 2 properties x 2 regions = 40.

For each SAT query, store the recovered pixel-vector witness so the audit pass
can score it. Audit metrics match audit_mnist_witnesses.py:
  - Classifier probability under the plausibility classifier
  - Nearest-real-image L_2 distance over the same-class test slice
  - Per-pair Lipschitz violation fraction at class-conditioned p_90 thresholds

Outputs:
  mnist_p1_p4_under_r3_r4.json  - per-query verdicts + per-witness audit + aggregates
"""
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from maraboupy import Marabou

import mnist_regions as mr


ONNX_PATH = "mnist_mlp.onnx"
DATA_PATH = "mnist_test_data.npz"
WEIGHTS_PATH = "mnist_mlp_weights.npz"
AE_PATH = "mnist_class_aes.npz"
CLASSIFIER_PATH = "mnist_plausibility_classifier_weights.npz"
OUTPUT_JSON = "mnist_p1_p4_under_r3_r4.json"

TIMEOUT_S = 180
N_PIX = 784

P4_TARGETS = {3: 8, 4: 9, 7: 1, 8: 3, 5: 6, 9: 4, 1: 7, 6: 5, 0: 6, 2: 7}


def numpy_forward(W0, b0, W1, b1, W2, b2, x):
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    return W2 @ h2 + b2


def runner_up_class(logits, true_c):
    other = logits.copy()
    other[true_c] = -np.inf
    return int(np.argmax(other))


def classifier_prob(X, w):
    h1 = np.maximum(X @ w["W0"].T + w["b0"], 0)
    h2 = np.maximum(h1 @ w["W1"].T + w["b1"], 0)
    logit = h2 @ w["W2"].T + w["b2"]
    return 1.0 / (1.0 + np.exp(-logit.flatten()))


def per_pair_lipschitz_violation(x, taus):
    diffs = np.array([abs(x[a] - x[b]) for a, b in mr.PAIRS])
    return float(np.mean(diffs > taus))


def calibrate_per_pair_lip(X_class, q=90):
    taus = np.zeros(len(mr.PAIRS))
    for k, (a, b) in enumerate(mr.PAIRS):
        taus[k] = np.percentile(np.abs(X_class[:, a] - X_class[:, b]), q)
    return taus


def base_net():
    net = Marabou.read_onnx(ONNX_PATH)
    ivars = [int(v) for v in np.asarray(net.inputVars[0]).flatten()]
    ovars = [int(v) for v in np.asarray(net.outputVars[0]).flatten()]
    return net, ivars, ovars


def solve(net):
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


def witness_from_vals(vals, ivars):
    return np.array([vals[int(ivars[i])] for i in range(N_PIX)], dtype=np.float64)


def run_query(true_c, target_c, region_name, attach_region, anchor_x, prop_name,
              Ws, audit_w, test_by_class, class_taus, baseline_rates):
    net, ivars, ovars = base_net()
    attach_region(net, ivars)
    net.addInequality([int(ovars[true_c]), int(ovars[target_c])], [1.0, -1.0], 0.0)
    r = solve(net)
    r.update({
        "property": prop_name, "region": region_name,
        "true_class": true_c, "target_class": target_c,
    })
    if r["status"] == "sat" and "_vals" in r:
        x_w = witness_from_vals(r["_vals"], ivars)
        logits_w = numpy_forward(*Ws, x_w)
        prob = float(classifier_prob(x_w[None, :], audit_w)[0])
        l2 = float(np.min(np.linalg.norm(test_by_class[true_c] - x_w[None, :], axis=1)))
        ppv = per_pair_lipschitz_violation(x_w, class_taus[true_c])
        r.update({
            "witness_pixels": x_w.round(5).tolist(),
            "witness_linf_from_anchor": float(np.max(np.abs(x_w - anchor_x))),
            "witness_argmax": int(np.argmax(logits_w)),
            "audit": {
                "classifier_prob": prob,
                "nearest_real_L2_same_class": l2,
                "pp_lip_violation_fraction": ppv,
                "pp_lip_baseline_rate": baseline_rates[true_c],
            },
        })
        del r["_vals"]
    return r


def main():
    print("Loading network + weights...", flush=True)
    data = np.load(DATA_PATH)
    X_te, y_te, preds_te = data["X_test"], data["y_test"], data["preds"]
    w = np.load(WEIGHTS_PATH)
    Ws = (w["W0"], w["b0"], w["W1"], w["b1"], w["W2"], w["b2"])
    ae = np.load(AE_PATH)
    audit_w = np.load(CLASSIFIER_PATH)

    print("Loading MNIST training set for PCA + audit baselines...", flush=True)
    (X_tr_raw, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.reshape(-1, 784).astype(np.float64) / 255.0

    print("Calibrating per-class per-pair Lipschitz thresholds + baseline rates...", flush=True)
    class_taus, baseline_rates = {}, {}
    for c in range(10):
        Xk = X_tr[y_tr == c]
        class_taus[c] = calibrate_per_pair_lip(Xk, q=90)
        rates = np.array([per_pair_lipschitz_violation(Xk[i], class_taus[c]) for i in range(min(500, len(Xk)))])
        baseline_rates[c] = float(rates.mean())
        print(f"  class {c}: pp-baseline {baseline_rates[c]:.4f}", flush=True)

    test_by_class = {c: X_te[y_te == c].astype(np.float64) for c in range(10)}

    print("\nSelecting representative anchors (one per class, same as mnist_p1p4)...", flush=True)
    correct_mask = preds_te == y_te
    correct_idx = np.where(correct_mask)[0]
    full_anchor_idx = []
    for c in range(10):
        cand = correct_idx[y_te[correct_idx] == c]
        for j in cand[:2]:
            full_anchor_idx.append(int(j))
    rep_anchor = {c: next(a for a in full_anchor_idx if y_te[a] == c) for c in range(10)}

    def make_R3(true_c):
        def attach(net, ivars):
            mr.add_R3_ae_decoder(
                net, ivars,
                ae[f"dec1_W_class{true_c}"], ae[f"dec1_b_class{true_c}"],
                ae[f"dec2_W_class{true_c}"], ae[f"dec2_b_class{true_c}"],
                ae[f"latent_lo_class{true_c}"], ae[f"latent_hi_class{true_c}"],
            )
        return attach

    def make_R4(x_ref, true_c):
        V_top, sigmas = mr.compute_local_pca(X_tr, y_tr, x_ref, true_c, d=8)
        def attach(net, ivars):
            mr.add_R4_local_pca(net, ivars, x_ref, V_top, sigmas, alpha=2.0)
        return attach

    queries = []
    print("\n" + "=" * 70, flush=True)
    print("Running P1 and P4 under R3 (class-AE) and R4 (local-PCA)...", flush=True)
    print("=" * 70, flush=True)
    for c in range(10):
        anchor_i = rep_anchor[c]
        x_ref = X_te[anchor_i].astype(np.float64)
        logits_ref = numpy_forward(*Ws, x_ref)
        p1_target = runner_up_class(logits_ref, c)
        p4_target = P4_TARGETS[c]
        attach_r3 = make_R3(c)
        attach_r4 = make_R4(x_ref, c)
        print(f"\nclass {c} (anchor {anchor_i}, P1 runner-up={p1_target}, P4 target={p4_target}):",
              flush=True)
        for prop_name, target in [("P1", p1_target), ("P4", p4_target)]:
            for region_name, attach in [("R3_class_AE", attach_r3), ("R4_local_PCA", attach_r4)]:
                r = run_query(c, target, region_name, attach, x_ref, prop_name,
                              Ws, audit_w, test_by_class, class_taus, baseline_rates)
                tok = "SAT" if r["status"] == "sat" else "UNSAT" if r["status"] == "unsat" else r["status"]
                aud = ""
                if "audit" in r:
                    a = r["audit"]
                    aud = (f"  [audit cls={a['classifier_prob']:.3f} "
                           f"L2={a['nearest_real_L2_same_class']:.2f} "
                           f"pp_viol={a['pp_lip_violation_fraction']:.3f} (baseline {a['pp_lip_baseline_rate']:.3f})]")
                print(f"  {prop_name} ({c}->{target}) {region_name:14s}: {tok:7s} t={r['time_seconds']}s{aud}",
                      flush=True)
                queries.append(r)

    by_region_prop = defaultdict(lambda: {"sat": 0, "unsat": 0, "timeout": 0, "other": 0})
    sat_witnesses_by_region = defaultdict(list)
    for q in queries:
        key = (q["region"], q["property"])
        s = q["status"]
        if s == "sat":
            by_region_prop[key]["sat"] += 1
            if "audit" in q:
                sat_witnesses_by_region[q["region"]].append(q["audit"])
        elif s == "unsat":
            by_region_prop[key]["unsat"] += 1
        elif "timeout" in s:
            by_region_prop[key]["timeout"] += 1
        else:
            by_region_prop[key]["other"] += 1

    def agg(rs):
        if not rs:
            return {"n": 0}
        probs = np.array([r["classifier_prob"] for r in rs])
        l2s = np.array([r["nearest_real_L2_same_class"] for r in rs])
        ppvs = np.array([r["pp_lip_violation_fraction"] for r in rs])
        return {
            "n": int(len(rs)),
            "classifier_prob_mean": float(probs.mean()),
            "classifier_prob_median": float(np.median(probs)),
            "classifier_above_0_5": int((probs > 0.5).sum()),
            "classifier_below_0_1": int((probs < 0.1).sum()),
            "nearest_real_L2_median": float(np.median(l2s)),
            "nearest_real_L2_mean": float(l2s.mean()),
            "pp_lip_violation_mean": float(ppvs.mean()),
            "pp_lip_violation_median": float(np.median(ppvs)),
        }

    print("\n" + "=" * 70, flush=True)
    print("Summary", flush=True)
    print("=" * 70, flush=True)
    for (region, prop), counts in sorted(by_region_prop.items()):
        n = counts["sat"] + counts["unsat"] + counts["timeout"] + counts["other"]
        print(f"  {prop} {region:14s}: SAT={counts['sat']}/{n}  UNSAT={counts['unsat']}/{n}  "
              f"TIMEOUT={counts['timeout']}/{n}", flush=True)

    by_region_aud = {r: agg(witnesses) for r, witnesses in sat_witnesses_by_region.items()}
    print("\nAggregate witness audit by region:", flush=True)
    for region, a in sorted(by_region_aud.items()):
        if a["n"] == 0:
            print(f"  {region:14s}: no SAT witnesses (nothing to audit)", flush=True)
            continue
        print(f"  {region:14s}: n={a['n']}  cls mean={a['classifier_prob_mean']:.3f}  "
              f"L2 med={a['nearest_real_L2_median']:.2f}  pp_viol mean={a['pp_lip_violation_mean']:.3f}",
              flush=True)

    out = {
        "provenance": {
            "script": "verify_mnist_p1_p4_under_r3_r4.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "purpose": "Extend MNIST P1/P4 grid to R3 and R4; audit SAT witnesses.",
            "p4_targets": {str(k): v for k, v in P4_TARGETS.items()},
            "rep_anchors": {str(c): rep_anchor[c] for c in range(10)},
        },
        "queries": queries,
        "summary_by_region_property": {f"{p}::{r}": v for (r, p), v in by_region_prop.items()},
        "witness_audit_by_region": by_region_aud,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
