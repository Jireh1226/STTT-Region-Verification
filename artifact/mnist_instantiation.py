"""
MNIST instantiation of the region_audit module.

Counterpart to taxinet_instantiation.py: the same network-agnostic module is
instantiated for the 784-32-16-10 MNIST MLP, and the smoke test reproduces
recorded verdicts from mnist_protocol_results.json across all six region
tags. Expected statuses are read from the recorded JSON rather than
hardcoded, so the check is literally "does the module reproduce the released
results".

Benchmark-specific code: dataset loading, the same-class-neighbor rule
(100 nearest same-class train digits, nearest excluded), per-class
calibration (class box, total-mass band, TV p90), and the ONNX reader.
All region constructions and audit signals come from region_audit.

Output: module_smoke_mnist.json
"""
import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from maraboupy import Marabou

import region_audit as ra

ONNX_PATH = "mnist_mlp.onnx"
DATA_PATH = "mnist_test_data.npz"
AE_PATH = "mnist_class_aes.npz"
CLF_PATH = "mnist_plausibility_classifier_weights.npz"
RESULTS_PATH = "mnist_protocol_results.json"
OUTPUT_JSON = "module_smoke_mnist.json"

SHAPE = (28, 28)
TIMEOUT = 180


def make_net():
    net = Marabou.read_onnx(ONNX_PATH)
    ivars = [int(v) for v in np.asarray(net.inputVars[0]).flatten()]
    ovars = [int(v) for v in np.asarray(net.outputVars[0]).flatten()]
    return net, ivars, ovars


def same_class_neighbors(X_tr, y_tr, x0, c, k=100):
    """MNIST neighbor rule: k nearest same-class train digits, nearest
    excluded (mirrors mnist_regions.get_same_class_neighbors)."""
    Xk = X_tr[y_tr == c].astype(np.float64)
    dists = np.linalg.norm(Xk - x0[None, :], axis=1)
    return Xk[np.argsort(dists)[1:k + 1]]


def main():
    data = np.load(DATA_PATH)
    X_te = data["X_test"].astype(np.float64)
    (X_tr_raw, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.reshape(-1, 784).astype(np.float64) / 255.0
    recorded = json.load(open(RESULTS_PATH))

    # Anchor: first recorded per-anchor query's anchor (class 0)
    first = recorded["per_anchor"][0]
    a_idx, true_c = first["anchor_idx"], first["true_class"]
    x0 = X_te[a_idx]

    def expected(region, prop="P3", eps=None, source="per_anchor"):
        pool = (recorded["per_anchor"] if source == "per_anchor"
                else recorded["per_class"][str(true_c)])
        for q in pool:
            if (q.get("region") == region and q.get("property") == prop
                    and q.get("eps") == eps
                    and q.get("anchor_idx", a_idx) == a_idx):
                return q["status"], q.get("target_class")
        raise KeyError(f"no recorded {prop}@{region} eps={eps}")

    # Per-class calibration for the class-level regions
    Xk = X_tr[y_tr == true_c]
    pairs = ra.grid_pairs(*SHAPE)
    tv_per_img = np.abs(Xk[:, [a for a, b in pairs]]
                        - Xk[:, [b for a, b in pairs]]).sum(axis=1)
    tv_p90 = float(np.percentile(tv_per_img, 90))
    sums = Xk.sum(axis=1)

    checks = []

    def check(name, exp, verdict):
        ok = verdict["status"] == exp
        checks.append({"name": name, "expected": exp,
                       "got": verdict["status"], "ok": ok,
                       "time_seconds": verdict["time_seconds"],
                       "region": verdict["region"]})
        print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: expected {exp}, "
              f"got {verdict['status']} ({verdict['time_seconds']:.2f}s)",
              flush=True)
        return verdict

    sat_witness = None

    # R1(x0, 0.05) per-image ball
    exp, tgt = expected("R1_Linf", eps=0.05)
    post = ra.class_confusion(true_c, tgt)
    v = check("P3@R1(x0,0.05)", exp,
              ra.run_query(make_net, ra.r1_linf_ball(x0, 0.05), post, TIMEOUT))
    if v["status"] == "sat":
        sat_witness = np.array(v["witness"])

    # R2(x0, 0.05): ball + class total-mass band (mu_k +- 2 sigma_k)
    exp, tgt = expected("R2_Linf", eps=0.05)
    r2 = ra.intersect(ra.r1_linf_ball(x0, 0.05),
                      ra.mass_band(sums.mean() - 2 * sums.std(),
                                   sums.mean() + 2 * sums.std()), tag="R2")
    check("P3@R2(x0,0.05)", exp,
          ra.run_query(make_net, r2, ra.class_confusion(true_c, tgt), TIMEOUT))

    # R3: class-AE decoder box composed in-Marabou
    exp, tgt = expected("R3_AE")
    ae = np.load(AE_PATH)
    r3 = ra.r3_inline_relu_decoder(
        ae[f"dec1_W_class{true_c}"], ae[f"dec1_b_class{true_c}"],
        ae[f"dec2_W_class{true_c}"], ae[f"dec2_b_class{true_c}"],
        ae[f"latent_lo_class{true_c}"], ae[f"latent_hi_class{true_c}"])
    check("P3@R3(class-AE)", exp,
          ra.run_query(make_net, r3, ra.class_confusion(true_c, tgt), TIMEOUT))

    # R4: local PCA, alpha = 2 sigma, d = 8
    exp, tgt = expected("R4_PCA")
    neighbors = same_class_neighbors(X_tr, y_tr, x0, true_c)
    V_top, sigmas, var_expl = ra.local_pca_basis(x0, neighbors, d=8)
    check("P3@R4(2sigma)", exp,
          ra.run_query(make_net, ra.r4_local_pca(x0, V_top, sigmas, 2.0),
                       ra.class_confusion(true_c, tgt), TIMEOUT))
    print(f"      (top-8 variance explained at this anchor: {var_expl:.3f})")

    # R5: same-class neighbor box
    exp, tgt = expected("R5_LabelBox")
    check("P3@R5(neighbor box)", exp,
          ra.run_query(make_net, ra.r5_neighbor_box(neighbors),
                       ra.class_confusion(true_c, tgt), TIMEOUT))

    # R6: class box + TV at class p90
    exp, tgt = expected("R6_TV")
    r6 = ra.intersect(ra.r1_subset_box(Xk, subset_name=f"class{true_c}"),
                      ra.tv_bound(SHAPE, tv_p90), tag="R6")
    check("P3@R6(class box+TV)", exp,
          ra.run_query(make_net, r6, ra.class_confusion(true_c, tgt), TIMEOUT))

    # P4 analog at class level: R1(S_k) class box, fixed confusable target
    exp, tgt = expected("R1_class_box", prop="P4", source="per_class")
    check("P4@R1(class box)", exp,
          ra.run_query(make_net, ra.r1_subset_box(Xk, subset_name=f"class{true_c}"),
                       ra.class_confusion(true_c, tgt), TIMEOUT))

    # Witness audit demo against the same-class real-digit baseline
    audit_block = None
    if sat_witness is not None:
        clf = ra.load_relu_classifier(
            CLF_PATH, keys=("W0", "b0", "W1", "b1", "W2", "b2"),
            transpose=True)
        X_te_class = X_te[data["y_test"] == true_c]
        audit = ra.WitnessAudit(X_te_class, SHAPE, classifier=clf,
                                pair_quantile=90)
        audit_block = {
            "witness_scores": audit.score(sat_witness),
            "real_baseline": audit.baseline(X_te_class[::10]),
        }
        print(f"  witness audit: {audit_block['witness_scores']}")
        print(f"  real baseline: {audit_block['real_baseline']}")

    n_ok = sum(c["ok"] for c in checks)
    out = {
        "provenance": {
            "script": "mnist_instantiation.py",
            "module": "region_audit.py",
            "anchor_idx": int(a_idx),
            "true_class": int(true_c),
            "expected_from": RESULTS_PATH,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "checks": checks,
        "summary": {"passed": n_ok, "total": len(checks)},
        "witness_audit": audit_block,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{n_ok}/{len(checks)} verdict reproductions passed. "
          f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
