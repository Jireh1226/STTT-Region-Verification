#!/usr/bin/env python3.11
"""
Fix the verification manifest BEFORE running Marabou (no cherry-picking after results).

Anchors: held-out TEST images that are correctly classified with a confidence margin,
balanced across the selected target classes, selected by ascending test index. For each anchor
the target class is the runner-up logit on the clean image. Region parameters and timeouts
are frozen here.
"""
import json, os, time
import numpy as np
from external_lard_common import (ROOT, load_dataset, load_model, forward,
                                  load_target, labels_split)

N_ANCHORS = int(os.environ.get("LARD_N_ANCHORS", "50"))
MARGIN = 0.25          # min (true_logit - runnerup_logit) so anchors are not boundary cases


def class_quotas(n_total, n_classes):
    base = n_total // n_classes
    rem = n_total % n_classes
    return [base + (1 if c < rem else 0) for c in range(n_classes)]


def main():
    d = load_dataset()
    Ws, bs, q1, q2, H, W = load_model()
    tgt = load_target()
    CLASS_NAMES = {i: n for i, n in enumerate(tgt["class_names"])}
    Xte = d["Xte"]
    yte = labels_split(d, "te", tgt)
    logits = forward(Xte, Ws, bs)
    pred = logits.argmax(1)

    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    correct = (pred == yte)

    anchors = []
    quotas = class_quotas(N_ANCHORS, 3)
    for c in range(3):
        cand = [i for i in range(len(Xte)) if correct[i] and yte[i] == c and margin[i] >= MARGIN]
        cand = sorted(cand)[:quotas[c]]
        if len(cand) < quotas[c]:
            raise RuntimeError(f"Only {len(cand)} eligible anchors for class {c}; need {quotas[c]}")
        for i in cand:
            row = logits[i].copy()
            row[c] = -np.inf
            t = int(row.argmax())          # runner-up target class
            anchors.append({
                "anchor_test_idx": int(i),
                "image": str(d["names_te"][i]),
                "airport": str(d["airport_te"][i]),
                "true_class": int(c),
                "true_class_name": CLASS_NAMES[c],
                "target_class": t,
                "target_class_name": CLASS_NAMES[t],
                "target_value": float(d[tgt["chosen_target"] + "_te"][i]),
                "yaw": float(d["yaw_te"][i]),
                "clean_logits": [float(v) for v in logits[i]],
                "clean_margin": float(margin[i]),
            })

    manifest = {
        "experiment": "external_lard_region_transfer",
        "purpose": "Test whether broad pixel regions admit class-confusion witnesses that "
                   "tighter data-conditioned regions remove, on an EXTERNAL runway dataset/model "
                   "(LARD V2 ges) independent of VerifyGAN/TinyTaxiNet.",
        "network": f"external_lard_mlp.nnet ({H*W}-64-32-3 ReLU MLP, trained by us on LARD V2 ges)",
        "dataset": "LARD V2 ges; see external_lard_provenance.json",
        "target": tgt["chosen_target"], "class_names": tgt["class_names"],
        "input": {"dim": H * W, "rows": H, "cols": W, "range": "[0,1] grayscale"},
        "selection": f"{len(anchors)} held-out anchors with class quotas {quotas}, correctly classified, "
                     f"clean margin >= {MARGIN}, ascending test index; train-only region calibration",
        "n_anchors": len(anchors),
        "property": "targeted class-confusion: assert Y_target >= Y_true (SAT = confusion witness exists)",
        "regions": {
            "R1_pixel_ball": {"type": "broad Linf box", "eps_grid": [0.1, 0.2, 0.3, 0.4], "clip": [0, 1]},
            "R2_mean_band": {"type": "R1 + class mean-brightness band", "eps_grid": [0.1, 0.2, 0.3, 0.4], "sigma": 2.0},
            "R5_knn_box": {"type": "label-conditioned per-pixel box over K same-class train NN", "K": 100, "clip": [0, 1]},
            "R4_local_pca": {"type": "per-anchor local PCA (composed)", "d": 8, "alphas": [1.0, 2.0, 3.0],
                             "neighbors": "same-class train, K=200 nearest", "variants": ["unbounded", "valid_pixel_bounded"]},
        },
        "solver": "maraboupy Marabou 2.0, snc, 4 workers, verbosity 0",
        "timeout_s": 120,
        "no_tuning_after_results": True,
        "seed": 0,
        "thresholds_train_tercile": [q1, q2],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchors": anchors,
    }
    with open(os.path.join(ROOT, "external_lard_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    from collections import Counter
    print("anchors:", len(anchors), "by true class:",
          dict(Counter(a["true_class_name"] for a in anchors)))
    print("targets:", Counter((a["true_class_name"], a["target_class_name"]) for a in anchors))
    print("WROTE external_lard_manifest.json")


if __name__ == "__main__":
    main()
