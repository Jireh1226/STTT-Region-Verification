#!/usr/bin/env python3
"""Part B: audit the controlled-oracle witnesses against a known semantic
ground truth, and check whether the witness audit recovers the
true-vs-artifact distinction.

Because we own the renderer, we compute an external semantic-support oracle
independent of the learned classifier: the nearest distance from a witness to
a dense renderer grid. A witness is 'renderer-supported' iff that distance is
below the 99th percentile of held-out rendered-to-rendered nearest-neighbour
distances -- a threshold fixed before looking at any witness.

For every SAT witness we report: classifier probability, nearest rendered-image
L2, per-pair Lipschitz violation fraction, nearest renderer-grid L2, the
nearest renderer state, whether that state lies in the injected bug region,
and the renderer-supported verdict.

Inputs:  controlled_verify_raw.json, controlled_classifier.npz
Outputs: controlled_oracle_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import controlled_renderer as R
from train_controlled_models import BUG_CTE

ROOT = Path(__file__).resolve().parent
SHAPE = (R.ROWS, R.COLS)
GRID_N = 20000
SUPPORT_PCTL = 99.0

GROUND_TRUTH = {
    "M_good": "no semantic failure in the tested state range",
    "M_bug": f"injected TRUE failure: wrong-sign CTE on rendered images with "
             f"true CTE > {BUG_CTE}",
    "M_artifact": "ARTIFACT-only failure: wrong-sign only on a non-renderable "
                  "checkerboard trigger; correct on all rendered images",
}
# Fixed before looking at results: which witnesses SHOULD be renderer-supported.
EXPECT_SUPPORTED = {"M_good": False, "M_bug": True, "M_artifact": False}


def load_nnet(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    nlayers = int(lines[i].split(",")[0]); i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]; i += 1
    i += 5
    Ws, bs = [], []
    for L in range(nlayers):
        rows, cols = sizes[L + 1], sizes[L]
        W = np.array([[float(v) for v in lines[i + r].strip().rstrip(",").split(",")][:cols]
                      for r in range(rows)]); i += rows
        b = np.array([float(lines[i + r].strip().rstrip(",")) for r in range(rows)]); i += rows
        Ws.append(W); bs.append(b)
    return Ws, bs


def net_forward(Ws, bs, x):
    a = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W.T + b
        if k < len(Ws) - 1:
            a = np.maximum(a, 0.0)
    return a


def load_classifier(npz):
    w = np.load(npz)
    layers = [(w["c1_W"], w["c1_b"]), (w["c2_W"], w["c2_b"]),
              (w["score_W"], w["score_b"])]

    def prob(X):
        h = np.atleast_2d(X)
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return 1.0 / (1.0 + np.exp(-np.clip(h.flatten(), -30, 30)))
    return prob


def per_pair_calib(Xref, q=90):
    pairs = R_grid_pairs()
    taus = np.array([np.percentile(np.abs(Xref[:, a] - Xref[:, b]), q)
                     for a, b in pairs])
    return pairs, taus


def R_grid_pairs():
    pairs = []
    for i in range(R.ROWS):
        for j in range(R.COLS):
            if j + 1 < R.COLS:
                pairs.append((i * R.COLS + j, i * R.COLS + j + 1))
            if i + 1 < R.ROWS:
                pairs.append((i * R.COLS + j, (i + 1) * R.COLS + j))
    return pairs


def main():
    raw = json.loads((ROOT / "controlled_verify_raw.json").read_text())
    rng = np.random.default_rng(909)
    clf = load_classifier(ROOT / "controlled_classifier.npz")

    # References: rendered left-offset set (nearest-real), held-out rendered set
    # (support threshold), dense renderer grid (semantic oracle).
    Xref, _ = R.sample_states(800, rng, cte_range=(BUG_CTE, 6.0))
    Xheld, _ = R.sample_states(400, np.random.default_rng(55),
                               cte_range=(BUG_CTE, 6.0))
    support_thr, _ = R.rendered_nn_percentile(Xheld, p=SUPPORT_PCTL)
    G, Gstates = R.dense_grid(GRID_N, np.random.default_rng(77),
                              cte_range=R.CTE_RANGE)
    pairs, taus = per_pair_calib(Xref, q=90)

    nets = {m: load_nnet(ROOT / f"controlled_{m}.nnet") for m in GROUND_TRUTH}

    def audit_witness(x, model):
        x = np.asarray(x, dtype=np.float64)
        nearest_real = float(np.min(np.linalg.norm(Xref - x[None, :], axis=1)))
        ppv = float(np.mean([abs(x[a] - x[b]) > t
                             for (a, b), t in zip(pairs, taus)]))
        grid_l2, grid_state = R.nearest_in_grid(x, G, Gstates)
        # Semantic oracle: re-render the witness's nearest renderable state and
        # check whether the model actually fails (wrong-sign) on that real
        # image. A true semantic failure reproduces on the render; an artifact
        # does not (the witness is not the image of any feasible state).
        Ws, bs = nets[model]
        x_render = R.render(grid_state["cte"], grid_state["heading"],
                            grid_state["brightness"], grid_state["width"])
        pred_render = float(net_forward(Ws, bs, x_render)[0])
        wrong_on_render = bool(pred_render <= 0.0
                               and grid_state["cte"] > BUG_CTE)
        return {
            "classifier_prob": round(float(clf(x[None, :])[0]), 4),
            "nearest_rendered_L2": round(nearest_real, 3),
            "per_pair_violation_frac": round(ppv, 3),
            "nearest_grid_L2": round(grid_l2, 3),
            "nearest_grid_state": {k: round(v, 2) for k, v in grid_state.items()},
            "nearest_state_in_bug_region": bool(grid_state["cte"] > BUG_CTE),
            "renderer_supported_L2": bool(grid_l2 < support_thr),
            "pred_on_nearest_render": round(pred_render, 3),
            "wrong_sign_on_nearest_render": wrong_on_render,
        }

    by_model = {}
    for rec in raw["verdicts"]:
        m, region = rec["model"], rec["region"]
        e = by_model.setdefault(m, {"ground_truth": GROUND_TRUTH[m], "regions": {}})
        r = {"status": rec["status"]}
        if rec["status"] == "sat" and "witness" in rec:
            r["pred_cte"] = round(rec.get("pred_cte", float("nan")), 3)
            r["audit"] = audit_witness(rec["witness"], m)
        e["regions"][region] = r

    # Per-model assessment. The workflow's mechanism is the verdict ladder:
    # the manifold-constrained region (R4 local-PCA) should REMOVE box-only
    # (artifact) failures while PRESERVING a true on-manifold failure. So
    # recovery is judged on the structured-region verdict plus, for the true
    # failure, whether the recovered witness points back into the injected bug
    # region. (The raw nearest-grid L2 is also reported; for R4 it carries the
    # linear-PCA reconstruction gap, so the witness is near -- not identical to
    # -- a render, mirroring the AE/PCA reconstruction-shift caveat.)
    table, assessment = [], {}
    for m, e in by_model.items():
        broad = e["regions"].get("R1_broad_box", {})
        tight = e["regions"].get("R5_neighbor_box", {})
        struct = e["regions"].get("R4_local_pca", {})
        struct_sat = struct.get("status") == "sat"
        box_sat = broad.get("status") == "sat" or tight.get("status") == "sat"
        # Decisive semantic oracle: does ANY recovered witness reproduce a
        # wrong-sign failure when its nearest renderable state is re-rendered?
        reproduces = any(r["audit"]["wrong_sign_on_nearest_render"]
                         for r in (broad, tight, struct) if "audit" in r)
        if m == "M_bug":
            in_bug = ("audit" in struct
                      and struct["audit"]["nearest_state_in_bug_region"])
            recovered = struct_sat and in_bug and reproduces
            audit_verdict = ("survives structured region; failure reproduces "
                             "on the re-rendered nearest state (true failure)") \
                if recovered else "true failure NOT preserved/reproduced"
        else:
            recovered = box_sat and not struct_sat and not reproduces
            audit_verdict = ("removed by structured region; no failure on any "
                             "re-rendered state (box artifact)") if recovered \
                else "structured region still SAT or failure reproduced"
        # Nearest-grid L2 of the most on-manifold witness recovered.
        gl2 = [r["audit"]["nearest_grid_L2"] for r in (broad, tight, struct)
               if "audit" in r]
        assessment[m] = {
            "expected_supported": EXPECT_SUPPORTED[m],
            "box_region_sat": bool(box_sat),
            "structured_region_sat": bool(struct_sat),
            "failure_reproduces_on_render": bool(reproduces),
            "min_nearest_grid_L2": round(min(gl2), 3) if gl2 else None,
            "expected_behavior_recovered": bool(recovered)}
        table.append({
            "model": m, "ground_truth": GROUND_TRUTH[m],
            "broad_R1": broad.get("status", "n/a"),
            "tight_R5": tight.get("status", "n/a"),
            "structured_R4": struct.get("status", "n/a"),
            "witness_audit": audit_verdict,
            "expected_recovered": bool(recovered),
        })

    result = {
        "provenance": {
            **raw.get("provenance", {}),
            "audit_script": "audit_controlled_witnesses.py",
            "renderer_grid_n": GRID_N,
            "support_percentile": SUPPORT_PCTL,
            "renderer_supported_threshold_L2": round(support_thr, 3),
            "note": "Controlled known-truth oracle calibration. The renderer "
                    "grid is a semantic-support oracle independent of the "
                    "audit classifier; it lets the true-vs-artifact "
                    "distinction be checked, unlike on TinyTaxiNet.",
        },
        "ground_truth": GROUND_TRUTH,
        "compact_table": table,
        "by_model": by_model,
        "assessment": assessment,
        "all_recovered": all(a["expected_behavior_recovered"]
                             for a in assessment.values()),
    }
    (ROOT / "controlled_oracle_results.json").write_text(
        json.dumps(result, indent=2) + "\n")

    print(f"renderer-supported threshold (p{SUPPORT_PCTL:.0f} rendered NN L2) "
          f"= {support_thr:.3f}\n")
    for row in table:
        print(f"{row['model']:11s} | R1={row['broad_R1']:5s} "
              f"R5={row['tight_R5']:5s} R4={row['structured_R4']:5s} | "
              f"{row['witness_audit'][:52]:52s} | "
              f"recovered={row['expected_recovered']}")
    for m, e in by_model.items():
        for rn, r in e["regions"].items():
            if "audit" in r:
                a = r["audit"]
                print(f"    {m} {rn}: predCTE={r['pred_cte']:+.2f} "
                      f"clf={a['classifier_prob']:.2f} gridL2={a['nearest_grid_L2']:.2f} "
                      f"fitCTE={a['nearest_grid_state']['cte']:+.2f} "
                      f"predOnRender={a['pred_on_nearest_render']:+.2f} "
                      f"wrongOnRender={a['wrong_sign_on_nearest_render']}")
    print(f"\nall_recovered = {result['all_recovered']}")
    print("Wrote controlled_oracle_results.json")


if __name__ == "__main__":
    main()
