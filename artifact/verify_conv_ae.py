"""
Head-to-head Conv-AE vs FC-AE comparison.

Runs the AE-composed verification suite (P1a, P1b, P4_left, P4_right, P5) under
the conv-AE region with mean(x)=0.5 enforced, plus the support audit
(reconstruction L_inf, output preservation |delta CTE|/|delta HE|, latent
coverage on held-out images, classifier acceptance of random latent decodes,
classifier scoring of any SAT witness's decoded pixel vector).

Output: verification_conv_ae.json (verdicts), conv_autoencoder_support_audit.json
(audit metrics), conv_vs_fc_comparison.json (side-by-side table).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json, time
from datetime import datetime, timezone

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
# NOTE: do NOT `from maraboupy import Marabou` at module load — it pollutes
# Keras's deserialization registry and breaks subsequent .keras file loads.
# We lazy-import inside the functions that need it.


DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
CLF_WEIGHTS = "plausibility_classifier.npz"
NUM_PIXELS = 128

AE_MAPPING = {
    "P1a": "centered",
    "P1b": "centered",
    "P4_left": "left",
    "P4_right": "right",
    "P5": "off_center",
}

POSTCONDITIONS = {
    "P1a": ("CTE", ">=", 10.0),
    "P1b": ("CTE", "<=", -10.0),
    "P4_left": ("CTE", "<=", 0.0),
    "P4_right": ("CTE", ">=", 0.0),
    "P5": "deadzone",
}

TIMEOUT_S = 600


def load_classifier():
    w = np.load(CLF_WEIGHTS)
    return [(w["c1_W"], w["c1_b"]),
            (w["c2_W"], w["c2_b"]),
            (w["score_W"], w["score_b"])]


def score_clf(x, layers):
    h = x[None, :] if x.ndim == 1 else x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h.flatten()[0])


def prob_from_logit(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def load_subsets():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte = labels[:, 0]
    return {
        "centered":   (images[np.abs(cte) < 1.0], labels[np.abs(cte) < 1.0]),
        "left":       (images[cte > 2.0],         labels[cte > 2.0]),
        "right":      (images[cte < -2.0],        labels[cte < -2.0]),
        "off_center": (images[np.abs(cte) > 1.0], labels[np.abs(cte) > 1.0]),
    }, images, labels


def solve_conv_ae_query(onnx_path, query_name, z_min, z_max):
    """Returns dict with status, time, and if SAT, the latent witness."""
    from maraboupy import Marabou
    net = Marabou.read_onnx(onnx_path)
    iv = np.array(net.inputVars[0]).flatten()
    ov = np.array(net.outputVars[0]).flatten()
    cte_var, he_var, sum_var = int(ov[0]), int(ov[1]), int(ov[2])

    for i in range(len(iv)):
        net.setLowerBound(int(iv[i]), float(z_min[i]))
        net.setUpperBound(int(iv[i]), float(z_max[i]))
    # mean = 0.5 ↔ sum = 64
    net.setLowerBound(sum_var, 64.0); net.setUpperBound(sum_var, 64.0)

    pc = POSTCONDITIONS[query_name]
    if pc == "deadzone":
        net.setLowerBound(cte_var, -0.01); net.setUpperBound(cte_var, 0.01)
        net.setLowerBound(he_var, -0.01);  net.setUpperBound(he_var, 0.01)
    else:
        var = cte_var if pc[0] == "CTE" else he_var
        if pc[1] == ">=":
            net.setLowerBound(var, float(pc[2]))
        else:
            net.setUpperBound(var, float(pc[2]))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {"status": f"error: {str(e)[:80]}", "time_seconds": round(time.time() - t0, 3)}
    elapsed = round(time.time() - t0, 3)
    s = str(status).strip().lower()
    out = {"status": s, "time_seconds": elapsed}
    if "sat" in s and vals:
        out["latent_witness"] = [round(float(vals[int(iv[i])]), 6) for i in range(len(iv))]
        out["CTE"] = round(float(vals[cte_var]), 4)
        out["HE"] = round(float(vals[he_var]), 4)
        out["pixel_sum"] = round(float(vals[sum_var]), 4)
    return out


def decode_with_keras(subset, latents):
    ae = keras.models.load_model(f"conv_autoencoder_{subset}.keras", compile=False, safe_mode=False)
    enc = keras.models.load_model(f"conv_encoder_{subset}.keras", compile=False, safe_mode=False)
    # Build decoder-only sub-model by indexing layers (since model is sequential-functional)
    # Easier: rebuild decoder from indices
    latents = np.atleast_2d(latents).astype(np.float32)
    # Step through decoder layers manually
    h = latents
    for ln in ["dec_dense1", "dec_dense2"]:
        W, b = ae.get_layer(ln).get_weights()
        h = h @ W + b
        h = np.maximum(h, 0)
    h = h.reshape(-1, 8, 16, 8)
    # dec_conv1
    W, b = ae.get_layer("dec_conv1").get_weights()
    h = tf.nn.conv2d(h.astype(np.float32), W, strides=1, padding="SAME").numpy() + b
    h = np.maximum(h, 0)
    # dec_conv2
    W, b = ae.get_layer("dec_conv2").get_weights()
    h = tf.nn.conv2d(h.astype(np.float32), W, strides=1, padding="SAME").numpy() + b
    return h.reshape(-1, NUM_PIXELS)


def support_audit(subset, images, clf):
    """For the conv AE: reconstruction L_inf, output preservation, latent
    coverage on held-out fraction, classifier acceptance of random latent decodes."""
    enc = keras.models.load_model(f"conv_encoder_{subset}.keras", compile=False, safe_mode=False)
    ae = keras.models.load_model(f"conv_autoencoder_{subset}.keras", compile=False, safe_mode=False)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(images))
    n_val = max(10, int(0.1 * len(images)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train, x_val = images[train_idx].astype(np.float32), images[val_idx].astype(np.float32)

    # Latent box from training subset (matches verification region)
    z_train = enc.predict(x_train, batch_size=256, verbose=0)
    z_min = z_train.min(axis=0); z_max = z_train.max(axis=0)
    z_val = enc.predict(x_val, batch_size=256, verbose=0)

    # Held-out coverage: fraction of validation latents inside training latent box
    inside = np.all((z_val >= z_min) & (z_val <= z_max), axis=1)
    held_out_coverage = float(inside.mean())

    # Reconstruction L_inf on full subset
    recon = ae.predict(images.astype(np.float32), batch_size=256, verbose=0)
    linf = np.max(np.abs(images - recon), axis=1)

    # Extract all decoder weights to numpy BEFORE Marabou is imported.
    # Marabou's TF1 session init closes the Keras session, so later
    # ae.get_layer(...).get_weights() calls would fail.
    decoder_weights = {ln: [w.copy() for w in ae.get_layer(ln).get_weights()]
                       for ln in ["dec_dense1", "dec_dense2", "dec_conv1", "dec_conv2"]}

    # Random-latent decoding using cached numpy weights (no Keras handle needed).
    rng = np.random.default_rng(1)
    K = 200
    z_random = z_min + (z_max - z_min) * rng.random((K, len(z_min)))
    h = z_random.astype(np.float32)
    for ln in ["dec_dense1", "dec_dense2"]:
        W, b = decoder_weights[ln]
        h = h @ W + b; h = np.maximum(h, 0)
    h = h.reshape(-1, 8, 16, 8)
    Wc, bc = decoder_weights["dec_conv1"]
    h = tf.nn.conv2d(h.astype(np.float32), Wc, strides=1, padding="SAME").numpy() + bc
    h = np.maximum(h, 0)
    Wc, bc = decoder_weights["dec_conv2"]
    h = tf.nn.conv2d(h.astype(np.float32), Wc, strides=1, padding="SAME").numpy() + bc
    decoded = h.reshape(-1, NUM_PIXELS)
    probs = np.array([prob_from_logit(score_clf(d, clf)) for d in decoded])
    accept_rate = float((probs > 0.5).mean())

    # NOW import Marabou for TinyTaxiNet output preservation
    from maraboupy import Marabou
    nn = Marabou.read_nnet(NNET_PATH)
    n = min(200, len(images))
    sub = images[:n].astype(np.float64)
    sub_recon = recon[:n].astype(np.float64)
    delta_cte, delta_he = [], []
    for a, b in zip(sub, sub_recon):
        out_a = np.array(nn.evaluate([a.tolist()])).flatten()
        out_b = np.array(nn.evaluate([b.tolist()])).flatten()
        delta_cte.append(abs(out_a[0] - out_b[0]))
        delta_he.append(abs(out_a[1] - out_b[1]))
    delta_cte = np.array(delta_cte); delta_he = np.array(delta_he)

    return {
        "subset": subset,
        "n_train_for_box": int(len(x_train)),
        "n_val": int(len(x_val)),
        "held_out_latent_coverage": held_out_coverage,
        "recon_linf_median": float(np.median(linf)),
        "recon_linf_max": float(linf.max()),
        "recon_linf_p95": float(np.percentile(linf, 95)),
        "ttn_output_preservation": {
            "delta_CTE_median": float(np.median(delta_cte)),
            "delta_CTE_p95": float(np.percentile(delta_cte, 95)),
            "delta_HE_median": float(np.median(delta_he)),
            "delta_HE_p95": float(np.percentile(delta_he, 95)),
        },
        "classifier_random_latent_accept_rate": accept_rate,
        "latent_box_widths": (z_max - z_min).tolist(),
        "z_min": z_min.tolist(),
        "z_max": z_max.tolist(),
    }


def _audit_one_subset_subprocess(subset):
    """Run support_audit in a subprocess to avoid Keras .keras-reload pollution."""
    import subprocess as sp
    print(f"\n--- support audit: {subset} (subprocess) ---", flush=True)
    out_path = f"_conv_audit_{subset}.json"
    r = sp.run(
        ["./prophecy_env/bin/python", "-u", __file__, "--audit", subset, out_path],
        capture_output=False, text=True)
    if r.returncode != 0:
        print(f"  subprocess for {subset} exited {r.returncode}", flush=True)
        return None
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        d = json.load(f)
    os.remove(out_path)
    return d


def main():
    import sys
    # Subprocess mode: run one subset's audit and dump to JSON
    if "--audit" in sys.argv:
        subset = sys.argv[sys.argv.index("--audit") + 1]
        out_path = sys.argv[sys.argv.index("--audit") + 2]
        subsets, _, _ = load_subsets()
        clf = load_classifier()
        imgs, _ = subsets[subset]
        rec = support_audit(subset, imgs, clf)
        with open(out_path, "w") as f:
            json.dump(rec, f, default=float)
        return
    if "--decode" in sys.argv:
        in_path = sys.argv[sys.argv.index("--decode") + 1]
        out_path = sys.argv[sys.argv.index("--decode") + 2]
        with open(in_path) as f:
            payload = json.load(f)
        clf = load_classifier()
        decoded = decode_with_keras(payload["subset"], np.array(payload["latent"]))
        logit = score_clf(decoded[0], clf)
        with open(out_path, "w") as f:
            json.dump({
                "classifier_prob": prob_from_logit(logit),
                "decoded_pixel_mean": float(decoded[0].mean()),
                "decoded_pixel_witness": [round(float(p), 4) for p in decoded[0]],
            }, f)
        return

    print("Loading data, classifier, AE artifacts...", flush=True)
    subsets, all_imgs, labels = load_subsets()
    clf = load_classifier()

    out = {
        "provenance": {
            "script": "verify_conv_ae.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
        },
        "support_audit": {},
        "verdicts": {},
    }

    # Per-subset support audit (one subprocess each to keep Keras state clean)
    for subset in ["centered", "left", "right", "off_center"]:
        if not os.path.exists(f"conv_autoencoder_{subset}.keras"):
            print(f"  [skip support audit] conv_autoencoder_{subset}.keras missing", flush=True)
            continue
        rec = _audit_one_subset_subprocess(subset)
        if rec is None:
            continue
        out["support_audit"][subset] = rec
        print(f"  held-out cov:  {rec['held_out_latent_coverage']:.3f}", flush=True)
        print(f"  recon Linf:    median={rec['recon_linf_median']:.4f}  "
              f"p95={rec['recon_linf_p95']:.4f}  max={rec['recon_linf_max']:.4f}", flush=True)
        op = rec["ttn_output_preservation"]
        print(f"  |dCTE|:        median={op['delta_CTE_median']:.4f}  p95={op['delta_CTE_p95']:.4f}", flush=True)
        print(f"  |dHE|:         median={op['delta_HE_median']:.4f}  p95={op['delta_HE_p95']:.4f}", flush=True)
        print(f"  random-latent classifier accept rate: {rec['classifier_random_latent_accept_rate']:.3f}",
              flush=True)

    # Verdicts on the 5 canonical queries
    print("\n--- conv-AE verdicts (mean=0.5 enforced) ---", flush=True)
    for q in ["P1a", "P1b", "P4_left", "P4_right", "P5"]:
        subset = AE_MAPPING[q]
        onnx_path = f"composed_conv_{subset}_with_sum.onnx"
        if not os.path.exists(onnx_path):
            out["verdicts"][q] = {"status": "onnx_missing", "subset": subset}
            print(f"  {q:10s}: ONNX missing", flush=True)
            continue
        aud = out["support_audit"].get(subset, {})
        if "z_min" not in aud:
            out["verdicts"][q] = {"status": "audit_missing", "subset": subset}
            continue
        z_min = np.array(aud["z_min"]); z_max = np.array(aud["z_max"])
        r = solve_conv_ae_query(onnx_path, q, z_min, z_max)
        r["subset"] = subset
        # Decode + classifier scoring done in a per-query subprocess to keep
        # the main process Keras-clean.
        if "latent_witness" in r:
            try:
                import subprocess as sp
                import tempfile
                tmp_in = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                tmp_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                json.dump({"subset": subset, "latent": r["latent_witness"]}, tmp_in)
                tmp_in.close(); tmp_out.close()
                sp.run(["./prophecy_env/bin/python", "-u", __file__,
                        "--decode", tmp_in.name, tmp_out.name],
                       capture_output=False, text=True)
                with open(tmp_out.name) as f:
                    dec = json.load(f)
                r["decoded_classifier_prob"] = dec.get("classifier_prob")
                r["decoded_pixel_mean"] = dec.get("decoded_pixel_mean")
                r["decoded_pixel_witness"] = dec.get("decoded_pixel_witness")
                os.remove(tmp_in.name); os.remove(tmp_out.name)
            except Exception as e:
                r["decode_error"] = str(e)[:80]
        out["verdicts"][q] = r
        cte_s = "" if "CTE" not in r else f"  CTE={r['CTE']}"
        clf_s = "" if "decoded_classifier_prob" not in r else f"  clf_prob={r['decoded_classifier_prob']:.3f}"
        print(f"  {q:10s}: {r['status']:8s} t={r['time_seconds']:.2f}s{cte_s}{clf_s}", flush=True)

    with open("verification_conv_ae.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved verification_conv_ae.json", flush=True)


if __name__ == "__main__":
    main()
