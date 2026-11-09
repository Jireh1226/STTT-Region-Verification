"""
Multi-seed stability check for plausibility-constrained verification.

Retrains each subset autoencoder with two additional seeds (7 and 13),
rebuilds composed ONNX networks for each seed, and reruns the same
verification queries with mean(x)=0.5 enforced.

Reports whether the key findings (P1a/P1b UNSAT, P4 SAT at wrong-sign
threshold 3.0) are stable across seeds.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import tempfile, shutil, subprocess
import numpy as np
import h5py

# -------- Autoencoder training helpers (same as train_subset_autoencoders.py) --------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
OUTPUT_JSON = "multi_seed_stability_results.json"

NUM_PIXELS = 128
LATENT_DIM = 8
HIDDEN_1 = 64
HIDDEN_2 = 16
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3

SEEDS = [42, 7, 13]      # 42 is already done; retrain 7 and 13 fresh
SUBSETS = ["centered", "left", "right", "off_center"]

AE_MAPPING = {
    "P1a": "centered",
    "P1b": "centered",
    "P4_C_left": "left",
    "P4_C_right": "right",
    "P5": "off_center",
}


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float32)
        labels = f["X_train"][:]
    return images, labels


def set_seed(s):
    np.random.seed(s)
    tf.random.set_seed(s)
    tf.keras.utils.set_random_seed(s)


def build_autoencoder():
    inp = keras.Input(shape=(NUM_PIXELS,), name="pixels_in")
    h = layers.Dense(HIDDEN_1, activation="relu", name="enc1")(inp)
    h = layers.Dense(HIDDEN_2, activation="relu", name="enc2")(h)
    z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(h)
    h = layers.Dense(HIDDEN_2, activation="relu", name="dec1")(z)
    h = layers.Dense(HIDDEN_1, activation="relu", name="dec2")(h)
    out = layers.Dense(NUM_PIXELS, activation="linear", name="pixels_out")(h)
    ae = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, z, name="encoder")
    return ae, encoder


def train_ae(subset_name, imgs, seed):
    set_seed(seed)
    ae, encoder = build_autoencoder()
    ae.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(imgs))
    n_val = max(10, int(0.1 * len(imgs)))
    x_val, x_train = imgs[idx[:n_val]], imgs[idx[n_val:]]

    hist = ae.fit(x_train, x_train, validation_data=(x_val, x_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    recon = ae.predict(imgs, batch_size=256, verbose=0)
    err = imgs - recon
    linf_max = float(np.max(np.abs(err), axis=1).max())
    latents = encoder.predict(imgs, batch_size=256, verbose=0)

    # Extract decoder weights
    decoder_w = {}
    for name in ["dec1", "dec2", "pixels_out"]:
        layer = ae.get_layer(name)
        W, b = layer.get_weights()
        decoder_w[f"{name}_W"] = W.astype(np.float64)
        decoder_w[f"{name}_b"] = b.astype(np.float64)

    return {
        "latent_min": latents.min(axis=0),
        "latent_max": latents.max(axis=0),
        "recon_linf_max": linf_max,
        "final_val_loss": float(hist.history["val_loss"][-1]),
        "decoder_weights": decoder_w,
    }


# -------- ONNX build (same as build_subset_onnx_with_sum.py) --------

def load_nnet_weights(path):
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while lines[i].startswith('//'):
        i += 1
    header = [int(x) for x in lines[i].split(',') if x.strip()]
    num_layers = header[0]
    i += 1
    layer_sizes = [int(x) for x in lines[i].split(',') if x.strip()]
    i += 1
    i += 5
    weights, biases = [], []
    for l in range(num_layers):
        rows = layer_sizes[l + 1]
        cols = layer_sizes[l]
        W = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            vals = [float(x) for x in lines[i].strip().rstrip(',').split(',')]
            W[r] = vals[:cols]
            i += 1
        b = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(','))
            i += 1
        weights.append(W); biases.append(b)
    return weights, biases, layer_sizes


def build_composed_with_sum(dec_weights):
    # dec_weights is a dict from train_ae
    dec_keras = []
    for name in ['dec1', 'dec2', 'pixels_out']:
        dec_keras.append((dec_weights[f'{name}_W'].astype(np.float32),
                          dec_weights[f'{name}_b'].astype(np.float32)))

    nnet_w, nnet_b, _ = load_nnet_weights(NNET_PATH)
    nnet_keras = [(W.T.astype(np.float32), b.astype(np.float32))
                  for W, b in zip(nnet_w, nnet_b)]

    inp = keras.Input(shape=(LATENT_DIM,), name="latent_in")
    h = layers.Dense(dec_keras[0][0].shape[1], activation='relu', name='dec1')(inp)
    h = layers.Dense(dec_keras[1][0].shape[1], activation='relu', name='dec2')(h)
    pixels = layers.Dense(dec_keras[2][0].shape[1], activation='linear', name='pixels')(h)
    pixel_sum = layers.Dense(1, activation='linear', use_bias=False, name='pixel_sum')(pixels)

    h2 = pixels
    for idx, (W, b) in enumerate(nnet_keras):
        activation = 'linear' if idx == len(nnet_keras) - 1 else 'relu'
        h2 = layers.Dense(W.shape[1], activation=activation, name=f'ttn{idx}')(h2)

    combined = layers.Concatenate(axis=-1, name='combined_out')([h2, pixel_sum])
    model = keras.Model(inputs=inp, outputs=combined)

    for name, (W, b) in zip(['dec1', 'dec2', 'pixels'], dec_keras):
        model.get_layer(name).set_weights([W, b])
    model.get_layer('pixel_sum').set_weights([np.ones((128, 1), dtype=np.float32)])
    for idx, (W, b) in enumerate(nnet_keras):
        model.get_layer(f'ttn{idx}').set_weights([W, b])

    return model


def export_to_onnx(model, onnx_path):
    saved_dir = tempfile.mkdtemp(prefix="saved_stab_")
    tf.saved_model.save(model, saved_dir)
    result = subprocess.run([
        "python3.11", "-m", "tf2onnx.convert",
        "--saved-model", saved_dir,
        "--output", onnx_path,
        "--opset", "13",
    ], capture_output=True, text=True)
    shutil.rmtree(saved_dir)
    return result.returncode == 0


# -------- Verification --------

from maraboupy import Marabou

def verify_property(onnx_path, kind, z_min, z_max, cte_threshold=None):
    """
    kind: 'P1a', 'P1b', 'P4_C_left', 'P4_C_right', 'P5'
    cte_threshold: for P4_C_right 'upper' direction, find witness with CTE >= threshold
                   for P4_C_left 'lower' direction, find witness with CTE <= -threshold
                   if None, use boundary (0 for P4, 10 for P1)
    """
    net = Marabou.read_onnx(onnx_path)
    input_vars = np.array(net.inputVars[0]).flatten()
    out = np.array(net.outputVars[0]).flatten()
    cte_var, he_var, sum_var = int(out[0]), int(out[1]), int(out[2])

    for i in range(len(input_vars)):
        net.setLowerBound(int(input_vars[i]), float(z_min[i]))
        net.setUpperBound(int(input_vars[i]), float(z_max[i]))
    net.setLowerBound(sum_var, 64.0)
    net.setUpperBound(sum_var, 64.0)

    if kind == "P1a":
        net.setLowerBound(cte_var, 10.0)
    elif kind == "P1b":
        net.setUpperBound(cte_var, -10.0)
    elif kind == "P4_C_left":
        thr = cte_threshold if cte_threshold is not None else 0.0
        net.setUpperBound(cte_var, -float(thr))
    elif kind == "P4_C_right":
        thr = cte_threshold if cte_threshold is not None else 0.0
        net.setLowerBound(cte_var, float(thr))
    elif kind == "P5":
        net.setLowerBound(cte_var, -0.01); net.setUpperBound(cte_var, 0.01)
        net.setLowerBound(he_var, -0.01); net.setUpperBound(he_var, 0.01)

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=300)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out["CTE"] = round(float(vals[cte_var]), 4)
    return out


def main():
    print("Loading data...")
    images, labels = load_data()
    cte = labels[:, 0]

    masks = {
        "centered": np.abs(cte) < 1.0,
        "left": cte > 2.0,
        "right": cte < -2.0,
        "off_center": np.abs(cte) > 1.0,
    }

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        seed_results = {"ae_training": {}, "verification": {}}

        # For seed 42, we already have trained AEs - load them and build composed ONNX if not exists
        # For other seeds, train fresh
        onnx_paths = {}
        for subset in SUBSETS:
            if seed == 42:
                print(f"  [{subset}] using existing seed 42 weights + ONNX")
                ae_weights = np.load(f"autoencoder_{subset}_weights.npz")
                lat_min = ae_weights["latent_min"]
                lat_max = ae_weights["latent_max"]
                onnx_paths[subset] = f"composed_{subset}_with_sum.onnx"
                seed_results["ae_training"][subset] = {
                    "latent_min": lat_min.tolist(), "latent_max": lat_max.tolist(),
                    "note": "existing artifact"
                }
            else:
                print(f"  [{subset}] training fresh (seed={seed})...")
                ae = train_ae(subset, images[masks[subset]], seed)
                seed_results["ae_training"][subset] = {
                    "latent_min": ae["latent_min"].tolist(),
                    "latent_max": ae["latent_max"].tolist(),
                    "recon_linf_max": ae["recon_linf_max"],
                    "final_val_loss": ae["final_val_loss"],
                }
                print(f"    recon L_inf max: {ae['recon_linf_max']:.3f}, val loss: {ae['final_val_loss']:.6f}")
                # Build composed ONNX
                print(f"  [{subset}] building composed ONNX...")
                model = build_composed_with_sum(ae["decoder_weights"])
                onnx_path = f"composed_{subset}_with_sum_seed{seed}.onnx"
                success = export_to_onnx(model, onnx_path)
                if not success:
                    print(f"    ERROR building ONNX for {subset} seed {seed}")
                    continue
                onnx_paths[subset] = onnx_path
                # Cache latent bounds for this seed
                seed_results["ae_training"][subset]["_latent_min_arr"] = ae["latent_min"]
                seed_results["ae_training"][subset]["_latent_max_arr"] = ae["latent_max"]

        # Run verification for each property
        print(f"\n  Verifying properties (seed {seed})...")
        queries = [
            ("P1a", None),
            ("P1b", None),
            ("P4_C_right", 3.0),   # sweep: is wrong-sign at 3.0 still reachable?
            ("P4_C_left", 3.0),
            ("P5", None),
        ]
        for kind, threshold in queries:
            subset = AE_MAPPING[kind]
            if subset not in onnx_paths:
                continue
            # Get latent bounds
            if seed == 42:
                ae_weights = np.load(f"autoencoder_{subset}_weights.npz")
                z_min = ae_weights["latent_min"]
                z_max = ae_weights["latent_max"]
            else:
                z_min = seed_results["ae_training"][subset]["_latent_min_arr"]
                z_max = seed_results["ae_training"][subset]["_latent_max_arr"]
            r = verify_property(onnx_paths[subset], kind, z_min, z_max, cte_threshold=threshold)
            name = f"{kind}_thr={threshold}" if threshold is not None else kind
            sat_str = r["status"] + (f" CTE={r.get('CTE', '?')}" if r['status'] == 'sat' else '')
            print(f"    {name:<20} {sat_str}  ({r['time_seconds']:.1f}s)")
            seed_results["verification"][name] = r

        # Clean up arrays (not JSON-serializable)
        for subset in SUBSETS:
            for k in list(seed_results["ae_training"][subset].keys()):
                if k.startswith("_"):
                    del seed_results["ae_training"][subset][k]
        all_results[f"seed_{seed}"] = seed_results

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # Summary
    print("\n" + "=" * 80)
    print("MULTI-SEED STABILITY SUMMARY")
    print("=" * 80)
    queries = ["P1a", "P1b", "P4_C_right_thr=3.0", "P4_C_left_thr=3.0", "P5"]
    print(f"{'Query':<22}", *[f"Seed {s:<4}" for s in SEEDS])
    print("-" * 80)
    for q in queries:
        row = [q]
        for s in SEEDS:
            r = all_results[f"seed_{s}"]["verification"].get(q)
            if r is None:
                row.append("-")
            elif r["status"] == "sat":
                row.append(f"sat CTE={r.get('CTE', '?')}")
            else:
                row.append(r["status"])
        print(f"{row[0]:<22}", *[f"{x:<10}" for x in row[1:]])


if __name__ == "__main__":
    main()
