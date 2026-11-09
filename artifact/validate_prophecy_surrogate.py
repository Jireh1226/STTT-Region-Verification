"""
Validate the flat Keras surrogate used for activation extraction in P1-patterned.

Checks two things on centered images (|CTE| < 1.0m):
1. Output agreement between the flat Keras model and TinyTaxiNet.nnet
2. Activation-pattern agreement between Keras hidden-layer outputs and
   Marabou's internal post-ReLU variables when inputs are fixed exactly

Usage:
  ./prophecy_env310/bin/python validate_prophecy_surrogate.py
"""

import json
import time

import h5py
import numpy as np
import tensorflow as tf
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
KERAS_PATH = "Prophecy/dataset_models/kj_tiny_taxinet/KJ_Taxinet_flat.h5"
OUTPUT_JSON = "prophecy_surrogate_validation.json"

CENTERED_THRESHOLD = 1.0
TOL = 1e-7

POST_RELU_RANGES = {
    "dense_0": (144, 160),
    "dense_1": (168, 176),
    "dense_2": (184, 192),
}


def load_centered_images():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:]
        labels = f["X_train"][:]

    centered_mask = np.abs(labels[:, 0]) < CENTERED_THRESHOLD
    centered_indices = np.where(centered_mask)[0]
    centered_images = images[centered_mask].reshape(-1, 128).astype(np.float32)
    return centered_images, centered_indices


def get_keras_outputs_and_activations(images):
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)

    intermediate_models = {
        layer_name: tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output,
        )
        for layer_name in ["dense_0", "dense_1", "dense_2"]
    }

    outputs = model.predict(images, verbose=0)
    activations = {
        layer_name: intermediate.predict(images, verbose=0)
        for layer_name, intermediate in intermediate_models.items()
    }
    return outputs, activations


def get_nnet_outputs(images):
    net = Marabou.read_nnet(NNET_PATH)
    outputs = [
        np.array(net.evaluateWithoutMarabou(images[i:i + 1])[0], dtype=np.float64)
        for i in range(len(images))
    ]
    return np.array(outputs, dtype=np.float64)


def validate_patterns(images, keras_outputs, keras_activations):
    options = Marabou.createOptions(verbosity=0)
    output_diffs = []
    layer_stats = {
        name: {
            "value_max_abs_diff": 0.0,
            "value_mean_abs_diff": 0.0,
            "pattern_mismatch_images": 0,
            "pattern_mismatch_neurons": 0,
        }
        for name in POST_RELU_RANGES
    }

    for image_idx, x0 in enumerate(images):
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()
        for i, val in enumerate(x0):
            net.setLowerBound(input_vars[i], float(val))
            net.setUpperBound(input_vars[i], float(val))

        status, vals, _ = net.solve(verbose=False, options=options)
        if status.strip().lower() != "sat":
            raise RuntimeError(f"Expected SAT for fixed input, got {status!r} on image {image_idx}")

        output_diffs.append(
            [
                abs(vals[192] - float(keras_outputs[image_idx, 0])),
                abs(vals[193] - float(keras_outputs[image_idx, 1])),
            ]
        )

        for layer_name, (lo, hi) in POST_RELU_RANGES.items():
            marabou_vals = np.array([vals[j] for j in range(lo, hi)], dtype=np.float64)
            keras_vals = np.array(keras_activations[layer_name][image_idx], dtype=np.float64)
            abs_diff = np.abs(marabou_vals - keras_vals)

            layer_stats[layer_name]["value_max_abs_diff"] = max(
                layer_stats[layer_name]["value_max_abs_diff"],
                float(abs_diff.max()),
            )
            layer_stats[layer_name]["value_mean_abs_diff"] += float(abs_diff.mean())

            marabou_pattern = marabou_vals > TOL
            keras_pattern = keras_vals > TOL
            mismatches = marabou_pattern != keras_pattern
            mismatch_count = int(np.count_nonzero(mismatches))
            if mismatch_count:
                layer_stats[layer_name]["pattern_mismatch_images"] += 1
                layer_stats[layer_name]["pattern_mismatch_neurons"] += mismatch_count

    output_diffs = np.array(output_diffs, dtype=np.float64)
    for layer_name in layer_stats:
        layer_stats[layer_name]["value_mean_abs_diff"] /= len(images)
    return output_diffs, layer_stats


if __name__ == "__main__":
    start = time.time()

    centered_images, centered_indices = load_centered_images()
    print(f"Centered images: {len(centered_images)}")

    keras_outputs, keras_activations = get_keras_outputs_and_activations(centered_images)
    nnet_outputs = get_nnet_outputs(centered_images)

    output_abs_diff = np.abs(nnet_outputs - keras_outputs)
    output_diffs_fixed, layer_stats = validate_patterns(centered_images, keras_outputs, keras_activations)

    results = {
        "n_centered_images": int(len(centered_images)),
        "centered_threshold": CENTERED_THRESHOLD,
        "output_validation": {
            "max_abs_diff_cte": float(output_abs_diff[:, 0].max()),
            "max_abs_diff_he": float(output_abs_diff[:, 1].max()),
            "mean_abs_diff_cte": float(output_abs_diff[:, 0].mean()),
            "mean_abs_diff_he": float(output_abs_diff[:, 1].mean()),
            "fixed_input_solve_max_abs_diff_cte": float(output_diffs_fixed[:, 0].max()),
            "fixed_input_solve_max_abs_diff_he": float(output_diffs_fixed[:, 1].max()),
        },
        "activation_validation": layer_stats,
        "runtime_seconds": round(time.time() - start, 4),
        "sample_indices_preview": centered_indices[:10].tolist(),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\nOutput agreement:")
    print(f"  max |ΔCTE| via evaluateWithoutMarabou: {results['output_validation']['max_abs_diff_cte']:.10f}")
    print(f"  max |ΔHE|  via evaluateWithoutMarabou: {results['output_validation']['max_abs_diff_he']:.10f}")
    print(f"  max |ΔCTE| via fixed-input solve:      {results['output_validation']['fixed_input_solve_max_abs_diff_cte']:.10f}")
    print(f"  max |ΔHE|  via fixed-input solve:      {results['output_validation']['fixed_input_solve_max_abs_diff_he']:.10f}")

    print("\nActivation agreement:")
    for layer_name in ["dense_0", "dense_1", "dense_2"]:
        stats = results["activation_validation"][layer_name]
        print(
            f"  {layer_name}: max |Δ|={stats['value_max_abs_diff']:.10f}, "
            f"pattern mismatch images={stats['pattern_mismatch_images']}, "
            f"mismatch neurons={stats['pattern_mismatch_neurons']}"
        )

    print(f"\nSaved {OUTPUT_JSON}")
