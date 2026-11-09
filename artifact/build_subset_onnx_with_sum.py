"""
Build composed ONNX models (subset AE + TinyTaxiNet) with an EXTRA output
exposing the pixel sum. This lets Marabou constrain sum(pixels) = 64
(i.e., mean(x) = 0.5) on the decoded pixel layer.

Outputs:
  composed_centered_with_sum.onnx
  composed_left_with_sum.onnx
  composed_right_with_sum.onnx
  composed_off_center_with_sum.onnx

Each ONNX network produces a 3-element output: [CTE, HE, pixel_sum].
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"


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
    i += 1   # move past layer_sizes line
    i += 5   # skip symmetric + means + ranges + mins + maxes
    weights = []
    biases = []
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
        weights.append(W)
        biases.append(b)
    return weights, biases, layer_sizes


def build_composed_with_sum(ae_weights_path):
    ae = np.load(ae_weights_path)
    dec_weights = []
    for name in ['dec1', 'dec2', 'pixels_out']:
        W = ae[f'{name}_W'].astype(np.float32)
        b = ae[f'{name}_b'].astype(np.float32)
        dec_weights.append((W, b))

    nnet_w, nnet_b, _ = load_nnet_weights(NNET_PATH)
    nnet_keras = [(W.T.astype(np.float32), b.astype(np.float32))
                  for W, b in zip(nnet_w, nnet_b)]

    latent_dim = dec_weights[0][0].shape[0]

    inp = keras.Input(shape=(latent_dim,), name="latent_in")
    h = layers.Dense(dec_weights[0][0].shape[1], activation='relu', name='dec1')(inp)
    h = layers.Dense(dec_weights[1][0].shape[1], activation='relu', name='dec2')(h)
    pixels = layers.Dense(dec_weights[2][0].shape[1], activation='linear', name='pixels')(h)

    # Pixel sum as a linear Dense(1) with all-ones weights, no bias
    pixel_sum = layers.Dense(1, activation='linear', use_bias=False, name='pixel_sum')(pixels)

    # TinyTaxiNet on pixels
    h2 = pixels
    for idx, (W, b) in enumerate(nnet_keras):
        activation = 'linear' if idx == len(nnet_keras) - 1 else 'relu'
        h2 = layers.Dense(W.shape[1], activation=activation, name=f'ttn{idx}')(h2)
    ttn_out = h2  # shape (batch, 2) = [CTE, HE]

    # Concatenate to form [CTE, HE, pixel_sum] - 3 outputs
    combined = layers.Concatenate(axis=-1, name='combined_out')([ttn_out, pixel_sum])
    model = keras.Model(inputs=inp, outputs=combined, name='composed_with_sum')

    # Set decoder weights
    for name, (W, b) in zip(['dec1', 'dec2', 'pixels'], dec_weights):
        model.get_layer(name).set_weights([W, b])
    # Set pixel_sum to all-ones
    ones_W = np.ones((128, 1), dtype=np.float32)
    model.get_layer('pixel_sum').set_weights([ones_W])
    # Set TinyTaxiNet weights
    for idx, (W, b) in enumerate(nnet_keras):
        model.get_layer(f'ttn{idx}').set_weights([W, b])

    return model, latent_dim


def main():
    import tf2onnx
    import tempfile, shutil
    for subset in ["centered", "left", "right", "off_center"]:
        print(f"\nBuilding composed ONNX (with pixel sum) for {subset}...")
        ae_path = f"autoencoder_{subset}_weights.npz"
        onnx_path = f"composed_{subset}_with_sum.onnx"
        model, latent_dim = build_composed_with_sum(ae_path)

        # Save as SavedModel first (more robust for tf2onnx)
        saved_model_dir = tempfile.mkdtemp(prefix=f"saved_{subset}_")
        tf.saved_model.save(model, saved_model_dir)

        # Convert SavedModel -> ONNX via CLI (more stable than direct Keras->ONNX)
        import subprocess
        result = subprocess.run([
            "python3.11", "-m", "tf2onnx.convert",
            "--saved-model", saved_model_dir,
            "--output", onnx_path,
            "--opset", "13",
        ], capture_output=True, text=True)
        shutil.rmtree(saved_model_dir)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            continue
        print(f"  Saved: {onnx_path}")

        # Sanity check: load in Marabou
        from maraboupy import Marabou
        net = Marabou.read_onnx(onnx_path)
        print(f"  Marabou: input {np.array(net.inputVars[0]).shape}, output {np.array(net.outputVars[0]).shape}")


if __name__ == "__main__":
    main()
