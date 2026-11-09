"""
Compose each trained Conv AE decoder with TinyTaxiNet into a single ONNX
network that takes latent z in R^8 and outputs [CTE, HE, pixel_sum].

Uses the saved Keras AE (conv_autoencoder_{subset}.keras), extracts the
decoder graph by slicing at the `latent` layer, attaches a pixel-sum head and
the TinyTaxiNet Dense layers, and exports via tf2onnx.

Outputs: composed_conv_{subset}_with_sum.onnx
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tempfile, shutil, subprocess
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
    i += 1; i += 5
    weights = []; biases = []
    for l in range(num_layers):
        rows = layer_sizes[l + 1]; cols = layer_sizes[l]
        W = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            vals = [float(x) for x in lines[i].strip().rstrip(',').split(',')]
            W[r] = vals[:cols]; i += 1
        b = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(',')); i += 1
        weights.append(W); biases.append(b)
    return weights, biases


def extract_decoder_weights(subset):
    """Load AE, extract decoder layer weights as numpy, then drop the AE."""
    ae = keras.models.load_model(f"conv_autoencoder_{subset}.keras", compile=False, safe_mode=False)
    weights = {ln: [w.copy() for w in ae.get_layer(ln).get_weights()]
               for ln in ["dec_dense1", "dec_dense2", "dec_conv1", "dec_conv2"]}
    latent_dim = ae.get_layer("latent").output.shape[-1]
    del ae
    keras.backend.clear_session()
    return weights, latent_dim


def build_composed(subset):
    """Extract decoder weights, clear Keras session, then rebuild composed graph
    from scratch. Two-phase to avoid Keras 'Functional' deserialization issues
    when multiple .keras files are loaded sequentially."""
    decoder_weights, latent_dim = extract_decoder_weights(subset)
    nnet_w, nnet_b = load_nnet_weights(NNET_PATH)

    inp = keras.Input(shape=(latent_dim,), name="latent_in")
    h = layers.Dense(32, activation="relu", name="dec_dense1")(inp)
    h = layers.Dense(8 * 16 * 8, activation="relu", name="dec_dense2")(h)
    h = layers.Reshape((8, 16, 8), name="dec_reshape")(h)
    h = layers.Conv2D(4, 3, padding="same", activation="relu", name="dec_conv1")(h)
    h = layers.Conv2D(1, 3, padding="same", activation="linear", name="dec_conv2")(h)
    pixels = layers.Flatten(name="pixels_out")(h)

    pixel_sum = layers.Dense(1, activation="linear", use_bias=False, name="pixel_sum")(pixels)

    h2 = pixels
    for idx, (W, b) in enumerate(zip(nnet_w, nnet_b)):
        act = "linear" if idx == len(nnet_w) - 1 else "relu"
        h2 = layers.Dense(W.shape[0], activation=act, name=f"ttn{idx}")(h2)
    ttn_out = h2

    combined = layers.Concatenate(axis=-1, name="combined")([ttn_out, pixel_sum])
    composed = keras.Model(inp, combined, name=f"composed_conv_{subset}")

    # Copy decoder weights from trained AE
    for name in ["dec_dense1", "dec_dense2", "dec_conv1", "dec_conv2"]:
        composed.get_layer(name).set_weights(decoder_weights[name])

    composed.get_layer("pixel_sum").set_weights([np.ones((128, 1), dtype=np.float32)])

    for idx, (W, b) in enumerate(zip(nnet_w, nnet_b)):
        composed.get_layer(f"ttn{idx}").set_weights(
            [W.T.astype(np.float32), b.astype(np.float32)])

    return composed, latent_dim


def build_one_subset(subset):
    """Single-subset build: only called from a fresh subprocess to avoid
    Keras-Functional reload pollution."""
    print(f"Building composed conv ONNX for {subset}...", flush=True)
    model, latent_dim = build_composed(subset)
    sm_dir = tempfile.mkdtemp(prefix=f"saved_conv_{subset}_")
    tf.saved_model.save(model, sm_dir)
    onnx_path = f"composed_conv_{subset}_with_sum.onnx"
    r = subprocess.run([
        "python3.11", "-m", "tf2onnx.convert",
        "--saved-model", sm_dir,
        "--output", onnx_path,
        "--opset", "13",
    ], capture_output=True, text=True)
    shutil.rmtree(sm_dir)
    if r.returncode != 0:
        print(f"  ERROR tf2onnx: {r.stderr[-400:]}", flush=True)
        return
    print(f"  Saved {onnx_path}", flush=True)
    from maraboupy import Marabou
    net = Marabou.read_onnx(onnx_path)
    print(f"  Marabou: input {np.array(net.inputVars[0]).shape}, "
          f"output {np.array(net.outputVars[0]).shape}", flush=True)


def main():
    """Dispatch: if called with --subset NAME, build that one (subprocess-friendly).
    Otherwise spawn a subprocess per subset to keep Keras state clean."""
    import sys
    if "--subset" in sys.argv:
        subset = sys.argv[sys.argv.index("--subset") + 1]
        build_one_subset(subset)
        return
    for subset in ["centered", "left", "right", "off_center"]:
        r = subprocess.run(
            ["./prophecy_env/bin/python", "-u", __file__, "--subset", subset],
            capture_output=False, text=True)
        if r.returncode != 0:
            print(f"  subprocess for {subset} exited {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
