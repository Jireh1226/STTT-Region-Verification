"""
Build a composed Keras model (Decoder -> TinyTaxiNet) and export to ONNX.
The resulting single network takes an 8-D latent input and outputs (CTE, HE).

This avoids Marabou's composition-via-equations fragility by presenting the
whole pipeline as one feedforward ReLU network.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
AE_WEIGHTS = "autoencoder_weights.npz"
ONNX_PATH = "composed_decoder_tinytaxi.onnx"


def load_nnet_weights(path):
    """Parse TinyTaxiNet.nnet and return list of (W, b) for each layer.
    .nnet format: layer sizes [128, 16, 8, 8, 2], ReLU on hidden, linear output.
    """
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while lines[i].startswith('//'):
        i += 1
    # Header: numLayers, numInputs, numOutputs, maxLayerSize
    header = [int(x) for x in lines[i].split(',') if x.strip()]
    num_layers = header[0]
    i += 1
    layer_sizes = [int(x) for x in lines[i].split(',') if x.strip()]
    i += 1
    # Skip symmetric flag
    i += 1
    # Means, ranges, mins, maxes - 4 lines
    i += 4

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


def build_composed_model():
    # Load autoencoder decoder weights
    ae = np.load(AE_WEIGHTS)
    dec_weights = []
    for name in ['dec1', 'dec2', 'pixels_out']:
        W = ae[f'{name}_W'].astype(np.float32)   # (in, out) in Keras
        b = ae[f'{name}_b'].astype(np.float32)
        dec_weights.append((W, b))

    # Load TinyTaxiNet weights from .nnet
    nnet_w, nnet_b, nnet_sizes = load_nnet_weights(NNET_PATH)
    # Keras expects (in, out). nnet gives (out, in).
    nnet_keras = [(W.T.astype(np.float32), b.astype(np.float32))
                  for W, b in zip(nnet_w, nnet_b)]

    latent_dim = dec_weights[0][0].shape[0]    # 8

    # Build Keras sequential model: latent -> decoder -> TinyTaxiNet
    inp = keras.Input(shape=(latent_dim,), name="latent_in")
    # Decoder
    h = layers.Dense(dec_weights[0][0].shape[1], activation='relu', name='dec1')(inp)
    h = layers.Dense(dec_weights[1][0].shape[1], activation='relu', name='dec2')(h)
    h = layers.Dense(dec_weights[2][0].shape[1], activation='linear', name='pixels')(h)
    # TinyTaxiNet layers (128 -> 16 -> 8 -> 8 -> 2 with ReLU on hidden, linear output)
    for idx, (W, b) in enumerate(nnet_keras):
        activation = 'linear' if idx == len(nnet_keras) - 1 else 'relu'
        h = layers.Dense(W.shape[1], activation=activation, name=f'ttn{idx}')(h)
    out = h
    model = keras.Model(inp, out, name="composed")

    # Now load weights
    for name, (W, b) in zip(['dec1', 'dec2', 'pixels'], dec_weights):
        model.get_layer(name).set_weights([W, b])
    for idx, (W, b) in enumerate(nnet_keras):
        model.get_layer(f'ttn{idx}').set_weights([W, b])

    return model, latent_dim


def main():
    print("Building composed Keras model...")
    model, latent_dim = build_composed_model()
    model.summary()

    # Sanity check: does the composed model give the right output on a training latent?
    ae = np.load(AE_WEIGHTS)
    train_latents = ae['latents'][:5]
    predictions = model.predict(train_latents, verbose=0)
    print(f"\nSanity check - first 5 training latents produce outputs:")
    print(predictions)

    # Export to ONNX
    print(f"\nExporting to ONNX at {ONNX_PATH}...")
    import tf2onnx
    spec = (tf.TensorSpec((None, latent_dim), tf.float32, name="latent_in"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                                opset=13, output_path=ONNX_PATH)
    print(f"Saved ONNX model ({len(model_proto.graph.node)} nodes).")

    # Verify Marabou can read it
    print("\nTest-loading ONNX into Marabou...")
    from maraboupy import Marabou
    net = Marabou.read_onnx(ONNX_PATH)
    print(f"  Marabou network loaded. Input vars: {net.inputVars[0].shape}")
    print(f"  Output vars: {net.outputVars[0].shape if hasattr(net.outputVars[0], 'shape') else net.outputVars[0]}")

    # Forward check: does Marabou's forward agree with Keras?
    z_test = train_latents[0:1].astype(np.float32)
    marabou_out = net.evaluate(z_test, options=Marabou.createOptions(verbosity=0))
    keras_out = model.predict(z_test, verbose=0)
    print(f"  Keras output:   {keras_out.flatten()}")
    print(f"  Marabou output: {marabou_out.flatten()}")
    diff = np.max(np.abs(keras_out - marabou_out))
    print(f"  Max diff: {diff:.8f}")


if __name__ == "__main__":
    main()
