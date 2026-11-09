"""
Build 4 composed Keras models (one per subset AE) and export each to ONNX.

Each composed network: z -> decoder_subset -> TinyTaxiNet -> (CTE, HE)

The result is 4 ONNX files:
  composed_centered.onnx   (for P1a, P1b)
  composed_left.onnx       (for P4 C_left)
  composed_right.onnx      (for P4 C_right)
  composed_off_center.onnx (for P5)
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


def build_composed(ae_weights_path):
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
    h = layers.Dense(dec_weights[2][0].shape[1], activation='linear', name='pixels')(h)
    for idx, (W, b) in enumerate(nnet_keras):
        activation = 'linear' if idx == len(nnet_keras) - 1 else 'relu'
        h = layers.Dense(W.shape[1], activation=activation, name=f'ttn{idx}')(h)
    model = keras.Model(inp, h)

    for name, (W, b) in zip(['dec1', 'dec2', 'pixels'], dec_weights):
        model.get_layer(name).set_weights([W, b])
    for idx, (W, b) in enumerate(nnet_keras):
        model.get_layer(f'ttn{idx}').set_weights([W, b])

    return model, latent_dim


def main():
    import tf2onnx
    for subset in ["centered", "left", "right", "off_center"]:
        print(f"\nBuilding composed ONNX for {subset}...")
        ae_path = f"autoencoder_{subset}_weights.npz"
        onnx_path = f"composed_{subset}.onnx"
        model, latent_dim = build_composed(ae_path)
        spec = (tf.TensorSpec((None, latent_dim), tf.float32, name="latent_in"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)
        print(f"  Saved: {onnx_path}")


if __name__ == "__main__":
    main()
