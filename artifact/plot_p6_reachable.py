"""
Generate P6 reachable-set visualization: CTE/HE 2D plot showing:
  - P6 reachable envelope (formal bounds)
  - P1 safety envelope (design intent)
  - Nominal training data distribution (centered images)
  - Full training data (all images)

This visualizes the gap between what the network CAN output (P6 bounds),
what it SHOULD output (P1 safety), and what it DOES output on real data.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from maraboupy import Marabou

DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
NNET_PATH = 'VerifyGAN/models/TinyTaxiNet.nnet'


def load_data():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]
        labels = f['X_train'][:]
    return images.reshape(-1, 128), labels


def run_network(images):
    """Run TinyTaxiNet on a batch of images to get predictions."""
    net = Marabou.read_nnet(NNET_PATH)
    predictions = []
    for img in images:
        options = Marabou.createOptions(verbosity=0)
        pred = net.evaluate(img.reshape(1, -1), options=options)
        predictions.append(pred.flatten())
    return np.array(predictions)


def main():
    images, labels = load_data()
    cte_labels = labels[:, 0]
    he_labels = labels[:, 1]

    # Centered subset (used by P1/P6)
    centered_mask = np.abs(cte_labels) < 1.0
    centered_images = images[centered_mask]

    # Predict on all images (ground truth labels) - we use labels directly
    # Actually we want NETWORK predictions, not ground truth
    # But running all 10000 through maraboupy is slow. Let's sample.
    print("Sampling predictions from dataset...")

    # Sample 500 centered and 500 non-centered for visualization
    rng = np.random.RandomState(42)
    centered_idx = rng.choice(np.where(centered_mask)[0], 500, replace=False)
    other_idx = rng.choice(np.where(~centered_mask)[0], 500, replace=False)

    # For speed, evaluate in batches using Keras-style forward pass
    # Since we have the .nnet, let's just load weights manually
    import struct

    # Read nnet weights
    def load_nnet_weights(path):
        with open(path) as f:
            lines = f.readlines()
        # Skip comment lines
        i = 0
        while lines[i].startswith('//'):
            i += 1
        # Header: numLayers, numInputs, numOutputs, maxLayerSize
        num_layers, num_inputs, num_outputs, _ = [int(x) for x in lines[i].split(',') if x.strip()]
        i += 1
        layer_sizes = [int(x) for x in lines[i].split(',') if x.strip()]
        i += 1
        # symmetric
        i += 1
        # means (num_inputs+1)
        i += 1
        # ranges (num_inputs+1)
        i += 1
        # mins
        i += 1
        # maxs
        i += 1

        weights = []
        biases = []
        for l in range(num_layers):
            rows = layer_sizes[l+1]
            cols = layer_sizes[l]
            W = np.zeros((rows, cols))
            for r in range(rows):
                vals = [float(x) for x in lines[i].strip().rstrip(',').split(',')]
                W[r] = vals[:cols]
                i += 1
            b = np.zeros(rows)
            for r in range(rows):
                b[r] = float(lines[i].strip().rstrip(','))
                i += 1
            weights.append(W)
            biases.append(b)
        return weights, biases

    try:
        weights, biases = load_nnet_weights(NNET_PATH)

        def forward(x):
            h = x
            for i, (W, b) in enumerate(zip(weights, biases)):
                h = h @ W.T + b
                if i < len(weights) - 1:
                    h = np.maximum(h, 0)  # ReLU
            return h

        # Get predictions for all centered images
        preds_centered = np.array([forward(img) for img in centered_images])
        # Get predictions for all off-center images
        other_mask = ~centered_mask
        preds_other = np.array([forward(img) for img in images[other_mask]])

    except Exception as e:
        print(f"Manual forward failed: {e}, falling back to labels")
        preds_centered = labels[centered_mask]
        preds_other = labels[other_mask]

    # ============================================================
    # Plot 1: CTE vs HE scatter + envelopes
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Nominal training data predictions
    ax.scatter(preds_other[:, 0], preds_other[:, 1], c='lightgray', s=3, alpha=0.3, label='Off-center predictions')
    ax.scatter(preds_centered[:, 0], preds_centered[:, 1], c='steelblue', s=6, alpha=0.6, label='Centered predictions')

    # P1 safety envelope (design intent)
    safety = Rectangle((-10, -90), 20, 180, linewidth=2, edgecolor='green',
                       facecolor='none', linestyle='--', label='P1 safety envelope ($\pm 10$m, $\pm 90^\circ$)')
    ax.add_patch(safety)

    # P6 reachable set (formal bound over C_centered)
    p6 = Rectangle((-24.33, -90.10), 24.33 + 23.30, 90.10 + 84.07,
                   linewidth=2, edgecolor='red', facecolor='none', linestyle='-',
                   label='P6 reachable set over $C_{centered}$')
    ax.add_patch(p6)

    # P1 witnesses
    witnesses = [
        (10.0, -52.47, 'P1a'),
        (-10.0, 14.98, 'P1b'),
        (18.48, -90.0, 'P1d'),
    ]
    for cte, he, name in witnesses:
        ax.scatter(cte, he, c='red', s=100, marker='X', zorder=5, edgecolor='black', linewidth=1)
        ax.annotate(name, (cte, he), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight='bold')

    ax.set_xlabel('CTE (m)', fontsize=12)
    ax.set_ylabel('HE (deg)', fontsize=12)
    ax.set_title('TinyTaxiNet reachable output set vs. safety envelope and training data', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Set reasonable axis limits
    ax.set_xlim(-30, 30)
    ax.set_ylim(-100, 100)

    plt.tight_layout()
    plt.savefig('p6_reachable_set.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: p6_reachable_set.png")

    # Summary stats
    print(f"\nTraining data CTE range: [{preds_centered[:,0].min():.2f}, {preds_centered[:,0].max():.2f}]")
    print(f"Training data HE range: [{preds_centered[:,1].min():.2f}, {preds_centered[:,1].max():.2f}]")
    print(f"\nP6 formal bounds:")
    print(f"  CTE: [-24.33, 23.30]")
    print(f"  HE: [-90.10, 84.07]")

    # Ratio: formal bound width / empirical width
    emp_cte_width = preds_centered[:,0].max() - preds_centered[:,0].min()
    emp_he_width = preds_centered[:,1].max() - preds_centered[:,1].min()
    formal_cte_width = 23.30 - (-24.33)
    formal_he_width = 84.07 - (-90.10)
    print(f"\nFormal / empirical width ratio (how much over-approximation):")
    print(f"  CTE: {formal_cte_width / emp_cte_width:.2f}x")
    print(f"  HE:  {formal_he_width / emp_he_width:.2f}x")


if __name__ == '__main__':
    main()
