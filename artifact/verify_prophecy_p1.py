"""
Prophecy-style P1 Safety Verification of TinyTaxiNet using Marabou 2.0

Implements Kadron et al. (VSTTE 2021) "sequence property" approach:
1. Enumerate FULL activation patterns across ALL hidden layers (dense_0, dense_1, dense_2)
2. For each pattern among centered images, compute per-pixel input bounds
3. Fix ALL ReLU activations to match the pattern (fully linearizing the network)
4. Verify P1 safety: |CTE| < 10.0 AND |HE| < 90.0

When all ReLUs are fixed, the network is piecewise-linear within each region,
so Marabou solves a pure LP — no case splits needed.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import time
import h5py
from maraboupy import Marabou, MarabouUtils, MarabouCore

# ── Configuration ──────────────────────────────────────────────────────
NNET_PATH = './VerifyGAN/models/TinyTaxiNet.nnet'
DATA_PATH = './VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
CTE_BOUND = 10.0
HE_BOUND = 90.0
CENTERED_THRESHOLD = 1.0  # |true CTE| < 1.0m for "centered"
OUTPUT_JSON = 'prophecy_p1_results.json'

# Variable layout for .nnet with sizes [128, 16, 8, 8, 2]:
#   Inputs:  0-127
#   Layer 0 (128→16): pre-ReLU 128-143, post-ReLU 144-159
#   Layer 1 (16→8):   pre-ReLU 160-167, post-ReLU 168-175
#   Layer 2 (8→8):    pre-ReLU 176-183, post-ReLU 184-191
#   Output (8→2):     192-193
LAYER_INFO = [
    ('dense_0', 128, 16),  # (name, pre_relu_start, num_neurons)
    ('dense_1', 160, 8),
    ('dense_2', 176, 8),
]

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
with h5py.File(DATA_PATH, 'r') as f:
    images = np.array(f['y_train']).reshape(-1, 128)  # (10000, 128)
    labels = np.array(f['X_train'])  # (10000, 3)

# ── Load Keras model for activation extraction ─────────────────────────
print("Loading Keras model for activation extraction...")
import tensorflow as tf
model = tf.keras.models.load_model(
    './Prophecy/dataset_models/kj_tiny_taxinet/KJ_Taxinet_flat.h5',
    compile=False
)
preds = model.predict(images.astype(np.float32), verbose=0)

# Get activations at all hidden layers
acts_all = {}
for layer_name, _, _ in LAYER_INFO:
    intermediate = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    acts_all[layer_name] = intermediate.predict(images.astype(np.float32), verbose=0)

# ── Enumerate FULL patterns among centered images ──────────────────────
centered_mask = np.abs(labels[:, 0]) < CENTERED_THRESHOLD
centered_imgs = images[centered_mask]
centered_preds = preds[centered_mask]

# Concatenate activation patterns from all layers: [16 + 8 + 8 = 32 bits]
centered_patterns = np.hstack([
    (acts_all[name][centered_mask] > 0).astype(int)
    for name, _, _ in LAYER_INFO
])

unique_patterns = np.unique(centered_patterns, axis=0)
print(f"\nCentered images: {len(centered_imgs)}")
print(f"Unique FULL activation patterns (all 3 layers): {len(unique_patterns)}")

# ── Verify P1 on each pattern region ───────────────────────────────────
results = {
    'config': {
        'cte_bound': CTE_BOUND,
        'he_bound': HE_BOUND,
        'centered_threshold': CENTERED_THRESHOLD,
        'nnet_path': NNET_PATH,
        'total_centered_images': int(np.sum(centered_mask)),
        'total_patterns': len(unique_patterns),
        'approach': 'full_sequence_property',
        'layers_constrained': ['dense_0 (16 neurons)', 'dense_1 (8 neurons)', 'dense_2 (8 neurons)'],
        'total_neurons_constrained': 32,
    },
    'patterns': []
}

total_verified = 0
total_violated = 0
total_images_covered = 0
total_queries = 0
total_time = 0

for pat_idx, pattern in enumerate(unique_patterns):
    pat_mask = np.all(centered_patterns == pattern, axis=1)
    pat_images = centered_imgs[pat_mask]
    pat_preds = centered_preds[pat_mask]
    count = len(pat_images)
    total_images_covered += count

    # Split pattern back into per-layer patterns
    pat_dense0 = pattern[:16]
    pat_dense1 = pattern[16:24]
    pat_dense2 = pattern[24:32]

    print(f"\n--- Pattern {pat_idx+1}/{len(unique_patterns)} ({count} images) ---")
    print(f"  dense_0: {pat_dense0.tolist()}")
    print(f"  dense_1: {pat_dense1.tolist()}")
    print(f"  dense_2: {pat_dense2.tolist()}")

    # Compute per-pixel input bounds for this pattern's images
    input_mins = pat_images.min(axis=0)
    input_maxs = pat_images.max(axis=0)

    # Observed output ranges
    obs_cte_range = (float(pat_preds[:, 0].min()), float(pat_preds[:, 0].max()))
    obs_he_range = (float(pat_preds[:, 1].min()), float(pat_preds[:, 1].max()))
    print(f"  Observed CTE: [{obs_cte_range[0]:.3f}, {obs_cte_range[1]:.3f}]")
    print(f"  Observed HE:  [{obs_he_range[0]:.3f}, {obs_he_range[1]:.3f}]")

    pattern_result = {
        'pattern_idx': pat_idx,
        'pattern_dense0': pat_dense0.tolist(),
        'pattern_dense1': pat_dense1.tolist(),
        'pattern_dense2': pat_dense2.tolist(),
        'image_count': count,
        'observed_cte_range': obs_cte_range,
        'observed_he_range': obs_he_range,
        'sub_queries': []
    }

    sub_queries = [
        ('P1a_CTE_lower', 0, 'upper', -CTE_BOUND),
        ('P1b_CTE_upper', 0, 'lower', CTE_BOUND),
        ('P1c_HE_lower',  1, 'upper', -HE_BOUND),
        ('P1d_HE_upper',  1, 'lower', HE_BOUND),
    ]

    pattern_verified = True

    for query_name, output_idx, bound_type, bound_val in sub_queries:
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()
        output_vars = net.outputVars[0].flatten()

        # Set input bounds
        for i in range(128):
            net.setLowerBound(input_vars[i], float(input_mins[i]))
            net.setUpperBound(input_vars[i], float(input_maxs[i]))

        # Fix ALL ReLU activations across all 3 hidden layers
        layer_patterns = [
            (128, 16, pat_dense0),  # dense_0
            (160, 8,  pat_dense1),  # dense_1
            (176, 8,  pat_dense2),  # dense_2
        ]
        for pre_relu_start, num_neurons, layer_pat in layer_patterns:
            for j in range(num_neurons):
                pre_relu_var = pre_relu_start + j
                if layer_pat[j] == 1:
                    net.setLowerBound(pre_relu_var, 0.0)
                else:
                    net.setUpperBound(pre_relu_var, 0.0)

        # Set negated output constraint
        if bound_type == 'upper':
            net.setUpperBound(output_vars[output_idx], float(bound_val))
        else:
            net.setLowerBound(output_vars[output_idx], float(bound_val))

        # Solve
        start = time.time()
        result_str, vals, stats = net.solve()
        elapsed = time.time() - start
        total_queries += 1
        total_time += elapsed

        if result_str == 'unsat':
            status = 'UNSAT'
            print(f"  {query_name}: UNSAT [{elapsed:.4f}s]")
        elif result_str == 'sat':
            status = 'SAT'
            pattern_verified = False
            cx = [vals[input_vars[j]] for j in range(128)]
            cx_out = [vals[output_vars[j]] for j in range(2)]
            print(f"  {query_name}: SAT CTE={cx_out[0]:.4f}, HE={cx_out[1]:.4f} [{elapsed:.4f}s]")
        else:
            status = str(result_str)
            pattern_verified = False
            print(f"  {query_name}: {status} [{elapsed:.4f}s]")

        sub_result = {
            'query': query_name,
            'status': status,
            'time_s': round(elapsed, 4),
        }
        if status == 'SAT':
            sub_result['counterexample_output'] = {'CTE': cx_out[0], 'HE': cx_out[1]}
            sub_result['counterexample_input_mean'] = float(np.mean(cx))
        pattern_result['sub_queries'].append(sub_result)

    pattern_result['verified'] = pattern_verified
    if pattern_verified:
        total_verified += 1
        print(f"  → VERIFIED")
    else:
        total_violated += 1
        print(f"  → VIOLATED")

    results['patterns'].append(pattern_result)

# ── Summary ────────────────────────────────────────────────────────────
coverage = total_images_covered / len(centered_imgs) * 100

results['summary'] = {
    'total_patterns': len(unique_patterns),
    'verified_patterns': total_verified,
    'violated_patterns': total_violated,
    'images_covered': total_images_covered,
    'coverage_pct': round(coverage, 2),
    'total_queries': total_queries,
    'total_time_s': round(total_time, 4),
    'avg_time_per_query_s': round(total_time / total_queries, 4) if total_queries > 0 else 0,
}

print(f"\n{'='*60}")
print(f"PROPHECY-STYLE P1 VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"Total activation patterns (centered): {len(unique_patterns)}")
print(f"Verified (UNSAT on all 4 sub-queries): {total_verified}/{len(unique_patterns)}")
print(f"Violated (SAT on some sub-query):      {total_violated}/{len(unique_patterns)}")
print(f"Image coverage: {total_images_covered}/{len(centered_imgs)} ({coverage:.1f}%)")
print(f"Total queries: {total_queries}")
print(f"Total time: {total_time:.2f}s")
print(f"Avg time per query: {total_time/total_queries:.4f}s")

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_JSON}")
