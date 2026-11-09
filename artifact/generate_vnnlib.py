"""
Generate VNN-LIB property files for TinyTaxiNet P1-P6.

VNN-LIB format (SMTLIB2-based):
  - Declare input variables X_0 ... X_127
  - Declare output variables Y_0 (CTE), Y_1 (HE)
  - Assert input bounds (precondition)
  - Assert NEGATED postcondition (the unsafe / violation region)

The verifier is asked: does there exist an input satisfying the input
bounds such that the output constraints also hold? If SAT, a violation
exists (the property fails). If UNSAT, no such input exists (the property
holds for all inputs in the region). This matches how Marabou encodes
queries internally.

Usage: python3.11 generate_vnnlib.py
"""

import numpy as np
import h5py
import os

NNET_PATH = 'VerifyGAN/models/TinyTaxiNet.nnet'
DATA_PATH = 'VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5'
OUTPUT_DIR = 'vnnlib'

N_INPUTS = 128
N_OUTPUTS = 2


def load_data():
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['y_train'][:]
        labels = f['X_train'][:]
    return images, labels


def write_vnnlib(filepath, input_bounds, output_constraints, comment=""):
    """
    Write a VNN-LIB file.

    input_bounds: list of (lo, hi) for each input variable
    output_constraints: list of disjunctive clauses, where each clause is
                        a list of (var_name, op, value) like ("Y_0", ">=", 10.0)
                        Clauses are OR'd together; within each clause, conditions are AND'd.
    """
    with open(filepath, 'w') as f:
        # Standard normalization/preprocessing header
        f.write("; === Preprocessing and Normalization ===\n")
        f.write("; Model: TinyTaxiNet.nnet (128 -> [16,8,8] -> 2, ReLU activations)\n")
        f.write("; .nnet normalization: none (all input means=0, ranges=1)\n")
        f.write("; Marabou setting: normalize=False\n")
        f.write("; Dataset preprocessing: per-image mean-centering to 0.5, then clip to [0,1]\n")
        f.write("; Note: these specs do not enforce the mean=0.5 preprocessing invariant\n")
        f.write("; Inputs: X_0..X_127 = flattened 8x16 grayscale pixel values\n")
        f.write("; Outputs: Y_0 = CTE (meters from centerline), Y_1 = HE (degrees off alignment)\n")
        f.write(";\n")

        if comment:
            for line in comment.strip().split('\n'):
                f.write(f"; {line}\n")
            f.write("\n")

        # Declare inputs
        for i in range(N_INPUTS):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare outputs
        for i in range(N_OUTPUTS):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Input bounds
        for i, (lo, hi) in enumerate(input_bounds):
            f.write(f"(assert (>= X_{i} {lo:.10f}))\n")
            f.write(f"(assert (<= X_{i} {hi:.10f}))\n")
        f.write("\n")

        # Output constraints (disjunctive normal form)
        if len(output_constraints) == 1:
            # Single conjunction - just assert each condition
            for var, op, val in output_constraints[0]:
                f.write(f"(assert ({op} {var} {val:.10f}))\n")
        else:
            # Multiple disjuncts - wrap in (assert (or ...))
            f.write("(assert (or\n")
            for clause in output_constraints:
                if len(clause) == 1:
                    var, op, val = clause[0]
                    f.write(f"  ({op} {var} {val:.10f})\n")
                else:
                    f.write("  (and\n")
                    for var, op, val in clause:
                        f.write(f"    ({op} {var} {val:.10f})\n")
                    f.write("  )\n")
            f.write("))\n")

    print(f"  Written: {filepath}")


def generate_p1(images, labels):
    """P1-global: C_centered -> |CTE| < 10 AND |HE| < 90
    Negated: CTE >= 10 OR CTE <= -10 OR HE >= 90 OR HE <= -90
    """
    cte = labels[:, 0]
    mask = np.abs(cte) < 1.0
    centered = images[mask].reshape(-1, N_INPUTS)
    pixel_mins = centered.min(axis=0)
    pixel_maxs = centered.max(axis=0)

    bounds = [(float(pixel_mins[i]), float(pixel_maxs[i])) for i in range(N_INPUTS)]

    # 4 separate files, one per sub-query.
    subqueries = [
        ("p1a_cte_upper", [("Y_0", ">=", 10.0)], "P1a: CTE >= 10 (negation of CTE < 10)"),
        ("p1b_cte_lower", [("Y_0", "<=", -10.0)], "P1b: CTE <= -10 (negation of CTE > -10)"),
        ("p1c_he_upper", [("Y_1", ">=", 90.0)], "P1c: HE >= 90 (negation of HE < 90)"),
        ("p1d_he_lower", [("Y_1", "<=", -90.0)], "P1d: HE <= -90 (negation of HE > -90)"),
    ]

    for name, constraints, desc in subqueries:
        comment = f"Property: P1-global Safety Bound\n"
        comment += f"Sub-query: {desc}\n"
        comment += f"Input region: C_centered (per-pixel box from {mask.sum()} images with |CTE| < 1.0m)\n"
        comment += f"Network: TinyTaxiNet.nnet (128 -> [16,8,8] -> 2)\n"
        comment += f"Output: Y_0 = CTE (meters), Y_1 = HE (degrees)\n"
        comment += f"Expected: SAT = safety violated, UNSAT = safety holds for this direction"
        write_vnnlib(
            os.path.join(OUTPUT_DIR, f"{name}.vnnlib"),
            bounds,
            [[c] for c in constraints] if len(constraints) > 1 else [constraints],
            comment
        )


def generate_p2(images, labels):
    """P2: ||x - x0||_inf <= 0.02 -> |CTE - CTE*| < 1.0 AND |HE - HE*| < 5.0
    Generate one file per image (20 images x 4 sub-queries = 80 files, or 20 with disjunction).
    For tractability, generate 20 files with disjunctive output constraints.
    """
    cte = labels[:, 0]
    he = labels[:, 1]
    near_center_mask = np.abs(cte) < 2.0
    near_center_indices = np.where(near_center_mask)[0]
    sample_indices = near_center_indices[
        np.linspace(0, len(near_center_indices) - 1, 20, dtype=int)
    ]

    epsilon = 0.02
    cte_bound = 1.0
    he_bound = 5.0

    for img_num, img_idx in enumerate(sample_indices):
        x0 = images[img_idx].flatten()
        cte_true = float(cte[img_idx])
        he_true = float(he[img_idx])

        bounds = []
        for i in range(N_INPUTS):
            lo = max(0.0, x0[i] - epsilon)
            hi = min(1.0, x0[i] + epsilon)
            bounds.append((float(lo), float(hi)))

        # Negated postcondition (disjunction):
        # CTE >= CTE* + 1.0 OR CTE <= CTE* - 1.0 OR HE >= HE* + 5.0 OR HE <= HE* - 5.0
        output_constraints = [
            [("Y_0", ">=", cte_true + cte_bound)],
            [("Y_0", "<=", cte_true - cte_bound)],
            [("Y_1", ">=", he_true + he_bound)],
            [("Y_1", "<=", he_true - he_bound)],
        ]

        comment = f"Property: P2 Local Correctness\n"
        comment += f"Image {img_num+1}/20 (dataset index {img_idx})\n"
        comment += f"Ground truth: CTE* = {cte_true:.6f}, HE* = {he_true:.6f}\n"
        comment += f"Input region: L-inf ball epsilon={epsilon} around training image, clipped to [0,1]\n"
        comment += f"Postcondition: |CTE - CTE*| < {cte_bound} AND |HE - HE*| < {he_bound}\n"
        comment += f"Negated (encoded below): CTE >= {cte_true+cte_bound:.6f} OR CTE <= {cte_true-cte_bound:.6f} OR HE >= {he_true+he_bound:.6f} OR HE <= {he_true-he_bound:.6f}\n"
        comment += f"Network: TinyTaxiNet.nnet\n"
        comment += f"Expected: SAT = correctness violated"

        write_vnnlib(
            os.path.join(OUTPUT_DIR, f"p2_img{img_num+1:02d}.vnnlib"),
            bounds,
            output_constraints,
            comment
        )


def generate_p3(images, labels):
    """P3: ||x - x0||_inf <= eps -> |CTE| < 10 AND |HE| < 90
    Generate files for eps in {0.05, 0.10}, 20 images each.
    """
    cte = labels[:, 0]
    centered_mask = np.abs(cte) < 1.0
    centered_indices = np.where(centered_mask)[0]
    sample_indices = centered_indices[
        np.linspace(0, len(centered_indices) - 1, 20, dtype=int)
    ]

    for epsilon in [0.05, 0.10]:
        for img_num, img_idx in enumerate(sample_indices):
            x0 = images[img_idx].flatten()
            cte_true = float(cte[img_idx])

            bounds = []
            for i in range(N_INPUTS):
                lo = max(0.0, x0[i] - epsilon)
                hi = min(1.0, x0[i] + epsilon)
                bounds.append((float(lo), float(hi)))

            output_constraints = [
                [("Y_0", ">=", 10.0)],
                [("Y_0", "<=", -10.0)],
                [("Y_1", ">=", 90.0)],
                [("Y_1", "<=", -90.0)],
            ]

            comment = f"Property: P3 Robustness Under Input Noise\n"
            comment += f"Epsilon: {epsilon}\n"
            comment += f"Image {img_num+1}/20 (dataset index {img_idx}, CTE={cte_true:.4f})\n"
            comment += f"Input region: L-inf ball epsilon={epsilon} around centered training image\n"
            comment += f"Postcondition: |CTE| < 10 AND |HE| < 90\n"
            comment += f"Negated: CTE >= 10 OR CTE <= -10 OR HE >= 90 OR HE <= -90\n"
            comment += f"Network: TinyTaxiNet.nnet"

            eps_str = f"{int(epsilon * 100):03d}"  # 0.05->005, 0.10->010
            write_vnnlib(
                os.path.join(OUTPUT_DIR, f"p3_eps{eps_str}_img{img_num+1:02d}.vnnlib"),
                bounds,
                output_constraints,
                comment
            )


def generate_p4(images, labels):
    """P4: C_left -> CTE > 0, C_right -> CTE < 0
    Negated: CTE <= 0, CTE >= 0 respectively.
    """
    cte = labels[:, 0]
    flat = images.reshape(-1, N_INPUTS)

    for region_name, mask, post, neg_op, neg_val in [
        ('C_left', cte > 2.0, 'CTE > 0', '<=', 0.0),
        ('C_right', cte < -2.0, 'CTE < 0', '>=', 0.0),
    ]:
        region_imgs = flat[mask]
        pixel_mins = region_imgs.min(axis=0)
        pixel_maxs = region_imgs.max(axis=0)

        bounds = [(float(pixel_mins[i]), float(pixel_maxs[i])) for i in range(N_INPUTS)]

        comment = f"Property: P4 Directional Correctness ({region_name})\n"
        comment += f"Input region: {region_name} (per-pixel box from {mask.sum()} images)\n"
        comment += f"Postcondition: {post}\n"
        comment += f"Negated: CTE {neg_op} {neg_val}\n"
        comment += f"Network: TinyTaxiNet.nnet"

        write_vnnlib(
            os.path.join(OUTPUT_DIR, f"p4_{region_name.lower()}.vnnlib"),
            bounds,
            [[("Y_0", neg_op, neg_val)]],
            comment
        )


def generate_p5(images, labels):
    """P5: C_off_center -> NOT(|CTE| <= 0.01 AND |HE| <= 0.01)
    Negated: |CTE| <= 0.01 AND |HE| <= 0.01
    Encoded as: -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01
    """
    cte = labels[:, 0]
    flat = images.reshape(-1, N_INPUTS)
    mask = np.abs(cte) > 1.0
    region_imgs = flat[mask]
    pixel_mins = region_imgs.min(axis=0)
    pixel_maxs = region_imgs.max(axis=0)

    bounds = [(float(pixel_mins[i]), float(pixel_maxs[i])) for i in range(N_INPUTS)]

    # Negated postcondition as conjunction on outputs
    output_constraints = [[
        ("Y_0", ">=", -0.01),
        ("Y_0", "<=", 0.01),
        ("Y_1", ">=", -0.01),
        ("Y_1", "<=", 0.01),
    ]]

    comment = f"Property: P5 Deadzone Detection\n"
    comment += f"Input region: C_off_center (per-pixel box from {mask.sum()} images with |CTE| > 1.0m)\n"
    comment += f"Postcondition: NOT(|CTE| <= 0.01 AND |HE| <= 0.01)\n"
    comment += f"Negated (encoded below): -0.01 <= CTE <= 0.01 AND -0.01 <= HE <= 0.01\n"
    comment += f"Network: TinyTaxiNet.nnet\n"
    comment += f"SAT = deadzone input exists (property violated)"

    write_vnnlib(
        os.path.join(OUTPUT_DIR, "p5_deadzone.vnnlib"),
        bounds,
        output_constraints,
        comment
    )


def generate_p6(images, labels):
    """P6: Generate VNN-LIB files for the actual binary search queries that were run.
    Reads the real midpoints from verification_results_p5_p6.json so the exported
    .vnnlib files match the recorded verification run exactly.
    """
    import json as _json

    cte = labels[:, 0]
    mask = np.abs(cte) < 1.0
    centered = images[mask].reshape(-1, N_INPUTS)
    pixel_mins = centered.min(axis=0)
    pixel_maxs = centered.max(axis=0)

    bounds = [(float(pixel_mins[i]), float(pixel_maxs[i])) for i in range(N_INPUTS)]

    # Load actual binary search logs
    with open('verification_results_p5_p6.json', 'r') as f:
        p6_data = _json.load(f)['P6']['results']

    var_map = {'CTE': 'Y_0', 'HE': 'Y_1'}

    p6_files = []
    for out_name in ['CTE', 'HE']:
        for direction in ['upper', 'lower']:
            search_log = p6_data[out_name][f'{direction}_search_log']
            for entry in search_log:
                iteration = entry['iteration']
                mid = entry['mid']
                status = entry['status']

                if direction == 'upper':
                    op = ">="
                    desc = f"Is {out_name} >= {mid:.6f} reachable?"
                else:
                    op = "<="
                    desc = f"Is {out_name} <= {mid:.6f} reachable?"

                comment = f"Property: P6 Output Bound Tightening\n"
                comment += f"Search: {out_name} {direction} bound, iteration {iteration}\n"
                comment += f"Query: {desc}\n"
                comment += f"Input region: C_centered (same as P1-global, {mask.sum()} images)\n"
                comment += f"Recorded result: {status.upper()}\n"
                comment += f"Tolerance: 0.01"

                fname = f"p6_{out_name.lower()}_{direction}_iter{iteration:02d}.vnnlib"
                write_vnnlib(
                    os.path.join(OUTPUT_DIR, fname),
                    bounds,
                    [[(var_map[out_name], op, mid)]],
                    comment
                )
                p6_files.append(fname)

    return p6_files


def generate_instances_csv(p6_files):
    """Generate an instances.csv listing all exported properties."""
    model = "VerifyGAN/models/TinyTaxiNet.nnet"
    instances = []

    # P1
    for name in ['p1a_cte_upper', 'p1b_cte_lower', 'p1c_he_upper', 'p1d_he_lower']:
        instances.append(f"{model},{name}.vnnlib,60")

    # P2
    for i in range(1, 21):
        instances.append(f"{model},p2_img{i:02d}.vnnlib,60")

    # P3
    for eps_str in ['005', '010']:
        for i in range(1, 21):
            instances.append(f"{model},p3_eps{eps_str}_img{i:02d}.vnnlib,60")

    # P4
    instances.append(f"{model},p4_c_left.vnnlib,60")
    instances.append(f"{model},p4_c_right.vnnlib,60")

    # P5
    instances.append(f"{model},p5_deadzone.vnnlib,60")

    # P6 (full binary search family)
    for fname in p6_files:
        instances.append(f"{model},{fname},60")

    filepath = os.path.join(OUTPUT_DIR, "instances.csv")
    with open(filepath, 'w') as f:
        for line in instances:
            f.write(line + "\n")
    print(f"  Written: {filepath} ({len(instances)} instances)")


if __name__ == '__main__':
    print("Generating VNN-LIB property files for TinyTaxiNet P1-P6")
    print()

    images, labels = load_data()

    print("P1 - Safety Bound (4 sub-queries):")
    generate_p1(images, labels)

    print("\nP2 - Local Correctness (20 images):")
    generate_p2(images, labels)

    print("\nP3 - Robustness (2 epsilons x 20 images = 40 files):")
    generate_p3(images, labels)

    print("\nP4 - Directional Correctness (2 regions):")
    generate_p4(images, labels)

    print("\nP5 - Deadzone Detection (1 query):")
    generate_p5(images, labels)

    print("\nP6 - Output Bound Tightening (full binary search family):")
    p6_files = generate_p6(images, labels)

    print("\nInstances CSV:")
    generate_instances_csv(p6_files)

    print(f"\nAll files written to {OUTPUT_DIR}/")
    print("To verify with Marabou CLI: ./prophecy_env/bin/Marabou VerifyGAN/models/TinyTaxiNet.nnet vnnlib/<file>.vnnlib")
