#!/usr/bin/env python3
"""Run the full P3 matched random-subspace control in one Python process.

The driver pays the maraboupy import cost once, then invokes
verify_p3_random_subspace.py for the three experiment slices reported in the
paper:

  * all 893 centered anchors at alpha=1 sigma, seeds 0/1/2
  * all 893 centered anchors at alpha=2 sigma, seeds 0/1/2
  * deterministic 200-anchor subset at alpha=3 sigma, seeds 0/1/2

It does not analyze results; run analyze_p3_random_control.py afterward.
"""

import verify_p3_random_subspace as random_control


RUNS = [
    [
        "--num-images", "all",
        "--alphas", "1.0",
        "--seeds", "0,1,2",
        "--output", "verification_p3_random_subspace_all_alpha1_seeds012.json",
    ],
    [
        "--num-images", "all",
        "--alphas", "2.0",
        "--seeds", "0,1,2",
        "--output", "verification_p3_random_subspace_all_alpha2_seeds012.json",
    ],
    [
        "--num-images", "200",
        "--alphas", "3.0",
        "--seeds", "0,1,2",
        "--output", "verification_p3_random_subspace_200_alpha3_seeds012.json",
        "--manifest", "p3_random_subspace_alpha3_200_manifest.json",
    ],
]


def main():
    for argv in RUNS:
        print("\n=== verify_p3_random_subspace.py " + " ".join(argv) + " ===", flush=True)
        random_control.main(argv)


if __name__ == "__main__":
    main()
