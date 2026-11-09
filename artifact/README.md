# Region-Relative Verification with Marabou 2.0

This repository contains the code, result artifacts, and reusable region-audit
module for a region-relative Marabou verification study. The main case study is
TinyTaxiNet, a compact open-source runway-perception surrogate that maps
8x16 grayscale images to cross-track and heading error. The artifact also
includes a small MNIST ReLU MLP instantiation used as a transfer check.

The central reusable component is `region_audit.py`, which factors the region
constructors and witness-audit signals away from the TinyTaxiNet- and
MNIST-specific scripts. The per-experiment scripts and JSON files remain the
source artifacts for the reported numerical results.

## Contents

### VNN-LIB export

`vnnlib/` contains the 118 baseline specifications in standard SMTLIB2 format
plus an `instances.csv` mapping each spec to the network. Composition:

| Property | Files |
| --- | --- |
| P1       | 4 sub-queries (`p1a`-`p1d`) |
| P2       | 20 per-image files (`p2_img01`-`p2_img20`) |
| P3       | 40 files (`p3_eps005_img*`, `p3_eps010_img*`) |
| P4       | 2 regional files (`p4_left`, `p4_right`) |
| P6       | 52 binary-search midpoints (`p6_*_iter*`) |

Every file embeds a preprocessing header (no `.nnet` normalization, Marabou
`normalize=False`, dataset mean-centered to 0.5 and clipped to [0, 1]) and the
caveat that `mean(x)=0.5` is **not** enforced in the baseline files.
Some script names still include P5, a deadzone variant from the original
artifact. The unpublished manuscript reports P1-P4 and P6; P5 is retained only
for historical reproducibility of the earlier runs.

### Reusable region-audit module

`region_audit.py` implements the reusable part of the workflow for flattened
image-input networks verified through `maraboupy`.

Region constructors:

- `r1_subset_box(images)` for dataset-subset pixel boxes, `R1(S)`.
- `r1_linf_ball(x0, eps)` for per-anchor perturbation balls, `R1(x0, eps)`.
- `r3_inline_relu_decoder(...)` and `r3_latent_box(...)` for decoder-box
  regions, `R3`.
- `r4_local_pca(x0, basis, sigmas, alpha)` for local PCA regions, `R4`.
- `r5_neighbor_box(neighbor_images)` for label- or class-conditioned boxes,
  `R5`.

Composable constraints:

- `mass_equality(...)` and `mass_band(...)` for exact or statistical mass
  constraints, used as the `R2` layer.
- `tv_bound(...)`, `fourier_hf_bound(...)`, `photometric_bound(...)`,
  `per_pair_bound(...)`, and `per_row_mean_band(...)` for the `R6` math-region
  family.
- `intersect(region, *constraints)` to combine any base region with these
  constraints under a new tag.

Tagged queries and audits:

- `run_query(make_net, region, postcondition, timeout_s)` attaches a tagged
  region, applies the negated property, solves with Marabou, and returns the
  verdict with the region tag, calibration parameters, and SAT witness when one
  is returned.
- `WitnessAudit(reference_images, shape, classifier)` scores witnesses using
  classifier probability, nearest-real-image L2 distance, and per-pair
  Lipschitz violation fraction; `baseline()` computes the same quantities on
  real reference images.

The two released instantiations show how to supply the application-specific
pieces: dataset loading, subset predicates, neighbor rules, calibration data,
and network readers.

```
python taxinet_instantiation.py
python mnist_instantiation.py
```

These scripts write `module_smoke_taxinet.json` and `module_smoke_mnist.json`.
The current smoke tests pass 8/8 TinyTaxiNet checks and 7/7 MNIST checks.

### Verification scripts

| Script | Property |
| --- | --- |
| `verify_p1_p2_v2.py` | P1 (global box) and P2 (per-image L_inf) |
| `verify_prophecy_p1.py` | P1-patterned (activation-pattern partitioning, 24 patterns) |
| `verify_p3_p4.py` | P3 (unconstrained baseline) and P4 (regional sign) |
| `verify_p5_p6.py` | Legacy P5 (deadzone, superseded in the manuscript) and P6 (binary-search reachable bounds) |
| `verify_mean_constrained.py` | Mean-constrained reruns of P1, P4, and legacy P5 |
| `verify_p3_mean_constrained.py` | P3 with `mean(x)=0.5` and clipping enforced |
| `verify_p3_certified_radius.py` | Per-image binary search for the P3 safety radius |
| `verify_p3_label_box.py` | P3 over per-image label-conditioned per-pixel boxes |
| `verify_p3_local_pca.py` | P3 over per-image 8-D local-PCA regions |
| `region_audit.py` | Reusable region constructors, tagged query runner, and witness-audit signals |
| `taxinet_instantiation.py` | TinyTaxiNet adapter and smoke tests for `region_audit.py` |
| `mnist_instantiation.py` | MNIST adapter and smoke tests for `region_audit.py` |

### Plausibility-constrained pipeline

| Script | Purpose |
| --- | --- |
| `train_subset_autoencoders.py` | Train one autoencoder per input subset (centered, left, right, off-center) |
| `build_subset_onnx_with_sum.py` | Compose subset decoder with TinyTaxiNet into an ONNX network with auxiliary pixel-sum output |
| `verify_ae_with_mean_constraint.py` | Verify P1/P4 and legacy P5 over the AE-composed regions with `mean(x)=0.5` enforced |
| `verify_ae_strong_wrong_sign.py` | P4 threshold sweep on the AE-composed regions |
| `multi_seed_stability.py` | Retrain autoencoders at seeds 7, 13 and rerun the AE-composed verifications |
| `train_plausibility_classifier.py` | Train the real-vs-synthetic plausibility classifier |
| `score_counterexamples.py`, `rescore_mean_constrained_witnesses.py` | Score SAT witnesses against the plausibility classifier |
| `validate_witnesses_nearest_neighbor.py` | Compare AE-composed P4 witnesses to nearest real images |
| `score_pca_witnesses.py` | Re-extract local-PCA SAT witnesses, score each against the plausibility classifier and the nearest real image; output `pca_witness_scores.json` |
| `audit_autoencoder_support.py` | Per-subset support audit of the trained autoencoders: held-out latent-box coverage, reconstruction `L_inf`, TinyTaxiNet output preservation `|f(x) - f(AE(x))|`, classifier acceptance of random latent decodes; output `autoencoder_support_audit.json` |
| `verify_ae_latent_quantile_sweep.py` | Reruns the AE-composed P1a/P1b, P4 left/right, and legacy P5 verifications at four latent-box widths (min/max, 0.5%-99.5%, 1%-99%, 5%-95%) to test whether SAT verdicts are corner artifacts; output `verification_ae_latent_quantile_sweep.json` |
| `verify_p1_p4_p5_with_tv_bound.py` | P1/P4 and legacy P5 verifications under a total-variation bound (spatial smoothness, calibrated from centered subset); output `verification_tv_bound.json` |
| `verify_p1_p4_p5_with_fourier_bound.py` | P1/P4 and legacy P5 verifications under a high-frequency Fourier L1 bound (low-spectral-content constraint); output `verification_fourier_bound.json` |
| `verify_p1_p4_p5_with_photometric_basis.py` | P1/P4 and legacy P5 verifications under a photometric basis residual bound (image = global brightness + horizontal gradient + bounded residual); output `verification_photometric_bound.json` |
| `verify_p1_p4_p5_combined_math.py` | P1/P4 and legacy P5 with TV + Fourier + photometric constraints imposed simultaneously; output `verification_math_combined.json` |
| `verify_per_row_mean_bound.py` | P1/P4 and legacy P5 with per-row mean bounds (vertical photometric structure from centered subset); output `verification_per_row_mean.json` |
| `verify_p4_right_strong_math.py` | P4-right strong wrong-sign sweep (`CTE >= +3`) under TV/Fourier/photometric, testing left-right magnitude asymmetry; output `verification_p4_right_strong_math.json` |
| `verify_p3_with_math_regions.py` | P3 at `eps=0.05` (per-image L_inf perturbation) under TV/Fourier/photometric at multiple tightnesses; output `verification_math_regions_p3_eps0p05.json` |
| `verify_p3_tight_tv.py` | P3 at `eps=0.05` under TV bounds `tau in {10, 8, 6, 4, 2}` (below empirical centered min); shows monotonic restoration 4/20 -> 20/20; output `verification_p3_tight_tv.json` |
| `verify_prophecy_p1_with_math.py` | P1-patterned verification (24 activation patterns) with additional TV/Fourier/photometric/combined math constraints; one pattern restored to UNSAT under tight constraints; output `verification_p1_patterned_math.json` |
| `verify_ae_with_math.py` | AE-composed verification with additional TV or photometric constraint on decoded pixels; verdicts unchanged (AE region already math-feasible on P1/P4 and legacy P5); output `verification_ae_plus_math.json` |
| `verify_math_exhaustive_batch.py` | Per-image TV on P3 at `eps=0.05`; TV on P3 at `eps=0.10`; TV on P2 (all fragile-correct images verifiable at `tau<=4`); TV on P6 axis-aligned bounds (envelope tightens 20-30% per axis); output `verification_math_exhaustive.json` |
| `verify_object_specific_math.py` | Object-specific math regions: per-cell mean bounds (2x2 and 4x4 patches) and per-pair Lipschitz bounds (each pixel pair has its own typical-difference bound calibrated from real images). Per-pair Lipschitz recovers the P4-left strong wrong-sign witness that global TV collapses; output `verification_object_specific_math.json` |
| `audit_math_witnesses.py` | Score every math-region SAT witness against the plausibility classifier and the nearest real centered image; 94.5% of math-region witnesses score "classifier-fake" (prob < 0.1); output `math_witnesses_classifier_audit.json` |
| `train_plausibility_classifier_holdout.py` | Retrain the plausibility classifier with a disjoint real-image split for held-out real acceptance checks |
| `audit_classifier_true_holdout.py` | Score math-region witnesses and unseen real images with the disjoint-split classifier |
| `audit_classifier_calibration_holdout.py` | Report classifier acceptance across TinyTaxiNet dataset slices |
| `audit_classifier_fifth_fake_family.py` | Stress-test both plausibility classifiers (original and disjoint-split) on smooth low-frequency, mean-normalized fake images not used during classifier training; output `classifier_fifth_fake_family_audit.json` |
| `diagnostic_postcondition_sweep.py` | Controlled postcondition sweep: holds the input region fixed and varies the CTE/HE bound to test whether the CTE-only failure pattern follows postcondition headroom; reports the headroom-equalized bound at which the asymmetry disappears; `--marabou` confirms the flips with the exact solver. Pure numpy/h5py (Tier 1); output `diagnostic_postcondition_sweep.json` |
| `select_vnnlib_calibration_slice.py` | Calibration A1: deterministic SAT/UNSAT slice of the VNN-COMP expected-results table (`review/sources/vnnlib_expected_results.csv`); outputs `vnnlib_calibration_slice.csv`, `vnnlib_calibration_slice_summary.json` |
| `prepare_vnnlib_calibration_assets.py` | Calibration A2: fetch/locate the ONNX+VNNLIB assets from the VNN-COMP 2023 mirror into `external/vnncomp/`; writes `MISSING_ASSETS.md` for anything unavailable |
| `run_vnnlib_calibration_slice.py` | Calibration A3: run the bundled Marabou 2.0 CLI on the slice (`--timeout`, `--snc`); outputs `vnnlib_calibration_results.{json,csv}` |
| `summarize_vnnlib_calibration.py` | Calibration A4: agreement of Marabou verdicts with the published labels; output `vnnlib_calibration_summary.json` |
| `controlled_renderer.py` | Controlled-oracle renderer: 8x16 lane images from known state, dense renderer grid + rendered-NN percentile (semantic-support oracle), non-renderable checkerboard trigger |
| `train_controlled_models.py` | Build `M_good`, `M_bug` (injected true failure), `M_artifact` (non-renderable trigger) as `.nnet`, plus a controlled plausibility classifier |
| `verify_controlled_oracle.py` | Verify the controlled models for wrong-sign under broad box, neighbor box, and manifold-constrained local-PCA regions (Marabou CLI on `.nnet`); output `controlled_verify_raw.json` |
| `audit_controlled_witnesses.py` | Audit controlled witnesses (classifier prob, nearest-rendered L2, per-pair, nearest renderer-grid L2 + state, re-render semantic check); output `controlled_oracle_results.json` |
| `audit_r5_label_box_witnesses.py` | Re-run and audit label-conditioned box witnesses |
| `train_mnist_marabou_mlp.py` | Train/export the small MNIST ReLU MLP used for the transfer check |
| `train_mnist_class_aes.py` | Train class-conditional MNIST autoencoders |
| `train_mnist_plausibility_classifier.py` | Train the MNIST real-vs-synthetic plausibility classifier |
| `verify_mnist_protocol.py` | Run the MNIST region-tagged protocol |
| `verify_mnist_p1_p4_under_r3_r4.py` | Fill the MNIST P1/P4 learned-region cells |
| `audit_mnist_witnesses.py` | Audit MNIST SAT witnesses |
| `audit_mnist_real_baselines.py` | Compute MNIST real-image audit baselines |
| `prepare_external_lard_dataset.py` | Download and preprocess the LARD V2 Google Earth Studio runway-image subset used for the external aerospace-data check |
| `select_external_lard_target.py` | Select the LARD classification target using a validation split from the training subset; records why pitch is used rather than a fine centerline-geometry target |
| `train_external_lard_mlp.py` | Train/export the compact LARD pitch-attitude ReLU MLP and Marabou-compatible `.nnet` |
| `build_external_lard_manifest.py` | Freeze the 50 held-out LARD anchors used in the paper, class-confusion targets, region parameters, and solver timeout before verification |
| `verify_external_lard.py` | Run the LARD region-tagged verification grid over R1, R2, R5, and local-PCA R4 variants |
| `audit_external_lard_witnesses.py` | Audit LARD SAT witnesses against held-out runway images using nearest-real distance, total variation, and per-pair violation |
| `summarize_external_lard.py` | Assemble the LARD verdict/audit summary in `external_lard_interpretation.json` |

### Calibration and analysis

| Script | Purpose |
| --- | --- |
| `calibrate_perturbation_scale.py` | Compute the perturbation calibration table (per-pixel std, same-state-NN L_inf, random-pair L_inf) |
| `analyze_box_coverage.py` | Per-pixel-box coverage statistics across the four data subsets |
| `analyze_p2_three_way.py` | P2 nominally-wrong / fragile / robust categorization |
| `analyze_margin_vs_radius.py` | Spearman correlation between nominal margin and certified radius (P3) |
| `analyze_counterexamples.py` | Summary of SAT witnesses per property |
| `plot_p6_reachable.py` | Generate the P6 reachable-set visualization from Marabou-certified bounds |
| `make_paper_summary_figures.py` | Generate the paper's region-transition heatmap and LARD witness-audit ratio figure from released result JSONs |
| `extract_images.py` | Decode SAT witnesses to images for visual inspection |
| `validate_prophecy_surrogate.py` | Validate the flat Keras surrogate used to extract activation patterns |
| `generate_vnnlib.py` | Produce the baseline VNN-LIB export from the verification configurations |

### Result artifacts

JSON files capture verification outcomes (status, witness, timing) and
calibration statistics. Examples: `verification_results_v2.json`,
`verification_results_p3_p4.json`, `verification_results_p5_p6.json` (legacy P5 plus P6),
`verification_p3_mean_constrained.json`, `verification_p3_certified_radius.json`,
`verification_p3_local_pca.json`, `verification_p3_label_box.json`,
`verification_subset_autoencoder_onnx.json`, `verification_strong_wrong_sign_sweep.json`,
`multi_seed_stability_results.json`, `mean_constrained_witness_scores.json`,
`witness_nearest_neighbor_validation.json`, `perturbation_scale_calibration.json`,
`p2_three_way_categorization.json`, `p3_margin_vs_radius.json`,
`prophecy_p1_results.json`.

Trained-model artifacts: `autoencoder_*_weights.npz`,
`plausibility_classifier.npz`, and the composed ONNX networks
(`composed_*_with_sum*.onnx`).

## External dependencies (not committed)

This project depends on two external repositories that are vendored locally
during development but kept outside this commit because of their size:

- **`VerifyGAN/`** (~46 MB): the released TinyTaxiNet model, dataset
  (`SK_DownsampledGANFocusAreaData.h5`), and reference code, from
  the [Stanford Intelligent Systems Laboratory](https://github.com/sisl/VerifyGAN).
  Required by the TinyTaxiNet experiments.
- **LARD V2 imagery**: fetched on demand by `prepare_external_lard_dataset.py`
  from the DEEL-AI LARD V2 dataset. This standalone repository includes the
  preparation code and saved provenance/results, with imagery and NumPy data obtained separately.
- **`Prophecy/`** (~289 MB): activation-pattern extraction tooling, from
  [SRI-CSL/prophecy-papers/Prophecy](https://github.com/SRI-CSL/prophecy-papers).
  Required by `verify_prophecy_p1.py` to extract the P1-patterned activation
  signatures.

Obtain the assets required by the experiment you intend to run. The
expected paths (referenced by the scripts) are:

```
VerifyGAN/models/TinyTaxiNet.nnet
VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5
```

## Verification environment

- Python 3.11
- `maraboupy` (Marabou 2.0 Python bindings)
- `numpy`, `h5py`, `tensorflow` (for autoencoder training and ONNX export),
  `tf2onnx`, `matplotlib`

A virtual environment named `prophecy_env/` is used during development and is
not committed. Reproduce with:

```
python3.11 -m venv prophecy_env
source prophecy_env/bin/activate
pip install maraboupy numpy h5py tensorflow tf2onnx matplotlib
```

## Reproducibility tiers (`make reproduce-headlines`)

The headline numbers are reproducible at three dependency depths, so a
reviewer can recompute the empirical and solver results **without TensorFlow
or ONNX**. The `Makefile` pins the interpreter via `PY` (default
`./prophecy_env/bin/python`); override it with any compatible interpreter,
e.g. `make PY=python3 reproduce-empirical`.

| Target | Tier | Dependencies | What it regenerates |
| --- | --- | --- | --- |
| `make reproduce-empirical` | 1 | numpy + h5py only (no maraboupy, no TF) | P4 wrong-sign / dead-band table, CTE/HE Jacobian + headroom medians, the controlled postcondition sweep, and both classifier support audits (fifth fake-family, disjoint-split held-out) |
| `make reproduce-solver` | 2 | + maraboupy reading `.nnet` (still no TF) | representative region verdicts through `region_audit.py`, and the exact-solver confirmation of the postcondition sweep |
| `make reproduce-learned` | 3 | + TensorFlow + ONNX | the MNIST instantiation |
| `make reproduce-calibration` | calib | numpy + Marabou CLI (no maraboupy/TF import); part A needs network | external VNN-COMP mnist_fc image-classifier expected-label slice (part A) and the controlled known-truth semantic oracle (part B) |
| `make reproduce-lard` | external | numpy + maraboupy + network access for first download | the LARD V2 runway-image external-data check |
| `make reproduce-headlines` | 1+2 | numpy + h5py + maraboupy | every TinyTaxiNet headline check, no TensorFlow used |
| `make reproduce-all` | 1+2+calib+3 | all of the above | every tier |

Run `make check-env` first to see which tiers the current interpreter
supports. Tier 1 covers every TinyTaxiNet *empirical* claim in the paper from
the released `.nnet` weights and dataset using a direct numpy forward pass, so
no solver or ML framework is required to audit those numbers.

**Calibration target.** `make reproduce-calibration` runs two known-outcome
checks (paper Section "Evidence Levels and Calibration"). Part A selects a
deterministic VNN-COMP mnist_fc expected-label slice (an image classifier,
`mnist-net_256x2`), fetches the assets, solves it
with the bundled Marabou 2.0 CLI, and reports agreement with the published
SAT/UNSAT labels (a solver/property-plumbing check, not a semantic oracle);
if the network or assets are unavailable it writes `MISSING_ASSETS.md` and
continues. The ACAS Xu slice remains reproducible with
`make reproduce-calibration CALIB_BENCH=acasxu_2023`. The expected SAT/UNSAT
label is a deterministic property of the (network, spec) files, so the
calibration must not be run with a modified epsilon or spec: that would void
the published label. If too few instances complete, report the timeouts
honestly rather than altering the spec. Part B is fully self-contained: it
trains three known-truth ReLU models on a tiny 8x16 renderer (`M_good`,
`M_bug` with an injected true failure, `M_artifact` with a non-renderable
trigger), verifies each under broad and manifold-constrained regions, and uses
a dense renderer grid plus a re-render check as a semantic-support oracle. The
calibration scripts are numpy-only and shell out to the Marabou CLI, so they
run under any `python3` with numpy (`CALIB_PY`, default `python3`); no
maraboupy or TensorFlow import is needed.

**TensorFlow note (Tier 3 only).** On the development machine, `import
tensorflow` in `prophecy_env` (TF 2.14.1 / protobuf 4.25.8 / numpy 1.26.4) is
slow-to-stalling, which is why the MNIST tier is isolated and nothing in Tiers
1-2 imports TensorFlow. If Tier 3 stalls, recreate a clean Tier-3 environment
with a pinned, mutually compatible stack:

```
python3.11 -m venv tf_env && source tf_env/bin/activate
pip install "tensorflow==2.15.1" "protobuf>=4.25,<5" "numpy<2" tf2onnx onnx maraboupy h5py
```

The MNIST model and ONNX export are regenerated by the Tier-3 training
scripts; the TinyTaxiNet results never depend on this environment.

## Reproducing the main results

The baseline P1-P6 results in the report come from:

```
python verify_p1_p2_v2.py
python verify_prophecy_p1.py
python verify_p3_p4.py
python verify_p5_p6.py
```

The plausibility-constrained results require training the autoencoders first:

```
python train_subset_autoencoders.py
python build_subset_onnx_with_sum.py
python verify_ae_with_mean_constraint.py
python verify_ae_strong_wrong_sign.py
python multi_seed_stability.py
python train_plausibility_classifier.py
python rescore_mean_constrained_witnesses.py
python validate_witnesses_nearest_neighbor.py
```

The P3 follow-ups (mean+clip, certified radius, label-conditioned box,
local-PCA) and the dataset calibration:

```
python calibrate_perturbation_scale.py
python verify_p3_mean_constrained.py
python verify_p3_certified_radius.py
python verify_p3_label_box.py
python verify_p3_local_pca.py
python analyze_p2_three_way.py
python analyze_margin_vs_radius.py
python analyze_box_coverage.py
```

Each script prints a summary and writes its results to a JSON file with the
matching name.

To run the reusable-module smoke tests:

```
python taxinet_instantiation.py
python mnist_instantiation.py
```

## Notes

All verification queries are over the post-preprocessed network input space
(after the SISL preprocessing pipeline of crop, resize, grayscale,
8x16 downsample, mean-centering, and clipping); we do not re-run the raw
preprocessing pipeline. SAT/UNSAT verdicts are sound only relative to the
encoded input region. The reusable module exposes the six region tags used by
the current paper; the application-specific instantiation supplies the data and
calibration choices for a new network.
