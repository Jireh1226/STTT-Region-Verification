# Results and code

Each entry links a reported finding to its saved results and the scripts that produced or analyze them.

[Full manuscript](manuscript/paper.pdf) | [Results summary](RESULTS_SUMMARY.json) | [Complete file index](FILE_INDEX.md)

## 1. Reusable verification module

The module, its TinyTaxiNet and MNIST adapters, and saved module checks.

Saved results:

- [module_smoke_taxinet.json](artifact/module_smoke_taxinet.json)
- [module_smoke_mnist.json](artifact/module_smoke_mnist.json)

Code:

- [region_audit.py](artifact/region_audit.py)
- [taxinet_instantiation.py](artifact/taxinet_instantiation.py)
- [mnist_instantiation.py](artifact/mnist_instantiation.py)

## 2. TinyTaxiNet comparison in portfolio Section 6

Pixel region: 186/893 verified. Local PCA: 847 verified, 43 violations, 3 timeouts. Random eight-dimensional regions: 893/893 verified for each seed.

Saved results:

- [p3_random_control_summary.json](artifact/p3_random_control_summary.json)
- [verification_full_sample_centered.json](artifact/verification_full_sample_centered.json)
- [verification_p3_local_pca_expanded.json](artifact/verification_p3_local_pca_expanded.json)
- [verification_p3_random_subspace_all_alpha1_seeds012.json](artifact/verification_p3_random_subspace_all_alpha1_seeds012.json)

Code:

- [verify_full_sample.py](artifact/verify_full_sample.py)
- [verify_p3_local_pca.py](artifact/verify_p3_local_pca.py)
- [verify_p3_random_subspace.py](artifact/verify_p3_random_subspace.py)
- [analyze_p3_random_control.py](artifact/analyze_p3_random_control.py)

## 3. Preprocessing and the heading-error result

Compare P1d in the original pixel-region results with P1d_mean after enforcing mean intensity 0.5.

Saved results:

- [verification_results_v2.json](artifact/verification_results_v2.json)
- [verification_mean_constrained.json](artifact/verification_mean_constrained.json)

Code:

- [verify_p1_p2_v2.py](artifact/verify_p1_p2_v2.py)
- [verify_mean_constrained.py](artifact/verify_mean_constrained.py)

## 4. Wrong-side predictions in the dataset

142 wrong-side predictions among 8,227 images with a true centerline offset greater than two metres.

Saved results:

- [reproduced_empirical_baseline_direct.json](artifact/reproduced_empirical_baseline_direct.json)
- [empirical_baseline.json](artifact/empirical_baseline.json)

Code:

- [reproduce_empirical_baseline_direct.py](artifact/reproduce_empirical_baseline_direct.py)
- [empirical_baseline.py](artifact/empirical_baseline.py)

## 5. MNIST handwritten-digit transfer test

Compact classifier: 20 test images, per-image and per-class verification records, and counterexample comparisons.

Saved results:

- [mnist_protocol_results.json](artifact/mnist_protocol_results.json)
- [mnist_p1_p4_under_r3_r4.json](artifact/mnist_p1_p4_under_r3_r4.json)
- [mnist_witness_audit.json](artifact/mnist_witness_audit.json)
- [mnist_real_baselines.json](artifact/mnist_real_baselines.json)

Code:

- [verify_mnist_protocol.py](artifact/verify_mnist_protocol.py)
- [verify_mnist_p1_p4_under_r3_r4.py](artifact/verify_mnist_p1_p4_under_r3_r4.py)
- [mnist_regions.py](artifact/mnist_regions.py)
- [audit_mnist_witnesses.py](artifact/audit_mnist_witnesses.py)

## 6. VNN-COMP MNIST benchmark

Local PCA at one standard deviation: all 50 image regions verified. Random-direction controls: 450 queries verified across scales and seeds.

Saved results:

- [external_mnistfc_matrix_analysis.json](artifact/external_mnistfc_matrix_analysis.json)
- [external_mnistfc_matrix_results.json](artifact/external_mnistfc_matrix_results.json)
- [external_mnistfc_1sigma_rest27_results.json](artifact/external_mnistfc_1sigma_rest27_results.json)
- [external_mnistfc_broad_timeout_rerun_results.json](artifact/external_mnistfc_broad_timeout_rerun_results.json)
- [external_mnistfc_random_subspace_results.json](artifact/external_mnistfc_random_subspace_results.json)

Code:

- [external_mnistfc_matrix.py](artifact/external_mnistfc_matrix.py)
- [external_mnistfc_1sigma_rest27.py](artifact/external_mnistfc_1sigma_rest27.py)
- [rerun_mnistfc_broad_timeouts.py](artifact/rerun_mnistfc_broad_timeouts.py)
- [external_mnistfc_random_subspace.py](artifact/external_mnistfc_random_subspace.py)
- [run_mnistfc_random_parallel.py](artifact/run_mnistfc_random_parallel.py)

## 7. LARD V2 / Google Earth Studio runway images

Broad pixel changes up to 0.4: 32 violations among 50 images. Valid-pixel local PCA: all 50 regions verified at each tested scale. Random-direction controls: 900 queries verified.

Saved results:

- [external_lard_provenance.json](artifact/external_lard_provenance.json)
- [external_lard_training_report.json](artifact/external_lard_training_report.json)
- [external_lard_interpretation.json](artifact/external_lard_interpretation.json)
- [external_lard_results.json](artifact/external_lard_results.json)
- [external_lard_random_subspace_results.json](artifact/external_lard_random_subspace_results.json)
- [external_lard_witness_audit.json](artifact/external_lard_witness_audit.json)

Code:

- [prepare_external_lard_dataset.py](artifact/prepare_external_lard_dataset.py)
- [train_external_lard_mlp.py](artifact/train_external_lard_mlp.py)
- [verify_external_lard.py](artifact/verify_external_lard.py)
- [verify_external_lard_random_subspace.py](artifact/verify_external_lard_random_subspace.py)
- [audit_external_lard_witnesses.py](artifact/audit_external_lard_witnesses.py)

## 8. Additional manuscript analyses

Dimension sweeps, counterexample audits, output-bound diagnostics, and the corresponding plots are included in the complete file index.

Saved results:

- [p3_dimension_sweep_results.json](artifact/p3_dimension_sweep_results.json)
- [p3_coverage_verification_frontier_data.json](artifact/p3_coverage_verification_frontier_data.json)
- [diagnostic_postcondition_sweep.json](artifact/diagnostic_postcondition_sweep.json)
- [classifier_true_holdout_audit.json](artifact/classifier_true_holdout_audit.json)

Code:

- [verify_p3_dimension_sweep.py](artifact/verify_p3_dimension_sweep.py)
- [build_p3_frontier.py](artifact/build_p3_frontier.py)
- [diagnostic_postcondition_sweep.py](artifact/diagnostic_postcondition_sweep.py)
- [audit_classifier_true_holdout.py](artifact/audit_classifier_true_holdout.py)
