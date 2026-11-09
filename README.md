# Region-relative neural-network verification

A neural-network verification result applies to the input region encoded for the solver. To make that scope explicit, I designed a reusable artifact that constructs input regions, attaches a requested property, and records the result together with the region's parameters. It also examines counterexamples using comparisons with reference images.

This repository contains the implementation, saved experimental results, and my unpublished draft manuscript, **Region-Relative Verification of TinyTaxiNet Using Marabou 2.0**, prepared for STTT. I designed the study and workflow, with AI assistance for implementation. The experiments use TinyTaxiNet, MNIST, a VNN-COMP MNIST network, and LARD V2 runway imagery.

## Start here

- [Revised unpublished draft](manuscript/paper.pdf)
- [Results and the code behind each finding](RESULTS_AND_CODE.md)
- [Reusable verification module](artifact/region_audit.py)
- [Complete file index](FILE_INDEX.md)

The [interactive file guide](START_HERE.html) can be opened locally after downloading the repository. On GitHub, use the Markdown results map above.

## Requirements and software documentation

| Document | Contents |
| --- | --- |
| [Requirements](docs/REQUIREMENTS.md) | Problem, expected behavior, inputs, outputs, and evidence |
| [Design](docs/DESIGN.md) | Region construction, constraints, query execution, and witness audits |
| [Setup and reproduction](docs/SETUP.md) | Saved-results review, dependencies, external assets, and experiment entry points |
| [Validation](docs/VALIDATION.md) | Package checks and recorded experiment results |
| [External work](docs/THIRD_PARTY.md) | Models, datasets, solver, and manuscript support files |
| [Release changes](docs/RELEASE_CHANGES.md) | Changes from the preserved research snapshot |

## Check the saved results

Run this from the repository root with Python 3.11 or later:

```sh
python3 summarize_saved_results.py
python3 check_artifact.py
```

Both commands use Python's standard library. The first recounts saved records; the second checks their summary and the release file hashes.

| Experiment | Saved result |
| --- | --- |
| TinyTaxiNet, pixel radius 0.05 | 186 verified regions and 707 violations across 893 anchors |
| TinyTaxiNet, local PCA at one standard deviation | 847 verified regions, 43 violations, 3 timeouts |
| TinyTaxiNet, matched random directions | 893 verified regions for each of three seeds |
| VNN-COMP MNIST, local PCA at one standard deviation | 50 verified regions |
| LARD V2, pixel radius 0.4 | 32 violations among 50 anchors |
| LARD V2, valid-pixel local PCA | 50 verified regions at each of three scales |

The [results map](RESULTS_AND_CODE.md) links these counts to their files and experimental context. They are guarantees or violations for the specified regions and properties.

## Run experiments

The [setup guide](docs/SETUP.md) separates reviewing saved results from running new solver queries or training models. Scripts and their saved outputs live together in `artifact/`. The full manuscript, LaTeX source, and figures live in `manuscript/`.

The manuscript was revised on November 8, 2026 for clarity and is explicitly labeled as an unpublished draft. Its PDF matches the version excerpted in the graduation portfolio. Experimental code and saved results are unchanged.

The selected source snapshot includes the PCA, random-direction, MNIST, and LARD extensions as well as the earlier verification core. Model weights and datasets are obtained from the upstream projects listed in the setup guide. The project uses the [MIT license](LICENSE) selected for this repository. Third-party notices remain with their respective files.
