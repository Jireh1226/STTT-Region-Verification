# Validation

The repository supplies saved experiment records and a standard-library check of the release contents. Run `python3 check_artifact.py` from the root to verify file hashes and compare the recounted results with `RESULTS_SUMMARY.json`.

The summary recounts TinyTaxiNet pixel-region, PCA, and random-direction outcomes; wrong-side predictions; compact MNIST results; VNN-COMP MNIST outcomes; LARD outcomes; and the saved module checks. [RESULTS_AND_CODE.md](../RESULTS_AND_CODE.md) gives the exact records behind each finding.

Repository preparation checks Python syntax, local documentation links, release-file hashes, and execution of the saved-results commands after extraction into a separate directory. These checks establish that the supplied package can be inspected and its saved counts reproduced. Full model training and solver experiments require the external assets and environment described in [SETUP.md](SETUP.md).

Current execution results are recorded in `VALIDATION.json` at the repository root.
