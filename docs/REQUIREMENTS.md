# Software requirements

## Problem

A solver verdict alone does not explain which inputs a neural-network guarantee covers. The artifact must therefore preserve the definition of the input region together with each result. When a query returns a counterexample, it should also support comparisons with the dataset used to construct the region.

## Expected behavior

| ID | Requirement | Implementation | Evidence |
| --- | --- | --- | --- |
| R1 | Construct parameterized regions for flattened image inputs | Region constructors in `artifact/region_audit.py` | TinyTaxiNet and MNIST instantiations; region-tagged records |
| R2 | Combine base regions with constraints on image structure or total intensity | `Constraint`, `intersect`, mass and mathematical constraints | Mean-constrained and mathematical-region comparisons |
| R3 | Attach a user-supplied negated property to a fresh network and run Marabou | `run_query` and postcondition helpers | `module_smoke_taxinet.json`, `module_smoke_mnist.json`, and experiment records |
| R4 | Record the verdict with the region tag and parameters, keeping timeouts separate | Query result dictionaries and experiment JSON | Per-image result files linked in the results map |
| R5 | Inspect counterexamples against reference images | `WitnessAudit`, nearest-image distance, classifier score, and adjacent-pixel checks | Witness-audit JSON and plots |
| R6 | Support model-specific data loading and region calibration outside the reusable module | TinyTaxiNet, MNIST, VNN-COMP, and LARD scripts | Transfer experiments and matched random-direction controls |
| R7 | Make reported findings traceable to code and saved records | `RESULTS_AND_CODE.md`, `FILE_INDEX.md`, summarizer | `check_artifact.py` checks the release and saved summary |

Inputs include a supported network, flattened image arrays, calibration data, region parameters, a property, and a timeout. Solver outputs describe the encoded region. Witness-audit scores provide comparisons with reference data; they do not extend a proof to inputs outside that region.

## Operating requirements

Saved-results review needs only Python. New verification queries need NumPy and Marabou, plus the model and data for the selected experiment. Training and ONNX experiments use additional dependencies described in [SETUP.md](SETUP.md). The study code and saved records are supplied as individual files for inspection.
