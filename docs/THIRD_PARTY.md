# External work and attribution

Jireh Katebe designed the study and reusable verification workflow. The manuscript and code cite the external methods, models, and datasets used in the experiments.

| Component | Source and role |
| --- | --- |
| Marabou | [NeuralNetworkVerification/Marabou](https://github.com/NeuralNetworkVerification/Marabou), the verification solver |
| TinyTaxiNet and source data | [SISL VerifyGAN](https://github.com/sisl/VerifyGAN), the existing runway-perception model and simulator data |
| MNIST and VNN-COMP | Handwritten-digit data and externally supplied benchmark network; provenance is recorded in the benchmark scripts and manifests |
| LARD V2 | [DEEL-AI/LARD_V2](https://huggingface.co/datasets/DEEL-AI/LARD_V2), runway images and annotations used for the external experiment |
| Prophecy | [SRI-CSL/prophecy-papers](https://github.com/SRI-CSL/prophecy-papers), tooling for the activation-pattern experiment |
| Springer manuscript support | `svjour3.cls`, `svglov3.clo`, bibliography styles, and `SpringerLarge.sty`; original headers are retained, including the existing local style modification |

External model weights, dataset archives, solver binaries, and library source are not bundled. Their use remains subject to their upstream terms. The manuscript support files retain their existing notices. The repository uses the [MIT license](../LICENSE) selected by the author. Third-party components retain their existing notices and terms.
