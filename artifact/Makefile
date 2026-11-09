# Reproduction targets for the TinyTaxiNet region-relative verification study.
#
# Three depth tiers, so a reviewer can reproduce the headline numbers without
# the heavy ML stack:
#
#   reproduce-empirical : Tier 1. Pure numpy + h5py. NO maraboupy, NO
#                         TensorFlow, NO ONNX. Recomputes every headline
#                         empirical summary (P4 wrong-sign / dead-band table,
#                         CTE/HE Jacobian + headroom, the controlled
#                         postcondition sweep, and the classifier support
#                         audits) directly from the released .nnet weights and
#                         dataset.
#   reproduce-solver    : Tier 2. Adds maraboupy reading the .nnet directly
#                         (still NO TensorFlow). Re-runs representative region
#                         verdicts through the reusable module and confirms the
#                         postcondition-sweep flips with the exact solver.
#   reproduce-learned   : Tier 3. Needs TensorFlow + ONNX (MNIST data load and
#                         ONNX export). Re-runs the MNIST instantiation.
#
#   reproduce-headlines : Tier 1 + Tier 2  (everything TinyTaxiNet, no TF).
#   reproduce-all       : all three tiers.
#
# The interpreter is pinned via PY. The default is the committed virtualenv,
# which carries numpy, h5py, and maraboupy. Tier 1 also runs under any plain
# `python3` that has numpy + h5py; override with e.g.  make PY=python3 reproduce-empirical
PY ?= ./prophecy_env/bin/python
# The calibration scripts are numpy-only and shell out to the Marabou CLI
# binary, so they run under any plain python3 with numpy (no maraboupy/TF
# import). Default to a fast system interpreter; override with CALIB_PY=...
CALIB_PY ?= python3
MARABOU := ./prophecy_env/bin/Marabou

.PHONY: reproduce-headlines reproduce-all reproduce-empirical reproduce-solver \
        reproduce-learned reproduce-calibration reproduce-lard check-env clean-repro

## ---- environment probe -----------------------------------------------------
check-env:
	@echo "Interpreter: $(PY)"
	@$(PY) -c "import sys; print('python', sys.version.split()[0])"
	@$(PY) -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo "numpy: MISSING (required for all tiers)"
	@$(PY) -c "import h5py; print('h5py', h5py.__version__)" 2>/dev/null || echo "h5py: MISSING (required for Tier 1/2)"
	@$(PY) -c "from maraboupy import Marabou; print('maraboupy: ok')" 2>/dev/null || echo "maraboupy: MISSING (required for Tier 2)"
	@$(PY) -c "import faulthandler; faulthandler.dump_traceback_later(30, exit=True); import tensorflow as tf; print('tensorflow', tf.__version__)" 2>/dev/null || echo "tensorflow: MISSING or import-stalled >30s (required only for Tier 3 / MNIST)"

## ---- Tier 1: empirical summaries, no TF, no maraboupy -----------------------
reproduce-empirical:
	@echo "== Tier 1: empirical summaries (numpy + h5py only) =="
	$(PY) reproduce_empirical_baseline_direct.py
	$(PY) diagnostic_postcondition_sweep.py
	$(PY) audit_classifier_fifth_fake_family.py
	$(PY) audit_classifier_true_holdout.py
	@echo "== Tier 1 complete. JSON summaries written next to the scripts. =="

## ---- Tier 2: solver proofs via maraboupy reading .nnet (no TF) --------------
reproduce-solver:
	@echo "== Tier 2: solver verdicts (maraboupy + .nnet, no TF) =="
	$(PY) taxinet_instantiation.py
	$(PY) diagnostic_postcondition_sweep.py --marabou
	@echo "== Tier 2 complete. =="

## ---- Tier 3: learned-region / MNIST, needs TF + ONNX -----------------------
reproduce-learned:
	@echo "== Tier 3: MNIST instantiation (requires TensorFlow + ONNX) =="
	@echo "   If TensorFlow import stalls on this machine, see README"
	@echo "   'Reproducibility tiers' for the pinned Tier-3 environment."
	$(PY) mnist_instantiation.py
	@echo "== Tier 3 complete. =="

## ---- Calibration: external known-labels + controlled semantic oracle -------
# CALIB_BENCH selects the external expected-label family. Default mnist_fc is
# the image-classifier slice the paper reports (VNN-COMP mnist_fc). The ACAS Xu
# slice remains reproducible with `make reproduce-calibration CALIB_BENCH=acasxu_2023`.
CALIB_BENCH ?= mnist_fc
reproduce-calibration:
	@echo "== Calibration part A: external VNN-COMP $(CALIB_BENCH) expected-label slice =="
	$(CALIB_PY) select_vnnlib_calibration_slice.py
	@echo "-- fetching $(CALIB_BENCH) assets (needs network; failures reported) --"
	-$(CALIB_PY) prepare_vnnlib_calibration_assets.py --benchmarks $(CALIB_BENCH)
	@echo "-- solving the slice (skips instances with missing assets) --"
	-$(CALIB_PY) run_vnnlib_calibration_slice.py --benchmarks $(CALIB_BENCH) --snc --timeout 120
	-$(CALIB_PY) summarize_vnnlib_calibration.py
	-cp -f vnnlib_calibration_summary.json $(CALIB_BENCH)_calibration_summary.json
	-cp -f vnnlib_calibration_results.json $(CALIB_BENCH)_calibration_results.json
	-cp -f vnnlib_calibration_results.csv $(CALIB_BENCH)_calibration_results.csv
	@echo "== Calibration part B: controlled known-truth semantic oracle =="
	$(CALIB_PY) train_controlled_models.py
	$(CALIB_PY) verify_controlled_oracle.py
	$(CALIB_PY) audit_controlled_witnesses.py
	@echo "== Calibration complete. Part A needs network/assets (else"
	@echo "   MISSING_ASSETS.md is written); Part B is self-contained. =="

## ---- External runway-data check -------------------------------------------
reproduce-lard:
	@echo "== External LARD V2 runway-image check (downloads public imagery if missing) =="
	$(CALIB_PY) prepare_external_lard_dataset.py
	$(CALIB_PY) select_external_lard_target.py
	$(CALIB_PY) train_external_lard_mlp.py
	$(PY) build_external_lard_manifest.py
	$(PY) verify_external_lard.py
	$(PY) audit_external_lard_witnesses.py
	$(CALIB_PY) summarize_external_lard.py
	@echo "== LARD check complete. JSON summaries written as external_lard_*.json. =="

## ---- aggregates ------------------------------------------------------------
reproduce-headlines: reproduce-empirical reproduce-solver
	@echo "== All TinyTaxiNet headline checks reproduced (no TensorFlow used). =="

reproduce-all: reproduce-headlines reproduce-calibration reproduce-learned
	@echo "== All tiers reproduced. =="

clean-repro:
	rm -f reproduced_empirical_baseline_direct.json \
	      diagnostic_postcondition_sweep.json \
	      classifier_fifth_fake_family_audit.json \
	      classifier_true_holdout_audit.json \
	      module_smoke_taxinet.json module_smoke_mnist.json
