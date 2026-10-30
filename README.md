# LEAP

Code for **Efficient LLM Inference Scheduling with Generalizable Length Prediction**.

LEAP predicts output length from a prefill hidden state using a linear readout, then updates the head with completion-supervised recursive least squares (RLS).

## Install

Python 3.10+; NumPy is the only core dependency. Run commands from this directory.

```bash
python -m pip install -e .
python examples/synthetic_demo.py
```

The demo checks RLS against a direct solve using synthetic data.

## Pretrained heads

```python
from leap import LinearReadout

head = LinearReadout.load("checkpoints/qwen3-8b-lmsys/head.npz")
lengths = head.predict(raw_features)  # [batch, 4096]; do not normalize again
```

Ten heads are included: LMSYS-source Qwen3-8B, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3; a separate ShareGPT-serving head; and six LMSYS Qwen scale controls. Layers, dimensions, targets, original precision and hashes are listed in [checkpoints](checkpoints/README.md). The default Qwen head has 4,097 FP32 parameters (16.39 KB; about 68 KB including statistics).

Prediction needs only the head. Online RLS also needs the matching inverse source precision `P`: 4,097 × 4,097 FP32 (67.1 MB) for the default head, held in host memory by the update thread. `P` is not bundled; `fit --rls-prior` exports it for a newly fitted head.

## Train and evaluate

Prepare features, labels and frozen splits as described in [data format](docs/DATA_FORMAT.md).

```bash
python scripts/prepare_cached_dataset.py \
  --features datasets/features.npz --labels datasets/labels.jsonl \
  --split datasets/splits.json --label-key length \
  --require-disjoint-parents --output datasets/cached_features.npz

leap fit --data datasets/cached_features.npz --output outputs/head \
  --feature-id qwen3_block30_last_prompt --model-revision MODEL_COMMIT --rls-prior

leap evaluate --head outputs/head/head.npz --data datasets/cached_features.npz \
  --split test --output outputs/metrics.json
```

Fitting selects regularization on source validation MAE and exports `head.npz`, `selection.json`, and optionally `rls_prior.npz`. Evaluation does not refit. The default mapping is `sqrt`; `log` and `identity` are offline alternatives. Output paths must be new.

## Collect features

```bash
python -m pip install -e '.[features]'
leap collect-features --model MODEL_DIRECTORY --revision MODEL_COMMIT \
  --prompts datasets/prompts.jsonl --output outputs/features.npz \
  --layer 30 --dtype bfloat16 --device cuda:0
```

This runs prefill only and captures the last valid prompt token at a one-based raw decoder block. Apply the intended chat template first. Files load locally unless `--allow-download` is set. Llama's two-block input is described in the checkpoint guide.

## Native readout

```bash
python scripts/build_native.py --output outputs/readout.so
```

Requires a C++17-capable `g++`; use `.dll` on Windows or `.dylib` on macOS. `leap.native.NativeReadout` supports width 4,096 and the square-root map. Engine integration uses `leap.scheduling` and `leap.ledger`; a complete GPU serving launcher is not included.

## Figures and tests

```bash
python -m pip install -e '.[plots]'
python scripts/reproduce_figures.py --output outputs/figures
python -m unittest discover -s tests -v
```

The plot script verifies bundled CSV digests and redraws Figures 5–8 and training curves, with PDF/PNG output and `PLOT_VALUES.csv`. It does not rerun hardware experiments. Optional native and local feature-interface tests require `g++` and the `features` extra respectively; missing dependencies are reported as skips.

## Layout

```text
src/leap/    Readout, RLS, features and scheduling interfaces
checkpoints/   Ten trained heads and metadata
native/        C++ readout
scripts/       Data preparation, native build and plotting
examples/      Synthetic RLS demo
configs/       Recorded experiment settings (not CLI launch files)
paper_data/    Aggregate measurements and input hashes
tests/         Numerical and interface checks
docs/          Data format and reproduction notes
```

[Reproduction notes](docs/REPRODUCIBILITY.md) cover fitting, integration and experiment protocols. Backbone weights, prompts, baseline checkpoints, dense RLS priors and full hardware logs are not included. The project license is not yet selected; [third-party terms](THIRD_PARTY_NOTICES.md) apply separately.
