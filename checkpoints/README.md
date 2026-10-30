# Pretrained linear readouts

The default checkpoint is **`qwen3-8b-lmsys/head.npz`**. LMSYS-source heads for the three main backbones, one separate ShareGPT-serving head, and six Qwen scale-control heads are included. Original coefficient precision and source-training statistics are preserved exactly in the portable `LinearReadout` schema. Conversion uses lossless ZIP compression, with no refitting or additional quantization.

## Main LMSYS-source heads

Each directory contains `head.npz`:

| Directory | Backbone | Raw block inputs | Width | Alpha | Bytes |
|---|---|---|---:|---:|---:|
| `qwen3-8b-lmsys/` | Qwen3-8B | 30 | 4,096 | 0.1 | 68,455 |
| `llama-3.1-8b-instruct-lmsys/` | Llama-3.1-8B-Instruct | 30 then 32, concatenated | 8,192 | 1.0 | 132,307 |
| `mistral-7b-instruct-v0.3-lmsys/` | Mistral-7B-Instruct-v0.3 | 28 | 4,096 | 1.0 | 68,127 |

These source fits use repeated-generation length centers. The original Llama and Mistral targets are `exp(fitted source log-t mu)`, as recorded in their training receipts. Validation uses source repeated-generation token MAE; target-test labels do not select the head. The Qwen head is the default for the LMSYS transfer and shared-configuration serving studies; the Llama and Mistral heads correspond to their LMSYS source-quality configurations.

## Separate source/control heads

`qwen3-8b-sharegpt/head.npz` is the distinct ShareGPT-trained serving head (block 30, width 4,096, alpha 1.0, 68,550 bytes). It is not the LMSYS head used for transfer. The paper includes both source-specific in-domain comparisons and LMSYS-source transfer; not every experiment trains on LMSYS.

The following **LMSYS single-label scale controls** use the observed first-rollout length as the training target. They are separate from the default repeated-generation source head, including at 8B. Their original FP64 coefficients are retained.

| Directory | Backbone | Raw block | Width | Bytes |
|---|---|---:|---:|---:|
| `qwen3-0.6b-lmsys-round0/` | Qwen3-0.6B | 24 | 1,024 | 23,402 |
| `qwen3-1.7b-lmsys-round0/` | Qwen3-1.7B | 24 | 2,048 | 43,801 |
| `qwen3-4b-lmsys-round0/` | Qwen3-4B | 32 | 2,560 | 54,149 |
| `qwen3-8b-lmsys-round0/` | Qwen3-8B | 30 | 4,096 | 84,801 |
| `qwen3-14b-lmsys-round0/` | Qwen3-14B | 36 | 5,120 | 105,187 |
| `qwen3-32b-lmsys-round0/` | Qwen3-32B | 60 | 5,120 | 105,058 |

All files include their original source-normalization statistics and metadata. `metadata.json` (format version 2) is keyed by checkpoint ID and records digests, backbone revisions, layers, dimensions, coefficient precision, training targets, and protocol groups. It contains no prompts or backbone weights.

## Load and predict

From the repository root, after `python -m pip install -e .`:

```python
import numpy as np
from leap import LinearReadout

head = LinearReadout.load("checkpoints/qwen3-8b-lmsys/head.npz")
with np.load("datasets/features.npz", allow_pickle=False) as features:
    predicted_lengths = head.predict(features["x"])
print(predicted_lengths)
```

For an NPZ with test features and labels in the [cached-data schema](../docs/DATA_FORMAT.md):

```bash
leap evaluate --head checkpoints/qwen3-8b-lmsys/head.npz \
  --data datasets/cached_features.npz --split test \
  --output outputs/pretrained_test_metrics.json
```

## Required features

For the **default Qwen3-8B** head:

- Backbone: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Layer: completed decoder block **30**, counted from 1.
- Token: last non-padding prompt token, before generating any output tokens.
- Dimension: 4,096; feature export passes through FP16 storage and then FP32 readout input.
- Mapping: clip the predicted square-root coordinate to `[1, sqrt(2048)]`, then square it and bound the result to `[1, 2048]`.
- Source standardization is already folded into `w` and `b`. Pass **raw features**, without normalizing them again.

Do not use these heads with a different backbone, layer, pooled feature, or chat-template convention and assume equivalent results. Select the source-domain head that matches the intended comparison; do not select a head using target-test results.

For the other heads, use the model revision and layer contract in `metadata.json`. The same `LinearReadout.load()` API accepts all bundled dimensions. The C++ readout is fixed to width 4,096 and cannot load the 8,192-dimensional Llama head or other widths.

### Llama feature order

Llama requires the raw completed outputs of block 30 **followed by** block 32 for the last valid prompt token, with the final RMS norm excluded. `hidden_states[-1]` is the normalized final state and is not a substitute for the raw block-32 output.

The supplied collector captures one raw decoder block per invocation. Collect blocks 30 and 32 with the pinned Llama revision and identical prompts, tokenization, and precision, then join by request ID in the same order:

```python
with np.load("datasets/llama_block30.npz", allow_pickle=False) as a, \
     np.load("datasets/llama_block32.npz", allow_pickle=False) as b:
    np.testing.assert_array_equal(a["ids"], b["ids"])
    raw_features = np.concatenate([a["x"], b["x"]], axis=1)

head = LinearReadout.load("checkpoints/llama-3.1-8b-instruct-lmsys/head.npz")
predicted_lengths = head.predict(raw_features)
```

## Scope

Ten frozen readouts and their normalization statistics are bundled. This is not a claim that all model/domain/ablation checkpoints have been released. Backbone weights, prompts, per-request feature caches, completion-adapted snapshots, and the dense RLS prior are not included. These checkpoints support frozen inference directly; they are not sufficient on their own to initialize source-precision RLS. That requires the matching source precision/prior from the original source-training features; an identity prior is not equivalent. The fitting workflow in the main README exports such a prior for newly fitted heads.

All files use `allow_pickle=False` loading. They follow the project's current licensing status; upstream backbone and dataset terms apply separately. See [third-party notices](../THIRD_PARTY_NOTICES.md).
