# Cached features, labels and checkpoints

## Training NPZ

Use NumPy arrays saved with `np.savez`, readable with `allow_pickle=False`:

| Key | Shape | Meaning |
|---|---|---|
| `train_x`, `val_x`, `test_x` | `[N_split, D]` | Raw last-prompt-token hidden features; one fixed model/layer/precision contract |
| `train_y`, `val_y`, `test_y` | `[N_split]` | Declared length target, finite in `[1,2048]` |
| `train_ids`, `val_ids`, `test_ids` | `[N_split]`, optional | Unique, mutually disjoint request IDs |

Training reads only `train_x/y` and `val_x/y`; the `evaluate` command reads the test split. When IDs are present, the loader checks that splits do not overlap. Requests from the same parent conversation should also stay in one split.

The paper fits offline on a length center computed from multiple source generations and adapts online on actual completion lengths. Apply the target mapping once, to the length itself.

`scripts/prepare_cached_dataset.py` joins feature IDs, a length-label JSONL, and a split JSON with `train`, `val`, `test` ID lists. It rejects unknown, duplicate, or overlapping IDs. `--require-disjoint-parents` requires a `parent_id` on every label row and checks parent overlap; `--label-key` selects the label field.

For example, `labels.jsonl` contains one row per request:

```json
{"id": "request-a", "parent_id": "conversation-a", "length": 127.5}
```

Here `length` must follow the declared offline label convention; a multi-generation source length center need not be an integer. A completion label for RLS is the actual integer output length instead. The feature NPZ contains `x` and `ids`, with `x[i]` belonging to `ids[i]`. A minimal split file has this structure (with enough requests for source fitting in practice):

```json
{
  "train": ["request-a", "request-b"],
  "val": ["request-c"],
  "test": ["request-d"]
}
```

The join preserves each split's listed order. Feature collection does not assign labels or choose train/validation/test membership.

## Prompt JSONL for optional feature collection

Each row contains `id` and `prompt`. The collector writes `x`, `ids`, and `metadata_json` (model revision, one-based layer, prompt-file digest, software versions, precision). It runs prefill only, so labels must be joined by request ID before building the split NPZ.

Prompts longer than `--max-prompt-tokens` are rejected, not truncated. Apply the same chat template to all prompts, since features from different templates are not comparable.

## Head and RLS prior

`head.npz` contains the folded `w` and `b`, the source-training `mean` and `scale`, `mapping`, `budget`, and JSON metadata (alpha, penalty `alpha * D`, source feature digest, feature identity). Coefficients are stored in FP32 and accumulated in FP64; inference takes raw features.

`rls_prior.npz` contains the FP32 precision inverse `P`, FP64 `coef`, and the same `mean`/`scale`. It is built offline from the source features used for fitting. For D=4096, `P` takes about 67 MB. Online updates do not store the completion history.

The loader does not accept pickle checkpoints; convert heads from other pipelines to this schema.

