# Reproduction notes

## Readout and RLS

Features use the source-training mean and sample standard deviation plus `1e-6`. The centered dual Ridge system is `X X^T / D + alpha I`, equivalent to a feature penalty `alpha * D` with an unpenalized intercept. Validation MAE selects alpha. Source normalization is folded into raw-feature coefficients; square-root predictions are clipped before squaring and bounded to `[1, 2048]`.

`RLSFeedbackCore` anchors updates to the deployed source head, including export rounding. Every eight ordered completions apply a Woodbury update. `P` is FP32; small solves and coefficients are FP64. The updater does not replay completion history. Tests compare it to a direct prior-anchored batch solve. Publication is versioned, slot/generation distinguishes request incarnations, and worker shutdown drains in-flight updates. Internal queue handling is in `_feedback.py`; historical bias/window-fit alternatives are omitted.

## Serving integration

The portable library supplies predictions and causal score consumption, not a full engine launcher. An integration must:

1. Capture the selected raw block's last prompt token; in vLLM, add branch and residual before the training-compatible FP16-to-FP32 export.
2. Retain request ID, slot/generation and feature-buffer ownership until the worker acknowledges it. Only the prediction worker waits for device-to-host completion.
3. Publish initial and RLS scores with timestamps and model versions. The scheduler reads only results accepted before its snapshot.
4. Subtract generated tokens once for countdown scheduling; missing scores fall back to FIFO. Keep admission, prefill, KV allocation and waiting protection in the engine.
5. Supply actual completion lengths in completion order and drain pending work before shutdown.

The paper used vLLM 0.8.5.post1, PyTorch 2.6.0+cu124 and Transformers 4.57.6, recorded in `requirements-paper-linux.txt`. Installing those dependencies alone does not supply engine wiring or model assets.

## Experiment settings

The JSON files in `configs/` record protocols; the CLI does not load them automatically. `readout_sqrt.json` matches the default fitting map and alpha grid.

Figure 6 has 20 points: FCFS/LEAP/TIE/LTR × five rates, the same first 1,024 LMSYS requests and 544,047 output tokens at every point, seed 0, resident limit 512, decode budget 64 and Aging 0.9 per 30 s. LEAP predicts on CPU; TIE/LTR use serving GPU physical batches of eight. Native B32 results are not merged into this panel. PTL is per-request end-to-end latency divided by actual output length before aggregation, not inter-token latency.

Figure 7 has 42 cells: seven methods × six rates, batch 8, a uniform native-clock polling producer with 1 s lead-in and 10 s supply, queue eight batches, drain up to 60 s and no expiration. Predictor, producer, receiver and sampler occupy separate cores. No serving model runs. Generator limitation means skipped batches exceed 1% or P95 attempt lag exceeds one batch interval; all published cells pass. Queue rejection and delivered throughput still determine overload behavior.

Figure 8 retains 12 rows covering two asynchronous paths, post-prefill and no-readout controls at three batch sizes. The plot shows six asynchronous ready fractions. Figure 5 retains its source/target cohorts and deduplicates repeated source rows. The appendix keeps 151 recorded epoch values without smoothing or imputation. Internal CSV method keys map to canonical display names in the plot script.

## Baselines

| Method | Recorded configuration |
|---|---|
| TIE | DeBERTa, two distribution heads, 20K LMSYS prompts with 20 outputs each; bounded expectation for quality, 10K MC draws without CVaR for the CPU component study |
| LTR | OPT-125M, ListMLE, 2,048-token input; source-validation isotonic mapping for length-based transfer |
| SSJF | BERT, L1, tail 512 tokens, three encoder-training then three frozen epochs; standardized source targets inverted at inference |
| TRAIL | 40-bin initial head from 2,880 requests; separate online head/filter with 1,390,439 training states derived from those requests |
| EGTP | Prefix-four-token feature, two-layer 20-bin head; EGTP-P reconstructs logits on CPU, EGTP-L assumes logits available |

Upstream baseline code and weights are not bundled; see [third-party notices](../THIRD_PARTY_NOTICES.md). The released backbone heads and single-label scale controls remain separate in [checkpoints](../checkpoints/README.md).

## Scope

`paper_data/PROVENANCE.json` binds aggregate CSV hashes, row counts and cohort definitions. Plotting reproduces their numerical values; fonts/layout may differ from the manuscript. Source paths in training-curve data are replaced with content hashes. Serving points and CPU cells each have one hardware run, without confidence intervals.

New hardware measurements require fixed model/checkpoint revisions, input IDs/order, RNG states, precision, thread counts, affinity, device and timing contracts. Predefine the grid, order and repeats; report all repetitions and their spread. Avoid unrelated jobs during timing windows. No prompts, generated trajectories or full hardware logs are released here.
