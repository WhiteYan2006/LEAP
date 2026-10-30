# Third-party components and assets

The trained LEAP linear readouts in `checkpoints/` are included. No vLLM, SGLang, Transformers, baseline repository, backbone model weights, or dataset is included in this repository. Each readout identifies its backbone and revision; applicable Qwen, Llama, Mistral, and dataset terms apply separately from the project's own license.

Runtime dependencies are installed separately. NumPy, Matplotlib, PyTorch, Transformers, Safetensors, and any serving engine retain their upstream licenses and notices. Model and dataset terms apply independently of this project's eventual license.

TRAIL: `https://github.com/harvard-cns/TRAIL-llm-scheduling` (Apache-2.0). TIE: upstream commit `ce6ddc4d7abe6a4a0821a03d463ad7af86113850` (Apache-2.0). Their code is not copied here.

EGTP: LP_Bench at `https://github.com/xiehuanyi/LP_Bench`, commit `170a47893e1351ecc062186aafc686ab3f8be3d1`. Its models and source are not included.

For LTR, SSJF, backbone models, and datasets, use the upstream releases and their licenses. Baseline configurations are in [reproduction notes](docs/REPRODUCIBILITY.md).

