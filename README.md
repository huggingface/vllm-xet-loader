# vLLM Zero-Copy Model Loader (xet)

A vLLM model loader that downloads safetensors weights via xet CAS directly into pinned host memory, bypassing disk entirely.

## How it works

1. Resolves xet hashes and CAS token from Hub API
2. Parses safetensors headers via small byte-range requests
3. Downloads each tensor individually in parallel (thread pool) via `hf_xet.download_to_buffer` byte-range requests into pinned host memory
4. Yields `torch.frombuffer` tensors for GPU loading via DMA

Key properties:
- **Parallel per-tensor downloads**: 16 concurrent CAS byte-range requests (configurable via `HF_ZEROCOPY_WORKERS`)
- **Server-side trimming**: CAS server trims responses at chunk granularity, so only needed data is transferred
- **Zero disk I/O**: data goes network -> pinned RAM -> GPU
- **Bounded memory**: peak pinned memory = N_workers * largest tensor size

## Requirements

- `hf_xet` with `download_to_buffer` byte-range support (xet-core PR #688)
- CUDA GPU
- `vllm` v0.8+

## Installation

Copy the loader into vLLM and register it:

```python
# In vllm/model_executor/model_loader/__init__.py:
from vllm.model_executor.model_loader.zerocopy_loader import ZeroCopyModelLoader

# Add "hf_zerocopy" to the LoadFormat Literal type
# Add to _LOAD_FORMAT_TO_MODEL_LOADER dict:
"hf_zerocopy": ZeroCopyModelLoader,
```

Then copy `zerocopy_loader.py` into `vllm/model_executor/model_loader/`.

## Usage

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --load-format hf_zerocopy --enforce-eager
```

## Benchmarks

### Per-tensor download (SmolLM2-1.7B, 1 shard 3.42 GB, g5.2xlarge A10G, us-east-1)

HF cache on local NVMe SSD for fair comparison:

| Method | Time | GB/s | Speedup |
|---|---|---|---|
| hf_hub download + safetensors load (NVMe) | 25.16s | 0.14 | 1.0x |
| Zero-copy parallel (8 workers) | 4.14s | 0.83 | 6.1x |
| **Zero-copy parallel (16 workers)** | **3.70s** | **0.92** | **6.8x** |

### End-to-end vLLM TTFT (Qwen2.5-7B-Instruct, 4 shards 15.2 GB)

| Path | TTFT | Speedup |
|---|---|---|
| Standard (xet -> disk -> load) | 43.0s | 1x |
| **Zero-copy (xet -> pinned mem -> DMA)** | **16.0s** | **2.7x** |
