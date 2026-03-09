# vLLM Zero-Copy Model Loader (xet)

A vLLM model loader that downloads safetensors weights via xet CAS directly into pinned host memory, bypassing disk entirely.

## How it works

1. Resolves xet hashes and CAS token from Hub API
2. Allocates pinned host memory (`torch.empty(pin_memory=True)`)
3. Downloads shards via `hf_xet.download_to_buffer` (single CAS roundtrip, parallel xorb block fetches)
4. Parses safetensors headers in-place and yields `torch.frombuffer` tensors
5. GPU loads tensors via DMA from pinned memory

Key optimizations:
- **Prefetch**: first shard download starts before model architecture initialization
- **Pipelining**: shard N+1 downloads while shard N's tensors are being consumed
- **Single CAS roundtrip**: all file terms resolved at once (no adaptive prefetch overhead)
- **Zero disk I/O**: data goes network -> pinned RAM -> GPU

## Requirements

- `hf_xet` with `download_to_buffer` support (xet-core PR #688)
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

## Benchmark (SmolLM2-1.7B, 3.4 GB, g5.xlarge A10G)

| Path | Median TTFT | Speedup |
|---|---|---|
| Standard (xet -> disk -> load) | 12.00s | 1x |
| **Zero-copy (xet -> pinned mem -> DMA)** | **5.75s** | **2.1x** |
