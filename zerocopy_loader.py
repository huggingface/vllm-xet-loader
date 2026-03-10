# SPDX-License-Identifier: Apache-2.0
"""Zero-copy model loader: xet CAS -> pinned memory -> GPU tensors.

Downloads safetensors shards via hf_xet directly into pinned host memory,
then yields tensors for GPU loading. Bypasses disk entirely.

Uses a thread pool to download multiple tensors in parallel, each into its
own pinned buffer. Peak pinned memory is bounded to N_WORKERS * largest tensor.

Usage:
    vllm serve model_id --load-format hf_zerocopy
"""
import itertools
import json
import os
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import HfApi
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

DTYPE_MAP = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
}


def _get_hf_token() -> str | None:
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _get_cas_token(hub_endpoint: str, repo: str, revision: str, token: str | None):
    """Get CAS JWT (endpoint + token) from Hub API."""
    url = f"{hub_endpoint}/api/models/{repo}/xet-read-token/{revision}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["casUrl"], data["accessToken"], data["exp"]


def _get_shard_info(model_name: str, revision: str | None, token: str | None):
    """Get safetensors shard filenames, sizes, and xet hashes from HF API."""
    hub_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    # Get file list with xet hashes
    api_url = f"{hub_endpoint}/api/models/{model_name}/tree/{revision or 'main'}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(api_url, headers=headers)
    resp.raise_for_status()
    all_files = {e["path"]: e for e in resp.json()}

    # Find safetensors files
    st_files = {k: v for k, v in all_files.items() if k.endswith(".safetensors")}

    # If sharded, filter to referenced shards
    if "model.safetensors.index.json" in all_files:
        api = HfApi()
        index_path = api.hf_hub_download(
            model_name, "model.safetensors.index.json", revision=revision
        )
        with open(index_path) as f:
            referenced = set(json.load(f).get("weight_map", {}).values())
        if referenced:
            st_files = {k: v for k, v in st_files.items() if k in referenced}

    result = []
    for fname in sorted(st_files.keys()):
        entry = st_files[fname]
        xet_hash = entry.get("xetHash")
        size = entry.get("size")
        if not xet_hash:
            raise ValueError(f"File '{fname}' has no xetHash (not stored via xet)")
        result.append((fname, size, xet_hash))

    return result


def _download_header(xet_hash, file_size, cas_url, cas_token, cas_exp):
    """Download and parse the safetensors header via two small range requests."""
    import hf_xet

    # First 8 bytes: header length (little-endian u64)
    size_buf = torch.empty(8, dtype=torch.uint8, pin_memory=True)
    hf_xet.download_to_buffer(
        hash=xet_hash, file_size=file_size,
        buf_ptr=size_buf.data_ptr(), buf_len=8,
        endpoint=cas_url, token_info=(cas_token, cas_exp),
        token_refresher=None, byte_range=(0, 8),
    )
    header_len = struct.unpack("<Q", size_buf.numpy().tobytes())[0]
    if header_len > file_size - 8:
        raise ValueError(
            f"Safetensors header ({header_len}B) exceeds file size ({file_size}B)"
        )

    # Download full header
    header_total = 8 + header_len
    header_buf = torch.empty(header_total, dtype=torch.uint8, pin_memory=True)
    hf_xet.download_to_buffer(
        hash=xet_hash, file_size=file_size,
        buf_ptr=header_buf.data_ptr(), buf_len=header_total,
        endpoint=cas_url, token_info=(cas_token, cas_exp),
        token_refresher=None, byte_range=(0, header_total),
    )
    metadata = json.loads(header_buf.numpy()[8:header_total].tobytes())
    return metadata, header_total


_DOWNLOAD_WORKERS = int(os.environ.get("HF_ZEROCOPY_WORKERS", "16"))


def _download_one_tensor(xet_hash, file_size, cas_url, cas_token, cas_exp,
                         data_offset, name, info):
    """Download a single tensor into a pinned buffer. Returns (name, tensor)."""
    import hf_xet

    dtype_str = info["dtype"]
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    torch_dtype, elem_size = DTYPE_MAP[dtype_str]
    start, end = info["data_offsets"]
    tensor_bytes = end - start
    count = tensor_bytes // elem_size

    buf = torch.empty(tensor_bytes, dtype=torch.uint8, pin_memory=True)
    hf_xet.download_to_buffer(
        hash=xet_hash, file_size=file_size,
        buf_ptr=buf.data_ptr(), buf_len=tensor_bytes,
        endpoint=cas_url, token_info=(cas_token, cas_exp),
        token_refresher=None,
        byte_range=(data_offset + start, data_offset + end),
    )

    tensor = torch.frombuffer(
        buf.numpy(), dtype=torch_dtype, count=count
    ).reshape(info["shape"])
    return name, tensor


def _yield_tensors(xet_hash, file_size, cas_url, cas_token, cas_exp):
    """Download tensors in parallel via per-tensor byte-range requests.

    Uses a sliding window of futures to bound peak pinned memory. The prefetch
    window is 4x the worker count, which benchmarks show matches submit-all
    throughput while capping in-flight buffers.
    """
    metadata, data_offset = _download_header(
        xet_hash, file_size, cas_url, cas_token, cas_exp
    )

    # Collect and sort tensors by their byte offset
    tensors = [
        (name, info) for name, info in metadata.items() if name != "__metadata__"
    ]
    tensors.sort(key=lambda x: x[1]["data_offsets"][0])

    if not tensors:
        return

    n_workers = min(_DOWNLOAD_WORKERS, len(tensors))
    prefetch = min(n_workers * 4, len(tensors))
    logger.info(
        "Shard: %d tensors (parallel download, %d workers, prefetch %d)",
        len(tensors), n_workers, prefetch,
    )

    def _submit(pool, name, info):
        return pool.submit(
            _download_one_tensor, xet_hash, file_size,
            cas_url, cas_token, cas_exp, data_offset, name, info,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        it = iter(tensors)
        pending = deque()

        # Prefill the window
        for name, info in itertools.islice(it, prefetch):
            pending.append(_submit(pool, name, info))

        # Yield completed, refill from remaining
        for name, info in it:
            yield pending.popleft().result()
            pending.append(_submit(pool, name, info))

        # Drain
        while pending:
            yield pending.popleft().result()


class ZeroCopyModelLoader(BaseModelLoader):
    """Model loader that downloads directly into pinned memory via xet CAS.

    Bypasses disk entirely. Downloads tensors in parallel via byte-range
    requests into pinned buffers, then yields for GPU loading via DMA.

    Key optimizations:
    - Parallel per-tensor downloads: N concurrent CAS requests via thread pool
    - Sliding window prefetch: bounds in-flight pinned buffers to 4x workers
    - Zero disk I/O: data goes network -> pinned RAM -> GPU
    - Multi-GPU: each rank downloads independently (no broadcast needed)
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @instrument(span_name="Load weights (zerocopy-xet)")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        token = _get_hf_token()
        hub_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        revision = model_config.revision or "main"

        shards = _get_shard_info(model_config.model, revision, token)
        cas_url, cas_token, cas_exp = _get_cas_token(
            hub_endpoint, model_config.model, revision, token
        )

        total_gb = sum(s for _, s, _ in shards) / 1e9
        logger.info(
            "Zero-copy (xet): %d shard(s), %.2f GB total",
            len(shards), total_gb,
        )

        t0 = time.perf_counter()

        def weights_iter():
            for filename, file_size, xet_hash in shards:
                logger.info("Loading %s (%.2f GB)...", filename, file_size / 1e9)
                t_shard = time.perf_counter()
                yield from _yield_tensors(
                    xet_hash, file_size, cas_url, cas_token, cas_exp,
                )
                dt = time.perf_counter() - t_shard
                logger.info(
                    "Loaded %s in %.2fs (%.2f GB/s)",
                    filename, dt, file_size / dt / 1e9 if dt > 0 else 0,
                )

        loaded_weights = model.load_weights(weights_iter())
        dt = time.perf_counter() - t0
        logger.info("Zero-copy weight loading completed in %.2fs", dt)

        if model_config.quantization is None and loaded_weights is not None:
            weights_to_load = {name for name, _ in model.named_parameters()}
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

    @instrument(span_name="Load model (zerocopy-xet)")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        from vllm.platforms import current_platform

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            self.load_weights(model, model_config)

            if current_platform.is_cuda():
                peak_memory = torch.cuda.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


# Late registration to avoid circular import when this file is placed inside
# vllm/model_executor/model_loader/
from vllm.model_executor.model_loader import register_model_loader  # noqa: E402

register_model_loader("hf_zerocopy")(ZeroCopyModelLoader)
