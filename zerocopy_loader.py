# SPDX-License-Identifier: Apache-2.0
"""Zero-copy model loader: xet CAS -> pinned memory -> GPU tensors.

Downloads safetensors shards via hf_xet.download_to_buffer directly into
pinned host memory, then yields tensors for GPU loading. Bypasses disk entirely.

Usage:
    vllm serve model_id --load-format hf_zerocopy
"""
import ctypes
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _parse_safetensors_header(buf_addr: int, buf_size: int) -> tuple[dict, int]:
    header_len = struct.unpack(
        "<Q", (ctypes.c_char * 8).from_address(buf_addr).raw
    )[0]
    if header_len > buf_size - 8:
        raise ValueError(
            f"Safetensors header ({header_len}B) exceeds buffer ({buf_size}B)"
        )
    header_json = (
        (ctypes.c_char * header_len).from_address(buf_addr + 8).raw.decode("utf-8")
    )
    return json.loads(header_json), 8 + header_len


class _ShardResult:
    """Holds the result of a background shard download."""

    __slots__ = ("buf", "file_size", "error", "download_time", "done")

    def __init__(self):
        self.buf: torch.Tensor | None = None
        self.file_size: int = 0
        self.error: Exception | None = None
        self.download_time: float = 0.0
        self.done = threading.Event()


def _download_shard(
    xet_hash: str,
    file_size: int,
    cas_url: str,
    cas_token: str,
    cas_exp: int,
    result: _ShardResult,
    buf: torch.Tensor | None = None,
):
    """Download a shard into pinned memory via hf_xet. Runs in a background thread."""
    try:
        import hf_xet

        if buf is None:
            buf = torch.empty(file_size, dtype=torch.uint8, pin_memory=True)
        result.buf = buf
        result.file_size = file_size

        t0 = time.perf_counter()
        hf_xet.download_to_buffer(
            hash=xet_hash,
            file_size=file_size,
            buf_ptr=buf.data_ptr(),
            buf_len=file_size,
            endpoint=cas_url,
            token_info=(cas_token, cas_exp),
            token_refresher=None,
        )
        result.download_time = time.perf_counter() - t0
    except Exception as e:
        result.error = e
    finally:
        result.done.set()


def _yield_tensors_from_shard(buf: torch.Tensor, file_size: int):
    """Parse safetensors header and yield (name, tensor) from pinned buffer."""
    buf_np = buf.numpy()
    header_len = struct.unpack("<Q", buf_np[:8].tobytes())[0]
    metadata = json.loads(buf_np[8 : 8 + header_len].tobytes())
    data_offset = 8 + header_len

    for name, info in metadata.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        if dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        torch_dtype, elem_size = DTYPE_MAP[dtype_str]
        start, end = info["data_offsets"]
        count = (end - start) // elem_size
        tensor = torch.frombuffer(
            buf_np, dtype=torch_dtype, offset=data_offset + start, count=count
        ).reshape(info["shape"])
        yield name, tensor


class ZeroCopyModelLoader(BaseModelLoader):
    """Model loader that downloads directly into pinned memory via xet CAS.

    Bypasses disk entirely. Uses hf_xet.download_to_buffer to download
    safetensors shards into pinned host memory, then yields tensors for
    GPU loading via DMA.

    Key optimizations:
    - Prefetch: first shard download starts before model architecture init
    - Pipelining: shard N+1 downloads while shard N's tensors are consumed
    - Zero-copy: CAS data lands directly in pinned memory, DMA to GPU
    - Single CAS roundtrip per shard for term resolution
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # Prefetch state
        self._shards: list[tuple[str, int, str]] | None = None
        self._cas_url: str | None = None
        self._cas_token: str | None = None
        self._cas_exp: int | None = None
        self._first_shard: _ShardResult | None = None

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _start_prefetch(self, model_config: ModelConfig):
        """Resolve shard info, CAS token, and start downloading first shard.

        Called BEFORE initialize_model() so download overlaps with model init.
        """
        t0 = time.perf_counter()
        token = _get_hf_token()
        hub_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        revision = model_config.revision or "main"

        self._shards = _get_shard_info(model_config.model, revision, token)
        self._cas_url, self._cas_token, self._cas_exp = _get_cas_token(
            hub_endpoint, model_config.model, revision, token
        )

        total_gb = sum(s for _, s, _ in self._shards) / 1e9
        logger.info(
            "Zero-copy (xet): %d shard(s), %.2f GB total",
            len(self._shards),
            total_gb,
        )

        if not self._shards:
            return

        # Allocate pinned buffer for first shard
        first_buf = torch.empty(
            self._shards[0][1], dtype=torch.uint8, pin_memory=True
        )

        # Start download in background (overlaps with model init)
        self._first_shard = _ShardResult()
        t = threading.Thread(
            target=_download_shard,
            args=(
                self._shards[0][2],  # xet_hash
                self._shards[0][1],  # file_size
                self._cas_url,
                self._cas_token,
                self._cas_exp,
                self._first_shard,
                first_buf,
            ),
            daemon=True,
        )
        t.start()

        dt = time.perf_counter() - t0
        logger.info("Zero-copy: prefetch started in %.2fs", dt)

    @instrument(span_name="Load weights (zerocopy-xet)")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        assert self._shards is not None, "_start_prefetch must be called first"

        t0 = time.perf_counter()

        def weights_iter():
            pending = self._first_shard

            for i, (filename, file_size, xet_hash) in enumerate(self._shards):
                assert pending is not None
                pending.done.wait()
                if pending.error:
                    raise pending.error

                logger.info(
                    "Downloaded %s (%.2f GB) in %.2fs (%.2f GB/s)",
                    filename,
                    pending.file_size / 1e9,
                    pending.download_time,
                    pending.file_size / pending.download_time / 1e9
                    if pending.download_time > 0
                    else 0,
                )

                # Start next shard download in background
                next_pending = None
                if i + 1 < len(self._shards):
                    next_pending = _ShardResult()
                    t = threading.Thread(
                        target=_download_shard,
                        args=(
                            self._shards[i + 1][2],
                            self._shards[i + 1][1],
                            self._cas_url,
                            self._cas_token,
                            self._cas_exp,
                            next_pending,
                        ),
                        daemon=True,
                    )
                    t.start()

                # Yield tensors from current shard
                yield from _yield_tensors_from_shard(pending.buf, pending.file_size)

                # Free pinned buffer
                del pending.buf
                pending = next_pending

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
        """Override load_model to start download BEFORE model architecture init."""
        from vllm.platforms import current_platform

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            # Start download in background FIRST
            self._start_prefetch(model_config)

            # Then initialize model architecture (download runs concurrently)
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # Load weights (first shard may already be downloaded)
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
