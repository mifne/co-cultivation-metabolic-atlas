"""Deterministic GPU assignment for parallel environment workers."""

from __future__ import annotations

import os
from typing import Iterable, Optional


def parse_gpu_ids(value: Optional[object] = None) -> list[int]:
    """Parse ``0,1,2``/``[0, 1, 2]`` into physical CUDA device IDs.

    If *value* is omitted, ``DFBA_GPU_IDS`` is used. An empty value means that
    CUDA visibility is unchanged.
    """

    if value is None:
        value = os.environ.get("DFBA_GPU_IDS", "")
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Iterable):
        raw = list(value)
    else:
        raw = [value]
    try:
        ids = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU IDs must be comma-separated integers, e.g. 0,1,2") from exc
    if any(item < 0 for item in ids):
        raise ValueError("GPU IDs must be non-negative")
    return ids


def assign_gpu_for_worker(
    rank: int,
    gpu_ids: Optional[object] = None,
    slots_per_device: Optional[int] = None,
) -> Optional[int]:
    """Pin one environment worker to one physical GPU.

    ``CUDA_VISIBLE_DEVICES`` is set before lazy cuOpt/PyTorch CUDA initialization
    in the worker. cuOpt then sees its assigned physical card as device 0.
    ``slots_per_device`` records a logical concurrency slot for observability;
    consumer RTX cards do not provide MIG-style hard memory partitioning, so
    actual isolation is cooperative process-level concurrency.
    """

    ids = parse_gpu_ids(gpu_ids)
    if not ids:
        return None
    if slots_per_device is None:
        slots_per_device = int(os.environ.get("DFBA_GPU_SLOTS_PER_DEVICE", "1"))
    slots_per_device = max(1, int(slots_per_device))
    assigned = ids[rank % len(ids)]
    logical_slot = (rank // len(ids)) % slots_per_device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned)
    os.environ["DFBA_ASSIGNED_GPU"] = str(assigned)
    os.environ["DFBA_GPU_SLOT"] = str(logical_slot)
    os.environ["DFBA_GPU_SLOTS_PER_DEVICE"] = str(slots_per_device)
    return assigned
