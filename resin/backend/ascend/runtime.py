"""Device initialization and single-node process groups for Ascend."""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

from resin import Parallelism
from resin.backend.topology import ParallelGroups


def initialize_device(index: int) -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "load the matching CANN environment and torch_npu first"
        ) from exc
    if not 0 <= index < torch.npu.device_count():
        raise ValueError(f"NPU index {index} is not visible to torch_npu")
    device = torch.device("npu", index)
    torch.npu.set_device(device)
    return device


def create_groups(dp: int, tp: int, world: int) -> ParallelGroups:
    if dp < 1 or tp < 1 or dp * tp != world or world > 8:
        raise ValueError("Ascend requires DP × TP == WORLD_SIZE, with 1–8 ranks")
    timeout = timedelta(minutes=2)
    dist.init_process_group("gloo", timeout=timeout)
    try:
        return ParallelGroups.create(
            Parallelism(world_size=world, attn_dp=dp, attn_tp=tp),
            device_backend="hccl",
            timeout=timeout,
        )
    except BaseException:
        dist.destroy_process_group()
        raise
