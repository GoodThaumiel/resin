"""Fixed-address inputs for the single-card TorchAir decode path."""

from __future__ import annotations

import importlib
import logging

import torch
from torch import Tensor

log = logging.getLogger(__name__)


class DecodeGraph:
    def __init__(self, forward, inputs: tuple[Tensor, ...]):
        try:
            importlib.import_module("torchair")
        except ModuleNotFoundError as exc:
            if exc.name != "torchair":
                raise
        self.inputs = tuple(torch.zeros_like(value) for value in inputs)
        self.forward = torch.compile(
            forward, backend="npugraph_ex", fullgraph=True, dynamic=False
        )
        # Capture on page zero, outside the Engine's one-based physical pages.
        self.forward(*self.inputs)
        self.forward(*self.inputs)
        torch.npu.synchronize(inputs[0].device)
        log.info("Ascend decode graph captured, page bucket=%d", inputs[2].numel())

    def replay(self, inputs: tuple[Tensor, ...]):
        for buffer, value in zip(self.inputs, inputs, strict=True):
            buffer.copy_(value)
        return self.forward(*self.inputs)
