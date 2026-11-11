"""Fixed-address inputs for single-card Ascend decode graphs."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
from torch import Tensor

GRAPH_MODES = ("eager", "aclgraph", "npugraph_ex")

Decode = Callable[..., tuple[Tensor, ...]]


class DecodeGraph:
    def __init__(self, forward: Decode, inputs: tuple[Tensor, ...], *, mode: str):
        # All-zero metadata routes warmup writes to page zero, outside the
        # Engine's one-based physical pages. Replay stages the live metadata.
        self.inputs = tuple(torch.zeros_like(value) for value in inputs)
        device = inputs[0].device
        if mode == "aclgraph":
            self._run = self._capture(forward, device)
        elif mode == "npugraph_ex":
            self._run = self._compile(forward)
        else:
            raise ValueError(f"unsupported decode graph mode: {mode!r}")
        torch.npu.synchronize(device)

    def _capture(
        self, forward: Decode, device: torch.device
    ) -> Callable[[], tuple[Tensor, ...]]:
        capture_stream = torch.npu.Stream(device=device)
        current_stream = torch.npu.current_stream(device)
        capture_stream.wait_stream(current_stream)
        with torch.npu.stream(capture_stream):
            forward(*self.inputs)
            forward(*self.inputs)
        current_stream.wait_stream(capture_stream)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream):
            output = forward(*self.inputs)

        def run() -> tuple[Tensor, ...]:
            # NPUGraph enqueues replay on the current stream, after input copies.
            graph.replay()
            return output

        return run

    def _compile(self, forward: Decode) -> Callable[[], tuple[Tensor, ...]]:
        try:
            importlib.import_module("torchair")
        except ModuleNotFoundError as exc:
            if exc.name != "torchair":
                raise
        compiled = torch.compile(
            forward, backend="npugraph_ex", fullgraph=True, dynamic=False
        )
        compiled(*self.inputs)
        compiled(*self.inputs)
        return lambda: compiled(*self.inputs)

    def replay(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        for buffer, value in zip(self.inputs, inputs, strict=True):
            buffer.copy_(value)
        return self._run()
