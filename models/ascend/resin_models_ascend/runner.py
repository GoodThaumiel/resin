"""Synchronous Qwen text execution against Engine-owned physical page leases."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheLayout,
    ResourceBinding,
    RunnerConfig,
    RunnerMetrics,
)
from resin.backend.ascend.graphs import GRAPH_MODES, DecodeGraph
from resin.backend.ascend.ops import NativePagedAttention
from resin.backend.plans import CacheGroupView, PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import (
    PromptScores,
    TorchSampler,
    format_output,
    prompt_scores,
)
from resin.backend.topology import ParallelismCapabilities
from resin_models_ascend.checkpoint import load_model
from resin_models_ascend.model import (
    GatedDeltaNet,
    RMSNorm,
    TensorParallel,
    prefill_gdn_segment,
)

log = logging.getLogger(__name__)
PAGE_TOKENS = 128
TARGET, STATE, MTP = range(3)


@dataclass
class LiveState:
    recurrent: Tensor
    convolution: Tensor
    hidden: Tensor


class QwenAscendRunner:
    # With ep=1, Parallelism still derives expert_tp from the full world.
    # Dense Qwen shards its FFNs on the attention TP group within each replica.
    parallelism_capabilities = ParallelismCapabilities(
        attn_dp=True, attn_tp=True, expert_tp=True
    )

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        tp: TensorParallel = TensorParallel(),
        cache_pages: int = 128,
        prefill_chunk: int = 256,
        graph_max_pages: int = 32,
        graph_capture_pages: Sequence[int] | None = None,
        graph_verify_batches: Sequence[int] | None = None,
        mtp: bool = False,
        num_speculative_tokens: int | None = None,
        quantization: str = "bf16",
        graph: str = "eager",
        attention: str = "torch",
        native_gdn_prefill: bool = False,
        fused_rmsnorm: bool | None = None,
        gdn_backend: str | None = None,
        triton_gdn: bool = False,
        memory_report: bool = False,
        prefill_graph: bool = False,
    ) -> None:
        if num_speculative_tokens is not None and (
            type(num_speculative_tokens) is not int
            or not 1 <= num_speculative_tokens < 2**32 - 1
        ):
            raise ValueError(
                "num_speculative_tokens must be positive and fit the decode lease"
            )
        draft_count = (
            num_speculative_tokens if num_speculative_tokens is not None else int(mtp)
        )
        mtp = draft_count > 0
        verify_width = 1 + draft_count
        if graph == "aclgraph" and verify_width > PAGE_TOKENS:
            raise ValueError("verify query width exceeds the ACLGraph scratch page")
        if min(cache_pages, prefill_chunk, graph_max_pages) < 1:
            raise ValueError(
                "cache pages, prefill chunk, and graph max pages must be positive"
            )
        if graph not in GRAPH_MODES or attention not in {
            "torch",
            "native",
        }:
            raise ValueError("unsupported Ascend graph or attention mode")
        capture_pages = None
        if graph_capture_pages is not None:
            if graph != "aclgraph":
                raise ValueError("graph_capture_pages requires graph='aclgraph'")
            if not graph_capture_pages or any(
                type(pages) is not int
                or not 1 <= pages <= graph_max_pages
                or pages & (pages - 1)
                for pages in graph_capture_pages
            ):
                raise ValueError(
                    "graph_capture_pages must contain positive powers of two "
                    "up to graph_max_pages"
                )
            capture_pages = tuple(sorted(set(graph_capture_pages)))
        if graph_verify_batches is not None:
            if not mtp or capture_pages is None:
                raise ValueError(
                    "graph_verify_batches requires MTP and startup capture"
                )
            if not graph_verify_batches or any(
                type(size) is not int or not 1 <= size <= PAGE_TOKENS // verify_width
                for size in graph_verify_batches
            ):
                raise ValueError(
                    f"verify batch sizes must be in [1, {PAGE_TOKENS // verify_width}]"
                )
        if graph != "eager" and (
            tp.size > 1 or device.type != "npu" or attention != "torch"
        ):
            raise ValueError(
                f"{graph} requires single-card decode with tensor attention"
            )
        if mtp and graph == "npugraph_ex":
            raise ValueError("MTP graph execution requires graph='aclgraph'")
        if attention == "native" and device.type != "npu":
            raise ValueError("native attention requires an NPU")
        if native_gdn_prefill and device.type != "npu":
            raise ValueError("native GDN prefill requires an NPU")
        if fused_rmsnorm is None:
            fused_rmsnorm = device.type == "npu"
        if triton_gdn:
            if gdn_backend not in (None, "triton"):
                raise ValueError("triton_gdn conflicts with gdn_backend")
            gdn_backend = "triton"
        if gdn_backend is None:
            gdn_backend = (
                "fla" if device.type == "npu" and graph != "npugraph_ex" else "off"
            )
        if gdn_backend not in ("fla", "triton", "off"):
            raise ValueError("unsupported GDN backend")
        if (fused_rmsnorm or gdn_backend != "off") and device.type != "npu":
            raise ValueError("fused RMSNorm and GDN require an NPU")
        if gdn_backend != "off" and graph == "npugraph_ex":
            raise ValueError("fused GDN supports eager execution and raw ACLGraph")
        if gdn_backend == "fla" and verify_width > 8:
            raise ValueError("FLA GDN supports up to 8 query tokens per request")
        if prefill_graph and (
            device.type != "npu"
            or tp.size != 1
            or graph == "npugraph_ex"
            or gdn_backend != "fla"
            or native_gdn_prefill
        ):
            raise ValueError("piecewise prefill requires single-card FLA GDN")
        self.device = device
        self.memory_report = memory_report
        self._report_memory("before_model")
        self.model = load_model(
            checkpoint, device, tp, mtp=mtp, quantization=quantization
        )
        if fused_rmsnorm:
            import torch_npu

            if not hasattr(torch_npu, "npu_rms_norm"):
                raise RuntimeError("fused RMSNorm requires torch_npu.npu_rms_norm")
            for module in self.model.modules():
                if isinstance(module, RMSNorm):
                    module.prepare_fused()
        if gdn_backend == "triton":
            from resin_models_ascend.triton_gdn import fused_recurrent_gdn

            for module in self.model.modules():
                if isinstance(module, GatedDeltaNet):
                    module.fused_recurrence = fused_recurrent_gdn
        if gdn_backend == "fla":
            from resin_models_ascend.fla_gdn import FlaGDN

            try:
                from resin_models_ascend.triton_gating import fused_gdn_gating
            except ModuleNotFoundError as error:
                if error.name != "triton":
                    raise
                fused_gdn_gating = None
                log.warning("Triton Ascend unavailable; using reference GDN gating")
            config = self.model.config
            if (config.linear_key_head_dim, config.linear_value_head_dim) != (128, 128):
                raise ValueError("FLA GDN requires 128-dimensional key and value heads")
            fla = FlaGDN()
            for module in self.model.modules():
                if isinstance(module, GatedDeltaNet):
                    module.fla_gdn = fla
                    module.fused_gating = fused_gdn_gating
                    module.packed_conv_weight = (
                        module.conv1d.weight.detach().squeeze(1).t().contiguous()
                    )
            log.info(
                "Ascend FLA: prefill=%s, recurrence=fla, conv=fla, gating=%s",
                "torch_npu_chunk" if native_gdn_prefill else "fla_chunk",
                "triton" if fused_gdn_gating is not None else "torch",
            )
        log.info("Ascend kernels: fused_rmsnorm=%s, gdn=%s", fused_rmsnorm, gdn_backend)
        self._report_memory("after_model")
        self.mtp = mtp
        self.num_speculative_tokens = draft_count
        self.native = NativePagedAttention(device) if attention == "native" else None
        self.native_gdn_prefill = native_gdn_prefill
        self.sampler = TorchSampler()
        groups = [
            CacheGroupLayout(PAGE_TOKENS),
            CacheGroupLayout(prefill_chunk, family=CacheGroupFamily.STATE),
        ]
        if mtp:
            groups.append(CacheGroupLayout(PAGE_TOKENS))
        self.layout = CacheLayout(parent_capacity=cache_pages, groups=groups)
        self.runner_config = RunnerConfig(
            cache_layout=self.layout,
            prefill_chunk_size=prefill_chunk,
            continuation_prefill=True,
            max_accepted_per_step=verify_width,
        )
        self.binding = CachePoolBinding(self.layout, resource="Ascend Qwen cache")
        self.states: dict[int, tuple[int, LiveState]] = {}
        self.graph_mode = graph
        self.graph_max_pages = graph_max_pages
        self.graph_capture_pages = capture_pages
        self.graph_verify_batches = tuple(sorted({1, *(graph_verify_batches or ())}))
        self.graphs: dict[tuple[str, int, int], DecodeGraph] = {}
        self.prefill_graph = prefill_graph
        self.prefill_chunk = prefill_chunk
        self.prefill_segments = {}
        self.graph_replays = 0
        self.graph_fallbacks = 0
        self.drafted = self.accepted = 0
        self.verify_rows = 0

    def bind_resources(self, binding: ResourceBinding) -> None:
        pool_id = self.binding.validate(binding.cache_pool_id, binding.cache_layout)
        if self.binding.pool_id is not None:
            return
        c = self.model.config
        shape = (
            self.layout.parent_capacity + 1,
            PAGE_TOKENS,
            c.num_key_value_heads,
            c.head_dim,
        )
        self.keys = torch.zeros(
            (c.full_layer_count, *shape), dtype=torch.bfloat16, device=self.device
        )
        self.values = torch.zeros_like(self.keys)
        if self.mtp:
            self.draft_keys = torch.zeros(
                shape, dtype=torch.bfloat16, device=self.device
            )
            self.draft_values = torch.zeros_like(self.draft_keys)
        self._report_memory("after_kv")
        if self.graph_capture_pages is not None:
            self._capture_startup_graphs(self.graph_capture_pages)
        if self.prefill_graph:
            self._capture_prefill_segments()
        self.binding.commit(pool_id)

    @torch.inference_mode()
    def _capture_prefill_segments(self) -> None:
        layers = tuple(self.model.text.layers)
        state = self._empty_state()
        hidden = state.hidden.new_zeros((self.prefill_chunk, state.hidden.shape[-1]))
        index = linear = 0
        while index < len(layers):
            if layers[index].kind == "full_attention":
                index += 1
                continue
            end = index + 1
            while end < len(layers) and layers[end].kind != "full_attention":
                end += 1
            count = end - index
            stop = linear + count
            label = f"prefill/layers={index}-{end - 1}/tokens={self.prefill_chunk}"
            log.info("Capturing Ascend %s", label)
            self._report_memory(f"before_capture/{label}")
            inputs = (
                hidden,
                state.recurrent[linear:stop],
                state.convolution[linear:stop],
            )
            graph = DecodeGraph(
                partial(prefill_gdn_segment, layers[index:end]), inputs, mode="aclgraph"
            )
            graph.replay(inputs)
            torch.npu.synchronize(self.device)
            self.prefill_segments[index] = count, graph.replay
            self._report_memory(f"after_capture/{label}")
            index, linear = end, stop

    def _report_memory(self, stage: str) -> None:
        if not self.memory_report or self.device.type != "npu":
            return
        npu = torch.npu
        free, total = npu.mem_get_info(self.device)
        mib = 1 << 20
        log.info(
            "Ascend memory stage=%s device=%s allocated_mib=%.1f reserved_mib=%.1f "
            "peak_allocated_mib=%.1f free_mib=%.1f total_mib=%.1f",
            stage,
            self.device,
            npu.memory_allocated(self.device) / mib,
            npu.memory_reserved(self.device) / mib,
            npu.max_memory_allocated(self.device) / mib,
            free / mib,
            total / mib,
        )

    def release(self, sequence_ids) -> None:
        for sequence in sequence_ids:
            self.states.pop(sequence, None)

    def metrics_snapshot(self) -> RunnerMetrics:
        return RunnerMetrics(
            spec_draft_tokens=self.drafted,
            spec_accepted_tokens=self.accepted,
            spec_verify_rows=self.verify_rows,
        )

    def _empty_state(self) -> LiveState:
        c = self.model.config
        return LiveState(
            torch.zeros(
                (
                    c.linear_layer_count,
                    c.linear_num_value_heads,
                    c.linear_key_head_dim,
                    c.linear_value_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.zeros(
                (c.linear_layer_count, c.linear_conv_dim, c.linear_conv_kernel_dim - 1),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.zeros((1, c.hidden_size), dtype=torch.bfloat16, device=self.device),
        )

    def _long(self, value) -> Tensor:
        # Native BatchPlan NumPy views are read-only.
        return torch.tensor(value, dtype=torch.long, device=self.device)

    @staticmethod
    def _page_metadata(
        group: CacheGroupView,
        row: int,
        start: int,
        length: int,
        *,
        bucket: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pages = group.block_slabs(row)[: (length + PAGE_TOKENS - 1) // PAGE_TOKENS]
        positions = np.arange(start, length, dtype=np.int64)
        slots = (
            pages[positions // PAGE_TOKENS].astype(np.int64) * PAGE_TOKENS
            + positions % PAGE_TOKENS
        )
        if bucket > len(pages):
            pages = np.pad(pages, (0, bucket - len(pages)))
        return positions, pages.astype(np.int64), slots

    def _page_table(
        self,
        group: CacheGroupView,
        row: int,
        start: int,
        length: int,
        *,
        bucket: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        positions, pages, slots = self._page_metadata(
            group, row, start, length, bucket=bucket
        )
        return self._long(positions), self._long(pages), self._long(slots)

    def _capture_graph(
        self,
        kind: str,
        bucket: int,
        inputs: tuple[Tensor, ...],
        *,
        startup: bool = False,
    ) -> DecodeGraph:
        batch_size = inputs[0].shape[0] if inputs[0].ndim == 2 else 1
        label = f"{kind}/pages={bucket}/batch={batch_size}"
        self._report_memory(f"before_capture/{label}")

        def forward(*args):
            if kind == "draft_chain":
                return (self._run_draft_chain(*args),)
            if kind == "draft":
                return (self.model.draft(*args, self.draft_keys, self.draft_values),)
            return self.model.target(
                *args,
                self.keys,
                self.values,
                verify=kind == "verify",
                decode=True,
            )

        scratch = [torch.zeros_like(value) for value in inputs]
        # Target inputs: ids, positions, pages, slots, recurrent, convolution.
        # Draft inputs: ids, previous hidden, positions, pages, slots.
        position_index, slot_index = (2, 4) if kind.startswith("draft") else (1, 3)
        shape = inputs[position_index].shape
        positions = torch.arange(
            inputs[position_index].numel(), device=self.device
        ).reshape(shape)
        scratch[position_index].copy_(positions)
        scratch[slot_index].copy_(positions)
        scratch = tuple(scratch)

        stage = "startup" if startup else "request"
        log.info(
            "Capturing Ascend decode graph, mode=%s, kind=%s, page bucket=%d, batch=%d, stage=%s",
            self.graph_mode,
            kind,
            bucket,
            batch_size,
            stage,
        )
        graph = DecodeGraph(
            forward, inputs, mode=self.graph_mode, warmup_inputs=scratch
        )
        if startup:
            graph.replay(scratch)
            torch.npu.synchronize(self.device)
        del scratch
        self._report_memory(f"after_capture/{label}")
        log.info(
            "Ascend decode graph captured, mode=%s, kind=%s, page bucket=%d, batch=%d, stage=%s",
            self.graph_mode,
            kind,
            bucket,
            batch_size,
            stage,
        )
        return graph

    @torch.inference_mode()
    def _capture_startup_graphs(self, buckets: tuple[int, ...]) -> None:
        state = self._empty_state()
        kinds = ("target",)
        if self.mtp:
            kinds += (
                "verify",
                "draft_chain" if self.num_speculative_tokens > 1 else "draft",
            )
        verify_width = 1 + self.num_speculative_tokens
        for bucket in buckets:
            for kind in kinds:
                zero = self._long([0] * (verify_width if kind == "verify" else 1))
                pages = zero.new_zeros((bucket,))
                inputs = (
                    (zero, state.hidden, zero, pages, zero)
                    if kind == "draft"
                    else (zero, zero, pages, zero, state.recurrent, state.convolution)
                )
                if kind == "draft_chain":
                    metadata = zero.new_zeros(self.num_speculative_tokens)
                    inputs = (zero, state.hidden, metadata, pages, metadata)
                self.graphs[kind, bucket, 1] = self._capture_graph(
                    kind, bucket, inputs, startup=True
                )
            if self.mtp:
                for size in self.graph_verify_batches:
                    if size == 1:
                        continue
                    ids = self._long(np.zeros((size, verify_width), dtype=np.int64))
                    inputs = (
                        ids,
                        ids,
                        ids.new_zeros((size, bucket)),
                        ids,
                        state.recurrent[:, None].expand(-1, size, -1, -1, -1),
                        state.convolution[:, None].expand(-1, size, -1, -1),
                    )
                    self.graphs["verify", bucket, size] = self._capture_graph(
                        "verify", bucket, inputs, startup=True
                    )

    def _graph_route(self, kind: str, bucket: int, eligible: bool):
        graph = self.graphs.get((kind, bucket, 1)) if eligible else None
        lazy_capture = (
            eligible
            and graph is None
            and self.graph_capture_pages is None
            and bucket <= self.graph_max_pages
        )
        return graph, lazy_capture

    def _replay_graph(
        self, graph: DecodeGraph, inputs, kind: str, bucket: int, batch_size: int = 1
    ):
        result = graph.replay(inputs)
        self.graph_replays += 1
        log.debug(
            "Ascend decode graph replay, mode=%s, kind=%s, page bucket=%d, batch=%d",
            self.graph_mode,
            kind,
            bucket,
            batch_size,
        )
        return result

    def _graph_fallback(self, kind: str, bucket: int, batch_size: int = 1) -> None:
        self.graph_fallbacks += 1
        log.debug(
            "Ascend decode graph fallback, kind=%s, page bucket=%d, batch=%d",
            kind,
            bucket,
            batch_size,
        )

    def _target(self, ids, group, row, context, state, *, decode=False, verify=False):
        length = context + len(ids)
        pages = (length + PAGE_TOKENS - 1) // PAGE_TOKENS
        bucket = 1 << (pages - 1).bit_length()
        eligible = (
            self.graph_mode != "eager"
            and decode
            and len(ids) == (1 + self.num_speculative_tokens if verify else 1)
            and (not verify or self.graph_mode == "aclgraph")
        )
        kind = "verify" if verify else "target"
        graph, lazy_capture = self._graph_route(kind, bucket, eligible)
        positions, page_table, slots = self._page_table(
            group,
            row,
            context,
            length,
            bucket=bucket if graph is not None or lazy_capture else 0,
        )
        inputs = (ids, positions, page_table, slots, state.recurrent, state.convolution)
        if lazy_capture:
            graph = self.graphs[kind, bucket, 1] = self._capture_graph(
                kind, bucket, inputs
            )
        if graph is not None:
            return self._replay_graph(graph, inputs, kind, bucket)
        if eligible:
            self._graph_fallback(kind, bucket)
        return self.model.target(
            *inputs,
            self.keys,
            self.values,
            self.native,
            length,
            verify=verify,
            native_prefill=self.native_gdn_prefill and not decode,
            decode=decode,
            prefill_segments=self.prefill_segments
            if not decode and len(ids) == self.prefill_chunk
            else None,
        )

    def _draft(self, ids, hidden, group, row, start, *, decode=False):
        length = start + len(ids)
        pages = (length + PAGE_TOKENS - 1) // PAGE_TOKENS
        bucket = 1 << (pages - 1).bit_length()
        eligible = self.graph_mode == "aclgraph" and decode and len(ids) == 1
        graph, lazy_capture = self._graph_route("draft", bucket, eligible)
        positions, pages, slots = self._page_table(
            group,
            row,
            start,
            length,
            bucket=bucket if graph is not None or lazy_capture else 0,
        )
        inputs = (ids, hidden, positions, pages, slots)
        if lazy_capture:
            graph = self.graphs["draft", bucket, 1] = self._capture_graph(
                "draft", bucket, inputs
            )
        if graph is not None:
            return self._replay_graph(graph, inputs, "draft", bucket)[0]
        if eligible:
            self._graph_fallback("draft", bucket)
        return self.model.draft(*inputs, self.draft_keys, self.draft_values)

    def _run_draft_chain(self, anchor, hidden, positions, pages, slots):
        tokens = [anchor]
        for step in range(self.num_speculative_tokens):
            hidden = self.model.draft(
                tokens[-1],
                hidden,
                positions[step : step + 1],
                pages,
                slots[step : step + 1],
                self.draft_keys,
                self.draft_values,
            )
            tokens.append(self.model.logits(hidden).argmax(dim=-1))
        return torch.cat(tokens)

    def _draft_candidates(self, anchor, hidden, group, row, context):
        if self.num_speculative_tokens == 1:
            hidden = self._draft(anchor, hidden, group, row, context, decode=True)
            return torch.cat((anchor, self.model.logits(hidden).argmax(dim=-1)))

        length = context + self.num_speculative_tokens
        pages = (length + PAGE_TOKENS - 1) // PAGE_TOKENS
        bucket = 1 << (pages - 1).bit_length()
        eligible = self.graph_mode == "aclgraph"
        graph, lazy_capture = self._graph_route("draft_chain", bucket, eligible)
        positions, table, slots = self._page_table(
            group, row, context, length, bucket=bucket if graph or lazy_capture else 0
        )
        inputs = (anchor, hidden, positions, table, slots)
        if lazy_capture:
            graph = self.graphs["draft_chain", bucket, 1] = self._capture_graph(
                "draft_chain", bucket, inputs
            )
        if graph is not None:
            # The next request can replay this graph before batch verify begins.
            return self._replay_graph(graph, inputs, "draft_chain", bucket)[0].clone()
        if eligible:
            self._graph_fallback("draft_chain", bucket)
        return self._run_draft_chain(*inputs)

    def _prime_draft(self, ids, hidden, group, row, start) -> None:
        positions, _, slots = self._page_metadata(group, row, start, start + len(ids))
        self.model.prime_draft(
            ids,
            hidden,
            self._long(positions),
            self._long(slots),
            self.draft_keys,
            self.draft_values,
        )

    def _verify_targets(self, group, rows):
        batch_size = len(rows)
        if batch_size == 1:
            row, context, state, ids = rows[0]
            hidden, *states = self._target(
                ids, group, row, context, state, decode=True, verify=True
            )
            return hidden[None], *(tensor[:, None] for tensor in states)
        if self.native is not None:
            # The native attention path currently takes a single host length.
            results = [
                self._target(ids, group, row, context, state, decode=True, verify=True)
                for row, context, state, ids in rows
            ]
            hidden, *states = zip(*results, strict=True)
            return torch.stack(hidden), *(torch.stack(items, dim=1) for items in states)

        pages = max(
            (context + len(ids) + PAGE_TOKENS - 1) // PAGE_TOKENS
            for _, context, _, ids in rows
        )
        bucket = 1 << (pages - 1).bit_length()
        graph = self.graphs.get(("verify", bucket, batch_size))
        metadata = [
            self._page_metadata(
                group,
                row,
                context,
                context + len(ids),
                bucket=bucket if graph else pages,
            )
            for row, context, _, ids in rows
        ]
        positions, page_table, slots = (
            self._long(np.stack(items)) for items in zip(*metadata, strict=True)
        )
        inputs = (
            torch.stack([ids for _, _, _, ids in rows]),
            positions,
            page_table,
            slots,
            torch.stack([state.recurrent for _, _, state, _ in rows], dim=1),
            torch.stack([state.convolution for _, _, state, _ in rows], dim=1),
        )
        if graph is not None:
            return self._replay_graph(graph, inputs, "verify", bucket, batch_size)
        if self.graph_mode == "aclgraph":
            self._graph_fallback("verify", bucket, batch_size)
        return self.model.target(*inputs, self.keys, self.values, verify=True)

    def _verify_rows(self, plan, cache, policies):
        rows = []
        draft_count = self.num_speculative_tokens
        for row, sequence in enumerate(plan.sequence_ids):
            context = int(plan.context_lengths[row])
            if not context or policies[row].temperature != 0:
                continue
            start, end = plan.token_range(row)
            ids = self._long(cache.batch.token_ids[start:end].astype(np.int64))
            state = self.states[int(sequence)][1]
            candidates = self._draft_candidates(
                ids, state.hidden, cache.groups[MTP], row, context
            )
            rows.append((row, context, state, candidates))
        if not rows:
            return {}

        hidden, recurrent, convolution = self._verify_targets(
            cache.groups[TARGET], rows
        )
        scores = self.model.logits(hidden.flatten(0, 1)).unflatten(0, hidden.shape[:2])
        drafts = torch.stack([ids[1:] for _, _, _, ids in rows])
        decisions = torch.cat((scores.argmax(dim=-1), drafts), dim=1).tolist()
        results = {}
        for index, ((row, context, _, _), tokens) in enumerate(
            zip(rows, decisions, strict=True)
        ):
            targets, proposals = tokens[: draft_count + 1], tokens[draft_count + 1 :]
            accepted = 0
            for target, proposal in zip(targets[:-1], proposals, strict=True):
                if target != proposal:
                    break
                accepted += 1
            self.drafted += draft_count
            self.accepted += accepted
            self.verify_rows += 1
            if accepted:
                # The draft chain used its own hidden states. Rebuild the
                # committed draft KV from the verified target hidden states.
                self._prime_draft(
                    drafts[index, :accepted],
                    hidden[index, :accepted],
                    cache.groups[MTP],
                    row,
                    context + 1,
                )
            results[row] = (
                scores[index],
                targets[: accepted + 1],
                LiveState(
                    recurrent[:, index, accepted],
                    convolution[:, index, accepted],
                    hidden[index, accepted : accepted + 1],
                ),
            )
        return results

    def _prepare(self, plan: BatchPlan) -> PagedCachePlan:
        if self.binding.pool_id is None:
            raise RuntimeError("Ascend Runner cache is not bound")
        if any(media is not None for media in plan.media):
            raise ValueError("Ascend Qwen supports text-only requests")
        decode = plan.phase == BatchPhase.DECODE
        cache = PagedCachePlan.from_plan(
            plan,
            self.layout,
            decode_lookahead=self.num_speculative_tokens if decode else 0,
        )
        for row, sequence in enumerate(plan.sequence_ids):
            if decode and cache.batch.query_lengths[row] != 1:
                raise ValueError("Ascend Qwen decode requires one anchor per row")
            if plan.context_lengths[row]:
                entry = self.states.get(int(sequence))
                if entry is None or entry[0] != int(
                    cache.groups[STATE].block_slabs(row)[0]
                ):
                    raise ValueError(
                        "Ascend Qwen is missing the live recurrent state for this lease"
                    )
        return cache

    def validate_distributed(self, plan, _context):
        return self._prepare(plan) if plan is not None else None

    def forward_distributed(self, plan, _context, *, prepared=None):
        if plan is not None:
            return self._forward(
                plan, prepared if prepared is not None else self._prepare(plan)
            )
        return None

    def forward(self, plan: BatchPlan):
        return self._forward(plan, self._prepare(plan))

    @torch.inference_mode()
    def _forward(self, plan: BatchPlan, cache: PagedCachePlan):
        logits, output_rows, prompt = [], [], []
        states = []
        decode = plan.phase == BatchPhase.DECODE
        policies = plan.sampling_params
        verified = (
            self._verify_rows(plan, cache, policies) if self.mtp and decode else {}
        )
        for row, sequence_id in enumerate(plan.sequence_ids):
            sequence = int(sequence_id)
            if row in verified:
                scores, tokens, state = verified[row]
                logits.append(scores)
                output_rows.append(tokens)
                states.append(
                    (
                        sequence,
                        int(cache.groups[STATE].block_slabs(row)[0]),
                        LiveState(
                            state.recurrent.clone(),
                            state.convolution.clone(),
                            state.hidden.clone(),
                        ),
                    )
                )
                continue
            start, end = plan.token_range(row)
            ids = self._long(cache.batch.token_ids[start:end].astype(np.int64))
            context = int(plan.context_lengths[row])
            state = self.states[sequence][1] if context else self._empty_state()
            hidden, recurrent, convolution = self._target(
                ids,
                cache.groups[TARGET],
                row,
                context,
                state,
                decode=decode,
            )
            last_hidden = hidden[-1:]
            logits.append(self.model.logits(last_hidden))
            output_rows.append([])
            if self.mtp:
                group = cache.groups[MTP]
                if not context:
                    first_page = int(group.block_slabs(row)[0])
                    self.draft_keys[first_page, 0].zero_()
                    self.draft_values[first_page, 0].zero_()
                skip = int(context == 0)
                if len(ids) > skip:
                    previous = torch.cat((state.hidden, hidden[:-1]))[skip:]
                    self._prime_draft(ids[skip:], previous, group, row, context + skip)
            if plan.needs_prompt_logprobs:
                # Shared scoring kernels use the CPU compiler here.
                def project_for_scores(selected: Tensor) -> Tensor:
                    return self.model.logits(selected.to(self.device)).float().cpu()

                prompt.append(
                    prompt_scores(
                        plan, hidden.cpu(), project_for_scores, token_start=start
                    )
                )
            # Graph outputs and slices may alias buffers overwritten on the next replay.
            states.append(
                (
                    sequence,
                    int(cache.groups[STATE].block_slabs(row)[0]),
                    LiveState(
                        recurrent.clone(), convolution.clone(), last_hidden.clone()
                    ),
                )
            )
        if any(
            not tokens and sample
            for tokens, sample in zip(output_rows, plan.sampling_mask, strict=True)
        ):
            final = torch.cat([scores[:1] for scores in logits])
            if any(policy.temperature != 0 for policy in policies):
                final = final.float().cpu()
            sampled = self.sampler(final, plan).tolist()
            for row, token in enumerate(sampled):
                if not output_rows[row]:
                    output_rows[row] = [token]
        for row, sample in enumerate(plan.sampling_mask):
            if not sample:
                # Engine requires one token per prefill row and discards it
                # when this chunk does not sample.
                output_rows[row] = [0]
        packed_logits = None
        if plan.needs_logprobs:
            width = max(scores.shape[0] for scores in logits)
            packed_logits = (
                torch.stack(
                    [
                        scores if scores.shape[0] == width else scores.expand(width, -1)
                        for scores in logits
                    ]
                )
                .float()
                .cpu()
            )
        result = format_output(
            packed_logits,
            output_rows,
            plan,
            prompt_logprobs=PromptScores.concat(prompt),
        )
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)
        self.states.update(
            (sequence, (slot, state)) for sequence, slot, state in states
        )
        return result
