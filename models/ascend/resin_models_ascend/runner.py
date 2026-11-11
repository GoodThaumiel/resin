"""Synchronous Qwen text execution against Engine-owned physical page leases."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
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
from resin_models_ascend.model import TensorParallel

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
        mtp: bool = False,
        quantization: str = "bf16",
        graph: str = "eager",
        attention: str = "torch",
        native_gdn_prefill: bool = False,
    ) -> None:
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
        self.model = load_model(
            checkpoint, device, tp, mtp=mtp, quantization=quantization
        )
        self.device = device
        self.mtp = mtp
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
            max_accepted_per_step=2 if mtp else 1,
        )
        self.binding = CachePoolBinding(self.layout, resource="Ascend Qwen cache")
        self.states: dict[int, tuple[int, LiveState]] = {}
        self.graph_mode = graph
        self.graph_max_pages = graph_max_pages
        self.graph_capture_pages = capture_pages
        self.graphs: dict[tuple[str, int], DecodeGraph] = {}
        self.graph_replays = 0
        self.graph_fallbacks = 0
        self.drafted = self.accepted = 0

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
        if self.graph_capture_pages is not None:
            self._capture_startup_graphs(self.graph_capture_pages)
        self.binding.commit(pool_id)

    def release(self, sequence_ids) -> None:
        for sequence in sequence_ids:
            self.states.pop(sequence, None)

    def metrics_snapshot(self) -> RunnerMetrics:
        return RunnerMetrics(
            spec_draft_tokens=self.drafted,
            spec_accepted_tokens=self.accepted,
            spec_verify_rows=self.drafted,
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

    def _page_table(
        self,
        group: CacheGroupView,
        row: int,
        start: int,
        length: int,
        *,
        bucket: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pages = group.block_slabs(row)[: (length + PAGE_TOKENS - 1) // PAGE_TOKENS]
        positions = np.arange(start, length, dtype=np.int64)
        slots = (
            pages[positions // PAGE_TOKENS].astype(np.int64) * PAGE_TOKENS
            + positions % PAGE_TOKENS
        )
        if bucket > len(pages):
            pages = np.pad(pages, (0, bucket - len(pages)))
        return (
            self._long(positions),
            self._long(pages.astype(np.int64)),
            self._long(slots),
        )

    def _capture_graph(
        self,
        kind: str,
        bucket: int,
        inputs: tuple[Tensor, ...],
        *,
        startup: bool = False,
    ) -> DecodeGraph:
        def forward(*args):
            if kind == "draft":
                return (self.model.draft(*args, self.draft_keys, self.draft_values),)
            return self.model.target(
                *args, self.keys, self.values, verify=kind == "verify"
            )

        scratch = [torch.zeros_like(value) for value in inputs]
        # Target inputs: ids, positions, pages, slots, recurrent, convolution.
        # Draft inputs: ids, previous hidden, positions, pages, slots.
        position_index, slot_index = (2, 4) if kind == "draft" else (1, 3)
        positions = torch.arange(inputs[0].numel(), device=self.device)
        scratch[position_index].copy_(positions)
        scratch[slot_index].copy_(positions)
        scratch = tuple(scratch)

        stage = "startup" if startup else "request"
        log.info(
            "Capturing Ascend decode graph, mode=%s, kind=%s, page bucket=%d, stage=%s",
            self.graph_mode,
            kind,
            bucket,
            stage,
        )
        graph = DecodeGraph(
            forward, inputs, mode=self.graph_mode, warmup_inputs=scratch
        )
        if startup:
            graph.replay(scratch)
            torch.npu.synchronize(self.device)
        log.info(
            "Ascend decode graph captured, mode=%s, kind=%s, page bucket=%d, stage=%s",
            self.graph_mode,
            kind,
            bucket,
            stage,
        )
        return graph

    @torch.inference_mode()
    def _capture_startup_graphs(self, buckets: tuple[int, ...]) -> None:
        state = self._empty_state()
        kinds = ("target", "verify", "draft") if self.mtp else ("target",)
        for bucket in buckets:
            for kind in kinds:
                zero = self._long([0, 0] if kind == "verify" else [0])
                pages = zero.new_zeros((bucket,))
                inputs = (
                    (zero, state.hidden, zero, pages, zero)
                    if kind == "draft"
                    else (zero, zero, pages, zero, state.recurrent, state.convolution)
                )
                self.graphs[kind, bucket] = self._capture_graph(
                    kind, bucket, inputs, startup=True
                )

    def _graph_route(self, kind: str, bucket: int, eligible: bool):
        graph = self.graphs.get((kind, bucket)) if eligible else None
        lazy_capture = (
            eligible
            and graph is None
            and self.graph_capture_pages is None
            and bucket <= self.graph_max_pages
        )
        return graph, lazy_capture

    def _replay_graph(self, graph: DecodeGraph, inputs, kind: str, bucket: int):
        result = graph.replay(inputs)
        self.graph_replays += 1
        log.debug(
            "Ascend decode graph replay, mode=%s, kind=%s, page bucket=%d",
            self.graph_mode,
            kind,
            bucket,
        )
        return result

    def _graph_fallback(self, kind: str, bucket: int) -> None:
        self.graph_fallbacks += 1
        log.debug("Ascend decode graph fallback, kind=%s, page bucket=%d", kind, bucket)

    def _target(self, ids, group, row, context, state, *, decode=False, verify=False):
        length = context + len(ids)
        pages = (length + PAGE_TOKENS - 1) // PAGE_TOKENS
        bucket = 1 << (pages - 1).bit_length()
        eligible = (
            self.graph_mode != "eager"
            and decode
            and len(ids) == (2 if verify else 1)
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
            graph = self.graphs[kind, bucket] = self._capture_graph(
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
            graph = self.graphs["draft", bucket] = self._capture_graph(
                "draft", bucket, inputs
            )
        if graph is not None:
            return self._replay_graph(graph, inputs, "draft", bucket)[0]
        if eligible:
            self._graph_fallback("draft", bucket)
        return self.model.draft(*inputs, self.draft_keys, self.draft_values)

    def _prepare(self, plan: BatchPlan) -> PagedCachePlan:
        if self.binding.pool_id is None:
            raise RuntimeError("Ascend Runner cache is not bound")
        if any(media is not None for media in plan.media):
            raise ValueError("Ascend Qwen supports text-only requests")
        decode = plan.phase == BatchPhase.DECODE
        cache = PagedCachePlan.from_plan(
            plan, self.layout, decode_lookahead=int(self.mtp and decode)
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
        for row, sequence_id in enumerate(plan.sequence_ids):
            sequence = int(sequence_id)
            start, end = plan.token_range(row)
            ids = self._long(cache.batch.token_ids[start:end].astype(np.int64))
            context = int(plan.context_lengths[row])
            state = self.states[sequence][1] if context else self._empty_state()
            spec = (
                self.mtp and decode and context > 0 and policies[row].temperature == 0
            )
            if spec:
                draft_hidden = self._draft(
                    ids, state.hidden, cache.groups[MTP], row, context, decode=True
                )
                draft = self.model.logits(draft_hidden).argmax(dim=-1)
                verify_ids = torch.cat((ids, draft))
                hidden, recurrent, convolution, first_rec, first_conv = self._target(
                    verify_ids,
                    cache.groups[TARGET],
                    row,
                    context,
                    state,
                    decode=True,
                    verify=True,
                )
                scores = self.model.logits(hidden)
                targets = scores.argmax(dim=-1).tolist()
                accepted = targets[0] == int(draft.item())
                self.drafted += 1
                self.accepted += int(accepted)
                if accepted:
                    self._draft(
                        draft,
                        hidden[:1],
                        cache.groups[MTP],
                        row,
                        context + 1,
                        decode=True,
                    )
                else:
                    recurrent, convolution = first_rec, first_conv
                last_hidden = hidden[int(accepted) : int(accepted) + 1]
                output_rows.append(targets[: 1 + int(accepted)])
                logits.append(scores)
            else:
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
                        self._draft(
                            ids[skip:],
                            previous,
                            group,
                            row,
                            context + skip,
                            decode=decode,
                        )
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
