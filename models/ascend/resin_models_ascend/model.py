"""Dense Qwen3.5-family text model with explicit tensor-parallel projections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from resin.backend.ascend.ops import (
    NativePagedAttention,
    native_chunk_gdn,
    tensor_attention,
    write_kv,
)
from resin.models.qwen3_5.config import Qwen35Config


@dataclass(frozen=True)
class TensorParallel:
    size: int = 1
    rank: int = 0
    group: dist.ProcessGroup | None = None

    def local_config(self, config: Qwen35Config) -> Qwen35Config:
        fields = (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "intermediate_size",
        )
        local = {}
        for name in fields:
            count = getattr(config, name)
            if count % self.size:
                raise ValueError(
                    f"Qwen {name}={count} is not divisible by TP={self.size}"
                )
            local[name] = count // self.size
        return replace(config, **local)

    def sum(self, partial: Tensor) -> Tensor:
        if self.size > 1:
            dist.all_reduce(partial, group=self.group)
        return partial


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, *, centered: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size))
        self.eps = eps
        self.centered = centered
        self.register_buffer("fused_scale", None, persistent=False)

    def prepare_fused(self) -> None:
        weight = self.weight.detach().float()
        self.fused_scale = weight + 1 if self.centered else weight

    def forward(self, value: Tensor, gate: Tensor | None = None) -> Tensor:
        if self.fused_scale is not None:
            import torch_npu

            # Keep the gated normalization in FP32 until after SiLU scaling.
            x = value if gate is None else value.float()
            result, _ = torch_npu.npu_rms_norm(x, self.fused_scale, self.eps)
            if gate is not None:
                result = result * F.silu(gate.float())
            return result.to(value.dtype)
        normed = (
            value.float()
            * (value.float().square().mean(-1, keepdim=True) + self.eps).rsqrt()
        )
        scale = self.weight.float() + 1 if self.centered else self.weight.float()
        result = normed * scale
        if gate is not None:
            result = result * F.silu(gate.float())
        return result.to(value.dtype)


class Attention(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel) -> None:
        super().__init__()
        h, kv, d = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.heads, self.kv_heads, self.head_dim = h, kv, d
        self.q_proj = nn.Linear(config.hidden_size, 2 * h * d, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv * d, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv * d, bias=False)
        self.o_proj = nn.Linear(h * d, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(d, config.rms_norm_eps)
        self.k_norm = RMSNorm(d, config.rms_norm_eps)
        self.tp = tp

    def project_kv(self, hidden: Tensor, rotary: tuple[Tensor, Tensor]):
        k = self.k_proj(hidden).view(-1, self.kv_heads, self.head_dim)
        v = self.v_proj(hidden).view_as(k)
        return apply_rotary(self.k_norm(k), *rotary), v

    def forward(
        self,
        hidden: Tensor,
        rotary: tuple[Tensor, Tensor],
        cache: tuple[Tensor, Tensor],
        slots: Tensor,
        pages: Tensor,
        positions: Tensor,
        native: NativePagedAttention | None,
        length: int,
    ) -> Tensor:
        q, gate = (
            self.q_proj(hidden).view(-1, self.heads, 2 * self.head_dim).chunk(2, dim=-1)
        )
        k, v = self.project_kv(hidden, rotary)
        q = apply_rotary(self.q_norm(q), *rotary)
        if native is None:
            if positions.ndim == 2:
                q = q.unflatten(0, positions.shape)
                k = k.unflatten(0, positions.shape)
                v = v.unflatten(0, positions.shape)
            mixed = tensor_attention(q, k, v, *cache, slots, pages, positions)
            mixed = mixed.reshape_as(gate)
        else:
            mixed = native(q, k, v, *cache, slots, pages, length)
        return self.tp.sum(self.o_proj((mixed * gate.sigmoid()).flatten(1)))


def apply_rotary(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotary, remainder = value[..., : cos.shape[-1]], value[..., cos.shape[-1] :]
    left, right = rotary.chunk(2, dim=-1)
    rotated = rotary * cos + torch.cat((-right, left), dim=-1) * sin
    return torch.cat((rotated, remainder), dim=-1)


def delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    decay: Tensor,
    beta: Tensor,
    state: Tensor,
    *,
    verify: bool = False,
):
    repeats = v.shape[-2] // q.shape[-2]
    q = q.float()
    k = k.float()
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * q.shape[-1] ** -0.5
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q = q.repeat_interleave(repeats, dim=-2)
    k = k.repeat_interleave(repeats, dim=-2)
    outputs = []
    prefixes = []
    for i in range(q.shape[-3]):
        state = state * decay[..., i, :].exp()[..., None, None]
        key = k[..., i, :, :].unsqueeze(-1)
        delta = (v[..., i, :, :].float() - (state * key).sum(-2)) * beta[
            ..., i, :, None
        ]
        state = state + key * delta.unsqueeze(-2)
        if verify:
            prefixes.append(state)
        outputs.append((state * q[..., i, :, :].unsqueeze(-1)).sum(-2))
    if verify:
        # Insert the query axis before [heads, key_dim, value_dim].
        state = torch.stack(prefixes, dim=-4)
    return torch.stack(outputs, dim=-3).to(v.dtype), state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel) -> None:
        super().__init__()
        self.config, self.tp = config, tp
        hidden, width = config.hidden_size, config.linear_conv_dim
        heads = config.linear_num_value_heads
        self.in_proj_qkv = nn.Linear(hidden, width, bias=False)
        self.in_proj_z = nn.Linear(hidden, config.linear_value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden, heads, bias=False)
        self.in_proj_b = nn.Linear(hidden, heads, bias=False)
        self.conv1d = nn.Conv1d(
            width, width, config.linear_conv_kernel_dim, groups=width, bias=False
        )
        self.dt_bias = nn.Parameter(torch.empty(heads))
        self.A_log = nn.Parameter(torch.empty(heads))
        self.norm = RMSNorm(
            config.linear_value_head_dim, config.rms_norm_eps, centered=False
        )
        self.out_proj = nn.Linear(config.linear_value_dim, hidden, bias=False)
        self.fused_recurrence = None
        self.fla_gdn = None
        self.fused_gating = None
        self.register_buffer("packed_conv_weight", None, persistent=False)

    def forward(
        self,
        hidden: Tensor,
        state: Tensor,
        convolution: Tensor,
        native_prefill: bool,
        *,
        verify: bool = False,
        decode: bool = False,
    ):
        c = self.config
        width = c.linear_conv_kernel_dim
        qkv = self.in_proj_qkv(hidden)
        window = torch.cat((convolution, qkv.transpose(-2, -1)), dim=-1)
        if self.packed_conv_weight is not None:
            mixed = self.fla_gdn.convolve(
                qkv, convolution, self.packed_conv_weight, decode=decode
            )
        else:
            mixed = (
                window.float().unfold(-1, width, 1) * self.conv1d.weight.float()
            ).sum(-1)
            mixed = F.silu(mixed.to(hidden.dtype)).transpose(-2, -1)
        q, k, v = mixed.split(
            (c.linear_key_dim, c.linear_key_dim, c.linear_value_dim), dim=-1
        )
        q = q.unflatten(-1, (c.linear_num_key_heads, c.linear_key_head_dim))
        k = k.reshape_as(q)
        v = v.unflatten(-1, (c.linear_num_value_heads, c.linear_value_head_dim))
        a, b = self.in_proj_a(hidden), self.in_proj_b(hidden)
        if decode and self.fused_recurrence is not None:
            core, state = self.fused_recurrence(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                verify=verify,
            )
        else:
            if self.fused_gating is not None:
                decay, beta = self.fused_gating(a, b, self.A_log, self.dt_bias)
            else:
                decay = -self.A_log.float().exp() * F.softplus(
                    a.float() + self.dt_bias.float()
                )
                beta = b.sigmoid()
            if native_prefill:
                core, state = native_chunk_gdn(q, k, v, decay, beta, state)
            elif self.fla_gdn is not None:
                if decode:
                    core, state = self.fla_gdn.decode(
                        q, k, v, decay, beta, state, verify=verify
                    )
                else:
                    core, state = self.fla_gdn.prefill(q, k, v, decay, beta, state)
            else:
                core, state = delta_rule(q, k, v, decay, beta, state, verify=verify)
        gate = self.in_proj_z(hidden).reshape_as(core)
        mixed = self.tp.sum(self.out_proj(self.norm(core, gate).flatten(-2)))
        return mixed, state, window


class MLP(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.tp = tp

    def forward(self, hidden: Tensor) -> Tensor:
        return self.tp.sum(
            self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "full_attention":
            self.self_attn = Attention(config, tp)
        else:
            self.linear_attn = GatedDeltaNet(config, tp)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config, tp)

    def finish(self, residual: Tensor, mixed: Tensor) -> Tensor:
        hidden = residual + mixed
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def prefill_gdn_segment(
    layers: Sequence[DecoderLayer],
    hidden: Tensor,
    recurrent: Tensor,
    convolution: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    states, windows = [], []
    for layer, state, history in zip(layers, recurrent, convolution, strict=True):
        mixed, state, window = layer.linear_attn(
            layer.input_layernorm(hidden), state, history, False
        )
        hidden = layer.finish(hidden, mixed)
        states.append(state)
        windows.append(window[..., hidden.shape[0] :])
    return hidden, torch.stack(states), torch.stack(windows)


class TextModel(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, tp, kind) for kind in config.layer_types
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class MTP(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(hidden, eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden, eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layers = nn.ModuleList([DecoderLayer(config, tp, "full_attention")])
        self.norm = RMSNorm(hidden, eps)


class QwenModel(nn.Module):
    def __init__(self, config: Qwen35Config, tp: TensorParallel, *, mtp: bool) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.language_model = TextModel(config, tp)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        self.mtp = MTP(config, tp) if mtp else None
        self.register_buffer("inv_freq", None, persistent=False)

    @property
    def text(self) -> TextModel:
        return self.model.language_model

    def rotary(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        phase = positions.float()[:, None, None] * self.inv_freq
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().to(dtype), phase.sin().to(dtype)

    def logits(self, hidden: Tensor) -> Tensor:
        if self.lm_head is None:
            return F.linear(hidden, self.text.embed_tokens.weight)
        return self.lm_head(hidden)

    def target(
        self,
        ids: Tensor,
        positions: Tensor,
        pages: Tensor,
        slots: Tensor,
        recurrent: Tensor,
        convolution: Tensor,
        keys: Tensor,
        values: Tensor,
        native: NativePagedAttention | None = None,
        length: int = 0,
        verify: bool = False,
        native_prefill: bool = False,
        decode: bool = False,
        prefill_segments: dict[int, tuple[int, Callable]] | None = None,
    ):
        hidden = self.text.embed_tokens(ids.flatten())
        rotary = self.rotary(positions.flatten(), hidden.dtype)
        states, windows = [], []
        full = linear = 0
        skip_until = 0
        for index, layer in enumerate(self.text.layers):
            if index < skip_until:
                continue
            if prefill_segments is not None and index in prefill_segments:
                count, replay = prefill_segments[index]
                stop = linear + count
                hidden, segment_states, segment_windows = replay(
                    (hidden, recurrent[linear:stop], convolution[linear:stop])
                )
                states.extend(segment_states.unbind())
                windows.extend(segment_windows.unbind())
                linear = stop
                skip_until = index + count
                continue
            normalized = layer.input_layernorm(hidden)
            if layer.kind == "full_attention":
                mixed = layer.self_attn(
                    normalized,
                    rotary,
                    (keys[full], values[full]),
                    slots,
                    pages,
                    positions,
                    native,
                    length,
                )
                full += 1
            else:
                mixed, state, window = layer.linear_attn(
                    normalized.reshape(*ids.shape, -1),
                    recurrent[linear],
                    convolution[linear],
                    native_prefill,
                    verify=verify,
                    decode=decode or verify,
                )
                width = convolution.shape[-1]
                states.append(state)
                if verify:
                    # Drop the incoming window and keep one window per prefix.
                    windows.append(
                        window.unfold(-1, width, 1)[..., 1:, :].transpose(-3, -2)
                    )
                else:
                    windows.append(window[..., ids.shape[-1] :])
                mixed = mixed.reshape_as(hidden)
                linear += 1
            hidden = layer.finish(hidden, mixed)
        return (
            self.text.norm(hidden).reshape(*ids.shape, -1),
            torch.stack(states),
            torch.stack(windows),
        )

    def _draft_input(self, ids: Tensor, previous: Tensor) -> tuple[MTP, Tensor]:
        mtp = self.mtp
        if mtp is None:
            raise RuntimeError("MTP weights were not loaded")
        embedding = mtp.pre_fc_norm_embedding(self.text.embed_tokens(ids))
        hidden = mtp.fc(
            torch.cat((embedding, mtp.pre_fc_norm_hidden(previous)), dim=-1)
        )
        return mtp, hidden

    def prime_draft(
        self,
        ids: Tensor,
        previous: Tensor,
        positions: Tensor,
        slots: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> None:
        mtp, hidden = self._draft_input(ids, previous)
        layer = mtp.layers[0]
        rotary = self.rotary(positions - 1, hidden.dtype)
        k, v = layer.self_attn.project_kv(layer.input_layernorm(hidden), rotary)
        write_kv(keys, values, k, v, slots)

    def draft(
        self,
        ids: Tensor,
        previous: Tensor,
        positions: Tensor,
        pages: Tensor,
        slots: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        mtp, hidden = self._draft_input(ids, previous)
        layer = mtp.layers[0]
        mixed = layer.self_attn(
            layer.input_layernorm(hidden),
            self.rotary(positions - 1, hidden.dtype),
            (keys, values),
            slots,
            pages,
            positions,
            None,
            0,
        )
        return mtp.norm(layer.finish(hidden, mixed))
