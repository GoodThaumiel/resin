"""Adapt FLA NPU's state-slot ABI to Resin's request-local GDN state."""

import torch
from torch import Tensor


def _unit_norm(tensor: Tensor) -> Tensor:
    fp32 = tensor.float()
    return (fp32 * (fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(
        tensor.dtype
    )


class FlaGDN:
    def __init__(self) -> None:
        try:
            from fla_npu.ops.ascendc import (
                causal_conv1d_fn,
                causal_conv1d_update,
                chunk_gated_delta_rule_fwd,
                recurrent_gated_delta_rule,
            )
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "GDN requires the FLA NPU A2 wheel on 910B. "
                "Install a wheel compatible with your CANN stack or pass --gdn off."
            ) from error
        self.chunk = chunk_gated_delta_rule_fwd
        self.recurrent = recurrent_gated_delta_rule
        self.conv_prefill = causal_conv1d_fn
        self.conv_update = causal_conv1d_update
        self.metadata: dict[tuple[torch.device, int, int], tuple[Tensor, Tensor]] = {}
        self.initialized_conv_rows: dict[tuple[torch.device, int], Tensor] = {}

    def convolve(self, qkv: Tensor, history: Tensor, weight: Tensor, *, decode: bool):
        single = qkv.ndim == 2
        x = (qkv[None] if single else qkv).contiguous()
        state = (
            (history[None] if single else history)
            .transpose(-2, -1)
            .clone(memory_format=torch.contiguous_format)
        )
        # FLA updates its private state. The caller builds the live prefix
        # windows separately from the original, unmodified projections.
        if decode:
            output = self.conv_update(
                x, state, weight, activation="silu", out=torch.empty_like(x)
            )
        else:
            key = x.device, x.shape[0]
            if key not in self.initialized_conv_rows:
                # New requests already have a zero-filled history.
                self.initialized_conv_rows[key] = torch.ones(
                    x.shape[0], dtype=torch.bool, device=x.device
                )
            output = self.conv_prefill(
                x,
                weight,
                None,
                conv_states=state,
                has_initial_state=self.initialized_conv_rows[key],
                activation="silu",
            )
        return output[0] if single else output

    def decode(self, q, k, v, decay, beta, state, *, verify: bool):
        single = q.ndim == 3
        if single:
            q, k, v, decay, beta, state = (
                x[None] for x in (q, k, v, decay, beta, state)
            )
        batch, width = q.shape[:2]
        device = q.device
        key = device, batch, width
        if key not in self.metadata:
            # FLA's ABI takes [start_offset, len(row0), len(row1), ...].
            lengths = torch.tensor(
                [0] + [width] * batch, device=device, dtype=torch.int32
            )
            indices = torch.arange(batch * width, device=device, dtype=torch.int32)
            self.metadata[key] = lengths, indices
        lengths, indices = self.metadata[key]
        # Each row starts in its first slot. FLA overwrites one slot per query
        # token with the resulting prefix state, leaving the live state untouched.
        value_first = state.transpose(-2, -1)
        heads, value_dim, key_dim = value_first.shape[-3:]
        prefixes = torch.empty(
            (batch, width, heads, value_dim, key_dim), device=device, dtype=state.dtype
        )
        prefixes[:, 0].copy_(value_first)
        out = self.recurrent(
            _unit_norm(q).flatten(0, 1).contiguous(),
            _unit_norm(k).flatten(0, 1).contiguous(),
            v.flatten(0, 1).contiguous(),
            prefixes.flatten(0, 1),
            g=decay.flatten(0, 1).contiguous(),
            beta=beta.flatten(0, 1).contiguous(),
            scale=key_dim**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
        ).unflatten(0, (batch, width))
        next_state = (prefixes if verify else prefixes[:, -1]).transpose(-2, -1)
        return (out[0], next_state[0]) if single else (out, next_state)

    def prefill(self, q, k, v, decay, beta, state):
        # Single-row chunked prefill. Use the BNSD kernel layout and FP32 K/V
        # state as in the vLLM Ascend FLA prefill path.
        tokens = q.shape[0]
        chunks = tuple(part for i in range((tokens + 63) // 64) for part in (0, i))
        output, next_state, *_ = self.chunk(
            _unit_norm(q).transpose(0, 1)[None].contiguous(),
            _unit_norm(k).transpose(0, 1)[None].contiguous(),
            v.transpose(0, 1)[None].contiguous(),
            decay[None].contiguous(),
            beta[None].contiguous(),
            initial_state=state[None].contiguous(),
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=(0, tokens),
            chunk_indices=chunks,
            scale=q.shape[-1] ** -0.5,
            layout="BNSD",
        )
        return output[0], next_state[0].float()
