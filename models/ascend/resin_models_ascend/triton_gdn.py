"""Fused GDN recurrence for Ascend decode and speculative verification."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _recurrent_gdn(
    Q,
    K,
    V,
    A,
    B,
    ALog,
    DtBias,
    State,
    Out,
    NextState,
    q_stride: tl.constexpr,
    k_stride: tl.constexpr,
    v_stride: tl.constexpr,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    state_stride: tl.constexpr,
    steps: tl.constexpr,
    key_heads: tl.constexpr,
    value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    key_block: tl.constexpr,
    value_block: tl.constexpr,
    tiles: tl.constexpr,
    scale: tl.constexpr,
    verify: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // (tiles * value_heads)
    head = program // tiles % value_heads
    q_head = head // (value_heads // key_heads)
    keys = tl.arange(0, key_block)
    values = (program % tiles) * value_block + tl.arange(0, value_block)
    mask = (keys[:, None] < key_dim) & (values[None, :] < value_dim)
    matrix = keys[:, None] * value_dim + values[None, :]
    state = tl.load(
        State
        + row * state_stride[0]
        + head * state_stride[1]
        + keys[:, None] * state_stride[2]
        + values[None, :] * state_stride[3],
        mask,
        other=0,
    )
    dt = tl.load(DtBias + head)
    negative_rate = -tl.exp(tl.load(ALog + head))
    for step in range(steps):
        q = tl.load(
            Q
            + row * q_stride[0]
            + step * q_stride[1]
            + q_head * q_stride[2]
            + keys * q_stride[3],
            keys < key_dim,
            other=0,
        ).to(tl.float32)
        k = tl.load(
            K
            + row * k_stride[0]
            + step * k_stride[1]
            + q_head * k_stride[2]
            + keys * k_stride[3],
            keys < key_dim,
            other=0,
        ).to(tl.float32)
        v = tl.load(
            V
            + row * v_stride[0]
            + step * v_stride[1]
            + head * v_stride[2]
            + values * v_stride[3],
            values < value_dim,
            other=0,
        ).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        a = tl.load(A + row * a_stride[0] + step * a_stride[1] + head * a_stride[2])
        b = tl.load(B + row * b_stride[0] + step * b_stride[1] + head * b_stride[2])
        x = a.to(tl.float32) + dt
        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        # The reference sigmoid rounds to the projection dtype before recurrence.
        beta = (1.0 / (1.0 + tl.exp(-b.to(tl.float32)))).to(b.dtype).to(tl.float32)
        state = state * tl.exp(negative_rate * softplus)
        delta = (v - tl.sum(state * k[:, None], 0)) * beta
        state = state + k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], 0)
        token = row * steps + step
        tl.store(
            Out + (token * value_heads + head) * value_dim + values,
            out,
            values < value_dim,
        )
        if verify:
            base = (token * value_heads + head) * key_dim * value_dim
            tl.store(NextState + base + matrix, state, mask)
    if not verify:
        base = (row * value_heads + head) * key_dim * value_dim
        tl.store(NextState + base + matrix, state, mask)


def fused_recurrent_gdn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    *,
    verify: bool,
) -> tuple[Tensor, Tensor]:
    single = q.ndim == 3
    if single:
        q, k, v, a, b, state = (x.unsqueeze(0) for x in (q, k, v, a, b, state))
    batch, steps, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    state_shape = (batch, value_heads, key_dim, value_dim)
    if verify:
        state_shape = (batch, steps, value_heads, key_dim, value_dim)
    next_state = torch.empty(state_shape, dtype=torch.float32, device=v.device)
    value_block = 16
    tiles = triton.cdiv(value_dim, value_block)
    _recurrent_gdn[(batch * value_heads * tiles,)](
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        out,
        next_state,
        q.stride(),
        k.stride(),
        v.stride(),
        a.stride(),
        b.stride(),
        state.stride(),
        steps,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        triton.next_power_of_2(key_dim),
        value_block,
        tiles,
        key_dim**-0.5,
        verify,
    )
    return (out[0], next_state[0]) if single else (out, next_state)
