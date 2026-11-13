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
    q_stride_row: tl.constexpr,
    q_stride_step: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_key: tl.constexpr,
    k_stride_row: tl.constexpr,
    k_stride_step: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_key: tl.constexpr,
    v_stride_row: tl.constexpr,
    v_stride_step: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_value: tl.constexpr,
    a_stride_row: tl.constexpr,
    a_stride_step: tl.constexpr,
    a_stride_head: tl.constexpr,
    b_stride_row: tl.constexpr,
    b_stride_step: tl.constexpr,
    b_stride_head: tl.constexpr,
    state_stride_row: tl.constexpr,
    state_stride_head: tl.constexpr,
    state_stride_key: tl.constexpr,
    state_stride_value: tl.constexpr,
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
        + row * state_stride_row
        + head * state_stride_head
        + keys[:, None] * state_stride_key
        + values[None, :] * state_stride_value,
        mask,
        other=0,
    )
    dt = tl.load(DtBias + head)
    negative_rate = -tl.exp(tl.load(ALog + head))
    for step in range(steps):
        q = tl.load(
            Q
            + row * q_stride_row
            + step * q_stride_step
            + q_head * q_stride_head
            + keys * q_stride_key,
            keys < key_dim,
            other=0,
        ).to(tl.float32)
        k = tl.load(
            K
            + row * k_stride_row
            + step * k_stride_step
            + q_head * k_stride_head
            + keys * k_stride_key,
            keys < key_dim,
            other=0,
        ).to(tl.float32)
        v = tl.load(
            V
            + row * v_stride_row
            + step * v_stride_step
            + head * v_stride_head
            + values * v_stride_value,
            values < value_dim,
            other=0,
        ).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        a = tl.load(
            A + row * a_stride_row + step * a_stride_step + head * a_stride_head
        )
        b = tl.load(
            B + row * b_stride_row + step * b_stride_step + head * b_stride_head
        )
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
    # Pass scalar strides: older Triton Ascend frontends cannot index a
    # tuple wrapped in tl.constexpr while building kernel IR.
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
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *a.stride(),
        *b.stride(),
        *state.stride(),
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
