"""Fused decay and beta projection transforms for FLA GDN."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _gdn_gating(
    A,
    B,
    ALog,
    DtBias,
    Decay,
    Beta,
    count: tl.constexpr,
    heads: tl.constexpr,
    block: tl.constexpr,
):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    live = offset < count
    head = offset % heads
    a = tl.load(A + offset, live, other=0).to(tl.float32)
    b = tl.load(B + offset, live, other=0).to(tl.float32)
    x = a + tl.load(DtBias + head)
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    decay = -tl.exp(tl.load(ALog + head)) * softplus
    tl.store(Decay + offset, decay, live)
    tl.store(Beta + offset, tl.sigmoid(b), live)


def fused_gdn_gating(a: Tensor, b: Tensor, a_log: Tensor, dt_bias: Tensor):
    a, b = a.contiguous(), b.contiguous()
    decay = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b)
    block = 128
    _gdn_gating[(triton.cdiv(a.numel(), block),)](
        a,
        b,
        a_log,
        dt_bias,
        decay,
        beta,
        a.numel(),
        a.shape[-1],
        block,
    )
    return decay, beta
