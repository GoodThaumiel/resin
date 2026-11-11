"""Ascend operators and a tensor-only attention path for graph capture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DynamicInt8Linear(nn.Module):
    """ModelSlim symmetric per-channel weights and per-token activations."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        import torch_npu

        x = hidden.reshape(-1, hidden.shape[-1])
        quantized, scales = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        result = torch_npu.npu_quant_matmul(
            quantized,
            self.weight,
            self.weight_scale,
            pertoken_scale=scales,
            output_dtype=hidden.dtype,
        )
        return result.reshape(*hidden.shape[:-1], result.shape[-1])

    def prepare(self) -> None:
        self.weight = self.weight.t().contiguous()
        self.weight_scale = self.weight_scale.flatten()


def write_kv(
    keys: Tensor, values: Tensor, key: Tensor, value: Tensor, slots: Tensor
) -> None:
    keys.flatten(0, 1).index_copy_(0, slots, key)
    values.flatten(0, 1).index_copy_(0, slots, value)


def native_chunk_gdn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    import torch_npu

    def normalize(tensor: Tensor) -> Tensor:
        fp32 = tensor.float()
        scale = (fp32.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        return (fp32 * scale).to(tensor.dtype).contiguous()

    lengths = torch.full((1,), len(query), dtype=torch.int32, device=query.device)
    output, next_state = torch_npu.npu_chunk_gated_delta_rule(
        normalize(query),
        normalize(key),
        value.contiguous(),
        beta=beta.contiguous(),
        initial_state=state.transpose(-2, -1).unsqueeze(0).to(value.dtype).contiguous(),
        actual_seq_lengths=lengths,
        scale=query.shape[-1] ** -0.5,
        g=decay.contiguous(),
    )
    return output, next_state[0].transpose(-2, -1).float()


def tensor_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    keys: Tensor,
    values: Tensor,
    slots: Tensor,
    pages: Tensor,
    positions: Tensor,
) -> Tensor:
    """Attend to each row's paged KV with device-side causal positions."""
    single = query.ndim == 3
    if single:
        query, pages, positions = query[None], pages[None], positions[None]
    write_kv(keys, values, key.flatten(0, -3), value.flatten(0, -3), slots.flatten())
    batch, page_count = pages.shape
    key = keys.index_select(0, pages.flatten()).unflatten(0, (batch, page_count))
    value = values.index_select(0, pages.flatten()).unflatten(0, (batch, page_count))
    repeats = query.shape[-2] // key.shape[-2]
    key = key.flatten(1, 2).repeat_interleave(repeats, dim=2).transpose(1, 2)
    value = value.flatten(1, 2).repeat_interleave(repeats, dim=2).transpose(1, 2)
    visible = torch.arange(key.shape[-2], device=query.device) <= positions[..., None]
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key,
        value,
        attn_mask=visible[0] if single else visible[:, None],
    ).transpose(1, 2)
    return output[0] if single else output


class NativePagedAttention:
    """TND ACLNN path with host sequence lengths. Validate with the probe first."""

    def __init__(self, device: torch.device) -> None:
        import torch_npu

        self.scatter = torch_npu.npu_scatter_pa_kv_cache
        self.attention = torch_npu.npu_fused_infer_attention_score
        self.mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu_(1)

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        keys: Tensor,
        values: Tensor,
        slots: Tensor,
        pages: Tensor,
        length: int,
    ) -> Tensor:
        self.scatter(
            key.contiguous(),
            value.contiguous(),
            keys,
            values,
            slots.to(torch.int32),
            cache_mode="Norm",
        )
        causal = len(query) > 1
        output, _ = self.attention(
            query.contiguous(),
            keys.flatten(2),
            values.flatten(2),
            input_layout="TND",
            num_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            scale=query.shape[-1] ** -0.5,
            block_table=pages[None].to(torch.int32),
            block_size=keys.shape[1],
            actual_seq_lengths=[len(query)],
            actual_seq_lengths_kv=[length],
            sparse_mode=3 if causal else 0,
            atten_mask=self.mask if causal else None,
        )
        return output
