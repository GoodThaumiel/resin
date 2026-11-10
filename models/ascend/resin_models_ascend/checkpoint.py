"""Stream BF16 and ModelSlim W8A8_DYNAMIC checkpoint tensors into local TP shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from resin.backend.ascend.ops import DynamicInt8Linear
from resin.backend.loading import assign_model_tensor, read_config
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_ascend.model import QwenModel, TensorParallel


def shard_tensor(
    name: str, tensor: Tensor, config: Qwen35Config, tp: TensorParallel
) -> Tensor:
    if tp.size == 1:
        return tensor
    layer = name.rsplit(".", 2)[-2]
    if layer in {"o_proj", "out_proj", "down_proj"}:
        return (
            tensor.chunk(tp.size, dim=1)[tp.rank]
            if name.endswith(".weight")
            else tensor
        )
    if layer in {"in_proj_qkv", "conv1d"}:
        segments = tensor.split(
            (config.linear_key_dim, config.linear_key_dim, config.linear_value_dim)
        )
        return torch.cat([segment.chunk(tp.size)[tp.rank] for segment in segments])
    if layer in {
        "q_proj",
        "k_proj",
        "v_proj",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "gate_proj",
        "up_proj",
    }:
        return tensor.chunk(tp.size)[tp.rank]
    if name.endswith((".A_log", ".dt_bias")):
        return tensor.chunk(tp.size)[tp.rank]
    return tensor


def checkpoint_name(name: str, tensors: dict) -> str:
    if name in tensors:
        return name
    text_prefix = "model.language_model."
    if name.startswith(text_prefix):
        suffix = name.removeprefix(text_prefix)
        for prefix in ("language_model.model.", "model."):
            if prefix + suffix in tensors:
                return prefix + suffix
    if name.startswith("lm_head.") and "language_model." + name in tensors:
        return "language_model." + name
    return name


def load_model(
    root: Path,
    device: torch.device,
    tp: TensorParallel,
    *,
    mtp: bool,
    quantization: str,
) -> QwenModel:
    raw = read_config(root)
    config = Qwen35Config.from_hugging_face(raw)
    config.require_dense()
    if mtp:
        config.require_mtp()
    local = tp.local_config(config)
    desc_path = root / "quant_model_description.json"
    description = json.loads(desc_path.read_text()) if desc_path.is_file() else {}
    if not isinstance(description, dict):
        raise ValueError("quant_model_description.json must contain an object")
    if quantization not in {"bf16", "w8a8"}:
        raise ValueError(f"unsupported Ascend quantization mode {quantization!r}")
    if quantization == "w8a8" and not description:
        raise ValueError("W8A8 requires ModelSlim quant_model_description.json")
    if quantization == "bf16" and (description or raw.get("quantization_config")):
        raise ValueError("quantized checkpoints require --quantization w8a8")
    tensors = indexed_tensor_files(root)
    with torch.device("meta"):
        model = QwenModel(local, tp, mtp=mtp)
        quantized = set()
        for name, module in tuple(model.named_modules()):
            weight_name = checkpoint_name(name + ".weight", tensors)
            scheme = description.get(weight_name, "FLOAT")
            if scheme == "FLOAT":
                continue
            if scheme != "W8A8_DYNAMIC" or not isinstance(module, nn.Linear):
                raise ValueError(
                    f"unsupported quantization {scheme!r} for {weight_name}"
                )
            model.set_submodule(
                name, DynamicInt8Linear(module.in_features, module.out_features)
            )
            quantized.add(name)
    if quantization == "w8a8" and not quantized:
        raise ValueError("checkpoint contains no W8A8_DYNAMIC text projections")
    with SafetensorReader() as reader:
        for name, expected in model.state_dict().items():
            source = checkpoint_name(name, tensors)
            if source not in tensors:
                raise ValueError(f"checkpoint is missing {source}")
            tensor = reader.read(tensors[source], source)
            if expected.dtype == torch.int8:
                if tensor.dtype != torch.int8:
                    raise ValueError(f"{source} must have INT8 weights")
                offset_name = source.removesuffix(".weight") + ".weight_offset"
                if (
                    offset_name in tensors
                    and reader.read(tensors[offset_name], offset_name).count_nonzero()
                ):
                    raise ValueError(f"{offset_name} must be zero for symmetric W8A8")
            elif not tensor.is_floating_point() or tensor.element_size() < 2:
                raise ValueError(f"{source} must be BF16 or wider floating point")
            if name.endswith(".weight_scale"):
                if tensor.ndim not in (1, 2) or (
                    tensor.ndim == 2 and tensor.shape[1] != 1
                ):
                    raise ValueError(
                        f"{source} must contain one scale per output channel"
                    )
                tensor = tensor.reshape(-1, 1)
                if not (tensor.isfinite() & (tensor > 0)).all():
                    raise ValueError(f"{source} has nonpositive or nonfinite scales")
            tensor = shard_tensor(name, tensor, config, tp)
            fp32 = name.endswith((".weight_scale", ".A_log", ".dt_bias"))
            dtype = torch.float32 if fp32 else torch.bfloat16
            assign_model_tensor(
                model, name, tensor.contiguous(), device=device, dtype=dtype
            )
    for module in model.modules():
        if isinstance(module, DynamicInt8Linear):
            module.prepare()
    exponent = (
        torch.arange(0, config.rotary_dim, 2, dtype=torch.float32) / config.rotary_dim
    )
    model.inv_freq = (1 / config.rope_theta**exponent).to(device)
    return model.eval()
