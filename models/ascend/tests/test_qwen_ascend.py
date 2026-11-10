import json

import pytest
import torch
from safetensors.torch import save_file

from resin import Engine, SamplingParams
from resin.backend.ascend.ops import DynamicInt8Linear, tensor_attention
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_ascend.checkpoint import load_model, shard_tensor
from resin_models_ascend.model import QwenModel, TensorParallel
from resin_models_ascend.runner import QwenAscendRunner

CONFIG = {
    "model_type": "qwen3_5_text",
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 24,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 4,
    "linear_value_head_dim": 4,
    "linear_conv_kernel_dim": 3,
    "max_position_embeddings": 512,
    "eos_token_id": 31,
    "mtp_num_hidden_layers": 1,
    "layer_types": ["linear_attention", "full_attention"],
    "tie_word_embeddings": True,
}


def make_checkpoint(root, *, accept_draft=True, quantize=False):
    config = Qwen35Config.from_hugging_face(CONFIG)
    model = QwenModel(config, TensorParallel(), mtp=True)
    with torch.no_grad():
        for weight in model.parameters():
            weight.zero_()
        # Target always predicts token 3. The draft projection independently
        # chooses token 3 or token 4, covering commit and rollback.
        model.text.embed_tokens.weight[:, 0] = 1
        model.text.embed_tokens.weight[3, 0] = 3
        model.text.embed_tokens.weight[4, 1] = 3
        model.mtp.fc.weight[0 if accept_draft else 1, 0] = 1
    (root / "config.json").write_text(json.dumps(CONFIG))
    tensors = {
        name: weight.to(torch.bfloat16) for name, weight in model.state_dict().items()
    }
    if quantize:
        name = "model.language_model.layers.0.linear_attn.in_proj_qkv"
        tensors[name + ".weight"] = tensors[name + ".weight"].to(torch.int8)
        tensors[name + ".weight_scale"] = torch.ones(
            config.linear_conv_dim, dtype=torch.float32
        )
        (root / "quant_model_description.json").write_text(
            json.dumps({name + ".weight": "W8A8_DYNAMIC"})
        )
    save_file(tensors, root / "model.safetensors")
    return config


@pytest.mark.parametrize("accept_draft", [True, False])
def test_mtp_matches_target_across_page_and_prefill_chunks(tmp_path, accept_draft):
    make_checkpoint(tmp_path, accept_draft=accept_draft)

    def generate(mtp):
        runner = QwenAscendRunner(
            tmp_path,
            device=torch.device("cpu"),
            mtp=mtp,
            cache_pages=16,
            prefill_chunk=64,
        )
        with Engine(runner, context_length=256, enable_overlap=False) as engine:
            engine.submit(
                [1] * 127,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=5),
            )
            events = []
            for _ in range(12):
                events.extend(engine.step())
                if len(events) >= 5:
                    break
            tokens = [event.token_id for event in events]
            assert not runner.states
        return tokens, runner.drafted, runner.accepted

    target, _, _ = generate(False)
    mtp, drafted, accepted = generate(True)
    assert mtp == target == [3] * 5
    assert drafted > 0
    assert accepted == (drafted if accept_draft else 0)


def test_modelslim_qkv_tp_preserves_segment_and_scale_order(tmp_path):
    config = make_checkpoint(tmp_path, quantize=True)
    tp = TensorParallel(2, 1)
    model = load_model(tmp_path, torch.device("cpu"), tp, mtp=True, quantization="w8a8")
    qkv = model.text.layers[0].linear_attn.in_proj_qkv
    assert isinstance(qkv, DynamicInt8Linear)
    assert qkv.weight.shape == (config.hidden_size, config.linear_conv_dim // 2)
    assert qkv.weight_scale.shape == (config.linear_conv_dim // 2,)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale"
    scales = torch.arange(config.linear_conv_dim)
    segments = scales.split(
        (config.linear_key_dim, config.linear_key_dim, config.linear_value_dim)
    )
    assert torch.equal(
        shard_tensor(name, scales, config, tp),
        torch.cat([x.chunk(2)[1] for x in segments]),
    )


def test_padded_attention_reads_only_pages_in_chronological_order():
    keys = torch.zeros((4, 2, 1, 4))
    values = torch.zeros_like(keys)
    values[3] = 2
    values[1, 1] = 99
    values[0] = 88
    q = torch.zeros((1, 2, 4))
    kv = torch.zeros((1, 1, 4))
    output = tensor_attention(
        q,
        kv,
        torch.full_like(kv, 5),
        keys,
        values,
        torch.tensor([2]),
        torch.tensor([3, 1, 0]),
        torch.tensor([2]),
    )
    torch.testing.assert_close(output, torch.full_like(output, 3))
