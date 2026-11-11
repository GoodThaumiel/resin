import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from resin import Engine, Parallelism, SamplingParams
from resin.backend.ascend.graphs import DecodeGraph
from resin.backend.ascend.ops import DynamicInt8Linear, tensor_attention
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_ascend import runner as ascend_runner
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

NPU_CONFIG = {
    **CONFIG,
    "hidden_size": 64,
    "intermediate_size": 128,
    "head_dim": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    device = torch.device("npu", torch.npu.current_device())
    torch.npu.set_device(device)
    return device


@pytest.mark.parametrize("dp,tp", [(1, 2), (2, 1), (2, 4)])
def test_dense_runner_accepts_dp_and_tp_with_derived_expert_tp(dp, tp):
    topology = Parallelism(world_size=dp * tp, attn_dp=dp, attn_tp=tp)
    QwenAscendRunner.parallelism_capabilities.validate(
        topology, model=QwenAscendRunner.__name__
    )


@pytest.mark.parametrize("axis", ["cp", "ep"])
def test_dense_runner_rejects_cp_and_ep(axis):
    topology = Parallelism(world_size=2, **{axis: 2})
    with pytest.raises(ValueError, match=f"does not support {axis}=2"):
        QwenAscendRunner.parallelism_capabilities.validate(
            topology, model=QwenAscendRunner.__name__
        )


def make_checkpoint(root, *, accept_draft=True, quantize=False, raw_config=CONFIG):
    config = Qwen35Config.from_hugging_face(raw_config)
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
    (root / "config.json").write_text(json.dumps(raw_config))
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


def test_chunked_prefill_returns_tokens_only_for_completed_prompts(tmp_path):
    make_checkpoint(tmp_path)
    runner = QwenAscendRunner(
        tmp_path, device=torch.device("cpu"), cache_pages=16, prefill_chunk=64
    )
    params = SamplingParams(temperature=0, max_output_tokens=1)
    with Engine(
        runner, context_length=256, max_running_requests=2, enable_overlap=False
    ) as engine:
        long = engine.submit([1] * 65, sampling_params=params)
        short = engine.submit([1], sampling_params=params)

        assert [event.token_id for event in engine.step()] == [3]
        assert short.done and not long.done
        assert short.take_result() == [3]

        assert [event.token_id for event in engine.step()] == [3]
        assert long.done
        assert long.take_result() == [3]


def test_startup_aclgraph_captures_configured_buckets_before_decode(
    tmp_path, monkeypatch
):
    make_checkpoint(tmp_path)
    runner = QwenAscendRunner(tmp_path, device=torch.device("cpu"), cache_pages=16)
    # Exercise Engine binding and decode routing on CPU with a stand-in graph.
    runner.graph_mode = "aclgraph"
    runner.graph_capture_pages = (1, 4)
    captured = []

    class EagerGraph:
        def __init__(self, forward, inputs, *, mode, warmup_inputs):
            assert mode == "aclgraph"
            assert torch.is_inference_mode_enabled()
            assert all(not tensor.count_nonzero() for tensor in warmup_inputs)
            captured.append(inputs[2].numel())
            self.forward = forward

        def replay(self, inputs):
            return self.forward(*inputs)

    monkeypatch.setattr(ascend_runner, "DecodeGraph", EagerGraph)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda _device: None), raising=False
    )
    with Engine(runner, context_length=512, enable_overlap=False) as engine:
        assert captured == [1, 4]
        assert runner.graph_replays == 0
        request = engine.submit(
            [1] * 127,
            sampling_params=SamplingParams(temperature=0, max_output_tokens=4),
        )
        for _ in range(8):
            engine.step()
            if request.done:
                break
        assert request.take_result() == [3] * 4

    assert captured == [1, 4]
    assert runner.graph_replays == 1
    assert runner.graph_fallbacks == 2


@pytest.mark.parametrize("accept_draft", [True, False])
@pytest.mark.parametrize("graph", ["eager", "simulated_aclgraph", "npu_aclgraph"])
@pytest.mark.parametrize("draft_count", [1, 3])
def test_mtp_matches_target_across_page_and_prefill_chunks(
    tmp_path, monkeypatch, accept_draft, graph, draft_count
):
    device = npu_device() if graph == "npu_aclgraph" else torch.device("cpu")
    make_checkpoint(
        tmp_path,
        accept_draft=accept_draft,
        raw_config=NPU_CONFIG if device.type == "npu" else CONFIG,
    )
    if graph == "simulated_aclgraph":
        install_cpu_graph(monkeypatch)

    def generate(mtp):
        runner = QwenAscendRunner(
            tmp_path,
            device=device,
            mtp=mtp,
            num_speculative_tokens=draft_count if mtp else None,
            cache_pages=16,
            prefill_chunk=64,
        )
        if graph != "eager":
            runner.graph_mode = "aclgraph"
            runner.graph_capture_pages = (1, 2)
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
        if graph != "eager" and mtp:
            assert {kind for kind, _, _ in runner.graphs} == {
                "target",
                "verify",
                "draft",
            }
            assert runner.graph_replays > 0
            assert runner.graph_fallbacks == 0
        return tokens, runner.drafted, runner.accepted

    target, _, _ = generate(False)
    mtp, drafted, accepted = generate(True)
    assert mtp == target == [3] * 5
    assert drafted > 0
    assert accepted == (drafted if accept_draft else 0)


def install_cpu_graph(monkeypatch):
    """Keep static inputs/outputs to catch aliasing across rows and graph kinds."""

    def capture(graph, forward, _device):
        assert torch.is_inference_mode_enabled()
        output = tuple(value.clone() for value in forward(*graph.inputs))

        def replay():
            for buffer, value in zip(output, forward(*graph.inputs), strict=True):
                buffer.copy_(value)
            return output

        return replay

    monkeypatch.setattr(DecodeGraph, "_capture", capture)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda _device: None), raising=False
    )


@pytest.mark.parametrize("capture_pages", [(1, 2), (1,), None])
@pytest.mark.parametrize(
    "device_kind,quantization", [("cpu", "bf16"), ("npu", "bf16"), ("npu", "w8a8")]
)
@torch.inference_mode()
def test_mtp_aclgraph_nonzero_states_and_scratch_isolation(
    tmp_path, monkeypatch, capture_pages, device_kind, quantization
):
    device = npu_device() if device_kind == "npu" else torch.device("cpu")
    make_checkpoint(
        tmp_path,
        raw_config=NPU_CONFIG if device_kind == "npu" else CONFIG,
        quantize=quantization == "w8a8",
    )
    if device_kind == "cpu":
        install_cpu_graph(monkeypatch)
    runner = QwenAscendRunner(
        tmp_path, device=device, mtp=True, quantization=quantization
    )
    # Nonzero projections make recurrence rollback and stale output observable.
    torch.manual_seed(17)
    for parameter in runner.model.parameters():
        parameter.normal_(0, 0.15)
    if quantization == "w8a8":
        qkv = runner.model.text.layers[0].linear_attn.in_proj_qkv
        qkv.weight.copy_(
            torch.randint(-5, 6, qkv.weight.shape, dtype=torch.int8).to(device)
        )
        qkv.weight_scale.fill_(0.03)
    c = runner.model.config
    shape = (4, 128, c.num_key_value_heads, c.head_dim)
    runner.keys = torch.randn(
        (c.full_layer_count, *shape), dtype=torch.bfloat16, device=device
    )
    runner.values = torch.randn_like(runner.keys)
    runner.draft_keys = torch.randn(shape, dtype=torch.bfloat16, device=device)
    runner.draft_values = torch.randn_like(runner.draft_keys)
    before = tuple(
        tensor.clone()
        for tensor in (
            runner.keys,
            runner.values,
            runner.draft_keys,
            runner.draft_values,
        )
    )
    runner.graph_mode = "aclgraph"
    runner.graph_capture_pages = capture_pages
    if capture_pages is not None:
        runner._capture_startup_graphs(capture_pages)
        for actual, expected in zip(
            (runner.keys, runner.values, runner.draft_keys, runner.draft_values), before
        ):
            # Warmup/replay can only change reserved physical page zero.
            index = (
                (slice(None), slice(1, None)) if actual.ndim == 5 else slice(1, None)
            )
            torch.testing.assert_close(actual[index], expected[index])
        count = len(runner.graphs)

    saved = None
    for context, pages in [(3, [3]), (127, [3, 1]), (128, [1, 2]), (4, [2])]:
        group = SimpleNamespace(block_slabs=lambda _row: np.array(pages))
        state = runner._empty_state()
        state.recurrent.normal_(0, 0.1)
        state.convolution.normal_(0, 0.1)
        state.hidden.normal_(0, 0.1)
        original_rec, original_conv = state.recurrent.clone(), state.convolution.clone()
        ids = torch.tensor([5, 7], device=device)
        positions, table, slots = runner._page_table(group, 0, context, context + 2)
        expected = runner.model.target(
            ids,
            positions,
            table,
            slots,
            state.recurrent,
            state.convolution,
            runner.keys.clone(),
            runner.values.clone(),
            verify=True,
        )
        actual = runner._target(ids, group, 0, context, state, decode=True, verify=True)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        # Rejection commits exactly the anchor's recurrent and convolution state.
        anchor = runner.model.target(
            ids[:1],
            positions[:1],
            table,
            slots[:1],
            state.recurrent,
            state.convolution,
            runner.keys.clone(),
            runner.values.clone(),
        )
        tolerance = (
            dict(rtol=0, atol=0) if device_kind == "cpu" else dict(rtol=0.02, atol=1e-4)
        )
        torch.testing.assert_close(actual[1][:, 0], anchor[1], **tolerance)
        torch.testing.assert_close(actual[2][:, 0], anchor[2], **tolerance)
        torch.testing.assert_close(state.recurrent, original_rec, rtol=0, atol=0)
        torch.testing.assert_close(state.convolution, original_conv, rtol=0, atol=0)
        if saved is not None:
            for value, copy in zip(saved[0], saved[1]):
                torch.testing.assert_close(value, copy, rtol=0, atol=0)
        committed = tuple(value.clone() for value in actual)
        saved = committed, tuple(value.clone() for value in committed)

        for offset in (0, 1):
            pos, table, slot = runner._page_table(
                group, 0, context + offset, context + offset + 1
            )
            hidden = actual[0][offset : offset + 1]
            expected_draft = runner.model.draft(
                ids[offset : offset + 1],
                hidden,
                pos,
                table,
                slot,
                runner.draft_keys.clone(),
                runner.draft_values.clone(),
            )
            actual_draft = runner._draft(
                ids[offset : offset + 1],
                hidden,
                group,
                0,
                context + offset,
                decode=True,
            )
            torch.testing.assert_close(actual_draft, expected_draft, rtol=0, atol=0)
        # Target-only decode remains available to non-greedy requests.
        target = runner._target(ids[:1], group, 0, context, state, decode=True)
        for result, reference in zip(target, anchor):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    if capture_pages is not None:
        assert len(runner.graphs) == count
    assert (runner.graph_fallbacks > 0) == (capture_pages == (1,))


def test_mtp_npugraph_ex_rejected_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="MTP graph execution requires"):
        QwenAscendRunner(
            tmp_path, device=SimpleNamespace(type="npu"), mtp=True, graph="npugraph_ex"
        )


@pytest.mark.parametrize("mode", ["eager", "simulated_aclgraph", "npu_aclgraph"])
@pytest.mark.parametrize("draft_count", [1, 3])
@torch.inference_mode()
def test_mtp_batch_verify_matches_independent_rows(
    tmp_path, monkeypatch, mode, draft_count
):
    device = npu_device() if mode == "npu_aclgraph" else torch.device("cpu")
    make_checkpoint(tmp_path, raw_config=NPU_CONFIG if device.type == "npu" else CONFIG)
    if mode == "simulated_aclgraph":
        install_cpu_graph(monkeypatch)
    runner = QwenAscendRunner(
        tmp_path, device=device, num_speculative_tokens=draft_count
    )
    torch.manual_seed(29)
    for parameter in runner.model.parameters():
        parameter.normal_(0, 0.15)
    c = runner.model.config
    shape = (5, 128, c.num_key_value_heads, c.head_dim)
    runner.keys = torch.randn(
        (c.full_layer_count, *shape), dtype=torch.bfloat16, device=device
    )
    runner.values = torch.randn_like(runner.keys)
    runner.draft_keys = torch.zeros_like(runner.keys[0])
    runner.draft_values = torch.zeros_like(runner.draft_keys)
    if mode != "eager":
        before = (runner.keys[:, 1:].clone(), runner.values[:, 1:].clone())
        runner.graph_mode = "aclgraph"
        runner.graph_capture_pages = (2,)
        runner.graph_verify_batches = (1, 2)
        runner._capture_startup_graphs((2,))
        torch.testing.assert_close(runner.keys[:, 1:], before[0])
        torch.testing.assert_close(runner.values[:, 1:], before[1])

    tables = (np.array([4, 1]), np.array([3]))
    group = SimpleNamespace(block_slabs=lambda row: tables[row])
    rows = []
    for row, context in enumerate((127, 8)):
        state = runner._empty_state()
        state.recurrent.normal_(0, 0.1)
        state.convolution.normal_(0, 0.1)
        rows.append(
            (
                row,
                context,
                state,
                runner._long(list(range(5 + row, 6 + row + draft_count))),
            )
        )

    short_row = (rows[0][0], 3, *rows[0][2:])
    for batch in (rows, list(reversed(rows)), rows[:1], [short_row, rows[1]]):
        ref_keys, ref_values = runner.keys.clone(), runner.values.clone()
        expected = []
        for row, context, state, ids in batch:
            recurrent, convolution = state.recurrent, state.convolution
            hidden_steps, recurrent_steps, conv_steps = [], [], []
            for offset in range(len(ids)):
                start = context + offset
                metadata = runner._page_table(group, row, start, start + 1)
                hidden, recurrent, convolution = runner.model.target(
                    ids[offset : offset + 1],
                    *metadata,
                    recurrent,
                    convolution,
                    ref_keys,
                    ref_values,
                )
                hidden_steps.append(hidden)
                recurrent_steps.append(recurrent)
                conv_steps.append(convolution)
            expected.append(
                (
                    torch.cat(hidden_steps),
                    torch.stack(recurrent_steps, dim=1),
                    torch.stack(conv_steps, dim=1),
                )
            )
        hidden, *states = runner._verify_targets(group, batch)
        for index, (reference, *ref_states) in enumerate(expected):
            torch.testing.assert_close(hidden[index], reference, rtol=0.03, atol=0.004)
            for state, ref in zip(states, ref_states, strict=True):
                torch.testing.assert_close(state[:, index], ref, rtol=0.03, atol=0.004)
        torch.testing.assert_close(
            runner.keys[:, 1:], ref_keys[:, 1:], rtol=0.03, atol=0.004
        )
        torch.testing.assert_close(
            runner.values[:, 1:], ref_values[:, 1:], rtol=0.03, atol=0.004
        )
    assert runner.graph_replays == (0 if mode == "eager" else 3)
    assert runner.graph_fallbacks == (0 if mode == "eager" else 1)


@torch.inference_mode()
def test_aclgraph_rejects_input_changes_before_copy(monkeypatch):
    install_cpu_graph(monkeypatch)
    graph = DecodeGraph(lambda x: (x + 1,), (torch.zeros(2),), mode="aclgraph")
    for bad in ((torch.ones(1),), (torch.ones(2, dtype=torch.long),), ()):
        with pytest.raises(ValueError, match="input shape, dtype, or device"):
            graph.replay(bad)
        assert not graph.inputs[0].count_nonzero()
    (result,) = graph.replay((torch.tensor([2.0, 3.0]),))
    torch.testing.assert_close(result, torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("stop_on_eos", [False, True])
@pytest.mark.parametrize("accept_draft", [False, True])
@pytest.mark.parametrize("draft_count", [1, 3])
def test_mtp_aclgraph_mixed_requests_stop_and_release(
    tmp_path, monkeypatch, stop_on_eos, accept_draft, draft_count
):
    make_checkpoint(tmp_path)
    install_cpu_graph(monkeypatch)

    def generate(graph):
        runner = QwenAscendRunner(
            tmp_path,
            device=torch.device("cpu"),
            num_speculative_tokens=draft_count,
            cache_pages=32,
        )
        with torch.no_grad():
            embedding = runner.model.text.embed_tokens.weight
            embedding.zero_()
            # Target transitions 1 -> 5 -> 3, so EOS occurs during verify.
            embedding[1, 0] = 1
            embedding[5, :2] = torch.tensor([2.0, 1.0])
            embedding[3, 1] = 6
            runner.model.mtp.fc.weight.zero_()
            runner.model.mtp.fc.weight[1 if accept_draft else 0, 0] = 1
        if graph:
            runner.graph_mode = "aclgraph"
            runner.graph_capture_pages = (1, 2)
            runner.graph_verify_batches = (1, 2)
        with Engine(
            runner, context_length=256, max_running_requests=4, enable_overlap=False
        ) as engine:
            requests = [
                engine.submit(
                    [1] * length,
                    sampling_params=SamplingParams(
                        temperature=temperature,
                        top_k=1,
                        seed=7,
                        max_output_tokens=8 if stop_on_eos else 2,
                        stop_token_ids=[3] if stop_on_eos else [],
                    ),
                )
                for length, temperature in [(127, 0), (3, 0), (4, 0.7)]
            ]
            # Token 3 has no component at input feature zero, so its draft
            # is rejected in the same batch where token 5 can be accepted.
            requests.append(
                engine.submit(
                    [3] * 5,
                    sampling_params=SamplingParams(temperature=0, max_output_tokens=2),
                )
            )
            engine.step()
            assert all(not request.done for request in requests)
            assert engine.cancel(requests[1])
            for _ in range(8):
                engine.step()
                if all(requests[i].done for i in (0, 2, 3)):
                    break
            results = [requests[i].take_result() for i in (0, 2, 3)]
            assert not runner.states
            # Reusing the pool after cancellation/EOS must not reuse live state.
            again = engine.submit(
                [1] * 129,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=3),
            )
            for _ in range(8):
                engine.step()
                if again.done:
                    break
            results.append(again.take_result())
            assert not runner.states
        assert runner.drafted > 0
        assert runner.drafted == runner.verify_rows * draft_count
        # The token-5 rows accept draft 3, then reject any further proposal.
        assert runner.accepted == (2 if accept_draft else 0)
        return results

    assert generate(True) == generate(False) == [[5, 3], [5, 3], [3, 3], [5, 3, 3]]


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


@torch.inference_mode()
def test_aclgraph_replays_paged_attention_with_updated_metadata():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    device = torch.device("npu", torch.npu.current_device())
    keys = torch.zeros((4, 128, 1, 256), dtype=torch.bfloat16, device=device)
    values = torch.zeros_like(keys)
    values[1, 0].fill_(2)
    values[3].fill_(8)
    eager_keys, eager_values = keys.clone(), values.clone()
    query = torch.zeros((1, 4, 256), dtype=keys.dtype, device=device)
    key = torch.zeros_like(keys[:1, 0])

    def forward(query, key, value, slot, pages, position):
        return (
            tensor_attention(query, key, value, keys, values, slot, pages, position),
        )

    def inputs(value, slot, pages, position):
        return (
            query,
            key,
            torch.full_like(key, value),
            torch.tensor([slot], device=device),
            torch.tensor(pages, device=device),
            torch.tensor([position], device=device),
        )

    first = inputs(4, 129, [1, 0], 1)
    second = inputs(137, 256, [3, 2], 128)
    graph = DecodeGraph(forward, first, mode="aclgraph")
    torch.testing.assert_close(values[1:].cpu(), eager_values[1:].cpu())

    for args in (first, second, first):
        expected = tensor_attention(*args[:3], eager_keys, eager_values, *args[3:])
        (actual,) = graph.replay(args)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(values[1:].cpu(), eager_values[1:].cpu())
