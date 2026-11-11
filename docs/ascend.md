# Ascend 910B bring-up

The Ascend runner serves Dense Qwen3.5-family text checkpoints through the
existing Resin Engine and HTTP server. Start with Qwen3.5-4B BF16. Use the same
local checkpoint directory for weights, config, and tokenizer.

## Check the host

Activate the server's CANN and TorchNPU environment first. The first host
probe used CANN and Ops 9.1.0, Torch 2.10.0+cpu, TorchNPU 2.10.0.post4,
and eight Ascend 910B4 devices. It passed the elementary `npugraph_ex`
check using the backend shipped with TorchNPU. Run the standalone probe
with the active environment's Python:

```bash
python3 scripts/probe_ascend.py --output ascend-probe.json
python3 scripts/probe_ascend.py --output ascend-probe-hccl.json --hccl-ranks 2
```

The JSON includes package versions, device inventory, and a result for each
operator, graph backend, and HCCL test. Review it before sharing. The script
writes its report even when a required check fails and then exits nonzero.
Use `--device` to select the first visible NPU and `--timeout` to extend the
per-check timeout. To rerun only recurrent GDN:

```bash
python3 scripts/probe_ascend.py --check gdn_recurrent --output ascend-probe-gdn.json
```

The repository's core package currently declares `torch>=2.13.0`. On a server
whose CANN stack provides Torch 2.10, preserve that installed Torch/TorchNPU
pair for bring-up. The explicit source-install procedure below bypasses normal
dependency resolution for the local Resin packages. It requires Python 3.11+
and a Rust toolchain to build the extension. Make sure both `rustc --version`
and `cargo --version` work in the shell. The probe checks whether they are
on `PATH`. The build installs Maturin into its isolated build environment.
If Rust is missing, install the stable toolchain using the
[Rust installation guide](https://rust-lang.org/install).

```bash
python3 -m pip install 'blake3>=1,<2' 'numpy>=1.26,<3' 'safetensors>=0.5' 'transformers>=5.14.1'
python3 -m pip install --no-deps . ./models/ascend
python3 -c 'import resin, torch, torch_npu; print(resin.__version__, torch.__version__, torch_npu.__version__)'
```

## Start the service

```bash
resin-ascend-serve /models/Qwen3.5-4B

torchrun --standalone --nproc-per-node=2 -m resin_models_ascend.serve \
  /models/Qwen3.5-4B --tp 2

torchrun --standalone --nproc-per-node=8 -m resin_models_ascend.serve \
  /models/Qwen3.5-4B --dp 2 --tp 4
```

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-ascend","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":24}'
```

- Use `--quantization w8a8` with a ModelSlim dynamic INT8 checkpoint that has
  `quant_model_description.json`. Floating-point layers keep BF16 weights.
- Use `--mtp` with the bundled MTP weights. Set request temperature to zero
  to exercise speculation. Positive temperatures use target-only sampling.
- Use `--recipe qwen3_8` for Qwen3.8 checkpoints.
- Use `--attention native` after the probe passes both KV write and paged
  attention. Default eager execution uses the tensor attention and GDN
  reference paths. These paths favor correctness checks over throughput.
- Use `--native-gdn-prefill` after the `gdn_chunk` probe passes. Compare
  continued prefill against the reference path because the fused operator
  uses BF16 state at chunk boundaries.
- Use `--graph npugraph_ex --verbose` for one-card target-only decode with
  tensor attention. Compare the same prompts in eager and graph modes, including
  prompts crossing a 128-token page boundary. Graph capture uses a reserved
  scratch page. The configured graph page limit bounds the capture buckets.

For HTTP throughput and TTFT, run the existing stress harness after installing
its `httpx` dependency. Its report contains per-request timing and token counts
for decode latency calculations. The server logs graph counts, MTP acceptance
counts, and per-rank peak allocated NPU memory at shutdown.

```bash
python3 scripts/stress_http.py http://127.0.0.1:8000 \
  --model qwen-ascend --concurrency 1 4 --requests 16 --output ascend-http.json
```

Start with BF16 single-card eager, then compare eager native attention and
native GDN prefill individually. Next run single-card graph decode, ModelSlim
W8A8, TP2, TP4, DP2, and DP2×TP4. Compare greedy MTP against target-only on
the same precision and topology. The native operator call sites are based on
[vllm-ascend v0.30.0](https://github.com/vllm-project/vllm-ascend/tree/v0.30.0).
