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

## Validate ACLGraph decode

```bash
resin-ascend-serve /models/Qwen3.5-4B --graph aclgraph \
  --graph-capture-pages 1 2 4 8 --verbose
```

This mode captures each one-token target forward using TorchNPU's
`NPUGraph`. `--graph npugraph_ex` uses the TorchNPU compile backend with the
same decode eligibility. Both modes use tensor attention on one card, with
eager prefill and sampling. Capture writes to a reserved scratch page. Each
replay updates the input buffers with the live token, position, page table,
write slot, and recurrent state.

### MTP with ACLGraph

```bash
resin-ascend-serve /models/Qwen3.5-4B --num-speculative-tokens 3 --graph aclgraph \
  --graph-capture-pages 1 2 4 8 --graph-verify-batches 2 4 8 --verbose
```

The checkpoint must include MTP weights. `--mtp` enables one draft token.
`--num-speculative-tokens K` enables MTP and sets the draft count to K.
For greedy requests (`temperature=0`), the same MTP head runs K times.
Each step feeds back its token and MTP hidden state to the next step.
The target verifies the anchor and K drafts in a single forward, commits
the longest matching prefix, and emits the next token from the target.
The verifier retains a recurrent and convolution state at each prefix.
Accepted draft KV is rebuilt using the verified target hidden states.
Writes beyond the accepted prefix are masked and overwritten on later steps.

Each page bucket captures one-token `target`, one-step `draft`, and
`verify` graphs at startup. Verify warmup uses distinct slots in the
reserved page, so a captured batch must fit `B × (K + 1) <= 128`.

With tensor attention, greedy MTP rows in a decode step share one batched
target forward over K + 1 query tokens per row. Their KV
tables and GDN states remain independent. Draft generation, draft cache
updates after acceptance, and ordinary target decode run per request.
`--graph-verify-batches` captures additional verify graphs for exact request
counts, with one graph for each combination of batch size and configured KV
page bucket. At K=3, a verify batch of four requests runs 16 query tokens. The
page bucket follows the longest sequence in the verify batch. If either
dimension misses, verify runs as a batch in eager mode. A batch of one still
uses the per-request graph. The eager native attention path uses
per-request verify.

Logits projection, argmax, acceptance decisions, multi-token draft KV repair,
state commit, and prefill
remain eager. Positive-temperature requests use target-only sampling and
maintain the draft cache. MTP graphs require one NPU and tensor attention;
`--mtp --graph npugraph_ex`, TP graphs, and native-attention graphs remain
unsupported. BF16 and dynamic W8A8 use the same graph routing, but each
precision requires independent validation on the target software stack.

The graph bucket uses the whole context including the verification tokens.
For example, verification at context 127 needs bucket 2, even though its
anchor alone fits in bucket 1. Missing buckets fall back to eager for that
operation, without runtime capture when startup buckets are configured.
Every extra verify batch size adds one graph per configured KV page bucket.

After rebuilding both packages, run the CPU and NPU tests on the server:

```bash
python3 -m pip install --no-deps --force-reinstall . ./models/ascend
python3 -m pytest -q models/ascend/tests/test_qwen_ascend.py -k 'mtp or aclgraph'
```

NPU cases skip if TorchNPU or a device is unavailable. They cover actual
capture/replay with changed tokens, page tables and recurrent state, scratch
isolation, forced draft acceptance/rejection, and a KV page boundary. CPU
cases emulate static graph output buffers to check lifetime and routing.
Also compare the real checkpoint's greedy token IDs for target-only eager,
MTP eager, and MTP ACLGraph, including concurrent requests, EOS and output
limits, before measuring speed. A toy checkpoint test does not establish
real-model accuracy or W8A8 compatibility.

The buffer lifetime rules follow TorchAir's
[graph memory documentation](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/memory_reuse.md).
This implementation uses the existing raw TorchNPU `NPUGraph` API; it does not
enable TorchAir graph-pool sharing between draft and verify.

### Page buckets and validation

With `--graph-capture-pages`, Resin captures and replays each listed bucket
once on the scratch page before the HTTP listener starts. Sizes are powers of
two in KV pages of 128 tokens. For example, `1 2 4 8` covers decode lengths
through 1,024 tokens, counting prompt and generated history. Requests whose
bucket is absent use eager decode. Omit the option to capture lazily on the
first request for each bucket. The startup option applies to `--graph aclgraph`.

Send the same requests at temperature zero to eager and ACLGraph servers.
Compare the generated output, including a decode that crosses a 128-token
page boundary. With `--verbose`, look for `mode=aclgraph` capture and replay
messages. Startup captures are marked `stage=startup`. `--graph-max-pages`
controls the largest graph page bucket.
Requests above that limit use eager decode and log a fallback.

The paged-attention replay test varies the page table and position within a
single captured graph. Run it in the server environment with pytest installed:

```bash
python3 -m pytest -q models/ascend/tests/test_qwen_ascend.py -k aclgraph
```

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
