# Resin Ascend 后端性能与完整性评估

本轮代码审查基于 `codex/ascend-910b`，包含本次新增的 MTP ACLGraph。
这是实现评估，不是 910B profiling 报告。实机已反馈单卡 eager、TP2 eager、
target-only ACLGraph 能生成正常文本，graph decode 比 eager 更快；尚无完整
吞吐、精度和显存对照数据。以下优先级按当前实现的结构性成本判断，收益需测量。

## 最值得做的性能工作

| 优先级 | 工作 | 当前代码证据及影响 | 验收方式 |
|---|---|---|---|
| P0 | 批量执行及 batch graph buckets | Tensor attention 的 MTP verify 已合并批内请求，`--graph-verify-batches` 配置图的请求数。`--graph-capture-pages` 控制 KV 上下文容量。普通 decode、draft 和 prefill 仍逐条执行。 | 将批量执行扩展至普通 decode 和 draft，支持不同长度和空槽，比较并发 1/4/8/16 的输出吞吐及 p95 TPOT。 |
| P0 | 原生 paged attention 接入 graph | `ops.py::tensor_attention` 每层 `index_select` 搬运历史 KV、`repeat_interleave` 扩展 GQA 头，再调用 SDPA；graph 只能减少 Host 下发，无法消除这些显存读写。已有 `--attention native` 只用于 eager target，draft 仍用 tensor 路径。 | 先校验 FIA 在分块 prefill、GQA、变长、verify 和跨页时的结果，再解决设备长度元数据或任务参数更新；同长度、精度、并发下测时间和峰值显存。 |
| P0 | GDN recurrent、chunk prefill 与卷积融合 | `model.py::delta_rule` 逐 token 做多次 FP32 逐元素运算和归约；参考 prefill 的 token 循环在每个 GDN 层执行。卷积用 unfold、乘法、归约组合。原生 recurrent probe 仍返回零，不能直接打开。 | 先定位 recurrent probe 的状态布局、序列边界和写回语义；逐步替换 prefill/recurrent/conv，覆盖非零初始状态、连续 chunk、MTP 两步与拒绝回滚。 |
| P1 | 减少每步同步、元数据传输及状态复制 | 每行新建 token/position/page/slot tensor，MTP `.tolist()`、`.item()` 取回结果，step 末尾设备 synchronize；全量 recurrent/conv/hidden 又 clone，图输入还 copy 一遍。 | 固定元数据 staging buffer、批量 H2D、设备采样/接受判定、预分配状态池或双缓冲；先保证 lease 完成后再回收，不能直接删同步。 |
| P1 | 融合投影、norm、RoPE、激活和 LM head 路径 | Q/K/V、gate/up、GDN 多路投影分开调用；norm/rotary 使用多次小算子。LM head 和采样仍在图外，未采样的 prefill chunk 也计算最后一个 token 的 logits。 | 先跳过无需输出的 LM head，再做 QKV/gate-up 融合与原生 norm/RoPE；W8A8 同时验证权重/scale 排列，测小 batch 延迟。 |
| P1 | TP 图模式与通信优化 | graph 被限制为单卡；TP 每层 attention/GDN 和 FFN 后同步 all-reduce，embedding/LM head 复制在各 rank。 | 先把一致的 batch/graph 调度接入 TP2/TP4，检验不同页长度和 MTP 分支的 collective 顺序，再考虑通信计算重叠与 vocab parallel。 |
| P1 | 图与缓存显存规划 | 按页数配置缓存，未根据模型权重、GDN state、graph 私有池和 workspace 自动预算。MTP 每个 bucket 三张图，长上下文 tensor attention 会进一步放大中间张量。 | 启动记录权重、KV、状态、每 bucket graph 的显存增量；增加预留空间和启动容量报告，OOM 在接流量前发现。暂不盲目共享 draft/verify 图池，因为 verify 输出需要跨 draft replay 存活。 |
| P2 | 混合模型 prefix cache | 服务明确 `prefix_cache=False`，recurrent 状态只保存在 sequence 的 live map 中；不能仅打开 Engine 开关就得到安全前缀复用。 | 同时保存和恢复 KV、GDN、卷积、hidden，启用 MTP 时也需 draft 缓存。重复系统提示词工作负载衡量 TTFT。 |
| P2 | overlap 与调度预算调优 | 服务 `enable_overlap=False`；Ascend CLI 未暴露总 prefill token budget，虽然 Rust 调度器有相关能力。 | 完成异步生命周期协议后再启用 overlap；分别测长 prefill 对已有 decode 的 p95/p99 影响，加入总 token budget 与混合负载测试。 |

低并发低延迟优先做原生 GDN、attention 和图内/图外的小算子成本。
AISBench 并发吞吐优先做真正的 batch 执行。大模型多卡优先考虑 TP graph、
通信和显存预算。MTP 是否加速取决于接受率与 draft/verify 成本，不能只看
`accepted/drafted`，还要比较同一输出长度的端到端耗时。

## 需要补齐的正确性与运维缺口

1. **缺少真实权重的独立精度基线。** 输出通顺不能排除 RoPE、GDN 状态、
   TP 分片、量化或者 MTP hidden 对齐的问题。现有受控测试用小模型覆盖路径，
   应增加 Transformers 或经验证的参考实现的 logits/层输出对照，覆盖 Qwen3.5
   和两个 27B checkpoint。先分离“与 eager 一致”和“与正确模型一致”。
2. **实机回归矩阵还不完整。** BF16/W8A8、target/MTP、eager/graph、TP/DP
   的支持声明应分别标记已实现与已验证。补齐混合长度、多轮页复用、取消、
   EOS、输出预算为 1、跨页接受/拒绝、两副本一侧空闲和 worker 失败测试。
   本次增加 graph 路由、非零状态与原生 NPU capture 测试，不能替代真实模型回归。
3. **量化描述需要更严格地核验。** `checkpoint.py` 只遍历实际构建的模块查询
   描述条目，未对所有描述 key 做消费核对。未匹配/拼错的条目可能未被诊断；
   需区分明确允许忽略的视觉/MTP 条目和无法识别的文本层条目，并输出层级清单。
   当前只支持对称动态 W8A8，不能据此推断所有 ModelSlim 导出都兼容。
4. **依赖安装仍处于 bring-up 状态。** 根包声明 `torch>=2.13.0`，服务器实际
   使用 2.10，当前用 `--no-deps` 安装。应把共享 Python 路径在 2.10 的回归做齐，
   调整基础包约束，并让 CUDA 模型包保留自己的限制，固定可复现安装组合。
5. **容量和状态开销没有统一计入启动预算。** GDN live state 按请求单独分配，
   FP32 recurrent 约为 `linear_layers × value_heads × key_dim × value_dim × 4`
   字节/请求，还不含 clone、图输入/输出、卷积和临时空间。KV 页数够不代表显存够。
6. **故障时的诊断粒度有限。** 原生 Engine 在执行契约破坏时 poison/quarantine
   是正确保护，不应绕过；需要持久记录 graph kind/bucket、阶段、rank、缓存容量
   和软件版本，并补服务级故障响应/退出回归。自动从已经部分写入缓存的 graph
   失败重试 eager 不安全，本次仍让错误向上抛出。
7. **可观测性尚不足以指导优化。** 当前有通用 HTTP/Engine 指标与 spec 计数，
   graph replay/fallback 和显存主要在日志/退出时汇总；应增加按 kind/bucket
   的捕获耗时、回放/回退次数、draft/verify/commit 时间、KV/state 使用量。
   用 profiling 时分清 tensor 运算、数据传输、Host gap 与 HCCL 等待。

## 本次 MTP ACLGraph 的边界

单卡 ACLGraph 支持 greedy MTP，`--mtp` 默认一个草稿 token，
`--num-speculative-tokens K` 配置重复运行同一个 MTP head 的步数。启动预捕获建立
target、draft、verify 图，verify 为每个前缀保留状态以处理拒绝。MTP 权重
缺失时仍在加载阶段报错。prefill、LM head、接受判断、提交和采样仍在图外。
TP graph、native attention graph、`--mtp --graph npugraph_ex` 暂未开放。
有启动 bucket 配置时，未覆盖的 bucket 走 eager，不动态捕获。

验收用同一 checkpoint、输入 token、输出预算和精度，依次比较 target eager、
MTP eager、MTP ACLGraph 的 token IDs、结束原因、TTFT、TPOT、总耗时、吞吐、
接受率和峰值显存。先并发 1，再 4/8；先 BF16，再 W8A8。固定 warmup，性能轮
关闭 verbose。应报告失败/取消请求，不能只统计成功请求的吞吐。

实现位置：`models/ascend/resin_models_ascend/{runner,model,checkpoint,serve}.py`、
`resin/backend/ascend/{graphs,ops,runtime}.py`。上述瓶颈判断来自本地源码，
没有将未测量的收益写成性能结论。
