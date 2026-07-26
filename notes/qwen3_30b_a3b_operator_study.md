# Qwen3-30B-A3B 在 vLLM 中的算子学习路线

> 整理日期：2026-07-12  
> 模型：[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)  
> 对应 vLLM 分支：`learn`，整理时 HEAD 为 `9160962493b5`  
> 官方 vLLM 基线：`upstream/main`，整理时为 `8df14cfc8c8a`

本文的主目标是学习 Qwen3-30B-A3B 原始 BF16 MoE 算子的
Triton/CUDA kernel 实现。为了让实际调用链稳定，下面默认使用
`--moe-backend triton`，不把 FlashInfer MoE 作为学习分支。

## 1. 先建立正确的模型画像

官方 `config.json` 和模型卡给出的关键参数如下：

| 项目 | 数值 | 对算子的影响 |
|---|---:|---|
| 总参数量 / 激活参数量 | 30.5B / 3.3B | 性能重点是稀疏 MoE，而不是把 30B 参数全部用于每个 token |
| Decoder 层数 | 48 | 每层均包含 Attention 和 MoE |
| Hidden size | 2048 | Attention、RMSNorm、router 和 expert GEMM 的输入宽度 |
| Q heads / KV heads | 32 / 4 | GQA；KV cache 比标准 MHA 小 |
| Head dim | 128 | Q/K 的 RMSNorm 和 RoPE 都在这个维度上工作 |
| Experts | 128 | router 每个 token 对 128 个 expert 打分 |
| Top-K | 8 | 每个 token 实际进入 8 个 expert |
| MoE intermediate size | 768 | expert 的 `gate/up/down` GEMM 维度 |
| Dense intermediate size | 6144 | 只有 dense MLP 层才使用；本模型 `decoder_sparse_step=1`，正常 48 层均走 MoE |
| 激活 | SiLU/SwiGLU | expert GEMM1 后执行 `silu(gate) * up` |
| 最大位置 | 40960 | 原生有效上下文为 32768，模型配置额外保留输出空间 |
| RoPE theta | 1,000,000 | 使用标准 Qwen3 RoPE；长上下文时可额外启用 YaRN |
| 权重类型 | BF16 | 原始模型应先研究 unquantized/BF16 路径，FP8/AWQ/NVFP4 放到第二阶段 |

官方资料：

- [Qwen3-30B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [原始 config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

## 2. 总调用链

先沿下面这条主线读，不要一开始就陷入某个 CUDA kernel：

```text
Qwen3MoeForCausalLM
  -> Qwen3MoeModel
    -> 48 x Qwen3MoeDecoderLayer
       |
       +-> RMSNorm / fused residual + RMSNorm
       +-> Qwen3MoeAttention
       |    -> QKVParallelLinear
       |    -> Q RMSNorm + K RMSNorm
       |    -> RotaryEmbedding
       |    -> Attention backend (FlashAttention / Triton / FlashInfer ...)
       |    -> RowParallelLinear
       |
       +-> RMSNorm / fused residual + RMSNorm
       +-> Qwen3MoeSparseMoeBlock
            -> router linear: [tokens, 2048] x [2048, 128]
            -> softmax + top-8
            -> token/expert 对齐和 padding
            -> expert gate_up GEMM: 2048 -> 2 x 768
            -> SwiGLU: 1536 -> 768
            -> expert down GEMM: 768 -> 2048
            -> top-8 加权求和
```

模型注册入口：

- `vllm/model_executor/models/registry.py`
- `vllm/model_executor/models/qwen3_moe.py`

`qwen3_moe.py` 是第一优先级。重点读这些类：

- `Qwen3MoeAttention`
- `Qwen3MoeSparseMoeBlock`
- `Qwen3MoeDecoderLayer`
- `Qwen3MoeModel`
- `Qwen3MoeForCausalLM`

## 3. 第一重点：MoE 路径

Qwen3-30B-A3B 与普通 Qwen3 dense 模型最大的差异就在这里。

### 3.1 FusedMoE 的组装与执行框架

按顺序阅读：

1. `vllm/model_executor/models/qwen3_moe.py`
   - `Qwen3MoeSparseMoeBlock.__init__`
   - `Qwen3MoeSparseMoeBlock.forward`
2. `vllm/model_executor/layers/fused_moe/layer.py`
   - 创建 router、parallel config、`FusedMoEConfig`、`RoutedExperts` 和 runner。
3. `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
   - 串起 routing、prepare、expert 执行和 finalize。
4. `vllm/model_executor/layers/fused_moe/routed_experts.py`
   - expert 权重和 quant method 的实际承载层。
5. `vllm/model_executor/layers/fused_moe/modular_kernel.py`
   - 理解当前 modular MoE 的 prepare/finalize 与 experts 接口。
6. `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
   - 原始 BF16 模型的权重布局、后端选择和调用入口。

### 3.2 Router：128 选 8

Qwen3-30B-A3B 使用普通 softmax top-k，不使用 DeepSeek 风格的 grouped top-k 或 correction bias。因此默认走 `FusedTopKRouter`。

阅读顺序：

1. `vllm/model_executor/layers/fused_moe/router/router_factory.py`
2. `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
3. `vllm/_custom_ops.py` 中的 `topk_softmax`
4. `csrc/libtorch_stable/moe/torch_bindings.cpp`
5. `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`

核心输入输出：

```text
router_logits: [num_tokens, 128]
topk_weights:  [num_tokens, 8], float32
topk_ids:      [num_tokens, 8], int32
```

这里有一个容易走错的地方：`moeTopKFuncs.cuh` **不在这个模型的标准
softmax top-8 调用链上**。它当前只被 `grouped_topk_kernels.cu` include，
服务于 grouped top-k。Qwen3-30B-A3B 没有 grouped top-k，因此应重点看
`topk_softmax_kernels.cu` 中的 `topkGating`、
`topkGatingLauncherHelper` 和 `topkGatingKernelLauncher`。

对本模型 `num_experts=128`，launcher 命中编译期特化的：

```cpp
case 128:
    LAUNCH_TOPK(128, 4, 16);
```

这条路径把 softmax、重复 8 次 warp 内 argmax、清除已选最大值，以及
top-8 权重重新归一化放在同一个 `topkGating` kernel 中，不需要
softmax workspace。`moeSoftmax + moeTopK` 是非特化 expert 数量的 fallback，
不是 E=128 的主路径。

### 3.3 Token 对齐：当前打开文件正处于这里

阅读顺序：

1. `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
2. `vllm/_custom_ops.py` 中的 `moe_align_block_size`
3. `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`
4. `csrc/libtorch_stable/moe/moe_ops.h`
5. `csrc/libtorch_stable/moe/torch_bindings.cpp`

它把 `[num_tokens, top_k]` 的 expert id 展平并按 expert 分组，再把每个 expert 的 token 数 padding 到 GEMM block size 的整数倍，产生：

```text
sorted_token_ids
expert_ids                 # 每个 GEMM block 对应哪个 expert
num_tokens_post_padded
```

在 `moe_align_sum_kernels.cu` 中重点看：

- `_moe_align_block_size`
- `_moe_align_block_size_small_batch_expert`
- `moe_align_block_size_kernel`
- `moe_align_block_size_small_batch_expert_kernel`
- `moe_sum_vec_kernel`
- `moe_sum_vec_dynamic_kernel`
- `moe_sum_scalar_kernel`
- host launcher `moe_align_block_size` 和 `moe_sum`

对 Qwen3-30B-A3B 还要结合 `_prepare_expert_assignment` 的分支判断：

- 无 EP 且 `M * 8 * 4 <= 128`，也就是 `M <= 4` 时，走
  `naive_block_assignment`，会完全跳过 `moe_align_block_size`。这常见于
  小 batch decode。
- `M > 4` 或启用 expert map/EP 时才真正调用
  `moe_align_block_size`。
- CUDA launcher 中的 `small_batch_expert_mode` 还要求
  `num_experts <= 64`。本模型有 128 个 experts，所以即使 token 很少也
  不会进入 `_moe_align_block_size_small_batch_expert`。
- 本模型进入 align 时应重点跟踪通用的 `moe_align_block_size_kernel`、
  `count_and_sort_expert_tokens_kernel` 和 `cumsum_buffer`。

需要特别理解两点：

1. 一个 token 被复制到 8 条 expert 路径中，`sorted_token_ids` 实际编码的是展平后的 `token_id * top_k + topk_slot`。
2. 第二次 expert GEMM 后结果形如 `[num_tokens, 8, hidden_size]`，`moe_sum` 把 8 条已乘 router weight 的结果合并为 `[num_tokens, hidden_size]`。
3. `top_k=8` 会在 `moe_sum` launcher 中命中
   `moe_sum_vec_kernel<scalar_t, 8>` 特化；只有内存不满足 16-byte
   vectorize 条件时才回退到 `moe_sum_scalar_kernel`。

### 3.4 Expert GEMM：默认 Triton 路径

阅读顺序：

1. `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`
   - `TritonExperts.apply`
   - `invoke_fused_moe_triton_kernel` 的两次调用
   - `TritonExperts.moe_sum`
2. `vllm/model_executor/layers/fused_moe/fused_moe.py`
   - `fused_moe_kernel`
   - `invoke_fused_moe_triton_kernel`
   - `_prepare_expert_assignment`
   - `try_get_optimal_moe_config`
3. `vllm/model_executor/layers/fused_moe/activation.py`
4. `vllm/model_executor/layers/fused_moe/configs/`

原始 BF16 expert 的核心数据流：

```text
x: [M, 2048]
  -> align top-8 token/expert assignment
  -> w13 grouped GEMM: [128, 1536, 2048]
  -> intermediate: [M, 8, 1536]
  -> SwiGLU
  -> [M, 8, 768]
  -> w2 grouped GEMM: [128, 2048, 768]
  -> [M, 8, 2048]
  -> multiply router weights + moe_sum
  -> [M, 2048]
```

这里的 `M` 是当前 forward 中的 token 数。Decode 时 `M` 较小，prefill 时 `M` 较大，因此同一个模型也需要不同 kernel config。

### 3.5 `fused_moe_kernel` 应该怎样读

`vllm/model_executor/layers/fused_moe/fused_moe.py` 中的
`fused_moe_kernel` 是两次 expert GEMM 共用的 Triton kernel。先只看 BF16
分支，暂时跳过所有 FP8、INT8、INT4、bias 和 `SWAP_AB` 代码。

建议按以下顺序逐段理解：

1. `pid -> (pid_m, pid_n)`：使用 `GROUP_SIZE_M` 重排 program id，改善
   expert 权重的 L2 reuse。
2. `offs_token`：从 `sorted_token_ids` 取出当前 expert block 中的展平
   token/top-k slot；naive decode 路径则直接由 `pid_m` 构造。
3. `off_experts = expert_ids[pid_m]`：一个 program block 只选择一个 expert
   的权重矩阵。
4. `a_ptrs`：通过 `offs_token // top_k` 找回 A 的实际行。
5. `b_ptrs`：通过 `off_experts * stride_be` 定位 expert 权重，再构造
   `[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile。
6. K 循环：`tl.load(A tile)`、`tl.load(B tile)`、`tl.dot`，FP32 累加。
7. 第二次 GEMM 的 `MUL_ROUTED_WEIGHT=True`：从扁平
   `topk_weights[offs_token]` 读取路由权重，在转换回 BF16 前相乘。
8. 输出仍按 `offs_token` 写回，使最后的 `moe_sum` 可以直接沿 top-k 维
   求和。

两次调用使用同一个 kernel，但参数语义不同：

| 项目 | GEMM1 / w13 | GEMM2 / w2 |
|---|---|---|
| A | `[M, 2048]` | `[M*8, 768]` |
| B | `[128, 1536, 2048]` | `[128, 2048, 768]` |
| `top_k` kernel constexpr | 8 | 1 |
| `offs_token // top_k` | 恢复原 token id | 保持展平 token/top-k slot |
| `MUL_ROUTED_WEIGHT` | false | true |
| C | `[M, 8, 1536]` | `[M, 8, 2048]` |

第二次调用传 `top_k=1` 很关键：此时 A 已经是 `[M*8, 768]` 的展平
SwiGLU 输出，不应再除以 8；但 `offs_token` 本身仍携带原始 top-k slot，
因此可以同时索引 `topk_weights` 和最终 `[M, 8, 2048]` workspace。

### 3.6 BF16 MoE 的五类实际 kernel

固定 `--moe-backend triton` 后，学习范围可以收敛为：

| 阶段 | 实际 kernel | 源文件 |
|---|---|---|
| Router | `topkGating<..., 128, ...>` | `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` |
| Assignment | naive，或 align + sort | `fused_moe.py`、`moe_align_sum_kernels.cu` |
| Expert GEMM1/2 | `fused_moe_kernel` | `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| SwiGLU | `act_and_mul_kernel` | `csrc/libtorch_stable/activation_kernels.cu` |
| Reduce | `moe_sum_vec_kernel<..., 8>` | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` |

这五类 kernel 就是本文后续 profiling 和源码精读的边界。

## 4. 第二重点：QK-Norm + GQA Attention

Qwen3 的 Attention 与 Qwen2 的一个重要区别是 Q、K 在 RoPE 前分别做 RMSNorm。

### 4.1 模型层顺序

在 `Qwen3MoeAttention.forward` 中：

```text
QKV projection
  -> split Q/K/V
  -> per-head Q RMSNorm
  -> per-head K RMSNorm
  -> RoPE(Q, K)
  -> attention(Q, K, V) + KV cache
  -> output projection
```

对应 shape：

```text
Q: [tokens, 32 * 128]
K: [tokens,  4 * 128]
V: [tokens,  4 * 128]
```

Tensor Parallel 后 head 数按 rank 切分；当 TP 大于 KV head 数时，KV heads 会复制而不是继续切碎。

### 4.2 RMSNorm 与 residual fusion

阅读顺序：

1. `vllm/model_executor/layers/layernorm.py`
2. `vllm/ir/ops/layernorm.py`
3. `csrc/libtorch_stable/layernorm_kernels.cu`
4. `csrc/libtorch_stable/torch_bindings.cpp`

需要区分：

- Q/K Norm：最后一维是 `head_dim=128`，无 residual。
- Decoder pre/post norm：最后一维是 `hidden_size=2048`，常走 residual-add + RMSNorm 融合。
- 当前代码先表达为 vLLM IR op，实际实现可由 IR/编译后端选择；不要只看 Python reference 就认为线上逐算子执行。

### 4.3 RoPE

阅读顺序：

1. `vllm/model_executor/layers/rotary_embedding/__init__.py`
2. `vllm/model_executor/layers/rotary_embedding/base.py`
3. `vllm/_custom_ops.py` 中的 `rotary_embedding`
4. `csrc/libtorch_stable/pos_encoding_kernels.cu`

默认路径是原地更新 Q/K。若启用 FlashInfer rotary，则会改走 `torch.ops.vllm.flashinfer_rotary_embedding`。

### 4.4 Attention backend

入口和选择逻辑：

- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backends/registry.py`

Qwen3-30B-A3B 是普通 GQA，不是 DeepSeek MLA。CUDA 上优先理解：

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/backends/flashinfer.py`

Attention backend 同样是运行时选择的。学习时用 `--attention-backend FLASH_ATTN` 或 `TRITON_ATTN` 固定后端，再用 profiler 确认 kernel 名称。

## 5. 第三重点：线性层、激活和并行通信

这些代码通用性很强，但要理解整体性能必须读：

- `vllm/model_executor/layers/linear.py`
  - `QKVParallelLinear`
  - `MergedColumnParallelLinear`
  - `RowParallelLinear`
  - `ReplicatedLinear`
- `vllm/model_executor/layers/activation.py`
  - `SiluAndMul`
- `csrc/libtorch_stable/activation_kernels.cu`
- `vllm/distributed/`

在 Qwen3-30B-A3B 中：

- Attention 的 QKV projection 是 column/tensor parallel。
- Attention output projection 是 row parallel，后面可能产生 all-reduce。
- Router 是 replicated linear，使各 rank 得到一致的 expert 选择。
- MoE 可使用 TP、EP，或者 DP+EP all-to-all；是否启用 EP 会显著改变调用链。

研究单卡算子时先关闭 EP，把 routing、align、两次 GEMM、SwiGLU、sum 看明白；随后再进入：

- `vllm/model_executor/layers/fused_moe/prepare_finalize/`
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`
- `docs/serving/expert_parallel_deployment.md`

## 6. 量化路径放在第二阶段

原模型是 BF16。掌握 BF16 后再对照研究：

| 方向 | 重点目录/文件 |
|---|---|
| FP8 MoE | `vllm/model_executor/layers/fused_moe/oracle/fp8.py`、`experts/cutlass_moe.py`、`experts/deep_gemm_moe.py` |
| AWQ/GPTQ/Marlin | `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`、`csrc/quantization/` |
| NVFP4 | `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`、`experts/trtllm_nvfp4_moe.py`、`experts/cutlass_moe.py` |

仓库已经提供大量 Qwen3-30B-A3B 的后端组合配置，可参考：

- `tests/evals/gsm8k/configs/moe-refactor/`
- `tests/evals/gsm8k/configs/moe-refactor-dp-ep/`
- `tests/evals/gsm8k/configs/humming/`

不要把某个量化模型的 kernel 路径误认为原始 BF16 模型的默认路径。

## 7. 推荐的实际学习顺序

### 阶段 A：一层模型结构

1. 从 `Qwen3MoeDecoderLayer.forward` 手画一层数据流。
2. 标出所有 shape：2048、32/4 heads、128 head dim、128 experts、top-8、768 expert dim。
3. 确认 residual、RMSNorm、Attention、MoE 的先后顺序。

### 阶段 B：完整 MoE 单卡路径

1. `Qwen3MoeSparseMoeBlock`
2. `FusedMoE`/runner/router
3. `topk_softmax_kernels.cu` 中 E=128 的 `topkGating` 特化
4. `moe_align_block_size.py`
5. `moe_align_sum_kernels.cu`
6. `TritonExperts.apply`
7. `fused_moe_kernel`
8. `moe_sum`

### 阶段 C：Attention

1. QKV projection 与 TP head 切分
2. Q/K RMSNorm
3. RoPE
4. KV cache layout
5. prefill 和 decode 的不同 Attention kernel

### 阶段 D：并行与量化

1. TP 下的 all-reduce
2. EP 下的 expert map 和 all-to-all
3. FP8/INT4/NVFP4 的权重布局、量化 scale 与 backend oracle

## 8. 定位、测试与 benchmark 命令

快速查调用链：

```bash
rg -n "Qwen3Moe|FusedMoE|topk_softmax|moe_align_block_size|moe_sum" \
  vllm/model_executor vllm/_custom_ops.py csrc/libtorch_stable/moe
```

查看模型专用 MoE benchmark shape：

```bash
rg -n "Qwen3-30B-A3B|E=128.*N=768|N=768.*K=2048" \
  benchmarks/kernels tests/kernels
```

重点 benchmark：

```bash
uv run benchmarks/kernels/benchmark_moe.py --help
uv run benchmarks/kernels/benchmark_moe_align_block_size.py --help
uv run benchmarks/kernels/benchmark_fused_topk.py --help
uv run benchmarks/kernels/benchmark_rmsnorm.py --help
uv run benchmarks/kernels/benchmark_paged_attention.py --help
```

重点正确性测试：

```bash
.venv/bin/python -m pytest tests/kernels/moe/test_fused_topk.py -v
.venv/bin/python -m pytest tests/kernels/moe/test_moe_align_block_size.py -v
.venv/bin/python -m pytest tests/kernels/moe/test_moe.py -v
.venv/bin/python -m pytest tests/kernels/core/test_layernorm.py -v
.venv/bin/python -m pytest tests/kernels/core/test_rotary_embedding.py -v
```

实际 profiling 时建议分别固定：

```text
--moe-backend triton
--attention-backend FLASH_ATTN
```

然后分别采集 prefill 和 decode，确认以下 kernel 是否出现：

```text
topk_softmax
moe_align_block_size / count_and_sort_expert_tokens (M > 4 或 EP)
fused_moe_kernel (w13)
act_and_mul_kernel (silu_and_mul)
fused_moe_kernel (w2)
moe_sum_vec_kernel<..., 8>
rms_norm / fused_add_rms_norm
rotary_embedding
attention prefill/decode kernel
```

## 9. Git 上游与 learn 分支结论

仓库 remote：

```text
origin   git@github.com:zzb610/vllm.git
upstream https://github.com/vllm-project/vllm.git
```

因此：

- 官方上游仓库是 `https://github.com/vllm-project/vllm.git`。
- `origin` 是个人 fork `zzb610/vllm`。
- `learn` 没有设置 `branch.learn.remote/merge` tracking 配置。
- 但提交历史明确表明 `learn` 以官方 `upstream/main` 为学习基线，并周期性 merge `upstream/main`。
- 2026-07-12 已将最新 `upstream/main@8df14cfc8c8a` 合并到本地 `learn`。
- 更新后的 merge commit 是 `9160962493b5`；此时 `learn` 相对 `upstream/main` 为 ahead 3、behind 0。
- 未推送 `origin/learn`，因为“更新本地学习分支”不等于授权发布远端。

复核命令：

```bash
git remote -v
git branch -vv
git log -1 --decorate --oneline
git rev-list --left-right --count learn...upstream/main
```
