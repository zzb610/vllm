# vLLM 源码学习：PD 分离里的 NixlConnector

这篇笔记对应当前仓库里的 vLLM v1 代码。旧版笔记里的主线还是对的：PD 分离把 prefill 和 decode 拆开，NIXL 负责在两边搬 KV cache。但代码已经变了很多，最明显的是 `NixlConnector` 不再是一个大文件里的单一路径。现在它被拆成了 scheduler、worker、metadata、pull、push、TP mapping 等模块，默认 `NixlConnector` 仍然走 pull/READ；新增的 `NixlPushConnector` 走 push/WRITE。

这不是简单的文件重命名。新版代码把几件事分清了：

- `kv_role` 明确表达实例身份：Prefill 一般是 `kv_producer`，Decode 一般是 `kv_consumer`。`kv_both` 还兼容，但 NIXL 里已经提示不推荐继续用。
- side-channel 只负责 NIXL 握手，不负责 HTTP 请求路由。
- scheduler 负责决定“哪些 token 可以来自远端、要分配哪些本地 block、请求是否进入等待状态”。
- worker 负责注册本地 KV 内存、和远端 agent 握手、提交 NIXL READ 或 WRITE。
- Prefill 侧 KV block 的延迟释放从单纯超时，变成了短租约加 heartbeat 续租。

两张图可以配合看：

- [pd_nixl_module_map.excalidraw](pd_nixl_module_map.excalidraw)：当前 NIXL connector 的模块拆分。
- [pd_nixl_pull_push_flow.excalidraw](pd_nixl_pull_push_flow.excalidraw)：pull/READ 和 push/WRITE 的流程对照。

## 先把几个结论说清楚

`NixlConnector` 现在只是 `NixlPullConnector` 的兼容别名。注册表里同时有 `NixlConnector`、`NixlPullConnector`、`NixlPushConnector`。如果配置里写的是 `"kv_connector": "NixlConnector"`，实际走的是 pull 模式，也就是 Decode worker 主动对 Prefill worker 发 NIXL `READ`。

`NixlPushConnector` 不是把 pull 里的方向简单反过来。它多了 Decode 侧注册、Prefill 侧匹配、`nixl-push-writer` 后台线程、`PUSH_REG:<msgpack>` 通知和注册 watchdog。Decode scheduler 先分配好本地 block，然后让 Decode worker 把“我的 engine、host、port、TP size、本地 block id”发给 Prefill worker。Prefill request 结束后，Prefill worker 匹配注册信息和已完成 block，再用 NIXL `WRITE` 写到 Decode 的本地 KV cache。

`kv_transfer_params` 是 HTTP 控制面和 NIXL 数据面的桥。proxy 先把请求发给 Prefill，Prefill 在输出里返回 `remote_block_ids`、`remote_engine_id`、`remote_request_id`、`remote_host`、`remote_port`、`tp_size`、`remote_num_tokens` 等字段。proxy 再把这些字段带到 Decode 请求。NIXL side-channel 不参与这一步。

Decode 请求进入 `WAITING_FOR_REMOTE_KVS` 时，`request.num_computed_tokens` 已经被 scheduler 逻辑上推进了。但对应 block 还没有写入 prefix cache。等 worker 上报 `finished_recving` 后，下一轮 scheduler 才会调用 `_update_waiting_for_remote_kv()`，把成功加载的 block 缓存起来，再把请求放回 `WAITING` 或 `PREEMPTED`。

传输失败现在也不只是等超时。worker 会上报 `invalid_block_ids`，scheduler 再根据 `kv_load_failure_policy` 选择失败请求，或者截断 `num_computed_tokens` 后重算失败 block。

## 读源码时先看哪些文件

主线文件如下：

- `vllm/config/kv_transfer.py`
- `vllm/distributed/kv_transfer/kv_connector/factory.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu/kv_connector.py`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`

对应的测试和文档也值得看：

- `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py`
- `tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py`
- `tests/v1/kv_connector/unit/test_nixl_connector.py`
- `tests/v1/kv_connector/unit/test_nixl_push_connector.py`
- `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`
- `docs/features/nixl_connector_usage.md`
- `docs/features/nixl_connector_compatibility.md`
- `docs/design/nixl_kv_cache_lease.md`
- `docs/design/nixl_kv_push_connector.md`

### 用 understand-anything 的图谱视角做覆盖检查

`understand-anything` 生成的是代码知识图谱。它不适合替代源码阅读，但很适合拿来检查这篇笔记有没有漏掉跨文件关系。对 NIXL 这条线，不建议直接把整个 vLLM 仓库一次性喂进去；仓库里 connector、worker、测试和文档太多，完整图会掺进大量与 PD 分离无关的节点。更实用的做法是以仓库根目录作为 project root，但用 `.understand-anything/.understandignore` 把范围收窄到上面这些文件，再用 `--language zh` 生成中文摘要。

如果后面要正式跑，可以按这个方向：

```bash
/understand /Users/bowenyuchi/Documents/vllm_dev/vllm --full --language zh
```

第一次运行时，skill 会先生成 `.understand-anything/.understandignore` 并要求确认。这里应该只放行 NIXL connector、scheduler/worker 入口、proxy 测试和 NIXL 官方文档。跑完以后，不需要逐个看所有节点，重点看图里有没有形成下面几组边：

- 配置边：`KVTransferConfig` -> `KVConnectorFactory` -> `NixlPullConnector` / `NixlPushConnector`
- 调度边：`Scheduler.schedule()` -> `get_num_new_matched_tokens()` -> `update_state_after_alloc()` -> `build_connector_meta()`
- metadata 边：`NixlConnectorMetadata` 把 scheduler 侧的 `_reqs_need_recv`、`_reqs_need_send`、`push_registrations`、`push_finished_blocks` 交给 worker
- worker 边：`bind_connector_metadata()` -> `start_load_kv()` -> `get_finished()` -> `get_block_ids_with_load_errors()`
- 控制面边：`toy_proxy_server.py` 把 Prefill 输出里的 `kv_transfer_params` 带到 Decode 请求
- 生命周期边：`request_finished()`、heartbeat、lease、finished notification 共同决定 Prefill block 什么时候能释放
- 失败恢复边：worker 的 `invalid_block_ids` 回到 scheduler，再由 `kv_load_failure_policy` 决定 fail 还是 recompute

这几组边能连起来，说明笔记的主线基本完整。如果图里只有 `connector.py` 和 `worker.py` 互相引用，却没有 `Scheduler.schedule()`、`KVConnectorMetadata`、`toy_proxy_server.py`、`request_finished()` 这些节点，那只能说明图谱范围收得太窄，不能拿它判断 PD 分离的完整行为。

用图谱分层看，本文后面的内容可以对应成这样：

| 层 | 关键节点 | 这层回答的问题 |
| --- | --- | --- |
| 配置入口 | `KVTransferConfig`、`KVConnectorFactory`、`connector.py` | 用户配置怎样选到 pull 或 push connector |
| HTTP 控制面 | `toy_proxy_server.py`、`kv_transfer_params` | Prefill 输出怎样变成 Decode 输入 |
| 调度面 | `Scheduler.schedule()`、`Nixl*ConnectorScheduler` | 远端 token 怎样变成 `WAITING_FOR_REMOTE_KVS` |
| metadata 通道 | `NixlConnectorMetadata`、`ReqMeta`、`PushRegistration` | scheduler 怎样把计划交给 worker |
| worker 数据面 | `Nixl*ConnectorWorker`、`NixlBaseConnectorWorker` | 本地 block 怎样注册，READ/WRITE 怎样提交 |
| 兼容和映射 | `tp_mapping.py`、`utils.py`、`NixlAgentMetadata` | TP、block size、layout、HMA/SWA 怎样对齐 |
| 收尾和恢复 | `request_finished()`、heartbeat、`invalid_block_ids` | 成功、失败、abort、多 batch 下 block 怎样回收 |

如果只想顺着一次请求看，建议按这个顺序读：

1. `KVTransferConfig.__post_init__()`
2. `KVConnectorFactory.create_connector()`
3. `Scheduler.schedule()`
4. `NixlPullConnectorScheduler.get_num_new_matched_tokens()`
5. `NixlPullConnectorScheduler.update_state_after_alloc()`
6. `NixlBaseConnectorScheduler.build_connector_meta()`
7. `NixlPullConnectorWorker.start_load_kv()`
8. `NixlPullConnectorWorker._read_blocks_for_req()`
9. `NixlPullConnectorWorker._read_blocks()`
10. `Scheduler._update_from_kv_xfer_finished()`
11. `Scheduler._update_waiting_for_remote_kv()`
12. `NixlPullConnectorScheduler.request_finished()`

Push 模式再加读：

1. `NixlPushConnectorScheduler.update_state_after_alloc()`
2. `NixlPushConnectorScheduler.request_finished()`
3. `NixlPushConnectorWorker._push_writer_loop()`
4. `NixlPushConnectorWorker._handle_push_reg_notif()`
5. `NixlPushConnectorWorker._do_start_push_kv()`
6. `NixlPushConnectorWorker._xfer_blocks()`

## 配置入口：KVTransferConfig 和 factory

`KVTransferConfig` 现在有几类字段：

- 选择 connector：`kv_connector`
- 标识实例：`engine_id`
- 标识角色：`kv_role`
- 选择缓冲区设备：`kv_buffer_device`
- 传 connector 私有配置：`kv_connector_extra_config`
- 失败策略：`kv_load_failure_policy`

`kv_role` 很重要。`KVProducer = "kv_producer" | "kv_both"`，`KVConsumer = "kv_consumer" | "kv_both"`。Prefill 节点通常配置成 producer，Decode 节点通常配置成 consumer。只要设置了 `kv_connector`，就必须设置 `kv_role`。

`KVConnectorFactory` 把名字映射到类：

- `"NixlConnector"` -> `vllm.distributed.kv_transfer.kv_connector.v1.nixl.NixlConnector`
- `"NixlPullConnector"` -> `NixlPullConnector`
- `"NixlPushConnector"` -> `NixlPushConnector`

`connector.py` 里最后一句是关键：

```python
NixlConnector = NixlPullConnector
```

所以旧配置名没坏，但含义要按 pull 来理解。

Factory 创建出来的 connector 分两份：

- scheduler process 里是 `KVConnectorRole.SCHEDULER`
- worker process 里是 `KVConnectorRole.WORKER`

这两个对象不是同一个东西。scheduler 侧不碰 NIXL 内存，worker 侧不做请求调度。中间靠 `KVConnectorMetadata` 传计划。

## HTTP 控制面：proxy 怎么串起 P 和 D

`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` 是最直接的参考。一次普通 PD 请求会被 proxy 拆成两段。

第一段发给 Prefill：

```json
{
  "do_remote_decode": true,
  "do_remote_prefill": false,
  "remote_engine_id": null,
  "remote_block_ids": null,
  "remote_host": null,
  "remote_port": null
}
```

Prefill 只跑 prompt，通常 `max_tokens=1`。请求结束时，scheduler 通过 connector 的 `request_finished()` 返回新的 `kv_transfer_params`。这份参数会出现在 engine output 里，proxy 取出来，再放进第二段 Decode 请求。

第二段发给 Decode：

```json
{
  "do_remote_prefill": true,
  "do_remote_decode": false,
  "remote_block_ids": "...",
  "remote_engine_id": "...",
  "remote_request_id": "...",
  "remote_host": "...",
  "remote_port": 5600,
  "tp_size": 1,
  "remote_num_tokens": 123
}
```

这里的 `remote_request_id` 不能省。Decode worker 读完或确认不需要读以后，给 Prefill worker 发通知时必须用 Prefill 侧 request id，而不是 Decode 侧自己的 request id。测试里专门覆盖了这个问题，避免 MLA 和异构 TP 场景下通知打错对象。

## NIXL side-channel：只做握手

NIXL 数据传输需要远端 agent metadata 和远端内存描述信息。vLLM 用一个 ZMQ side-channel 做这件事。

worker 初始化时会注册本地 KV cache，并生成 `NixlHandshakePayload`。EngineCore 初始化 scheduler 后，会调用 executor 的 `get_kv_connector_handshake_metadata()`，从所有 worker 收集这份 payload。scheduler 再通过 `set_xfer_handshake_metadata_pp_aware()` 把它们交给 NIXL scheduler 侧对象。

`NixlBaseConnectorScheduler.set_xfer_handshake_metadata()` 会启动 `nixl_handshake_listener` 线程。Decode worker 后面要连 Prefill 时，会向 `remote_host:remote_port` 发 `GET_META_MSG` 和目标 TP rank。listener 返回对应 rank 的 `NixlHandshakePayload`。

这个 payload 分两层：

- `NixlHandshakePayload`
  - `compatibility_hash`
  - `agent_metadata_bytes`
- `NixlAgentMetadata`
  - `engine_id`
  - `agent_metadata`
  - `kv_caches_base_addr`
  - `device_id`
  - `num_blocks`
  - `block_lens`
  - `kv_cache_layout`
  - `block_size`
  - `ssm_sizes`
  - `attn_backend_name`
  - `physical_blocks_per_logical_kv_block`

先比 `compatibility_hash`，再解 `NixlAgentMetadata`。这样 P/D 版本或模型配置不兼容时，错误会停在握手阶段，不会等到传输时才暴露。

hash 覆盖的因素包括 vLLM 版本、NIXL connector version、模型名、dtype、KV heads、head size、hidden layers、attention backend、cache dtype、是否 cross-layer blocks、是否启用 HMA。TP size、block size、KV cache layout 没放进 hash，因为它们可以在运行时由 `TransferTopology` 和 handshake 校验处理。

## Scheduler 主流程：远端 KV 如何进入 WAITING_FOR_REMOTE_KVS

入口在 `Scheduler.schedule()` 的 waiting request 调度部分。

当一个 request 第一次被调度，`request.num_computed_tokens == 0`，scheduler 会先问本地 prefix cache：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

如果有 connector，再问 connector：

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, num_new_local_computed_tokens
)
```

`ext_tokens` 表示除了本地 cache 命中之外，还能从远端拿多少 token 的 KV。`load_kv_async=True` 表示这次不是马上跑模型，而是先发起异步 KV load。

接着 scheduler 调 `allocate_slots()`：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    ...
)
```

`delay_cache_blocks=load_kv_async` 是这里的核心。远端 KV 要搬到本地，总得先有本地 block 作为落点；但传输没完成之前，这些 block 不能写入 prefix cache。否则其它请求可能会把一块还没填好的 KV 当成可复用 cache。

分配完成后，scheduler 调：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

NIXL scheduler 会把请求和 block id 存到 `_reqs_need_recv` 或 push 的注册表里。随后 scheduler 发现 `load_kv_async=True`，就把请求设成：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
request.num_computed_tokens = num_computed_tokens
```

这个 `num_computed_tokens` 是逻辑上的。它告诉 scheduler：“等 KV 回来以后，可以从这里继续调度”。真正 cache blocks 的动作要等到后面。

## Metadata：scheduler 怎么把计划交给 worker

`NixlBaseConnectorScheduler.build_connector_meta()` 会生成 `NixlConnectorMetadata`。这份对象是 scheduler 到 worker 的唯一正式通道。

主要字段：

- `reqs_to_recv`: worker 需要加载远端 KV 的请求
- `reqs_to_save`: 使用 host buffer 时，Prefill 侧需要先把 device KV 拷到 host buffer
- `reqs_to_send`: Prefill 侧或 bidirectional D 侧已经完成、但 block 暂时不能释放的请求和过期时间
- `reqs_in_batch`: 本轮参与过 batch 的请求，用来配合租约和完成通知
- `reqs_not_processed`: aborted 或不再需要处理的请求
- `heartbeat_by_engine`: Decode 侧发给 Prefill 侧的 heartbeat 批次
- `push_registrations`: push 模式下 Decode worker 要发给 Prefill worker 的注册数据
- `push_finished_blocks`: push 模式下 Prefill worker 要匹配和发送的已完成 blocks

worker 侧的调用点有两个版本。GPU worker 走 `vllm/v1/worker/gpu/kv_connector.py`，其它 model runner 也可以通过 `KVConnectorModelRunnerMixin` 走同样生命周期：

1. `bind_connector_metadata()`
2. `start_load_kv()`
3. 正常 forward，或者 no-forward 时只推进 KV connector
4. `wait_for_save()`
5. `get_finished()`
6. `get_block_ids_with_load_errors()`
7. `clear_connector_metadata()`

NIXL 不是 layer-wise connector，`wait_for_layer_load()` 和 `save_kv_layer()` 是空实现；真正的异步传输在 `start_load_kv()` 里提交，在 `get_finished()` 里轮询。

## Pull 模式：默认 NixlConnector 的 READ 路径

Pull 模式下，Decode worker 从 Prefill worker 读 KV。关键类是：

- `NixlPullConnector`
- `NixlPullConnectorScheduler`
- `NixlPullConnectorWorker`

### Decode 侧：get_num_new_matched_tokens

普通 PD Decode 请求带着 `do_remote_prefill=True`。`NixlPullConnectorScheduler.get_num_new_matched_tokens()` 看到这个字段，就按 prompt 长度算可远端加载 token 数。

对普通 attention 模型，远端 token 数基本就是 prompt token 数减本地命中 token 数。对 Mamba hybrid 模型，有一个特殊处理：Decode 必须重算最后一个 token，所以 `_mamba_prefill_token_count()` 会返回 `N - 1`。

同一个函数里还有 bidirectional KV transfer 的逻辑。多轮对话里，第二轮开始 P 可能拿到 D 上一轮缓存的 block。这时请求在 P 侧仍然带 `do_remote_decode=True`，但也带了 `remote_block_ids` 等字段。scheduler 会计算从 D 拉回来的 token 数；如果小于 `kv_recompute_threshold`，就不拉，直接本地重算，避免传输成本超过收益。

### Decode 侧：update_state_after_alloc

当 `do_remote_prefill=True` 且有 `remote_block_ids`，pull scheduler 会：

1. 取本地已经分配好的 unhashed block id。
2. 如果 HMA/SWA 需要，调用 `get_sw_clipped_blocks()` 裁掉滑窗外 block。
3. 写入 `_reqs_need_recv[request_id] = (request, local_block_ids)`。
4. 把 `params["do_remote_prefill"] = False`。
5. 把 `params["_remote_blocks_processed"] = True`。

后两个字段是防重复的。请求因为 preemption 或下一轮调度再次进入这段代码时，不会重复生成 recv 任务。

如果本地 prefix cache 已经全命中，`num_external_tokens` 可能是 0。这时仍然可能进 `_reqs_need_recv`，但 `local_block_ids` 为空。worker 后面不会发 READ，只会通知 Prefill：这批远端 block 可以释放。

### Worker 侧：start_load_kv

`NixlPullConnectorWorker.start_load_kv()` 处理 `metadata.reqs_to_recv`。

每个请求先把本地逻辑 block id 映射成 kernel 物理 block id：

```python
meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
    meta.local_block_ids
)
```

然后检查远端 engine 是否握过手：

- 没握过手：调用 `_background_nixl_handshake()`，提交到单线程 executor，握手完成后把请求放进 `_ready_requests`。
- 已握手：直接调用 `_read_blocks_for_req()`。

函数末尾还会把 `_ready_requests` 里已经握手完成的请求拿出来，继续发起 READ。

### Worker 侧：_read_blocks_for_req

`_read_blocks_for_req()` 主要做 TP mapping。

NIXL 支持 P/D tensor parallel size 不同。`compute_tp_mapping()` 会告诉当前本地 rank 应该从哪些远端 TP rank 读。如果 `D_TP > P_TP`，多个 D rank 会从同一个 P rank 读取不同 KV head slice。如果 `P_TP > D_TP`，一个 D rank 可能要从多个 P rank 读。MLA 由于 KV replicated，可以少读一些冗余 rank。

函数为每个远端 rank 构造 `ReadSpec`：

```python
ReadSpec(
    remote_rank=rank,
    local_block_ids=[...],
    remote_block_ids=[...],
)
```

然后选择本地 dlist handle 和远端 dlist handle，进入 `_read_blocks()`。

### Worker 侧：_read_blocks

`_read_blocks()` 是真正下发 NIXL READ 的地方。

它会先处理几类 block id 差异：

- P/D block size 不同：用 `get_mapped_blocks()` 展开本地 block。
- 逻辑 block 和 kernel block 不同：用 `physical_blocks_per_logical_kv_block` 做映射。
- 本地 prefix cache 部分命中：`_apply_prefix_caching()` 会裁掉远端前缀，只读未命中的尾部 block。
- Mamba hybrid：不能简单按 token 前缀裁，因为 SSM state 不是普通 per-token KV。

随后计算 NIXL descriptor id：

```python
remote_block_descs_ids = self._compute_desc_ids(...)
local_block_descs_ids = self._compute_desc_ids(...)
```

最后提交异步传输：

```python
handle = self.nixl_wrapper.make_prepped_xfer(
    "READ",
    local_xfer_side_handle,
    local_block_descs_ids,
    remote_xfer_side_handle,
    remote_block_descs_ids,
    notif_msg=notif_id,
)
self.nixl_wrapper.transfer(handle)
self._recving_transfers[request_id].append(handle)
```

`notif_id` 是：

```python
f"{remote_request_id}:{self.world_size}".encode()
```

这里用的是 Prefill 侧的 `remote_request_id`。`:self.world_size` 告诉 Prefill 侧有多少 consumer TP worker，Prefill 侧据此判断是否所有应读的消费者都完成了。

如果 `local_block_ids` 为空，说明 Decode 本地已经全命中，不需要读。worker 会直接 `send_notif()`，让 Prefill 可以释放那边的 block。

## Prefill 侧：request_finished 和延迟释放

Prefill 请求结束时，scheduler 会调用 `_connector_finished()`，最后进 NIXL scheduler 的 `request_finished()`。

Pull 模式里，它会判断：

```python
is_p_node = bool(params.get("do_remote_decode"))
is_d_node = not is_p_node
```

普通 Prefill 请求是 `do_remote_decode=True`，所以 `is_p_node=True`。如果请求是正常结束，并且有 block id，connector 会：

1. 设置 `delay_free_blocks=True`，告诉 scheduler 不能马上释放 block。
2. 在 `_reqs_need_send[request_id]` 里记录过期时间。
3. 通过 `get_sw_clipped_blocks()` 裁剪 HMA/SWA block。
4. 返回给上层一份新的 `kv_transfer_params`：

```python
{
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": block_ids,
    "remote_engine_id": self.engine_id,
    "remote_request_id": request.request_id,
    "remote_host": self.side_channel_host,
    "remote_port": self.side_channel_port,
    "tp_size": tensor_parallel_size,
    "remote_num_tokens": request.num_computed_tokens,
}
```

这些字段会跟着 response 回到 proxy，再进入 Decode 请求。

真正释放 Prefill blocks 的条件有两个：

- Decode worker 发来的完成通知已经满足 fan-in 数量。
- 租约过期。

新版默认不是等 480 秒。`kv_lease_duration` 默认 30 秒，Decode 侧会在等待队列里周期性 heartbeat，Prefill 侧收到 `"HB:req1,req2"` 后延长租约。Decode 崩掉或网络断了，heartbeat 停止，Prefill 很快释放 block；Decode 只是排队久，heartbeat 会保住 block。

heartbeat 从 `NixlBaseConnectorScheduler.on_new_request()` 开始追踪。只要 Decode 请求带 `do_remote_prefill=True` 且字段完整，scheduler 就按 `remote_engine_id` 分组记录。`build_connector_meta()` 按 `kv_lease_duration // 6` 的间隔把 heartbeat 数据交给 worker，worker 在 `_send_heartbeats()` 里通过 NIXL notification 发给 Prefill。

## Decode 侧：finished_recving 之后不是立刻运行

worker 的 `get_finished()` 会轮询 `_recving_transfers`。每个 NIXL handle 状态是：

- `"DONE"`：记录 telemetry，释放 handle。
- `"PROC"`：继续等。
- 其它状态或异常：标记失败，释放 handle，把 block id 放进 `_invalid_block_ids`。

所有 handle 完成后，worker 把 request id 放进 `finished_recving`。executor 侧会用 `KVOutputAggregator` 聚合所有 worker 的输出。只有所有相关 worker 都报完成，scheduler 才能看到这个 request 的 `finished_recving`。

`Scheduler.update_from_output()` 收到 `kv_connector_output` 后，会调用 `_update_from_kv_xfer_finished()`。这里也只是把 request id 放进 `finished_recving_kv_req_ids`。

下一轮 `schedule()` 遍历 waiting queue 时，遇到 `WAITING_FOR_REMOTE_KVS`，会调用 `_try_promote_blocked_waiting_request()`。如果 request id 已在 `finished_recving_kv_req_ids`，才进入 `_update_waiting_for_remote_kv()`：

- 成功路径：`cache_blocks(request, request.num_computed_tokens)`。
- 如果刚好 full prompt hit：把 `num_computed_tokens` 回退到 `num_tokens - 1`，让模型重算最后一个 token 来采样下一个 token。
- 失败恢复路径：如果 request 在 `failed_recving_kv_req_ids`，只缓存仍然有效的前缀，或者释放已分配 block。

最后状态回到 `WAITING` 或 `PREEMPTED`，请求才会在后续调度中继续跑。

## Push 模式：NixlPushConnector 的 WRITE 路径

Push 模式的目标是让 Prefill 主动写入 Decode 预分配的 KV block。它复用 base scheduler/base worker 的大部分能力，但多了一套注册和匹配机制。

关键类：

- `NixlPushConnector`
- `NixlPushConnectorScheduler`
- `NixlPushConnectorWorker`

Push 模式目前不支持 `bidirectional_kv_xfer`，构造 scheduler 时会直接 `NotImplementedError`。

### D scheduler：先注册本地落点

Decode 请求进来时仍然是 `do_remote_prefill=True`。`NixlPushConnectorScheduler.get_num_new_matched_tokens()` 也返回需要远端加载的 token 数，并让 scheduler 进入异步 load。

`update_state_after_alloc()` 分配完本地 block 后，不会准备 READ。它会把注册数据放进 `_push_pending_registrations`：

```python
{
    "request_id": request.request_id,
    "decode_engine_id": self.engine_id,
    "decode_host": self.side_channel_host,
    "decode_port": self.side_channel_port,
    "decode_tp_size": tensor_parallel_size,
    "local_block_ids": local_block_ids,
    "remote_engine_id": params["remote_engine_id"],
    "remote_host": params["remote_host"],
    "remote_port": params["remote_port"],
    "remote_tp_size": params["tp_size"],
}
```

同时它会启动 `_push_registration_deadlines` watchdog。如果 Decode 已经注册，但长时间没等到 Prefill 的 WRITE completion，`build_connector_meta()` 会清理过期注册，避免反复发送。

为了复用 base metadata 的结构，push scheduler 还会把 `params["remote_block_ids"] = ()`，并把请求放进 `_reqs_need_recv`。实际远端 block id 由 Prefill worker 在 WRITE 时决定。

### D worker：发送 PUSH_REG

`NixlPushConnectorWorker` 有一个后台线程，名字是 `nixl-push-writer`。worker 主线程在 `start_load_kv()` 里只做轻量处理：

- 把 `reqs_to_recv` 记录到 `_recving_metadata`。
- 把 `metadata.push_registrations` 放进 `_reg_send_inbox`。
- 如果有新数据，唤醒 writer。

writer 线程从 `_reg_send_inbox` 取注册数据。若还没和 Prefill 握手，先 `_ensure_handshake()`；握手完成后重新入队，再由 writer 调 `send_notif()`。

发送格式是：

```text
PUSH_REG:<msgpack-encoded dict>
```

这条消息走 NIXL notification，不走 HTTP，也不走 scheduler 的 ZMQ side-channel。

### P scheduler：请求结束后暂存 blocks

Prefill 侧 request 正常结束时，`NixlPushConnectorScheduler.request_finished()` 会：

1. 设置租约，延迟释放 blocks。
2. 把 block id 放进 `_finished_request_blocks`，用于 `has_pending_push_work()` 判断还有没有未完成 push。
3. 把 block id 放进 `_newly_finished_push_blocks`，下一轮 `build_connector_meta()` 交给 P worker。
4. 返回 `kv_transfer_params` 给 proxy，字段和 pull 模式保持一致。

`has_pending_push_work()` 很重要。Push 模式下，有时 engine 这一轮没有模型 forward，但后台 writer 还有待匹配或待提交的 WRITE。scheduler 需要继续 step，让 worker 的 `get_finished()` 和 writer 有机会推进。

### P writer：匹配注册和 blocks，提交 WRITE

P worker 的 writer 线程维护两个表：

- `_pending_d_registrations`: D 已经发来注册，但 P blocks 还没到。
- `_push_finished_blocks`: P blocks 已经完成，但 D 注册还没到。

两边谁先到都可以。writer 匹配 request id 时，先精确匹配；匹配不到，再用 `get_base_request_id()` 去掉尾部 8 位随机后缀再比。这是因为 P/D 两段 HTTP 请求共享 `X-Request-Id`，但每个 engine 内部还会追加自己的随机 suffix。

匹配成功后，P writer 调 `_do_start_push_kv()`：

1. 确保 P 已经和 Decode worker 握手。
2. 把 P 侧 block id 和 D 注册里的 block id 都规范成分组结构。
3. 构造 `ReqMeta`。
4. 调 `_xfer_blocks_for_req()`。

`_xfer_blocks()` 最终提交：

```python
handle = self.nixl_wrapper.make_prepped_xfer(
    "WRITE",
    local_xfer_side_handle,
    local_block_descs_ids,
    remote_xfer_side_handle,
    remote_block_descs_ids,
    notif_msg=notif_id,
)
self.nixl_wrapper.transfer(handle)
```

这个 `notif_id` 用的是 Decode 侧 request id。WRITE 完成后，Decode 侧收到 completion notif，会把空 handle entry 塞进 `_recving_transfers`，下一次 `get_finished()` 报 `finished_recving`。Prefill 侧自己也会轮询 `_sending_transfers`，WRITE 完成后报 `finished_sending`，scheduler 再释放 Prefill blocks。

## Worker 初始化：本地 KV cache 如何注册给 NIXL

`NixlBaseConnectorWorker.register_kv_caches()` 是 worker 侧最厚的一层。

它先创建 `TransferTopology`。这个对象记录本地 TP rank、world size、block size、engine id、是否 MLA、KV heads、attention backend、tensor shape、是否 Mamba。后面的 TP mapping、block size ratio、KV head split 都靠它。

然后计算 compatibility hash。如果 `prefer_cross_layer_blocks` 生效，hash 里也会包含这个信息。

接着决定传输缓冲区：

- `kv_buffer_device == "cuda"` 或 `"xpu"`：直接注册 device KV cache。
- `kv_buffer_device == "cpu"`：初始化 `host_xfer_buffers`，通过 `copy_kv_blocks` 在 device 和 host 间搬 block。
- CPU 平台上不会再额外使用 host buffer。

支持的组合在 `nixl/utils.py` 的 `_NIXL_SUPPORTED_DEVICE` 里：

```python
{
    "cuda": ("cuda", "cpu"),
    "tpu": ("cpu",),
    "xpu": ("cpu", "xpu"),
    "cpu": ("cpu",),
}
```

KV cache 注册时，代码会为每个 region 生成 `(base_addr, size, device_id, "")`，再交给：

```python
descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
```

之后还会为本地内存准备 xfer descriptor list：

```python
self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
    self.register_local_xfer_handler(self.block_size)
)
```

最后把 `NixlAgentMetadata` 编成 `NixlHandshakePayload`，等待 scheduler 收集并通过 side-channel 对外服务。

有一个优化路径值得单独记：如果多个 KV tensor 是同一个 backing storage 的不同 view，并且不是 Mamba，代码会走 `_register_packed_kv_cache()`。这种 packed allocation 每个 block 上跨层数据更连续，NIXL descriptor 数量也更少。

## HND、cross-layer blocks、HMA 和 SWA

NIXL 默认更喜欢 HND。`NixlBaseConnector.get_required_kvcache_layout()` 对非 MLA 模型返回 `"HND"`，日志也会提示设置 HND 是为了更好的传输性能。

`prefer_cross_layer_blocks` 的条件更苛刻：

- 不能有 MambaSpec。
- attention backend 必须是 `FLASH_ATTN`、`FLASHINFER` 或 `TRITON_ATTN`。
- 当前 KV cache layout 必须是 HND。
- `kv_connector_extra_config.enable_cross_layers_blocks` 要显式设成 `"true"`。

HMA 相关逻辑贯穿 scheduler 和 worker。`BlockIds` 在当前代码里不是单个 list，而是 `tuple[list[int], ...] | list[list[int]]`。每个 group 有自己的 block id。NIXL 的 metadata、descriptor id 计算、SWA clipping 都按 group 处理。

SWA 的裁剪在 `NixlBaseConnectorScheduler.get_sw_clipped_blocks()`。它按每个 group 的 sliding window 算最多保留多少 block。这样 Prefill 返回或 Decode 接收时，不会把滑窗外 block 误当成还需要传。

## Mamba hybrid 的特殊处理

Mamba hybrid 模型不是普通 KV cache。注意三个地方。

Decode 远端 prefill token 数用 `N - 1`。Prefill 侧会通过 `_truncate_mamba_request_for_prefill()` 去掉最后一个 prompt token，只计算到 `h(N-1)`；Decode 再重算最后一个 token 得到正确状态。

Mamba 的 conv state 传输要求 `VLLM_SSM_CONV_STATE_LAYOUT=DS`。`NixlBaseConnectorWorker.__init__()` 里会 assert 这个布局。

Mamba 的 descriptor 不是简单 K/V 两块。代码用 `derive_mamba_conv_split()` 把 conv 子投影和 SSM temporal state 拆开。`_build_mamba_local()` 和 `_build_mamba_remote()` 会生成多段 descriptor。当前注释里也写得很直白：Mamba 3-read transfer 对 block size ratio 不等于 1 的情况还没有测试。

## 异构 TP 和 block size 差异

异构 TP 的核心在 `tp_mapping.py`。

`compute_tp_mapping()` 根据本地 TP size、远端 TP size、KV heads 和 group spec 类型，算出：

- 当前 rank 要从哪些 remote rank 读。
- 每个 group 对应哪些 source ranks。
- attention head slot 的偏移。
- D_TP > P_TP 时的 head split offset。

Full attention 下，TP 不同时需要按 KV head 切片。MLA 的 KV cache 是 replicated，很多时候不用按 head 切。Mamba 的 SSM 和 attention group 又有自己的 source rank 选择。

block size 差异则靠 `block_size_ratio` 和 `physical_blocks_per_logical_kv_block`。例如 remote block size 比 local 小，local 一个逻辑 block 可能对应 remote 多个物理 kernel block。`_logical_to_kernel_block_ids()`、`_logical_to_remote_kernel_block_ids()`、`get_mapped_blocks()` 都是在处理这类映射。

限制也很清楚：

- HMA 要求 block size ratio 为 1。
- 非 HMA 场景支持 P block size < D block size 的一部分情况。
- 异构 TP head split 需要 HND，除非启用实验性的 `enable_permute_local_kv`。

## 失败恢复：invalid_block_ids 和 kv_load_failure_policy

NIXL READ/WRITE 不是总能成功。新版 worker 失败时会走 `_handle_failed_transfer()`：

- 如果当前请求有 recv metadata，且不是 HMA 场景，就把对应本地 logical block id 放进 `_invalid_block_ids`。
- 把 request id 放进 `_failed_recv_reqs`。
- 释放 transfer handle。
- 记录失败统计。

worker 输出会把 `invalid_block_ids` 带回 scheduler。`Scheduler.update_from_output()` 看到后调用 `_handle_invalid_blocks()`。

这里分两类请求：

- 还在 `WAITING_FOR_REMOTE_KVS` 的 async load 请求。
- 已经 running 的 sync load 请求。

scheduler 会扫描这些请求的 block table，找到第一个失败 block，把 `request.num_computed_tokens` 截断到失败 block 前面。之后看 `kv_load_failure_policy`：

- `"fail"`：返回失败 request id，后面主循环会让请求失败。
- `"recompute"`：把 async failed request 放进 `failed_recving_kv_req_ids`，等 `finished_recving` 到达后只 cache 成功前缀，再重新调度重算失败部分。

这里的 TODO 还提到 HMA 的失败处理没有完整覆盖，所以读这段时不要把非 HMA 逻辑外推到所有模型。

## abort_immediately：被拒请求也要释放 P 侧 block

Decode 请求可能在进入 scheduler 前后就被拒，比如服务层发现不能接。EngineCore 里有 `notify_kv_transfer_request_rejected()`，会构造一个 pre-aborted request，让 connector 的 `request_finished()` 有机会跑。

NIXL scheduler 里有一个专门处理：

```python
if params.get("do_remote_prefill"):
    self._reqs_need_recv[request.request_id] = (request, [])
    params["do_remote_prefill"] = False
    return False, None
```

这表示 Decode 侧虽然不再需要远端 KV，但仍要让 worker 发一个空 recv/通知，告诉 Prefill 那边可以释放已经 pin 住的 block。`test_abort_immediately_remote_prefill_enqueues_empty_recv()` 覆盖的就是这个路径。

## 多 batch 和延迟 free

`Scheduler.__init__()` 里有一个新的防护：

```python
multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

注释说得很清楚：async scheduling 或 pipeline parallel 下，某个 step 可能还在写一个已经释放请求的 KV block。consumer connector 又可能把这块 block 重新分配给远端 load。为了避免新旧写入乱序，consumer 在多 batch 场景会延迟释放 block，等 GPU 写入 fence 过去后再真正回收到 block pool。

这个细节和 NIXL 本身无关，但对 PD 场景很关键。否则 KV connector 会把“看起来空闲”的 block 当落点，实际上一轮 forward 还没完全写完。

## Pull 和 push 该怎么区分

从用户角度看，两者都要 proxy 先跑 P，再跑 D，也都把 Prefill 输出的 `kv_transfer_params` 交给 Decode。区别在 worker 数据面。

Pull 模式：

- 配置名：`NixlConnector` 或 `NixlPullConnector`
- Decode 侧拿到 remote block id 后，自己提交 NIXL `READ`
- Prefill 侧等 Decode 的完成 notif 或 heartbeat/lease 过期
- 代码主线更短，默认路径也是它

Push 模式：

- 配置名：`NixlPushConnector`
- Decode 侧先注册自己的本地 block
- Prefill 侧 prefill 结束后，由 writer 线程匹配注册和 blocks
- Prefill worker 提交 NIXL `WRITE`
- Decode 侧等 completion notif
- 多了注册 watchdog 和 writer-local matching table

Pull 更像“D 知道 P 在哪，自己去拿”；push 更像“D 先报地址，P 算完后送过去”。两者共享握手、metadata、TransferTopology、descriptor 构造、heartbeat 和很多失败处理逻辑。

## 当前代码里容易误读的地方

`NixlConnectorScheduler` 和 `NixlConnectorWorker` 还存在，但只是向后兼容 re-export，分别指向 `NixlPullConnectorScheduler` 和 `NixlPullConnectorWorker`。不要再按旧笔记里的单文件实现去找。

`VLLM_NIXL_SIDE_CHANNEL_PORT` 不是传 KV 的端口。它只是握手服务端口。真正 KV 搬运走 NIXL backend，例如 UCX、LIBFABRIC、GDS 等。

`remote_block_ids` 是逻辑 block id，不是直接拿来给 NIXL descriptor 用的物理 id。worker 会结合本地和远端的 `physical_blocks_per_logical_kv_block` 再展开。

`request.num_computed_tokens` 在 `WAITING_FOR_REMOTE_KVS` 时已经变大，不代表 KV 已经可读。它只是 scheduler 的状态承诺。

Prefill block 的释放现在主要看 lease 和 notif。旧版“超时释放”的说法不够准确，heartbeat 会延长 lease。

Push 模式下，`finished_sending` 不是 Decode 读完 Prefill block，而是 Prefill WRITE 完成、P 侧可以释放自己暂存的 blocks。

## 整体串起来

当前 vLLM 的 NIXL PD 分离可以这样理解：proxy 用 `kv_transfer_params` 串起 Prefill 和 Decode 两段 HTTP 请求；scheduler 根据这些参数把远端 token 算进已计算前缀，先分配本地 block，再用 metadata 把传输计划交给 worker。

worker 通过 side-channel 拿到远端 NIXL agent metadata。之后按 pull 模式发 READ，或按 push 模式先注册再由 Prefill 发 WRITE。完成、失败、heartbeat 和租约回收都通过 `KVConnectorOutput` 和 NIXL notification 回到 scheduler。请求最终从 `WAITING_FOR_REMOTE_KVS` 回到正常调度流。
