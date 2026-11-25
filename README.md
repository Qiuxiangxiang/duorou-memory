# duorou-memory

duorou-memory 是本地、证据优先的 Agent 记忆系统，提供 Python API 和 MCP 服务。

[ATM-Bench Hard 复现](benchmarks/atm-hard/README.md) · [逐题重评和用量](benchmarks/atm-hard/rejudge/README.md) · [榜单投稿 #13](https://github.com/atmbench/atmbench.github.io/issues/13)

Hard 31 题的实验选择器达到 **80.76% Recall@10**，官方评测器以 `gpt-5-mini` 重评得到 **45.04% QS**。选择器曾根据 Hard 错题迭代。下文保留 v1.0.0 发行版的原始实现说明。

## v1.0.0 实现说明

独立自研、证据优先的本地 Agent 记忆系统。代码版本为 1.0.0，**不是已通过 SOTA/生产验收的版本**。核心只依赖 Python 标准库与 SQLite（FTS5、JSON、STRICT 支持，SQLite >=3.37）；MCP SDK、语义模型和旧 benchmark 依赖按需安装。

## 代码结构

- `duorou/store.py`：唯一业务数据入口。作用域、权限、原始证据、事务、派生结构、时间版本、工作集、索引缓存。
- `duorou/retrieval.py`：作用域内 BM25 倒排索引、元数据/上下文索引、可选密集向量、RRF、有限关系扩展、证据预算。
- `duorou/server.py`：17 个 MCP 业务工具，官方 FastMCP stdio 适配。
- `schema/schema.sql`：沿用旧 0.2 数据层，启动时幂等添加退役记录表及 topic 标签定义。
- `benchmarks/v1_eval.py`：同条件逐题证据检索与外部预测排名评分；`benchmarks/v1_load.py`：合成负载与更新可见性。
- `benchmarks/` 旧版文件、`vendor/`：历史公开评测和评测 MCP 保留作对照；不是业务服务运行依赖。

## 快速启动

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m duorou.server --db memory.db --principal my-agent --grant project-a
python -m duorou.server --db memory.db --principal my-agent
```

最后一条由 MCP 客户端作为 stdio 子进程启动，而不是当 HTTP URL。操作员离线创建授权；不要向模型暴露 `--grant`、数据库文件或任意命令执行。`read` 与 `write` 是独立权限；`admin` 包含两者。每个进程绑定自己的 principal；MCP 参数无权指定认证主体。

客户端示例（将绝对路径换成实际位置）：

```json
{"mcpServers":{"duorou":{"command":"/absolute/path/.venv/bin/python","args":["-m","duorou.server","--db","/absolute/path/memory.db","--principal","my-agent"],"cwd":"/absolute/path/duorou-v1.0"}}}
```

`cwd` 的支持取决于 MCP 宿主；宿主不支持时从项目目录启动，或显式设置 PYTHONPATH。

不装 MCP SDK 也可以直接使用 Python API：

```python
from duorou import Memory
m = Memory('memory.db', 'my-agent')
m.provision('project-a')   # 仅可信操作员调用
m.ingest('project-a', 'msg-1', 'I plan to replace the Orion controller.',
         speaker_id='person-1', speaker='Alice', session='meeting-1', tags=['hardware'])
m.consolidate('project-a', 'msg-1')
print(m.search('project-a', 'Orion', top_k=5, budget=4096))
print(m.facts('project-a', subject_id='person-1'))
m.close()
```

## MCP 业务接口

| 工具 | 职责 |
|---|---|
| memory_ingest / memory_update / memory_delete | 原文写入、追加修订并退役旧版本、逻辑删除 |
| memory_search / memory_get_evidence / memory_timeline | 排名证据、原文、按记录时间查询 |
| memory_mention / memory_alias | 显式身份绑定及别名；不会自动按同名合并 |
| memory_event / memory_relate | 事件与带来源的关系 |
| memory_claim / memory_facts | 候选语义陈述、双时间与版本查询 |
| memory_intention / memory_intentions | 前瞻记忆及完成证据 |
| memory_consolidate | 保守的中文/英文第一人称计划和决定抽取 |
| memory_pin / memory_working_set | 私有任务工作集及预算 |

语义陈述是带来源的候选，不保证为真。`memory_search` 支持 `bm25`, `enriched`, `hybrid`, `dense` 四个显式模式；默认 `hybrid` 是尚待公开语料验证的实验配置。`dense` 需显式配置本地 embedding 模型。

可选模型：

```sh
python -m pip install -r requirements-semantic.txt
python -m duorou.server --db memory.db --principal my-agent --embedding-model /local/model
```

仅使用已准备的本地 sentence-transformers 模型目录。Python API 可直接提供 `embed(list[str]) -> matrix`；也可提供 `extractor(text) -> list[dict]`，字段必须为 `start,end,predicate,object_text,modality,polarity`。提取结果为候选 `model_inference`，固定到已知发言者及精确 Unicode 原文跨度，禁止自行指定身份。默认无外部模型调用。CLI 未提供通用 LLM 提取器的供应商封装；通过 Python API 配置后可使用同一 `register()` 启动 MCP。

默认检索预算为**返回证据正文 UTF-8 字节之和**，不计 JSON 元数据、提示词模板或模型生成。Python 可传入 `token_count`；评测 CLI 可传本地 tokenizer 目录；该情况下预算为独立证据文本 token 数之和。不能将字节预算当成 tokenizer token 计数。返回的图关联必须是实际叶子证据，不能将事件全部成员视为自动命中。

## 整体评测

```sh
python -m pip install -r requirements-bench.txt
python -m unittest discover -s tests -v
python -m benchmarks.v1_eval --fixture --top-k 5 --budget 4096 --output reports/reproduced-fixture
python -m benchmarks.v1_load --output reports/reproduced-load.json
```

正式公开语料复测（原始数据按既定 revision/hash 获取，不随包重新分发）：

```sh
python -m vendor.public_memory_bench.locomo --root .
python -m vendor.public_memory_bench.longmemeval --root .
for name in locomo longmemeval; do
  python -m benchmarks.v1_eval \
    --documents data/converted/$name.documents.jsonl \
    --queries data/converted/$name.queries.jsonl \
    --metadata data/converted/$name.query_metadata.json \
    --top-k 20 --budget 32768 --output reports/v1-$name
done
```

加入 `--embedding-model /local/model` 会测 dense 和含 dense 的 hybrid。若要控制实际正文 token 数，加入 `--tokenizer /local/tokenizer --budget 4096`。旧 v0.2.2 成绩是 ID-only MCP 请求且无相同正文预算，需重新跑同条件基线，不直接比较时延。`--external-rankings path.jsonl` 可按同一查询、证据与预算校验其他后端的实测排名，每条格式 `{"query_id":"...","ranked_ids":["..."],"latency_ms":12.3}`；必须覆盖全部题目，禁止跨 scope 和超预算。该评分不独立验证外部系统是否真正返回文本；需保留其原始调用回执。外部系统部署、统一 reader/judge 端到端 QA 与真实多身份 MCP 压测是后续整体评测工作，本次没有运行。

### 已执行

见 `reports/V1-VALIDATION.zh.md`、`unit-tests.txt`、`v1-fixture/summary.json`、`v1-load.json`。原始逐题数据为合成样例，不是公共 benchmark。已执行旧版与新增测试；语义模型和 MCP SDK 在执行环境中不可用，分别通过替身回调和工具注册函数测试，未假称真实模型/协议测试。公开数据下载失败，旧成绩仅保留为历史记录。

## 边界与工程约束

- SQLite 本地单服务，读写事务串行化由 RLock 与 SQLite 实现。缓存按 scope/历史模式建立；任何本服务写入及其他连接变化使缓存失效。首次查询或写入后需要重建整个 scope 索引，尚无增量 ANN 或分布式索引。不要把合成热点 QPS 视为生产容量。
- 数据库文件本身是信任边界；非远程认证服务。不能直接暴露公网。跨连接撤权在下一事务生效，在途事务遵循 SQLite 快照。源、标签或图内容皆是不可信数据，绝非操作指令。
- 默认原文修订保留历史、隐藏退役记录。删除是逻辑删除及外键级联，并清除可删除的派生语义事件。独立实体、会话、任务标识、历史备份、外部副本、WAL 和未清除的数据块不等于安全擦除；重新摄入同 ID 需要由上游删除策略控制。
- 自动抽取默认是窄规则，不是通用事实/人物识别。身份消歧由明确调用方标识完成；未验证事实不能作为操作授权。`consolidate` 多候选逐条事务，写入与提取不是同一个原子操作。意图完成引用只验证存在性，不证明业务完成。
- 检索当前支持一跳有界扩展、相关性和轻量会话多样性，尚无经过统一 benchmark 验证的自适应多轮路由、自动记忆演化、程序学习或真正生产级并发容量保证。
- 旧设计文档保留为历史设计依据，当前实现以本 README 和代码为准。LoCoMo 数据许可证为 CC BY-NC 4.0；按原协议下载和使用，竞品许可独立评估。参见 NOTICE。
