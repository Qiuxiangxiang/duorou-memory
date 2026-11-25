# duorou 与其他记忆系统：评测标准和公开效果

核查日期：2026-11-24。证据分为 duorou 已保存的本地实测、他方作者公开成绩、公开运行日志核对。此次没有运行新的竞品评测，也没有由命名变更产生新的性能成绩。

## 1. duorou 现在是什么

duorou 是本项目独立实现的记忆系统原型总名，包含三个阶段的工作：

| 组成 | 实际实现 | 现状 |
|---|---|---|
| 评测控制 MCP | 压测启动、状态、报告、取消、对比、写入可见性等 7 个工具 | 已运行与验证 |
| 公开数据检索目标 | 通过 MCP 查询的 BM25 / TF-IDF 基线、数据转换与评分 | 已跑公开数据 |
| 结构化记忆数据层 | SQLite、来源/span、实体、事件、陈述版本、意图、受控标签、索引与工作集引用 | 已有原型与真实导入验证 |

当前尚未接通完整的模型抽取、语义事件识别、实体消歧、归纳更新、图/向量混合检索与回答生成链路。新 schema 也尚未作为新 MCP 业务服务部署。已有压测 MCP 不等于记忆管理 Agent。

独立实现指项目代码与数据模型；BM25、TF-IDF、SQLite、MCP 及认知机制本身不是本项目发明。原始证据、事件、时间有效性和来源追踪也是其他系统使用的设计。例如 [Graphiti](https://github.com/getzep/graphiti) 已明确实现 episodes/provenance、时间有效窗口和混合检索。duorou 的效果与差异需要同协议测量，不能以概念名称证明。

## 2. 先分清指标

| 指标 | 测量对象 | 不代表什么 |
|---|---|---|
| Evidence Recall@k | 每题所需 gold 证据被召回的比例，再对题平均 | 不等于答案正确率 |
| AnyHit@k | 至少找到一条 gold 的题目比例 | 不代表多跳证据找齐 |
| AllHit@k | 全部 gold 都找到的题目比例 | 不保证回答模型正确使用证据 |
| Answer F1 | 输出答案与标准答案的词级匹配 | 不等于 gold 文档召回 |
| LLM judge accuracy | judge 认定最终答案正确的比例 | 依赖 reader、judge、提示与评分规则 |
| Answer-in-Context | 返回上下文中包含答案文本 | 不自动证明归属、时间或推理正确 |
| 检索延迟 | 查询到取回上下文的耗时 | 不一定含抽取、写入、回答生成 |

同一数据集名称不能替代完整协议。需固定数据版本、排除规则、reader/judge、证据粒度、返回 token、检索轮数和吞吐负载。

## 3. duorou 已有的真实结果

来源：`reports/baseline_snapshot.json`，保留上一轮结果文件的 SHA256。

| 数据集/方法 | 可评题数 | 证据单位 | Recall@20 | AllHit@20 |
|---|---:|---|---:|---:|
| LoCoMo / BM25 | 1,527 | turn | 62.03% | 56.84% |
| LoCoMo / TF-IDF | 1,527 | turn | 59.96% | 54.75% |
| LongMemEval / BM25 | 470 | session | 97.65% | 94.89% |
| LongMemEval / TF-IDF | 470 | session | 97.60% | 94.68% |

LoCoMo BM25 的 278 个 multi-hop 问题中，Recall@20 为 33.51%、AllHit@20 为 14.75%。这是需要优先改善的证据缺口。

LoCoMo 的 1,527 题排除了 adversarial 及缺失/无效 gold；常见他方 QA 口径为排除 adversarial 后的 1,540 题。LongMemEval 的 470 题排除了 30 个拒答问题，证据单位还是整场 session。因此 97.65% 不能用来宣称优于其他系统约 90% 的 QA 成绩。

历史性能测试是本地 MCP 链路、预加载索引，返回 ID/score，不含模型抽取、外部 embedding、回答生成或逐条完整文本传输。新数据层只验证导入与约束；当前 15 项约束测试通过，导入 5,882 条原文、272 个 session 结构分组、6,426 条确定性元数据标签，未产生语义 claim。

## 4. 其他系统的公开结果：不能当统一排行榜

以下均为各作者设置下的公开报告。本文只核对来源，没有独立复跑。

| 系统/版本 | LoCoMo QA | LongMemEval QA | 必须保留的条件 |
|---|---:|---:|---|
| Mem0 managed v3 | 92.5% | 94.4% | 作者 headline，top200；含 OSS 不具备的托管优化 |
| Mem0 OSS：GPT-5 抽取 | 未在此行引用 | 91.0% | Qwen 600M embedding、Qdrant、GPT-5 reader/judge |
| Zep 当前多路检索 | 94.7% | 90.2% | GPT-5.4 reader/judge，五路上下文；LoCoMo 分项与总计待核对 |
| Hindsight v0.4.19 单次检索 | 92.0% | 94.6% | 1,540/500 题，Gemini-3.1-Pro-Preview 回答、Gemini-2.5-Flash-Lite judge |
| Letta Filesystem，2025-08 | 74.0% | 此实验未报 | GPT-4o-mini；Agent 可多轮 search_files/grep |
| Rolling Memory，2026-11-16 | 85.26% | 此实验未报 | 1,313/1,540，deepseek-v4.1-flash 回答与 judge |

### 对应的一手来源

- Mem0 托管结果与 OSS 区别：[主仓库](https://github.com/mem0ai/mem0)；[评测库](https://github.com/mem0ai/memory-benchmarks)。评测库还报 top50 的 LongMemEval 94.8%，不能混用不同 top-k 配置。
- Zep：[当前方法与成绩](https://www.getzep.com/research/memory-benchmarks/)。平台成绩不能直接填到自部署 Graphiti 上。
- Hindsight：[v0.4.19 公告](https://github.com/vectorize-io/hindsight/blob/main/hindsight-docs/blog/2026-03-23-agent-memory-benchmark.mdx)；[运行汇总](https://github.com/vectorize-io/agent-memory-benchmark/blob/main/results-manifest.json)。
- Letta：[2025-08-12 文件检索实验](https://www.letta.com/blog/benchmarking-ai-agent-memory/)。
- Rolling Memory：[官方 README §4.2](https://github.com/rocareer/rolling-memory)。

### 成本与口径

Zep 的上述 LongMemEval 运行报上下文中位数 4,408 tokens、检索 p95 162ms；Hindsight 对应运行报上下文均值 43,624.5 tokens、平均检索 674.9ms。一个是中位数/p95，一个是均值，模型也不同，不能按这两行计算效率倍率。Hindsight 返回数万 token 后的回答生成成本仍应单独计算。

Rolling Memory 另报 ATM Main/Hard Recall@10 为 82.77%/52.43%，这是另一数据集、另一指标；其 LoCoMo 检索均值 371ms，P95 未记录，不应与 duorou 本地压测的 p95 直接相除。

### 公开数字的可追溯性核对

本次实际读取了 Mem0 官方 `results/platform` JSON：所附较早日志为 LoCoMo 1,410/1,540（91.558%）、LongMemEval 467/500（93.4%），与 README 当前 headline 不同。它们可能对应不同运行；现有材料不能确定原因，也不能说这些日志已逐题复现新 headline。

Zep 页面总计为 1,459/1,540，四类计数相加为 1,436/1,539。页面未解释这一差异，需要对应逐题日志才能核对。

Hindsight 两个公开 run 的头部已核实 reader、judge、题数、得分、检索耗时和上下文 token。但头部没有完整服务端抽取配置，不能据此称全系统已独立复现。

核对摘要及原始 URL 存在 `reports/external_result_audit.json`。该文件只保存摘要和元数据，没有复制完整问答语料。

## 5. 应采用哪些公开标准

| 基准 | 核心用途 | 评分边界 |
|---|---|---|
| LoCoMo | 长对话的单跳、多跳、时间和拒答 | 原始代码默认答案 F1；不少后续工作换用 LLM judge，必须注明 |
| LongMemEval V1 | 信息提取、多会话推理、知识更新、时间和拒答 | 官方 QA judge 与证据 recall 可分别运行；固定 S/M/cleaned revision |
| MemoryAgentBench | 增量交互、测试时学习、长程理解和冲突处理 | 子集使用 exact match、substring match、Recall@5 或 judge；没有可随意混用的单一 accuracy |
| LongMemEval-V2 | 从 Agent 轨迹提取状态、流程经验、环境陷阱与前提条件 | 451 题，固定 reader、限制返回上下文，测 QA 与 memory-query latency |

一手来源：[LoCoMo 评分代码](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)、[LongMemEval](https://github.com/xiaowu0162/LongMemEval)、[MemoryAgentBench 指标映射](https://github.com/HUST-AI-HYZ/MemoryAgentBench)、[LongMemEval-V2](https://xiaowu0162.github.io/longmemeval-v2/)。V1 已于 2025/09 再次清理，V2 于 2026/05 公布。文件名相同或都含 cleaned 不证明内容相同。

LongMemEval-V2 的同表 Small 结果提供了有用的成本取舍例子：RAG slice+notes 为 51.0% / 0.2 秒，AgentRunbook-R 为 58.6% / 26.9 秒，AgentRunbook-C 为 74.9% / 108.3 秒。延迟是平均 memory-query 时间，不含最终 reader 回答。它支持按准确率与延迟的共同边界比较，而非只报最高分。

上述 QA/轨迹任务不会自动证明 Agent team 的并发读写、身份隔离、来源删除或共享记忆冲突处理能力；这些仍需专门系统验证。

## 6. duorou 的下一轮对照协议

先建立相同 reader/judge、同一数据 revision、相同实际返回 token 预算和检索次数的比较：

1. 固定当前 BM25/TF-IDF 作基础对照；增加 full-context 和普通向量 RAG。
2. 为 duorou 接通 source/tag/event/claim 到可返回原文的查询链路。
3. 选择可复现版本的 Mem0 OSS、Graphiti、Hindsight，以及明确本地模式的 Rolling Memory；托管产品另列产品配置实验。
4. 先测证据完整性，再测固定 reader 的最终答案。多轮 Agent 检索独立成组。
5. 同时报告写入/抽取成本、更新可见性、检索 p50/p95/p99、回答成本、失败率、实际上下文 token、删除与隔离结果。
6. 不把被索引的整个事件祖先范围都算作实际返回的证据；只计交付的叶子文本。排除规则与 abstention 单独报。

我们已经查看过既有 LoCoMo/LongMemEval 结果，后续同集优化应称开发迭代。发布泛化结论还需要冻结方案后的独立评估。

当前合理的判断是：duorou 有自研代码、可复现的证据检索基线和可运行数据骨架；完整记忆闭环与同条件竞品效果仍待测量。名称、脑科学启发和更多标签都不自动构成性能优势。
