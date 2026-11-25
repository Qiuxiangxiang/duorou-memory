# 标签输入合同与 SQLite 映射

`schema/annotation.schema.json` 是待接入服务的 JSON 输入结构合同，不是当前已注册的 MCP 工具。`examples/annotation.json` 表达公开数据的结构性 session 序号；不包含模型抽取的事实。

## 输入到数据行的映射

| 输入字段 | SQL 字段/处理 |
|---|---|
| schema_version | 校验合同版本，不写入 tag_annotations；服务应记录所用 schema 版本 |
| scope_id | scope_id；服务先核验可信调用主体的写入权限 |
| annotation_id / namespace / tag_key | 同名列 |
| value | 规范化 JSON 序列化到 value_json，例如字符串存为 `"dialogue_turn"`、数值存为 `1` |
| target.kind=source | target.id → source_id，其余目标列 NULL |
| target.kind=event | target.id → event_id |
| target.kind=entity | target.id → entity_id |
| target.kind=claim | target.id → claim_id；target.version → claim_version |
| target.kind=intention | target.id → intention_id |
| target.kind=span | target.id → target_span_id |
| evidence_span_id | 同名列；省略时 NULL，model 来源不允许省略 |
| origin / status / extractor_version | 同名列 |
| extraction_confidence | 同名列；省略/未知存 NULL，不擅自补成 1 |

目标 span 和支持 span 有不同含义：前者是“给哪一段打标签”，后者是“哪段内容支持这个标签”。即使指向同一 span，也应明确表达。

## 三层校验职责

1. **JSON Schema**：目标唯一、必填字段、类型、origin/status 枚举、模型标签必须有证据引用；内置三个 demo key 的值和目标约束。
2. **SQLite**：同 scope 外键、词表存在、JSON 类型和值域、唯一目标、引用删除级联，以及模型证据引用要求。
3. **可信服务**：认证、授权、allowed namespace、标签值语义、证据是否支持标签、canonical JSON、模型标签如何审核，以及完整审计。

输入合同允许形状正确的新词表 key，SQLite 中未定义的 key 仍会被外键拒绝。`tag_registry.json` 的 extensions_not_loaded 只是设计提案；它们不在导入示例自动启用。任何内容 tag 都不能授权读写或改变 principal 身份。

SQL 的 number 类型允许整数和浮点数，而 session_ordinal 的服务合同进一步要求正整数。SQL 尚未按具体 key 约束目标类型；这由该 JSON 合同/服务完成。不要将某一层的校验误称为完整生产 API。

## 状态和置信度的解释

`accepted` 表示该标注进入当前可用词表，不证明被描述世界状态为真。`metadata` 表示标签来自来源元数据，不代表现实身份已经核验。`model` 标签有来源位置仍可能误读来源，需要抽样评估。

原型只导入 3 类确定性元数据标签。`extraction_confidence=NULL` 是刻意保留未知，不是漏算。置信度校准、自动实体链接、主题抽取和事实抽取均不在本次导入示例中。
