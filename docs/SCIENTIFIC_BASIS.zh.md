# 认知机制与工程转译依据

本文采用“认知机制提供设计动机，工程实验验证系统收益”的立场。人类记忆提取研究不能证明某个标签、索引或检索算法能够提升软件性能；从对话中抽取出更多记忆，也不等于检索或回答更准确。下列七项机制分别标明证据类型、可借鉴原则与推论边界，不把数据库组件命名为脑区，也不声称实现了人脑记忆。

## 1. 互补学习系统：具体经历与整合知识并存

**证据与类型：** McClelland、McNaughton 与 O’Reilly（1995）的计算理论解释快速学习具体关联与逐步整合结构化知识之间的分工，以及新学习对旧知识的干扰。

**工程映射：** 原始证据与具体事件及时记录；跨事件形成的事实、规则和概括作为派生表征另行保存，并链接回来源。异步整合负责去重、冲突识别与概括，原始经历不因生成摘要而消失。

**不能推出：** 海马等于缓存、皮层等于数据库；语义事实必须重复若干次才能成立；所有记忆都按固定流程迁移。单条明确陈述可以作为证据，整合过程仍须评估其来源和适用范围。

**一手来源：** McClelland, J. L., McNaughton, B. L., & O’Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory.* Psychological Review. [DOI: 10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419)；[作者全文](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)。

## 2. 事件分割：连续经验具有多种时间尺度

**证据与类型：** Baldassano 等（2017）的叙事观看与回忆实验结合脑成像分析，发现不同时间尺度的事件结构，以及事件边界与记忆过程的联系。

**工程映射：** 保存原始片段、事件与较长经历之间的成员关系；利用目标、人物、时间、地点和情境变化提出边界。允许一个事件跨越多次会话，也允许一段原文支持多个事件。

**不能推出：** 技术上的 session 等于认知事件；一条消息必定对应一个事件；固定长度分块具有脑科学上的最优性。具体分割器、层数及边界规则都属于待检验的工程假设。

**一手来源：** Baldassano, C., et al. (2017). *Discovering Event Structure in Continuous Narrative Perception and Memory.* Neuron. [DOI: 10.1016/j.neuron.2017.06.041](https://doi.org/10.1016/j.neuron.2017.06.041)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558154/)。

## 3. 关系绑定、模式分离与补全：联想而不混淆

**证据与类型：** Horner 等（2015）的行为与脑成像实验支持事件元素的绑定及部分线索引发关联内容恢复；Bakker 等（2008）的高分辨率脑成像结果与区分相似输入的模式分离机制相一致。

**工程映射：** 区分实体身份与事件发生实例，记录参与角色和上下文。检索命中线索后，沿明确的事件成员关系或证据关系有限扩展；对相似事件使用时间、主体及情境消歧。

**不能推出：** 文本相似就应合并实体或事件；共现就是因果；图检索必然优于其他方法。“补全”只能找回已有证据，不能生成缺失事实。脑成像结果也不等于对单一细胞机制的直接证明。

**一手来源：** Horner, A. J., et al. (2015). *Evidence for holistic episodic recollection via hippocampal pattern completion.* Nature Communications. [DOI: 10.1038/ncomms8462](https://doi.org/10.1038/ncomms8462)。Bakker, A., et al. (2008). *Pattern Separation in the Human Hippocampal CA3 and Dentate Gyrus.* Science. [DOI: 10.1126/science.1152882](https://doi.org/10.1126/science.1152882)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2829853/)。

## 4. 目标维持与认知控制：由任务选择工作上下文

**证据与类型：** Miller 与 Cohen（2001）的整合性理论综述强调，对目标和情境的主动维持能够影响信息处理与行为选择。

**工程映射：** 为当前任务维护目标、未解决问题及证据引用组成的工作集，以相关性和预算控制进入上下文的信息。工作集淘汰引用，不等于删除长期记录。

**不能推出：** 工作集对应单一脑区；人类记忆容量可以直接换算为 token 常数；高“重要性”记忆应进入每个任务。选择策略需要按任务验证。

**一手来源：** Miller, E. K., & Cohen, J. D. (2001). *An integrative theory of prefrontal cortex function.* Annual Review of Neuroscience. [DOI: 10.1146/annurev.neuro.24.1.167](https://doi.org/10.1146/annurev.neuro.24.1.167)。

## 5. 前瞻记忆：意图与已发生事实分开

**证据与类型：** McDaniel 与 Einstein（2000）的多过程理论框架认为，实现延迟意图可能依赖主动监测，也可能由适当线索触发。

**工程映射：** 单独记录行动意图、责任主体、触发条件、截止时间、状态及完成证据。时间和事件线索可以用于查找尚待完成的事项。

**不能推出：** “计划做”意味着“已经做”；截止时间过去意味着完成；从文本提取意图就获得执行授权。调度与授权是额外的工程和产品规则。

**一手来源：** McDaniel, M. A., & Einstein, G. O. (2000；在线发布于 2001). *Strategic and automatic processes in prospective memory retrieval: A multiprocess framework.* Applied Cognitive Psychology. [DOI: 10.1002/acp.775](https://doi.org/10.1002/acp.775)。

## 6. 来源监控：记录证据归属与认识状态

**证据与类型：** Johnson、Hashtroudi 与 Lindsay（1993）的理论框架讨论人如何判断记忆、知识与信念的来源，以及如何发生来源错误。它强调依据记忆内容的特征作出归因，**并非“大脑显式保存和检索来源标签”的理论**。

**工程映射：** 显式来源字段是增强软件可靠性的设计：记录原文位置、陈述者、讨论对象，以及观察、转述、推断、假设等性质。同一原文经多个代理转述仍是一条证据链。检索相关度、抽取置信度和来源可信度应分别表达。

**不能推出：** 有来源标签就必然真实；模型推断可当作用户陈述；重复检索或多次转述能够增加独立证据数量。

**一手来源：** Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). *Source monitoring.* Psychological Bulletin. [DOI: 10.1037/0033-2909.114.1.3](https://doi.org/10.1037/0033-2909.114.1.3)；[作者全文](https://memlab.yale.edu/sites/default/files/files/1993_Johnson_Hashtroudi_Lindsay_PsychBull.pdf)。

## 7. 更新与再巩固：允许修订，保留可追溯历史

**证据与类型：** Hupbach 等（2007）的人类行为实验观察到，提醒旧经历后进行新学习可以改变后续回忆。Sederberg 等（2011）的计算模型给出时间情境与关联绑定的替代解释，相关行为结果不能唯一证明细胞层面的再巩固。

**工程映射：** 新信息可以补充、反驳或取代派生结论，但保存来源和版本关系；区分事实有效时间与系统获知时间。检索可提出复核候选，不能自动改写原始证据。保留历史仍须服从明确的删除和保留规则。

**不能推出：** 每次回忆必定使所有记忆不稳定；版本控制是大脑的实际实现；系统应复现人类记忆失真。

**一手来源：** Hupbach, A., et al. (2007). *Reconsolidation of episodic memories: A subtle reminder triggers integration of new information.* Learning & Memory. [DOI: 10.1101/lm.365707](https://doi.org/10.1101/lm.365707)。Sederberg, P. B., et al. (2011). *Human memory reconsolidation can be explained using the temporal context model.* Psychonomic Bulletin & Review. [DOI: 10.3758/s13423-011-0086-9](https://doi.org/10.3758/s13423-011-0086-9)。

## 适用边界与验证原则

上述映射不将程序性知识、人格或情绪强行对应到脑区；“操作规程”也不等于人类程序性学习。它们若成为产品数据类型，应依据可观察证据和任务需要定义，不从文风推断稳定人格或敏感身份。

验证应逐项增加事件组织、实体与时间索引、关系扩展及派生表征，在相同证据和上下文预算下做消融，检查证据完整率、实体误合并、时间错误、来源错误、延迟与索引成本。结论只适用于实际测量的任务与数据；认知上的合理性不能替代这些测量。
