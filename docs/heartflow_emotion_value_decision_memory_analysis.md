# HeartFlow 情绪 / 值 / 决策 / 记忆链路分析

- 版本：1.0
- 日期：2026-05-07
- 状态：已完成
- 范围：`src/chat/heart_flow`、`src/modules/social_value`、`src/modules/modcore/dynamic_persona`、`src/core`、`src/memory_system`、`src/modules/perception`

---

## 1. 结论先看

这套系统不是单一状态机，而是多层并行的状态网络。

当前最重要的事实有四个：

1. **情绪和值不是一个源头**，而是多套系统并行维护同一语义的不同视图。
2. **决策不是单点判断**，而是 `维度投票 → 网关裁定 → 行为/模型/仪表盘纠偏 → 回复执行` 的复合链路。
3. **群体态势、个人关系、记忆重激活、世界快照** 已经接到主链上，但边界还不够清晰。
4. **最危险的问题不是“没有值”**，而是“同一值被多处重复定义”，导致语义漂移和默认值污染。

如果只保留一句话：

> 当前系统的核心问题不是缺模块，而是缺一个统一的“真值契约”。

---

## 2. 情绪和值系统的真实来源

### 2.1 频道氛围层

**文件**：`src/chat/heart_flow/emotion_stream.py`

这是频道级的情绪账本，核心对象是 `MoodLedger`。

#### 关键字段
- `category`
- `inquiry_tally`
- `tool_call_tally`
- `duplicate_hits`
- `privileged_ratio`
- `intensity`
- `vitality`
- `weariness`
- `vexation`

#### 更新入口
- `record_inquiry()`
- `record_tool_call()`
- `record_social_feedback()`
- `EmotionAxisDimension.on_event()`
- `EmotionAxisDimension.tick()`

#### 导出入口
- `fetch_mood()`
- `compose_mood_hint()`
- `dump_snapshot()`
- `serialize()`
- `get_state_summary()`

#### 语义映射
- `vitality` → 耐心/活力
- `weariness` → 疲劳
- `vexation` → 烦躁

这里的三轴是 **频道级** 状态，不是用户级情绪。

---

### 2.2 内在驱动层

**文件**：`src/chat/heart_flow/emotion_driven_core.py`

这是机器人自身的内在驱动核心，状态是连续累积的，不是单次触发器。

#### 关键字段
- `boredom`
- `environmental_fatigue`
- `loneliness`
- `social_desire`
- `mood`
- `energy`
- `curiosity`
- `proactive_willingness`
- `withdrawal_tendency`
- `internal_clock`

#### 更新入口
- `tick()`
- `on_user_message()`
- `on_bot_message()`
- `integrate_external_state()`

#### 导出入口
- `get_state()`
- `get_state_snapshot()`
- `get_all_states()`
- `should_proactive()`

#### 核心特征
- 无聊、孤独、环境疲劳不是“事件标记”，而是**真实累积量**。
- `proactive_willingness` 不是直接赋值，而是从内在时钟、情感驱动、疲劳抑制里涌现出来。
- 这个模块更接近“主动性源头”，而不是传统的规则开关。

---

### 2.3 用户情绪与关系层

**文件**：`src/modules/modcore/dynamic_persona/emotion_tracker.py`

这是**按用户**维护的核心关系/情绪状态。

#### 关键字段
- `affection`
- `trust_score` / `trust_value`
- `annoyance`
- `trauma_score`
- `psychological_pressure`
- `negative_emotion_aggregate`
- `inner_chaos`
- `surface_mask`
- `submission_level`
- `shyness_level`
- `relationship`
- `mood`

#### 更新入口
- `process_interaction()`
- `update_affection()`
- `update_trust()`
- `update_annoyance()`
- `update_psychological_pressure()`
- `update_trauma()`
- `process_interaction_with_llm()`

#### 导出入口
- `get_user_state()`
- `get_all_user_states()`
- `get_statistics()`

#### 关键判断
- 这个模块是“人际关系真值”的主要来源之一。
- 但它不是唯一关系源，后面还有 `social_value`、`world_snapshot`、`impression_evolution_hub` 等桥接层。

---

### 2.4 社交值层

**文件**：
- `src/modules/social_value/models.py`
- `src/modules/social_value/social_calculator.py`
- `src/modules/social_value/settlement_engine.py`
- `src/modules/social_value/social_affect_fuser.py`
- `src/modules/social_value/phase_tracker.py`

这是社交值的独立链路，核心对象不是情绪，而是“关系评分、阶段、印象和变化过程”。

#### 关键字段
`SocialValueRecord`：
- `value`
- `positive_dim`
- `negative_dim`
- `trust_value`
- `annoyance_value`
- `interaction_count`
- `last_interaction`

`BondDossier` / `PersonaImpression`：
- `current_phase`
- `trend_label`
- `recent_values`
- `transition_log`
- `peer_affinity`
- `impression`

#### 更新入口
- `SocialCalculator.calculate()`
- `SocialCalculator.record_outcome()`
- `SettlementEngine.settle_event()`
- `SettlementEngine.settle_from_behavior()`
- `SocialAffectFuser.evaluate()`
- `SocialAffectFuser.evaluate_from_behavior()`
- `PhaseTracker.ingest_score()`
- `PhaseTracker.refresh_impression()`

#### 导出入口
- `read_score()`
- `read_full_record()`
- `get_snapshot()`
- `get_phase_label()`
- `get_dossier()`
- `get_brief_summary()`

#### 结论
这条链路已经是“社交值权威管线”，但它和 `emotion_tracker`、`world_snapshot` 的关系字段仍然存在重叠。

---

## 3. 决策链路如何消费这些值

### 3.1 主循环入口

**文件**：`src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py`

这里是最上层的事件驱动循环。

#### 关键职责
- 预构建世界快照
- 夜间节律门控
- 能量门控
- 消息去重
- 空闲主动行为
- 维度网关调用
- 统一规划器调用

这里不是“做决定”的唯一地点，但它是**调度总入口**。

---

### 3.2 维度网关

**文件**：`src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py`

#### 核心调用链
- 构建 `EventContext`
- `DimensionDispatcher.broadcast_event(evt)`
- `DimensionDispatcher.collect_votes(evt)`
- `DecisionGateway.decide(votes, gw_ctx)`

#### 进入网关的关键信号
- `emotion_baseline`
- `rapport_trust`
- `boundary_block`
- `tempo_control`
- `counterparty_readiness`
- `group_context`
- `resource_ledger`
- `social_balance`
- `pending_engagement`
- `trauma_load`
- `surface_mask`

#### 作用
这个层把分散的维度值统一成一次裁定，不再让各模块各说各话。

---

### 3.3 决策网关

**文件**：`src/chat/heart_flow/decision_gateway.py`

#### 核心职责
- 强制触发
- 强制抑制
- LLM 意愿修正
- 概率融合
- 结果门控

#### 关键逻辑
- `decide()`
- `_check_force_triggers()`
- `_check_force_suppressions()`
- `_check_voice_intent()`
- `_compute_probability()`
- `_post_process()`

#### 主要消费字段
- `emotion_baseline.should_refuse`
- `emotion_baseline.emotion_proactive_willingness`
- `rapport_trust.proactive_willingness`
- `boundary_block.is_blocked`
- `boundary_block.proactive_suppression`
- `tempo_control.in_cooldown`
- `counterparty_readiness.want_to_chat_confidence`
- `group_context.engagement_boost`
- `resource_ledger.force_refuse`
- `social_balance.proactive_boost`
- `trauma_load.force_refuse`
- `surface_mask.quality_cap`

#### 结论
网关本质上是一个“维度信号裁定器”，不是单一的回复概率器。

---

### 3.4 最终行动收口

**文件**：`src/chat/heart_flow/enhanced_modules/strategy_action_state_mixin.py`

#### 核心职责
- 把 voice / behavior / dashboard / relation / model path 汇总成最终动作
- 统一 `should_reply`、`next_action`、`model_path`、`decision_stage`
- 在执行前做二次修正

#### 关键函数
- `_build_decision_runtime()`
- `_apply_execution_verdict_correction()`
- `_store_gate_runtime()`
- `_store_execution_runtime()`

#### 结论
这个层已经不是简单“补充判断”，而是在做最终策略收口。

---

### 3.5 主动行为链

**文件**：`src/chat/heart_flow/enhanced_modules/strategy_proactive_decision_mixin.py`

**统一决策器**：`src/chat/proactive/proactive_decider.py`

#### 信号来源
- `EmotionDrivenCore` 的主动意愿、无聊、孤独
- `EnergyChainDimension` 的能量比例
- `social_value` 的社交地位
- `intention_pool` 的意图驱动
- 内心独白 `voice`
- 群体模式 `pattern_evidence`

#### 核心函数
- `_evaluate_proactive_decision()`
- `ProactiveDecision.evaluate()`

#### 结论
主动行为不是单独一个开关，而是由情绪、能量、意图、关系、群体模式共同推出来的。

---

### 3.6 回复生成与发送

#### 相关文件
- `src/chat/heart_flow/reply_coordinator.py`
- `src/chat/heart_flow/heartFC_chat.py`
- `src/chat/heart_flow/enhanced_modules/interaction_core_mixin.py`

#### 关键职责
- 生成回复
- 选择风格路由
- 最终发送
- 回写状态

#### 结论
决策层和发送层已经分离，但中间仍存在大量“状态注入”逻辑，容易让提示词和策略边界混在一起。

---

## 4. 群体 / 个人 / 记忆 / 上下文如何接入

### 4.1 群体态势感知

**文件**：`src/modules/perception/group_sense.py`

#### 输出
- `activity_level`
- `active_user_count`
- `silence_duration_seconds`
- `controversy_detected`
- `needs_topic`
- `burst_detected`
- `topic_hints`

#### 结论
群体层主要负责“群聊是否适合介入”，它是频道级环境输入，不是个人关系值。

---

### 4.2 用户状态感知

**文件**：`src/modules/perception/user_state.py`

#### 输出
- `state`
- `willingness`
- `interest_level`
- `attention_level`
- `confidence`
- `rule_signals`

#### 结论
这是“对方现在想不想聊”的上层判断，和关系值、社交值是不同维度。

---

### 4.3 自身状态感知

**文件**：`src/modules/perception/self_sense.py`

#### 输出
- `time_of_day`
- `session_duration_minutes`
- `time_since_last_reply`
- `reply_count`
- `active_streams`
- `energy_level`
- `mood_indicator`

#### 结论
它描述的是 bot 自身的宏观状态，主要用于自我监控和仪表盘，不直接替代决策核心。

---

### 4.4 世界快照聚合层

**文件**：`src/core/world_snapshot.py`

#### 关键结构
- `SubjectState`
- `SelfResourceState`
- `TargetUserState`
- `SceneState`
- `WorldSnapshot`

#### 关键方法
- `to_relation_dict()`
- `to_canonical_state()`
- `build_world_snapshot()`

#### 作用
把主体、目标用户、场景资源、群体环境统一成一个可消费快照。

#### 关键风险
`TargetUserState` 同时包含：
- `social_value`
- `trust_value`
- `annoyance_value`
- `affection`
- `trust_score`
- `psychological_pressure`
- `trauma_score`
- `positive_dim`
- `negative_dim`

这已经不是单一真源，而是**桥接层拼接结果**。

---

### 4.5 状态耦合矩阵

**文件**：`src/core/state_coupling_matrix.py`

#### 主要联动链
1. `boredom → loafing`
2. `activity → thinking → behavior`
3. `chat_fuel → reply_style_tier`
4. `loafing → interruption_block`
5. `relation → visibility`
6. `group environment ↔ state`

#### 结论
这是一层从多个原始状态推导出行为风格和可见性偏置的耦合引擎，属于**派生层**，不应反向污染真值层。

---

### 4.6 记忆存取与重激活

#### 记忆核心
**文件**：`src/memory_system/memory_core.py`

##### 关键职责
- 存记忆
- 查记忆
- 重要性调权
- 海马体同步
- 重复检测

##### 关键函数
- `deposit_memory()`
- `query_memories()`
- `format_query_for_prompt()`
- `execute_routine_maintenance()`
- `_apply_emotion_weight()`
- `_sync_to_hippocampus()`

#### 记忆重激活门控
**文件**：`src/core/memory_reactivation_gate.py`

##### 关键函数
- `evaluate_topic_trigger()`
- `evaluate_user_return_trigger()`
- `evaluate_emotion_trigger()`
- `evaluate_boredom_trigger()`
- `evaluate_visibility_aware()`

#### 结论
记忆链已经具备“按话题 / 回归用户 / 情绪波动 / 无聊 / 可见性”激活旧内容的能力，这对主动行为和回复生成都在起作用。

---

### 4.7 统一规划器接入

**文件**：`src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py`

#### 输入给规划器的关键数据
- `world_snapshot`
- `memory_hint`
- `context_execution_block`
- `relation_snapshot`
- `active_intentions`
- `group_pattern_hint`
- `repeated_topic_pressure`
- `last_user_intent`

#### 结论
规划器不是孤立工作的，它吃的是“世界快照 + 记忆提示 + 关系提示 + 群体模式 + 意图池”的混合上下文。

---

## 5. 当前结构的割裂点

### 5.1 同语义多真源

同一语义被多处重复维护：
- `social_value`
- `trust_value`
- `trust_score`
- `affection`
- `fondness_value`
- `annoyance_value`
- `psychological_pressure`
- `trauma_score`

这会导致：
- 视图不一致
- 导出层互相覆盖
- 依赖方难以判断谁是权威字段

---

### 5.2 默认值污染

大量聚合层存在“`or 默认值`”和 truthy 判断，真实的 `0` / 负值 / 低值 会被洗回中性。

结果就是：
- 负面状态看起来没那么负面
- 低能量看起来像正常
- 未设置值与真实零值无法区分

---

### 5.3 软提示和硬门控混用

目前很多逻辑既是 prompt 提示，又是策略输入，甚至还会被反向作为执行条件。

这会让系统出现：
- 提示词污染决策
- 决策结果反向污染提示词
- 调试时不知道哪一层才是“真正判定”

---

### 5.4 主动公式重复

主动意愿至少有三套近似实现：
- `emotion_driven_core`
- `emotion_feedback_loop`
- `adaptive_threshold_learner_v2`

这不是“冗余实现”，而是**主动性定义不统一**。

---

### 5.5 桥接层职责过重

`world_snapshot.py`、`scene_planner_bridge_mixin.py`、`heartfc_state_exporter.py` 都承担了：
- 聚合
- 归一化
- 视图转换
- 提示词注入
- 兼容字段回填

这些层已经很容易变成“上下文污染中心”。

---

## 6. 优化优先级

### P0：先统一真值契约

先明确以下对象谁是权威源：
- 用户关系
- 频道氛围
- 主动意愿
- 群体态势
- 记忆重激活

没有这一步，后面所有重构都会继续分叉。

### P1：拆出“真值层 / 派生层 / 提示层”

建议分成三层：
1. **真值层**：唯一可写
2. **派生层**：可重算，可缓存
3. **提示层**：只给 LLM，看得见但不回写

### P2：合并主动行为计算入口

保留一个主动决策权威入口，其余都做适配。

### P3：整理 `world_snapshot` 的契约

把 `TargetUserState` 中的别名字段和权威字段明确区分：
- 哪个是事实
- 哪个是兼容别名
- 哪个只是导出视图

### P4：给记忆重激活单独定义边界

记忆激活只做“记忆取回”，不要顺手改决策策略。

---

## 7. 可直接复用的结论

1. **情绪系统**：频道级氛围 + bot 内在驱动 + 用户级关系情绪并存。
2. **值系统**：社交值、关系值、能量值、创伤值都存在，但语义边界未完全收口。
3. **决策系统**：已经形成从维度投票到网关裁定的收口链。
4. **记忆系统**：已经接入主动行为和规划器上下文，不再是旁路。
5. **最大问题**：不是缺功能，而是缺统一的真值边界与派生边界。

---

## 8. 关键文件索引

- `src/chat/heart_flow/emotion_stream.py`
- `src/chat/heart_flow/emotion_driven_core.py`
- `src/modules/modcore/dynamic_persona/emotion_tracker.py`
- `src/modules/social_value/models.py`
- `src/modules/social_value/social_calculator.py`
- `src/modules/social_value/settlement_engine.py`
- `src/modules/social_value/social_affect_fuser.py`
- `src/modules/social_value/phase_tracker.py`
- `src/chat/heart_flow/decision_gateway.py`
- `src/chat/heart_flow/enhanced_modules/strategy_action_state_mixin.py`
- `src/chat/heart_flow/enhanced_modules/strategy_proactive_decision_mixin.py`
- `src/chat/proactive/proactive_decider.py`
- `src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py`
- `src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py`
- `src/core/world_snapshot.py`
- `src/core/state_coupling_matrix.py`
- `src/memory_system/memory_core.py`
- `src/core/memory_reactivation_gate.py`
- `src/modules/perception/group_sense.py`
- `src/modules/perception/user_state.py`
- `src/modules/perception/self_sense.py`

---

## 9. 备注

这份文档只记录分析结论，不修改运行时逻辑。

后续如果要继续推进，建议下一步直接把“真值层 / 派生层 / 提示层”拆出来，再逐步收敛重复字段。

---

## 10. 可执行落地清单

这一节不是分析结论，而是下一轮真正适合落地的写法顺序。

### 10.1 先定真值契约

先为下面五类状态定权威源：
- 用户关系
- 频道氛围
- 主动意愿
- 群体态势
- 记忆重激活

每一类只允许一个可写入口，其余全部改成只读视图或派生视图。

### 10.2 再拆三层边界

建议直接拆成三层：
1. **真值层**：只负责存储和更新
2. **派生层**：只负责重算、聚合、缓存
3. **提示层**：只负责给 LLM 看，不反向回写决策

这一步的目标不是重构所有文件，而是先把职责边界钉死。

### 10.3 优先收口的字段

优先处理这些重复字段：
- `affection` / `fondness_value`
- `trust_value` / `trust_score`
- `annoyance_value` / `annoyance`
- `social_value` / `social_score`
- `psychological_pressure` / `inner_chaos`
- `trauma_score` / `negative_emotion_aggregate`

原则是：**保留一个权威字段，其它全部降级为别名或导出视图**。

### 10.4 主动行为只保留一个入口

当前主动行为已经有多个近似公式，下一步应该只保留一个权威决策入口。

建议顺序：
- 先统一 `emotion_driven_core` 的主动意愿定义
- 再让 `proactive_decider` 做融合器
- 其他模块只做适配，不再单独定义主动概率

### 10.5 记忆只做取回，不做裁定

`memory_core` 和 `memory_reactivation_gate` 的职责应该更窄：
- 负责找回相关记忆
- 负责给出重激活候选
- 不直接修改回复决策阈值

真正的裁定仍然交给决策网关和最终动作收口层。

### 10.6 文档后续写法

如果继续扩写这份文档，建议按下面顺序追加：
1. 真值契约表
2. 字段权威源清单
3. 派生层/提示层划分图
4. 决策链路调用顺序图
5. 需要优先拆分的文件列表

---

## 11. 结束语

当前分析已经足够支撑后续重构的第一轮切分。

真正该做的，不是继续加更多别名，而是把“谁能写、谁只能读、谁只能提示”分清楚。