# SRC 全局值体系/机制污染深度审计

日期: 2026-04-15
范围: `src/` 全仓核心状态系统、关系系统、夜间系统、主动系统、回复系统、提示词注入系统
目标: 识别所有核心值的来源、流转、消费点、约束层级、算法重复、上下文污染、默认值污染、职责混叠，并给出后续彻底修复的落地方案

---

## 0. 修复进展（2026-04-15 第四~第五轮落地）

本轮已完成“运行时断链 + 关系字段收口 + 记忆链中层收口”的硬修，且已通过静态与回归验证。

- 运行时断链清零（已落地）
  - `unified_planner` 改为 canonical 消息 API，不再依赖失效的 `message_repository.get_message_repository`。
  - `memory_core` 状态仓库入口改为 `acquire_persistent_store`。
  - `heartFC` 夜间整理入口改为 `acquire_nightly_tidier`，并在无单例时本地创建 `NightlyMemoryTidier` 兜底。
  - `heartFC` 自适应管线改为真实接口：`collect_event + run_pipeline`。
  - `heartflow_message_processor` 改用 `src.modules.perception.user_state.get_user_state_detector().feed_interaction(...)`。

- 关系真源收口（已落地）
  - `heartFC_chat_enhanced` 移除对 `get_social_value_core` 的回填依赖，统一改读 `world_snapshot / cached_relation_result`。
  - `adaptive_threshold_learner_v2` 主路径读取改为 canonical：`trust_value / annoyance_value`。
  - `group/private generator` 去除 `AffectionStageMapper` 失效依赖，改为本地阶段映射函数（仅消费 canonical rapport）。

- 记忆与行为链可激活性修复（已落地）
  - `group_generator` 内心规划器入口改为 `get_narration_planner`。
  - 记忆过载读取改为 `acquire_pressure_monitor + collect_overload_diagnostics`，并保留 `overload_governor` 增强提示。
  - 学习状态改为 `learning_hub` 真源，不再依赖失效的 `dynamic_vocabulary`。
  - `group_persona_manager` 群话题改读 `hippo_memorizer.summary_storage`。
  - `sticker_policy` 改用 `EmojiManager` 可用链路（初始化/情绪匹配/随机回退）与 `emoji_api.register_emoji` 收集。
  - `intrinsic_drive` 活力读取改为 `EnergyChainDimension`，移除失效 `engagement_energy`。
  - `emotion_tracker` 角色年龄读取改为 `global_config.personality.character_age`。

- 验证结果
  - `uv run python -m compileall -q src` 通过。
  - `uv run python -m ruff check src --target-version py312 --select F821,F823` 通过。
  - `uv run python -m unittest test_heart_flow_regressions.py`：`82/82` 通过。

---

## 1. 结论先看

这套系统当前不是“某一个模块有 bug”，而是存在一整条结构性问题链:

1. 同一语义的值被多个系统同时维护，没有单一真源。
2. 同一类决策公式被多个模块重复实现，且权重不同。
3. 大量聚合层通过 `or 默认值`、truthy 判断、别名桥接，把真实的 `0 / 负值 / 低值` 洗回中性。
4. `heartFC_chat_enhanced.py` 同时承担了采集、归一化、融合、回写、提示词注入、仪表盘输入、ACFN 输入、验证打点等职责，成为上下文污染中心。
5. 许多所谓“约束”实际上只是软提示或观测输出，并不是硬门控；但代码里没有清晰分层，导致逻辑互相打架。

这不是单点修补能彻底解决的问题，必须把“值体系”和“约束体系”拆层，明确:

- 哪些值是原始账本
- 哪些值是导出视图
- 哪些值是策略推导
- 哪些值只能做 prompt 提示，不能反过来当真值

---

## 2. 本轮仓内证据

### 2.1 默认值污染扫描

在以下范围内扫描:

- `src/chat/heart_flow/heartFC_chat_enhanced.py`
- `src/core`
- `src/chat/proactive`
- `src/chat/replyer`
- `src/modules/brain`
- `src/modules/social_value`

命中模式:

- `or 50.0`
- `or 0.5`
- `or 0.3`
- `or 0.30`
- `or 0.50`
- `or 999.0`

结果: 命中 `77` 处。

这说明“默认值污染”不是零散现象，而是贯穿聚合层、决策层、提示词层、视图层的结构性模式。

### 2.2 主动意愿公式重复

当前至少有 3 套相关实现:

1. `src/chat/heart_flow/emotion_driven_core.py`
2. `src/core/emotion_feedback_loop.py`
3. `src/core/adaptive_threshold_learner_v2.py`

三者都在计算或近似重建:

- `boredom`
- `loneliness`
- `social_desire`
- `proactive_willingness`

但权重和附加因子不一致。

### 2.3 关系值重复来源

当前至少同时存在以下并行关系源:

1. `src/modules/social_value/social_value_core.py`
2. `src/modules/social_value/social_affect_fuser.py`
3. `src/chat/heart_flow/fondness_trust.py`
4. `src/modules/modcore/dynamic_persona/emotion_tracker` 的 `affection / trust_score / annoyance`
5. `src/core/impression_evolution_hub.py`
6. `src/core/world_snapshot.py` 的别名桥接层

这导致:

- `social_value`
- `trust_value`
- `trust_score`
- `affection`
- `fondness_value`
- `annoyance_value`

并不是一套稳定的一一对应关系，而是多源拼接。

---

## 3. 全局值体系地图

### 3.1 资源链

#### A. D6 能量链: `src/chat/heart_flow/energy_manager.py`

主要字段:

- `chat_pool / chat_ceiling`
- `thinking_value / thinking_ceiling`
- `annoyance_level`
- `activity_level`
- `social_value`
- `combined_ratio()`

职责:

- 频道级资源账本
- 夜间打扰/唤醒阈值
- 回复、思考、被忽略、深夜打扰的即时资源变化
- 与夜间系统联动

特征:

- 明显带“硬资源账本”属性
- 已经有链式消耗和夜间适配
- 更像真实运行时资源层

关键公式:

- 恢复阻尼: `damping = max(0.15, combined * 0.6 + 0.25)`
- 夜间恢复再乘 `0.5`
- 回复成本倍率: `0.4 / 0.7 / 1.0`，取决于 `combined_ratio`
- 单次回复最多只能扣当前剩余的一半:
  - `final_chat_cost <= _chat_remaining * 0.5`
  - `final_think_cost <= _think_remaining * 0.5`

问题:

1. 它已经像“主资源账本”，但旁边又存在一套 `MetabolismEngine`。
2. `social_value` 被放在能量链里维护，但关系系统里也有 `social_value`。
3. `annoyance_level` 是频道级烦躁，和用户级 `annoyance_value` 语义不同，但主链里经常混着用。

#### B. 代谢引擎: `src/core/metabolism_engine.py`

主要字段:

- `chat_fuel`
- `thinking_fuel`
- `activity_gauge`
- `social_gauge`
- `boredom_level`
- `loafing_level`
- `fatigue_accumulator`
- 4 个额外资源池:
  - `group_observation_fuel`
  - `pattern_analysis_fuel`
  - `impression_evolution_fuel`
  - `scene_adaptation_energy`

职责:

- 行为消耗模型
- 恢复模型
- 产出行为约束摘要

关键公式:

- 行为成本表 `_BEHAVIOR_COST_TABLE`
- 成本会乘:
  - `intensity`
  - `fatigue_factor()`
  - `length_scale`
  - `repeat_mod`
  - `circadian_mod`
  - `social_mod`
  - `resonance_mod`
- 恢复会乘:
  - `quiet_bonus`
  - `night_mod`
  - `circadian_mod`
  - `group_boost`
  - `complexity_penalty`
  - `sleep debt modifier`

问题:

1. 与 D6 高度重叠:
   - `chat_fuel` vs `chat_pool`
   - `thinking_fuel` vs `thinking_value`
   - `activity_gauge` vs `activity_level`
2. `boredom / loafing` 在这里是 0~100，但 `emotion_driven_core` 里又有 0~1 的无聊链。
3. 这里只产出“行为约束摘要”，但主链又会把它进一步转写成 dashboard/raw/prompt/input，重复解释。

结论:

- D6 和 Metabolism 不能继续并列做“一等公民”。
- 必须二选一做主账本，另一套退化为派生视图或高级资源池。

---

### 3.2 情绪链

#### A. 情感驱动核心: `src/chat/heart_flow/emotion_driven_core.py`

主要字段:

- `boredom`
- `environmental_fatigue`
- `loneliness`
- `social_desire`
- `mood`
- `energy`
- `curiosity`
- `proactive_willingness`

职责:

- 自主感受层
- 依据沉默和未回复累积无聊/孤独
- 生成主动说话意愿

主动意愿公式:

- `boredom * 0.35`
- `loneliness * 0.28`
- `social_desire * 0.22`
- `mood * 0.05`
- `time_factor * 0.05`
- `curiosity * 0.05`
- `monitoring + 0.12`
- 低能量乘 `0.5` 或 `0.75`

问题:

1. 这是“原始主观感受层”，但不是唯一主观感受层。
2. `energy` 仍然是抽象 0~1 值，不直接绑定 D6 或 Metabolism 的唯一主账本。

#### B. 情感反馈环: `src/core/emotion_feedback_loop.py`

职责:

- 把 ACFN、情境解释、安全、代谢、夜间、主动结果等回写成情绪增量

问题核心:

它不是简单“同步”，而是在重算一套自己的情绪状态。

其 `compute_proactive_willingness()` 公式为:

- `boredom * 0.30`
- `loneliness * 0.25`
- `social_desire * 0.20`
- `mood * 0.10`
- `curiosity * 0.05`
- `-(1 - energy) * 0.10`

这与 `emotion_driven_core` 不一致。

额外问题:

1. `submit_from_metabolism()` 又把代谢结果转成情绪值:
   - `energy_ratio -> energy`
   - `boredom_level -> boredom`
   - `loafing_level -> social_desire` 的负向修正
2. `submit_from_night()` 又把夜间阶段重新写成 mood/energy/curiosity 压制
3. `submit_from_user_interaction()` 又再调 mood/energy/curiosity/social_desire

结论:

- 当前“情绪值”不是单一系统，而是 `emotion_driven_core + feedback_loop` 双系统叠加。
- 如果不统一，后续任何“无聊值/社交欲/主动意愿”调参都会出现前后层互相抵消或重复叠加。

#### C. 自适应阈值学习器 v2: `src/core/adaptive_threshold_learner_v2.py`

问题:

它再次维护:

- `_boredom_level`
- `_loneliness_level`
- `_social_desire_level`

并再次按“参考 emotion_driven_core”的方式计算主动意愿。

结论:

- 这是第三套同类逻辑。
- 这类模块应该读取主情绪状态，不应该自行再维护一份情绪账本。

---

### 3.3 关系链

#### A. 社交值核心: `src/modules/social_value/social_value_core.py`

职责:

- 管理 `social_value`
- 内容/行为 → 算法更新 → 阻尼 → 阶段加速 → 心理修正 → 持久化

核心算法链:

1. `base_score * intent_multiplier * severity * type_bonus`
2. 渐进限制
3. 阈值加速
4. 累积平衡

这是偏“量化真值层”的关系总分。

#### B. 社交情感融合器: `src/modules/social_value/social_affect_fuser.py`

职责:

- `social_value_core + phase_tracker + emotion_reader` 的融合入口

这本来应该是统一只读出口，但目前并没有彻底替代其它关系源。

#### C. 阶段追踪器: `src/modules/social_value/phase_tracker.py`

职责:

- 把 `social_score` 离散化为:
  - `HOSTILE`
  - `STRANGER`
  - `ACQUAINTANCE`
  - `FAMILIAR`
  - `CLOSE`
  - `TRUSTED`

它属于“阶段层”，不是原始值层。

#### D. D2 好感/信任系统: `src/chat/heart_flow/fondness_trust.py`

职责:

- 单独维护:
  - `fondness_value`
  - `trust_value`

关键问题:

1. 它也是 per-user-per-channel 关系系统。
2. 它通过桥接把 `fondness_value` 部分同步到 `emotion_tracker.affection`。
3. 它与 `social_value_core` 没有真正单向从属关系。

这意味着:

- `fondness_value`
- `affection`
- `social_value`

都可能在不同链上各自变化。

#### E. 印象枢纽: `src/core/impression_evolution_hub.py`

职责:

- 真值层 + 主观标签层 + 叙事层 + 行为映射层

问题:

1. 它不是纯读模型，而是会根据交互继续推导 `social_value / trust / affection / pressure / trauma` 的补足值。
2. `assemble_relation_snapshot()` 是一个多源聚合降级口，会在外部缺失时用本地印象推导值。
3. 这在工程上很容易演化成“假的权威源”。

#### F. WorldSnapshot: `src/core/world_snapshot.py`

职责:

- 统一导出快照

当前问题:

- `TargetUserState` 同时包含:
  - `social_value`
  - `favorability`
  - `trust_value`
  - `trust_score`
  - `affection`
  - `annoyance_value`

更糟的是，`to_rapport_dict()` 仍然有别名桥接:

- `affection = u.favorability or u.affection`
- `trust = u.trust_score or u.trust_value`

这意味着在统一出口层依然存在 alias 污染。

---

## 4. 当前硬约束 / 软约束 / 旁路回写

### 4.1 真正接近硬约束的部分

这些值或系统会直接阻止行为:

1. `energy_manager.is_exhausted()` / `combined_ratio`
2. `night_cycle_system` 的深睡/熬穿态
3. `safety_assessment.block_reply`
4. `state_dashboard` 的 L1 `process=False`
5. 主链中的管理员安全守卫和骚扰强拦截

### 4.2 软约束

这些更多是加权或提示，不应被当作主账本:

1. `emotion_feedback_loop`
2. `state_coupling_matrix`
3. `planner_prompt_injection`
4. `group_generator` / `private_generator` 的 behavioral directive
5. `impression_evolution_hub` 的行为映射层

### 4.3 旁路回写

这类系统不是门控，而是会反向污染主状态:

1. `emotion_feedback_loop.submit_from_metabolism`
2. `fondness_trust -> emotion_tracker.affection` 桥接
3. `impression_evolution_hub` 的本地补足关系快照
4. `heartFC_chat_enhanced.py` 里把多种 `_cached_*` 状态重新组装后再喂给 dashboard / ACFN / prompt

结论:

当前系统的最大问题不是“值多”，而是:

- 原始值
- 派生值
- 视图值
- prompt 值
- 回写值

没有边界。

---

## 5. 核心污染类型

### 5.1 默认值污染

表现:

- `value or 0.5`
- `value or 50.0`
- `value or 0.3`

后果:

- `0` 被当缺失
- 负值被中性化
- 真实低值被提升成“正常”

高风险区域:

- `heartFC_chat_enhanced.py`
- `world_snapshot.py`
- prompt 注入与仪表盘 raw 汇总段

### 5.2 上下文污染

表现:

- 一个 mega aggregator 从几十个 `_cached_*` 变量里抓值
- 同一个值既可能来自缓存 A，也可能来自缓存 B，也可能来自 fallback C
- 写入 dashboard、planner、ACFN 前再次做一轮默认化

高风险中心:

- `src/chat/heart_flow/heartFC_chat_enhanced.py`

这是当前最主要的“上下文污染汇聚点”。

### 5.3 机制污染

表现:

- 同一语义公式被多套系统重写
- 例如:
  - 主动意愿
  - 关系好感
  - 夜间压制
  - 社交开放度

后果:

- 参数改一处不生效
- 某值升高后被另一层压回
- 某系统以为自己在做硬约束，实际只是旁路提示

### 5.4 尺度污染

当前混用:

- `0~1`
- `0~100`
- `-100~100`
- 布尔态
- 枚举态

典型问题:

1. `boredom` 有 0~1 版本，也有 0~100 版本。
2. `social_value` 有:
   - `-100~100` 的关系总分
   - `0~100` 风格映射式代理值
   - `0~1` 决策归一化版本
3. `trust` 同时存在:
   - `trust_value`
   - `trust_score`
   - `phase_weight * social_score` 派生信任

---

## 6. 当前最严重的结构性问题

### P0-1. 双资源主账本并存

并存对象:

- `energy_manager.py`
- `metabolism_engine.py`

影响:

- 回复长度
- 活跃度
- 疲劳/休息
- 夜间响应
- 话痨抑制

这是资源链最根本的冲突。

### P0-2. 主动意愿三套实现

并存对象:

- `emotion_driven_core.py`
- `emotion_feedback_loop.py`
- `adaptive_threshold_learner_v2.py`

影响:

- 主动发言触发
- 接管决策
- 低能量时主动概率
- 夜间是否被唤醒

### P0-3. 关系真值层不唯一

并存对象:

- `social_value_core`
- `fondness_trust`
- `emotion_tracker`
- `impression_evolution_hub`
- `world_snapshot alias`

影响:

- 好感
- 信任
- 烦躁
- 关系等级
- prompt 语气
- 关系叙事

### P0-4. 主链装配过于集中

`heartFC_chat_enhanced.py` 当前不仅是聊天主链，还是:

- 值采集器
- 值归一化器
- 状态拼装器
- dashboard 原始输入构造器
- ACFN 输入构造器
- feedback 注册器
- validator 打点器

这会持续制造上下文污染。

### P1-1. 统一出口仍有别名污染

`world_snapshot.to_rapport_dict()` 仍然用 `or` 做别名桥接。

这说明:

- 即使上游值清洗过，统一出口层仍可能把值重新污染。

### P1-2. CrossEngineValidator 主要是观测，不是强执行

当前它更像“联动审计日志”，不是强约束框架。

问题:

- 能记录“谁产出谁消费”
- 但不能阻止值冲突
- 不能保证只有一个权威源

### P1-3. Dashboard/Planner/Prompt 层仍然在再解释值

理想情况下:

- 仪表盘应该只展示
- prompt 注入应该只消费

但当前它们仍在:

- 做默认值回退
- 做尺度推断
- 做二次映射

---

## 7. 修复总原则

后续彻底修复必须遵循下面 8 条规则:

1. 原始账本只允许单一写入源。
2. 派生视图只能读原始账本，不能再写回原始账本。
3. Prompt 层不能持有“真值”。
4. 视图层不能再做尺度转换推断。
5. 同名值只能有一种 canonical scale。
6. 所有 `0 / 负值 / False` 必须被视为合法值，不能通过 truthy 判断过滤。
7. 硬约束与软约束必须分层。
8. `heartFC_chat_enhanced.py` 不再直接拼 40+ 个 `_cached_*` 值给下游。

---

## 8. 建议的新值体系

### 8.1 层级划分

#### Layer 0: Raw Ledger 原始账本层

只允许以下系统作为原始账本:

1. 资源账本
   - 推荐保留 `energy_manager`
   - `metabolism_engine` 降级为“高级行为代谢派生器”
2. 情绪账本
   - 推荐保留 `emotion_driven_core`
   - `emotion_feedback_loop` 改成“增量提交器 + 单一回写器”，不再自己持有第二套状态定义
3. 关系账本
   - 推荐保留 `social_affect_fuser` 作为统一只读出口
   - `social_value_core` 为量化真值
   - `phase_tracker` 为阶段层
   - `emotion_tracker` 只保留情绪态，不再充当关系总分来源

#### Layer 1: Derived View 派生视图层

允许存在:

- `world_snapshot`
- `state_dashboard`
- `state_coupling_matrix`

但要求:

- 只能读 Layer 0
- 不能再决定默认真值
- 不能自创一份同名主值

#### Layer 2: Policy / Constraint 策略层

允许存在:

- `night_cycle`
- `ACFN`
- `decision_gateway`
- `proactive_decider`

但必须只读取 canonical state contract。

#### Layer 3: Prompt / UX 层

允许存在:

- `planner_prompt_injection`
- `group_generator`
- `private_generator`

职责只剩:

- 文本解释
- 风格包装
- UX 层展现

不准再改写原始状态。

---

## 9. 建议的 canonical contract

建议新增统一值协议，例如 `src/core/value_contract.py`

### 9.1 资源协议

```python
@dataclass
class ResourceLedger:
    chat_energy_100: float
    thinking_energy_100: float
    activity_100: float
    channel_annoyance_100: float
    channel_social_100: float
    night_energy_gate_01: float
```

规则:

- 原始资源只用 `0~100`
- 归一化只在消费边缘做
- `*_ratio` 不存储，只计算

### 9.2 情绪协议

```python
@dataclass
class EmotionLedger:
    boredom_01: float
    loneliness_01: float
    social_desire_01: float
    mood_01: float
    curiosity_01: float
    proactive_drive_01: float
```

规则:

- 只允许一套主动意愿公式
- `adaptive_threshold_learner_v2` 只能读取，不能再自建第二账本

### 9.3 关系协议

```python
@dataclass
class RelationLedger:
    social_value_m100_100: float
    trust_value_m100_100: float
    affection_m100_100: float
    annoyance_0_100: float
    trauma_0_100: float
    pressure_0_100: float
    phase: str
```
```

规则:

- 明确保留负值区间
- `trust_score` 与 `trust_value` 必须二选一做 canonical
- `favorability` 只能是兼容别名，不能继续广泛参与判断

---

## 10. 后续彻底修复计划

### P0. 冻结值协议

1. 定义 canonical dataclass
2. 把尺度写死
3. 禁止任何 `or 0.5 / or 50.0` 式回退进入核心链

### P1. 砍掉重复公式

1. 主动意愿只保留 1 套
2. 夜间压制只保留 1 套主入口
3. 关系总分只保留 1 套总账本

### P2. 拆分 mega aggregator

把 `heartFC_chat_enhanced.py` 拆成:

1. `state_ingest_service`
2. `state_normalize_service`
3. `decision_input_builder`
4. `prompt_input_builder`
5. `telemetry_recorder`

### P3. 视图层去写值

以下模块禁止再拥有主值决定权:

1. `state_dashboard`
2. `planner_prompt_injection`
3. `group_generator`
4. `private_generator`
5. `world_snapshot` 的兼容别名输出

### P4. 建立值污染测试

必须补的测试类型:

1. `0 / 负值` 保留测试
2. 0~1 与 0~100 不混尺度测试
3. 单一主账本写入测试
4. prompt 层只读测试
5. ACFN / dashboard / planner 输入一致性测试

---

## 11. 我建议的重构落地顺序

### 第一批: 先止血

1. 清掉 `heartFC_chat_enhanced.py` 里所有核心值的 `or 默认值` 污染
2. 清掉 `world_snapshot` 统一出口里的 alias 污染
3. 把 `dashboard` / `planner` / `replyer` 限制为只读消费层

### 第二批: 收主账本

1. 资源只留一套主账本
2. 情绪只留一套主账本
3. 关系只留一套主账本

### 第三批: 重接下游

1. ACFN 只读 canonical contract
2. Prompt 注入只读 canonical contract
3. 群聊/私聊生成器只读 canonical contract

### 第四批: 删除旧桥接

1. 删除 `favorability / trust_score` 之类的历史别名判断
2. 删除重复主动意愿公式
3. 删除主链中临时拼出来的“伪 relation snapshot”

---

## 12. 当前结论总结

这套系统现在最大的问题，不是某一个公式错了，而是:

- 同一个值被多套系统分别维护
- 同一个值被多层系统重复解释
- 同一个值在输出时又被默认值洗白
- 同一个值进入不同决策器时尺度还不统一

所以你前面感觉到的这些现象:

- 话痨和疲困互相打架
- 熬夜和清醒逻辑来回覆盖
- 好感和讨厌度不结算或结算错位
- 无聊、摸鱼、社交欲、活跃值互相串
- prompt 里像一个人，真正决策像另一个人

从代码层面看，都是合理的结构后果，不是错觉。

---

## 13. 下一步建议

如果下一轮继续实做，建议直接按下面顺序推进:

1. `heartFC_chat_enhanced.py` 核心值汇总段去污染
2. `world_snapshot` / `state_dashboard` / `planner_prompt_injection` 全出口 contract 化
3. `emotion_driven_core + emotion_feedback_loop + adaptive_threshold_learner_v2` 三套主动意愿统一
4. `energy_manager + metabolism_engine` 双资源链归一
5. `social_value_core + fondness_trust + impression_hub + emotion_tracker` 关系主账本归一

这份文档就是后面整轮“彻底修复”的施工底稿。

---

## 14. 第二轮补充证据

这部分是本轮继续全仓深排后新增确认的证据，重点覆盖:

- `heartFC_chat_enhanced.py` 以外的边缘污染链
- 模型层默认中值污染
- 单位/字段协议不一致
- 已经能够确认会真实改行为的高置信度 bug

### 14.1 默认值污染不只发生在读取口，也发生在数据模型定义口

上一轮主要抓的是 `or 0.5 / or 50.0` 型读取污染。

这一轮继续扫后确认，很多值在“尚未读取前”就已经被数据模型初始化成中性，后续所有消费者都会把它当成真状态。

按 `: float = 0.5 / : float = 50.0` 统计，命中较多的文件包括:

1. `src/core/impression_evolution_hub.py` `15` 处
2. `src/core/multi_factor_decision_engine.py` `12` 处
3. `src/modules/modcore/dynamic_persona/persona_generator.py` `11` 处
4. `src/core/state_coupling_matrix.py` `10` 处
5. `src/modules/recall/models.py` `8` 处
6. `src/core/night_cycle_system.py` `7` 处
7. `src/core/state_dashboard.py` `7` 处
8. `src/modules/perception/understand.py` `7` 处
9. `src/chat/heart_flow/heartflow_decision.py` `7` 处

这意味着:

1. 不是只有“脏回退”，还有“脏初始化”
2. 中值默认已经进入 dataclass / engine state / parser result 层
3. 后续即使把部分读取口修干净，只要模型层还在生成中性假值，污染还会继续流动

### 14.2 `heartFC_chat_enhanced.py` 仍然是核心污染中心，但不是唯一污染中心

按以下模式:

- `or 50.0`
- `or 0.5`
- `or 0.3`
- `or 0.30`
- `or 0.50`
- `or 999.0`

继续扫后，读路径污染文件分布为:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py` `73` 处
2. `src/memory_system/deep_optimizer.py` `3` 处
3. `src/core/multimodal_budgeter.py` `2` 处
4. `src/memory_system/memory_core.py` `1` 处
5. `src/modules/social_value/social_value_core.py` `1` 处

说明:

1. `heartFC_chat_enhanced.py` 仍然是最大污染放大器
2. 但真正危险的是“边缘模块也在洗值”，因为它们会把污染重新送回主链
3. `memory / multimodal / social_value` 这些边缘链并不只是旁枝，而是会真实影响主行为

### 14.3 单位混用已经不是个别现象，而是跨模块协议问题

#### A. `activity_level` 至少有三种单位

当前仓内同时存在:

1. 0~100 数值:
   - `src/chat/heart_flow/energy_manager.py`
   - `src/core/night_cycle_system.py`
   - `src/core/unified_planner.py`
2. 0~1 数值:
   - `src/modules/perception/understand.py`
   - `src/modules/perception/group_atmosphere.py`
3. 字符串等级:
   - `src/modules/perception/group_sense.py`
   - `src/modules/perception/message_preprocessor.py`

这会带来两个结果:

1. 同一个字段名在不同模块里不代表同一个尺度
2. 一旦某层遗漏归一化，就会把“0.5 活跃”当成“50 活跃”，或者把“冷清”直接塞进需要数值的地方

#### B. `relationship_depth / group_atmosphere` 在 recall 链里甚至同名异型

`src/modules/recall/models.py` 里的 `DimensionFactors`:

- `relationship_depth: float = 0.5`
- `group_atmosphere: float = 0.5`

而 `src/modules/recall/dimension_collector.py` 里的 `DimensionFactors`:

- `relationship_depth: str = "陌生"`
- `group_atmosphere: str = "普通"`

这不是“实现细节不同”，而是协议本身不一致:

1. 同名类型不同
2. 默认值语义不同
3. 下游如果引用错模型，会出现完全合法但完全错误的运行结果

#### C. `social_value` 也不是单一语义

当前它至少被当作:

1. 用户关系真值
2. 频道资源链里的社交状态
3. prompt / planner 的关系代理值

一旦这些不同语义共用同名字段，下游很难知道拿到的是:

- “对这个用户的关系”
- “当前频道整体的社交氛围”
- “某个聚合器拼出来的近似代理”

### 14.4 边缘模块也在持续洗值

#### A. 多模态预算器: `src/core/multimodal_budgeter.py`

问题:

- `context_relevance=0.0` 会被 `or 0.5` 洗成 `0.5`

影响:

1. 原本应低优先级跳过的图片/贴图会被误判成中等相关
2. 多图风暴节流时，抽样排序会偏向“被洗白”的素材
3. 图片预算器不再是被动资源管理，而是在主动扭曲语义优先级

#### B. 记忆优化链: `src/memory_system/deep_optimizer.py`

问题:

- `significance` 读取使用 `or 0.5`
- 低显著度记忆会被静默抬回中显著度

影响:

1. 垃圾清理阈值偏松
2. 重复记忆保留评分偏高
3. 低价值记忆更难被淘汰，长远会反向污染 recall / planner

#### C. 记忆输出桥: `src/memory_system/memory_core.py`

问题:

- `MemoryRecord.significance -> BufferedMemory.importance` 的桥接仍然用 `0.5`

影响:

1. 数据库存量记忆如果显著度真实为 `0` 或极低，会在输出层重新中性化
2. 后续格式化器、摘要器、提示词层看到的是“经过洗白的记忆重要性”

#### D. 决策脑: `src/modules/brain/decision_brain.py`

问题:

- `social_penalty` 缺失时按 `social_value=0.5` 计算

影响:

1. 关系数据缺失时，不会进入“未知风险”而是直接落入“中性风险”
2. 这会让风险决策对缺失关系状态过于宽松

### 14.5 记忆链虽然做了别名统一说明，但实际仍然存在桥接污染

记忆系统当前已经在注释层承认:

- `weight == significance`
- `importance == significance`

对应文件:

1. `src/memory_system/hippocampus_buffer.py`
2. `src/memory_system/memory_models.py`
3. `src/memory_system/memory_core.py`

这本来是好事，说明开发时已经意识到同义字段过多。

但现状仍然有两个问题:

1. “标准字段”并没有真正统一到单一名字，兼容别名依然大量存在
2. 一旦桥接层继续使用 `0.5`，兼容层本身就成了第二污染源

换句话说:

- 注释已经统一
- 运行时协议还没统一

### 14.6 `CrossEngineValidator` 目前仍主要是观测器，不是协议守门员

从实现看，它能做的是:

1. 注册引擎
2. 记录产出
3. 记录消费
4. 统计 orphan / delivered / expired

但它做不到:

1. 阻止两个引擎同时维护同一 canonical value
2. 阻止不同尺度的值进入同一决策输入
3. 阻止 prompt 层把导出值重新写回主账本

所以它当前更像:

- 数据流遥测
- 观测面板

而不是:

- 强协议边界
- 值体系闸门

---

## 15. 本轮新增确认的高置信度 bug

这部分不是“可能有风险”，而是已经从代码路径上能确认会造成语义错位或行为偏差的问题。

### 15.1 Recall 维度收集器把“用户关系值”和“频道社交值”写进同一个字段

文件: `src/modules/recall/dimension_collector.py`

问题链:

1. `_collect_social_factors()` 先把 `emotion_tracker.state.affection` 写进 `factors.social_value`
2. `_collect_psychological_factors()` 随后又把 `EnergyChainDimension._ch.social_value` 写进同一个 `factors.social_value`

结果:

1. 用户关系语义被频道资源语义覆盖
2. 同一个 prompt 里的“社交分”到底指用户关系还是频道社交值，取决于后写入者
3. 这是一个真实的字段污染 bug，不是抽象架构瑕疵

### 15.2 Recall 链存在两份同名 `DimensionFactors`，协议不兼容

文件:

1. `src/modules/recall/models.py`
2. `src/modules/recall/dimension_collector.py`

问题:

1. 同名 dataclass
2. 字段类型不同
3. 默认值语义不同

这会导致:

1. 后续维护者很容易导入错模型
2. 即使代码能跑，语义也会完全偏掉
3. 这类 bug 隐蔽而顽固，因为它不是类型错误，而是“错模型照样能运行”

### 15.3 `heartFC_chat_enhanced.py` 把 `social_willingness` 当 `loneliness` 的兜底

文件: `src/chat/heart_flow/heartFC_chat_enhanced.py`

问题位置:

在情绪快照异常时:

1. 从代谢约束回填 `boredom`
2. 再用 `presence_state.social_willingness` 去兜底 `_loneliness`

这两个值不是同一语义:

- `social_willingness` 更接近“愿不愿社交”
- `loneliness` 是“孤独积累/缺社交痛感”

后果:

1. 愿意社交高，不代表孤独高
2. 不愿意社交，也不代表孤独低
3. 会直接把主动意愿链的情绪驱动力算偏

### 15.4 `world_snapshot.to_rapport_dict()` 仍在做 alias 污染

文件: `src/core/world_snapshot.py`

问题:

1. `affection = favorability or affection`
2. `trust = trust_score or trust_value`

这会带来:

1. `0` 或低值可能被后备字段覆盖
2. 历史字段继续参与真值导出
3. 即使上游修干净，出口层仍然会重新污染

### 15.5 `multimodal_budgeter` 会把真实不相关素材洗成中相关

文件: `src/core/multimodal_budgeter.py`

问题:

1. `context_relevance=0.0` 会被回退成 `0.5`
2. 排序和预算判断都依赖这个值

结果:

1. 多图抽样策略偏离真实上下文
2. 不相关图片更容易被拿去分析
3. 边缘模块开始反向影响主聊天行为

### 15.6 `deep_optimizer` 会把低显著记忆抬回中显著

文件: `src/memory_system/deep_optimizer.py`

问题:

1. 低显著/零显著度通过 `or 0.5` 被抬高中性
2. 重复消除和垃圾清理都依赖该值

结果:

1. 低价值记忆更容易残留
2. 存量垃圾更难被收掉
3. 后续召回、规划和上下文窗口会持续被旧噪音干扰

### 15.7 多因子输入和耦合矩阵仍把缺失关系值默认成 `50`

文件:

1. `src/core/multi_factor_decision_engine.py`
2. `src/core/state_coupling_matrix.py`

问题:

1. `affection/trust_value` 默认就是 `50.0`
2. 一旦关系快照缺失，就落回中亲近/中信任

结果:

1. 真正“未知关系”会被伪装成“普通关系”
2. 冷启动和缺失状态下的行为会偏乐观
3. 关系门控会在最应该保守的时候变得不够保守

### 15.8 `attention_snapshot` 的生产协议和消费协议已经错位

文件:

1. `src/core/subjective_attention_flow.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`
3. `src/core/state_dashboard.py`

问题:

1. `AttentionFlowSnapshot.to_dict()` 输出的 `state` / `previous` 是中文标签，不是机器态枚举值
2. 下游却按 `fully_engaged / withdrawn / deep_withdrawal` 这类英文状态消费
3. `to_dict()` 没有输出 `visibility_threshold / process_ratio / withdrawal_depth / consecutive_empty_peeks`
4. 下游面板和 quick gate 却直接读取这些键

结果:

1. 注意力模式会频繁落回默认分支
2. `AttentionBar` 多个字段长期只是默认值
3. 这是高置信度 active bug，不是单纯“设计不优雅”

### 15.9 `deep_withdrawal / deep_withdrawn` 枚举错拼会让最深退出态失真

文件:

1. `src/core/subjective_attention_flow.py`
2. `src/core/state_dashboard.py`

问题:

1. 实际状态名是 `deep_withdrawal`
2. `state_dashboard` 映射表却写成了 `deep_withdrawn`

结果:

1. 最深退出态在面板层会掉回默认态
2. 部分“沉睡 / 深退场”判断会表现为还在扫描
3. 这是一个独立的 active bug

### 15.10 `ContextManager` 存在“请求 8 条，实际最多 5 条”的静默截断

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/modules/context_manager.py`

问题:

1. 主链明确请求 `get_recent_context(limit=8)`
2. `ContextManager` 内部会把上限强行压到 `_default_max_messages=5`
3. 调用方不会收到任何告警

结果:

1. 调用方以为自己拿到了 8 条上下文
2. 实际上最多只有 5 条
3. 这属于典型的静默协议污染

### 15.11 `SelfAwareness` 空内容写入会让“我刚做了什么”失去正文

文件:

1. `src/modules/recall/self_awareness.py`
2. `src/chat/proactive/proactive_integration_hub.py`

问题:

1. `SelfAwareness.record_action()` / `record_user_event()` 本来支持记录 `content`
2. 但 `ProactiveIntegrationHub` 调用时把 `content` 固定写成空串
3. 同时，全仓未见 `SelfAwareness.record_message()` 被主链稳定接入

结果:

1. 系统知道“做过一次 reply / 收到一次 user_message”
2. 却不知道具体说了什么、用户说了什么
3. 自我觉察更像空壳遥测，而不是正文级自我记忆

### 15.12 `world_snapshot` 出口之后，消费层仍继续二次造关系值

文件:

1. `src/core/world_snapshot.py`
2. `src/chat/heart_flow/inner_voice.py`
3. `src/chat/chat_core_base.py`

问题:

1. `world_snapshot` 已经聚合了 `social_affect_fuser + emotion_tracker`
2. 但 `inner_voice` 和 `chat_core_base` 仍从 `FondnessTrustDimension` 手搓 `trust/annoyance/affection`
3. 它们使用的简化公式和 `world_snapshot` 的输出并不一致

结果:

1. 统一出口并没有真正成为唯一消费入口
2. 下游会再次引入旧 proxy 关系值
3. 这是 relation chain 的 duplicate authority 问题

### 15.13 记忆检索链存在“跳过门局部化 + 上层强搜 + 下层粗搜”的策略互搏

文件:

1. `src/memory_system/memory_retrieval.py`
2. `src/memory_system/retrieval_tools/query_direct_memory.py`
3. `src/memory_system/memory_core.py`
4. `src/chat/replyer/group_generator.py`
5. `src/chat/replyer/private_generator.py`

问题:

1. 生成器层确实有 `should_skip_memory_retrieval()` 这道门
2. 但检索 prompt 又要求“短句必须检索、宁可多搜不可漏搜”
3. `query_direct_memory` 只取第一个 term 当关键词
4. `memory_core.query_memories()` 的长期库检索仍是 `content.contains(keyword)`

结果:

1. 系统不是单纯“全捞”，而是局部门控与整体策略互相打架
2. 检索次数、检索精度和 token 消耗没有统一 governor
3. 这是高置信度的策略级 active bug

### 15.14 `decision_messages` 会把当前消息、历史上下文和自我伪消息混进同一个样本池

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/modules/context_manager.py`

问题:

1. `_prepare_decision_messages()` 先从 `filtered_messages` 取当前轮消息
2. 然后直接把 `ContextManager.get_recent_context()` 和 `search_context()` 返回值包装成 pseudo message 追加进去
3. 接着又把 `_build_synthetic_self_messages()` 产出的 bot 自我伪消息继续追加进去
4. 三种来源最后只按 `timestamp` 排序，不再保留“当前轮 / 历史 / 自我回顾”的边界

结果:

1. 后续模块看到的是一池混合样本，不是纯粹的“本轮待判定消息”
2. 复读检测、骚扰检测、记忆预取、对象提取、旁白规划都会吃到这池混合数据
3. 这是高置信度的 active bug，不是单纯“实现风格比较随意”

### 15.15 当前对象提取会被历史上下文伪消息劫持

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/modules/context_manager.py`

问题:

1. `_integrate_narration_planning()` 会倒序扫描 `decision_messages`，取最后一个非 bot `user_id`
2. `_build_decision_context_packet()` 也会从同一池 `decision_messages` 里倒序找 `latest_user`
3. `_emit_target_profile()` 前的面板目标提取同样直接吃这池混合消息
4. 但历史上下文被包成 pseudo message 后并没有单独的来源优先级或硬标签

结果:

1. 系统本来在看当前发言人，却可能被较新的历史上下文伪消息改写成另一个对象
2. 负面情绪读取、个性标签读取、面板对象、续接对象都会一起漂移
3. 这会直接表现成“明明在回 A，却在读 B 的状态”

### 15.16 窥屏态内心独白绕过统一频控，只剩时间冷却

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/chat/heart_flow/inner_voice.py`

问题:

1. 主反应链平时会通过 `_should_run_voice()` 和 `_compute_llm_call_level()` 决定是否调用内心独白
2. 这套判断里本来会综合夜间、能量、场景、quiet_preference、群模式等状态
3. 但窥屏态 `_run_peek_with_reflection()` 明确写成“有消息到达时始终调用内心独白，仅保留冷却保护”
4. 它只检查 `_last_voice_ts`，绕过了主链的动态阈值 governor

结果:

1. 窥屏态会比主链更容易频繁触发第一层内心活动
2. 它还会写回同一个 `_last_voice_ts`，反过来影响主反应链后续是否还能正常跑独白
3. 这是“独白频控双轨制”的 active bug，也是你感觉它老在想、老在回的一条真实根因

---

## 16. 新增结构性判断

### 16.1 当前最大问题已经不仅是“哪个公式错”，而是“协议层没有冻结”

从本轮证据看，污染有四个入口:

1. 模型定义默认中值
2. 读取时 `or 默认值`
3. 别名桥接时再次回退
4. 语义相近但不等价字段被混写到同一键

只修其中一个入口，系统仍会继续脏。

### 16.2 现在最危险的不是主链，而是“边缘模块反向污染主链”

本轮确认的高风险边缘链:

1. `multimodal_budgeter`
2. `memory_core / deep_optimizer / hippocampus_buffer`
3. `recall.dimension_collector`
4. `perception.understand / group_atmosphere`

这些模块的共同点是:

1. 看起来像辅助模块
2. 但它们会把“解释后的状态”重新送回主决策链
3. 因此一旦写脏，比单纯展示层更危险

### 16.3 “硬约束”和“软约束”混用的本质原因，是输入 contract 不统一

现在很多模块都在接收一个“差不多像状态快照”的东西。

但这些快照之间并不统一:

1. 有的接 0~1
2. 有的接 0~100
3. 有的接字符串标签
4. 有的接 impression proxy
5. 有的接真正资源账本

于是下游看起来像:

- 夜间系统在压制
- 代谢系统在压制
- 主观注意力系统在鼓动
- 主动意愿系统在鼓动

但更深的原因是:

- 它们根本没有建立在同一份输入协议上

---

## 17. 下一阶段建议增加的硬修范围

如果下一轮进入硬修，不建议只修 `heartFC_chat_enhanced.py`，而要把以下几组一起打掉:

### 第一组: 主链汇总口止血

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/core/world_snapshot.py`
3. `src/core/state_coupling_matrix.py`
4. `src/core/multi_factor_decision_engine.py`

目标:

1. 冻结主决策输入 contract
2. 停止把缺失关系值默认成 `50`
3. 停止把 `social_willingness` 之类近似量冒充情绪真值

### 第二组: 边缘反向污染口止血

1. `src/core/multimodal_budgeter.py`
2. `src/memory_system/deep_optimizer.py`
3. `src/memory_system/memory_core.py`
4. `src/modules/brain/decision_brain.py`

目标:

1. 保留合法的 `0`
2. 缺失和中性必须严格区分
3. 不允许预算器/记忆器/风险器在缺失时自造中性真值

### 第三组: Recall 协议收敛

1. `src/modules/recall/dimension_collector.py`
2. `src/modules/recall/models.py`
3. `src/modules/recall/post_send_analyzer.py`

目标:

1. 合并重复 `DimensionFactors`
2. `social_value` 拆成:
   - `relation_social_value`
   - `channel_social_state`
3. 统一 `relationship_depth / group_atmosphere` 的字段类型

### 第四组: 感知层去伪中性

1. `src/modules/perception/understand.py`
2. `src/modules/perception/group_atmosphere.py`
3. `src/modules/perception/group_sense.py`

目标:

1. 明确区分:
   - 模型没给
   - 真的中等
   - 置信度不足
2. `activity_level` 统一 contract，不允许字符串和数值混写到同一层

---

这次补充后，这份文档已经不仅能说明“哪里乱”，也能直接作为下一轮分批硬修的路线图。

---

## 18. 人格化行为链与“像不像人”问题补充审计

这一节不再只看数值污染，而是直接对应你提的这些体感问题：

1. 它到底会不会看自己刚说过的话
2. 它什么时候知道该闭嘴
3. 它会不会一直重复干一件事
4. 它会不会把记忆全捞出来浪费 token
5. 它到底有没有“群里没人就别说、群里刷屏就别硬融”的判断

结论先说:

1. 这些机制大多“不是没有”
2. 但它们分散在多套链里，各自知道一点
3. 真正缺的是一个统一的人格化行为总闸门

### 18.1 `SelfAwareness` 目前更像记账器，不是行为决策器

文件: `src/modules/recall/self_awareness.py`

代码已经明写职责边界:

1. 只负责记录和查询
2. 不做决策判断
3. 决策由模型完成

这说明现在的“自我意识”本质上还是存档层，而不是控制层。

更关键的是:

1. `record_message()` 确实存在
2. 但截至本轮仓内搜索，没有发现主聊天链稳定调用它来记录 bot 自己发出的正文消息

这意味着:

1. 系统理论上有“我说过什么”的容器
2. 但主链并没有把自己的输出稳定写进去
3. 后续自然也就很难把“我刚说过这句”作为硬约束来防重复

这正是“它好像知道自己存在，但又不会拿这个知识约束自己”的典型来源。

### 18.2 `SelfReplyRecognizer` 能识别自消息，但还没升格为统一去重权威

文件: `src/core/self_reply_recognizer.py`

这个模块其实并不弱，它已经支持:

1. 标记 bot 自己发出的消息
2. 按 message_id 和内容哈希识别回流自消息
3. 维护 reply chain
4. 记录用户反应和回复质量

但它目前更像:

1. 一个识别器
2. 一个观察账本
3. 一个质量回写点

而不是:

1. 主回复链的总防重复闸门
2. 统一决定“这句我刚说过，不要再说”的权威入口

所以现在的状态是:

1. 自我识别模块知道一部分“自己说过什么”
2. 主行为链也有一部分“避免重复”的局部护栏
3. 但两边没有强绑定成一个总协议

### 18.3 “会不会看自己上下文”不是没有，而是有多本账

涉及文件:

1. `src/modules/context_manager.py`
2. `src/chat/heart_flow/heartflow_message_processor.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`
4. `src/core/unified_planner.py`

当前至少有三类上下文来源:

1. `ContextManager`
2. message repository 的 recent messages
3. self awareness / self reply 这一套自我记录

其中 `ContextManager` 的问题很典型:

1. `ConversationContext` 默认上限能到 20
2. 但 `ContextManager` 创建频道上下文时把 `_default_max_messages` 设成了 5
3. `get_recent_context()` 还会再次把读取上限硬卡在这个默认值内

也就是说:

1. 主链自以为在看“最近上下文”
2. 实际拿到的是一个最多 5 条的轻量缓存
3. 而其他模块可能又在看 8 条、20 条、或者别的历史来源

这会造成:

1. 有的模块觉得“前情还在”
2. 有的模块觉得“上下文已经断了”
3. 同一轮决策里，不同子系统对“最近发生了什么”根本没对齐

所以用户会体感成:

1. 它有时候像记得
2. 有时候又像完全不记得自己刚说过什么

### 18.4 “什么时候该闭嘴”已经有多套机制，但它们没有统一裁决

涉及文件:

1. `src/chat/heart_flow/frequency_control.py`
2. `src/chat/heart_flow/waiting_handler.py`
3. `src/core/inner_narration_planner.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

当前至少存在四条“闭嘴 / 暂停 / 等待 / 追问”的链:

1. `FrequencyControl`
2. `PendingOrchestrator`
3. `InnerNarrationPlanner`
4. `heartFC_chat_enhanced` 自己的一堆 burst / fatigue / penalty / cooldown 护栏

这几套东西各自都在管一部分:

1. 频率控制负责“冷却”和“连续跳过后强制触发”
2. 等待编排器负责“等多久、何时沉思、何时追问、何时放弃”
3. 旁白规划器负责“该回复 / 观察 / 延后”
4. 主链又额外做一层刷屏、话痨、夜间、主动回复疲劳等门控

所以问题不是“它完全不会闭嘴”，而是:

1. 会闭嘴
2. 但谁说了算并不固定
3. 而且不同链看到的输入也不完全一致

这正是“有时太话痨，有时又突然很闷”的根源之一。

### 18.5 内心所想并没有缺失，但它更像软约束，不是硬控制

文件: `src/core/inner_narration_planner.py`

`InnerNarrationPlanner` 已经会综合这些因素:

1. 理解等级
2. 可见性
3. 能量与社交意愿
4. 用户关系
5. @ / 提问
6. 话题兴趣
7. 学习优先
8. 场景适合度
9. 群体模式
10. 创伤 / 烦躁 / 误解风险 / 用户负面情绪

然后再产出:

1. `should_reply`
2. `should_observe`
3. `should_defer`
4. `inner_thought`
5. `action_intent`

这已经很接近“第一层内心所想”。

但问题在于:

1. 它是一个规划器
2. 不是整条链的唯一裁决器
3. 后面还有 visibility / gateway / autonomy / voice action / model takeover / final decision 等层继续覆盖

所以现在“内心所想”更像:

1. 一份主观建议
2. 一份带情绪的软约束

而不是:

1. 一锤定音的行为总开关

### 18.6 低信息复读护栏存在，但重复控制仍然是“多处各修一点”

涉及文件:

1. `src/chat/heart_flow/heartflow_message_processor.py`
2. `src/chat/heart_flow/inner_voice.py`
3. `src/chat/replyer/group_generator.py`

当前已经有的重复护栏包括:

1. 入口层 `_annotate_repeated_short_input()` 给短复读打标并附带前情提示
2. `inner_voice` 会压低 desire，并把独白收束成“先看看再说 / 无不无聊啊”
3. `group_generator` 会把近期 bot 的低信息短回复拿出来，构造“避免重复”的 prompt 约束

但这三层依然是:

1. 入口提示层
2. 独白层
3. 生成 prompt 层

缺少的是:

1. 一个统一的 repetition ledger
2. 一个全链共用的“我已经说过这个意思了”的 canonical signal

所以现在会出现:

1. 某层已经判断“像复读”
2. 另一层还是把它当新话题
3. 最终还是可能重复行动

### 18.7 群聊认知并不弱，但被拆成了多套场景引擎

涉及文件:

1. `src/core/group_scene_state.py`
2. `src/chat/heart_flow/group_atmosphere.py`
3. `src/modules/perception/group_sense.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

现在系统其实已经会感知:

1. 群里有几个人在说话
2. 有没有 dominant speaker
3. 有没有 burst
4. 有没有 controversy
5. 当前适不适合插话
6. 是否适合保持沉默
7. 是否在冷场 / 悼念 / 围观 / 吵架 / 刷屏

但问题是这些能力被拆散了:

1. `GroupSceneState` 是底层场景快照
2. `GroupAtmosphereDimension` 把它映射成五档氛围和 AI 定位
3. `GroupSense` 又自己做一套 burst / controversy / dominant users / needs_topic
4. `heartFC_chat_enhanced` 最终再把这些结果拼起来决定行为

于是系统会给人一种感觉:

1. 它不是不懂群聊
2. 而是群聊认知没有一个单一真源

这会直接导致:

1. 有的模块觉得“群里没人，别说”
2. 有的模块觉得“氛围一般，可以说”
3. 有的模块觉得“有话题空位，可以插”
4. 最终合成结果就容易摇摆

### 18.8 引用逻辑有改进，但“什么时候该引用”仍然偏启发式

涉及文件:

1. `src/chat/chat_core_base.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`
3. `src/core/message_semantic_router.py`
4. `src/core/multi_factor_decision_engine.py`

基础层里，引用判断仍然比较浅:

1. 明确提问更偏引用
2. hostile / harassing 不引用
3. 很短的消息直接接话

在 `heartFC_chat_enhanced.py` 里，这条逻辑已经比基础层更细，包括:

1. 不引用 bot 自己的消息
2. 对方本身就在引用链里时更倾向引用
3. hostile / harassing 场景关闭引用

但整体仍然存在两个问题:

1. 引用判断没有统一放在“对话线程管理”层
2. 还没有真正把“这条旧消息是否仍然值得接、是否会显得尴尬、是否会打断当前群流”做成主协议

所以现在的引用更像:

1. 风格选择
2. 回复形式选择

还不像:

1. 线程感知
2. 社交距离感知
3. 群聊融入策略

### 18.9 记忆检索不是完全“全捞”，但整体策略确实偏激进

涉及文件:

1. `src/memory_system/memory_core.py`
2. `src/memory_system/retrieval_tools/query_direct_memory.py`
3. `src/memory_system/memory_retrieval.py`
4. `src/core/unified_planner.py`

先说好的部分:

1. `query_memories()` 自身有 `ceiling`
2. `query_direct_memory()` 也会限制分类和条数
3. `should_skip_memory_retrieval()` 也确实在试图跳过一些无意义检索
4. thinking cache 也有避免短时间重复探测的逻辑

但整体还是偏激进，原因在于:

1. 记忆检索 prompt 里明确写着“宁可多搜不可漏搜”
2. 对短句、模糊句、重复关键词都强烈鼓励去搜
3. `unified_planner` 在长时上下文索引时会直接 `query_memories(ctx.channel_id, ceiling=20)`

这就导致:

1. 单个工具虽然有限流
2. 但架构层面对“总检索预算”没有建立统一上限
3. 多层都在各自决定要不要搜

所以用户体感“它会把所有记忆都拎出来”并不完全准确，
但“它整体上过度偏向检索、容易浪费 token”是成立的。

### 18.10 “像不像人”的根本缺口，不是没有情绪和状态，而是没有统一行为 contract

从这一轮证据看，当前系统已经有:

1. 情绪值
2. 社交意愿
3. 无聊 / 摸鱼 / 烦躁 / 疲惫
4. 夜间 / 熬夜 / 等待 / 追问
5. 群聊认知
6. 自我识别
7. 上下文检索
8. 复读护栏

真正缺的不是“变量数量”，而是:

1. 哪些值是硬约束
2. 哪些值只是 prompt 提示
3. 哪些值只影响主观独白
4. 哪些值能直接中止一次行为
5. 哪些值能覆盖别的层

如果这层 contract 不冻结，继续往里加机制只会更像:

1. 左脑想闭嘴
2. 右脑想接话
3. 记忆层想去搜
4. prompt 层又想表现得积极

最后就会越来越“像有很多人格模块，但没有统一人格”。

---

## 19. 新增高置信度问题清单（行为链视角）

### 19.1 `self_awareness.record_message()` 未接主输出正文链

文件: `src/modules/recall/self_awareness.py`

截至本轮搜索结果，仓内没有看到主聊天发送链稳定调用它记录 bot 自己发出的正文消息。

结果:

1. 自我意识无法成为可靠的“我刚说过什么”真源
2. 后续 recall / 去重 / 撤回 / 自查都会偏弱

### 19.2 `proactive_integration_hub` 只取自我意识统计，不取正文语义

文件: `src/chat/proactive/proactive_integration_hub.py`

现在它从 `self_awareness` 里取的主要是:

1. recent_messages_count
2. recent_actions_count
3. interaction stats

同时 bot 发消息后的回写，只记录 action type，`content=""`。

结果:

1. 主动链知道“最近做过几次动作”
2. 但不知道“最近到底说了什么”
3. 无法把“内容重复”提升为主动行为硬约束

### 19.3 `ContextManager` 默认只保 5 条消息，和其他上下文源不一致

文件: `src/modules/context_manager.py`

问题:

1. 默认 `_default_max_messages = 5`
2. 读取 recent context 还会被这个值再次裁断

结果:

1. 主链最近上下文过短
2. 和 message repository 的 20 条、其他缓存来源不一致
3. 上下文断裂感会直接传导到行为判断

### 19.4 `chat_core_base` 的引用判断仍偏浅层

文件: `src/chat/chat_core_base.py`

当前引用逻辑主要看:

1. 问号 / “怎么 / 为什么 / 什么”
2. hostile / harassing
3. 文本长度

结果:

1. 能处理一部分基础场景
2. 但不够支撑群聊线程感知和社交融入

### 19.5 `unified_planner` 长时记忆索引默认直拉 20 条

文件: `src/core/unified_planner.py`

问题:

1. `_index_context()` 里直接 `query_memories(..., ceiling=20)`
2. 没有先按本轮问题是否真的需要长时记忆做强预算门控

结果:

1. 长时上下文容易过肥
2. 规划链会被历史噪音污染
3. token 成本容易虚高

### 19.6 频率 / 等待 / 旁白 / 主链惩罚并存，但没有统一优先级表

涉及文件:

1. `src/chat/heart_flow/frequency_control.py`
2. `src/chat/heart_flow/waiting_handler.py`
3. `src/core/inner_narration_planner.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

结果:

1. 行为一致性取决于当前哪条链先命中
2. 容易出现“上一层说等一下，下一层又把话接上”的覆盖问题

---

## 20. 面向“像人一样说话和闭嘴”的下一阶段硬修分组

如果下一轮不是只做文档，而是直接硬修，我建议把工作分成下面四组。

### 第一组: 自我认知总账本收口

文件:

1. `src/modules/recall/self_awareness.py`
2. `src/core/self_reply_recognizer.py`
3. `src/chat/proactive/proactive_integration_hub.py`
4. `src/chat/heart_flow/heartflow_message_processor.py`

目标:

1. 所有 bot 输出统一写入 self ledger
2. 区分“发过”与“被回流识别到”
3. 给主链输出统一 `self_recent_content / self_recent_actions / self_repeat_risk`

### 第二组: 上下文与记忆预算收口

文件:

1. `src/modules/context_manager.py`
2. `src/core/unified_planner.py`
3. `src/memory_system/memory_retrieval.py`
4. `src/memory_system/retrieval_tools/query_direct_memory.py`
5. `src/memory_system/memory_core.py`

目标:

1. 冻结“最近上下文”与“长时记忆”的职责边界
2. 先看最近，再决定要不要搜长期
3. 建立统一 retrieval budget，而不是每层自己搜

### 第三组: 说话/闭嘴统一裁决表

文件:

1. `src/chat/heart_flow/frequency_control.py`
2. `src/chat/heart_flow/waiting_handler.py`
3. `src/core/inner_narration_planner.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

目标:

1. 明确哪些是硬否决
2. 明确哪些是软降权
3. 明确哪些可以被 @ / 强唤醒 / 安全事件覆盖
4. 形成统一 `speak / wait / observe / chase / silent / leave` contract

### 第四组: 群聊认知与引用线程收口

文件:

1. `src/core/group_scene_state.py`
2. `src/chat/heart_flow/group_atmosphere.py`
3. `src/modules/perception/group_sense.py`
4. `src/chat/chat_core_base.py`
5. `src/chat/heart_flow/heartFC_chat_enhanced.py`

目标:

1. 合并群聊活跃度 / 氛围 / dominant speaker / burst / controversy 的真源
2. 统一“适不适合插话”“适不适合引用”“适不适合继续追同一线程”
3. 让“融入群聊”从 prompt 文案，升级为结构化策略

---

补到这里，这份文档已经把“值系统污染”和“人格化行为链分裂”两大块都串起来了。

---

## 21. 情绪链三账本并存，且公式并不真正一致

这一轮确认，主动意愿相关状态至少有三套并行账本：

1. `src/chat/heart_flow/emotion_driven_core.py`
2. `src/core/emotion_feedback_loop.py`
3. `src/core/adaptive_threshold_learner_v2.py`

它们都在维护：

1. `boredom`
2. `loneliness`
3. `social_desire`
4. `proactive_willingness`

但并不是同一实现被复用，而是三套独立实现。

### 21.1 `emotion_driven_core` 与 `adaptive_threshold_learner_v2` 标称“对齐”，实则已经分叉

文件:

1. `src/chat/heart_flow/emotion_driven_core.py`
2. `src/core/adaptive_threshold_learner_v2.py`

`adaptive_threshold_learner_v2.update_boredom_state()` 的注释写着：

1. 与 `emotion_driven_core` 的 boredom 累积逻辑对齐

但实际差异已经存在：

1. `emotion_driven_core` 的 loneliness 使用 `max(0.0, silence_min - 1.5) * 0.06`
2. `adaptive_threshold_learner_v2` 改成了 `max(0.0, silence_min - 3.0) * 0.04`
3. `emotion_driven_core` 的 boredom 在 mood 高时还会乘 `0.8`
4. `adaptive_threshold_learner_v2` 没有这条高 mood 抑制

更重要的是主动意愿公式也不一致：

1. `emotion_driven_core`：
   - `boredom*0.35 + loneliness*0.28 + social_desire*0.22 + mood*0.05 + time_factor*0.05 + curiosity*0.05`
   - monitoring 加成 `+0.12`
2. `adaptive_threshold_learner_v2`：
   - `boredom*0.30 + loneliness*0.25 + social_desire*0.20 + time_factor*0.10 + curiosity*0.05`
   - 不含 `mood`
   - monitoring 加成 `+0.08`

这不是“近似实现”，而是：

1. 同名概念
2. 相似字段
3. 不同权重
4. 不同触发条件

所以当两个模块都在影响“主动想说话”的判断时，行为本身就天然会漂。

### 21.2 `emotion_feedback_loop` 又维护了第三套主动意愿公式

文件: `src/core/emotion_feedback_loop.py`

`EmotionStateSnapshot.compute_proactive_willingness()` 里：

1. `boredom*0.30`
2. `loneliness*0.25`
3. `social_desire*0.20`
4. `mood*0.10`
5. `curiosity*0.05`
6. `+(1.0 - energy) * -0.10`

这又与前两者不同：

1. 它没有 `time_factor`
2. energy 用线性负项，而不是最终整体乘法压制
3. mood 权重比 `emotion_driven_core` 更高

结果就是：

1. 同样一轮状态
2. 三套主动意愿会得出不同结论
3. 下游谁引用哪套，系统就会表现成哪种人格

### 21.3 `emotion_feedback_loop` 只回写部分字段，情绪账本会天然分叉

文件:

1. `src/core/emotion_feedback_loop.py`
2. `src/chat/heart_flow/emotion_driven_core.py`

`emotion_feedback_loop.run_feedback_cycle()` 最后只同步：

1. `mood`
2. `energy`
3. `curiosity`

给 `emotion_driven_core.integrate_external_state()`。

但它自己内部已经更新了：

1. `boredom`
2. `social_desire`
3. `loneliness`
4. `proactive_willingness`

这些值没有同步回去。

这意味着：

1. 情绪反馈环有一套 boredom/loneliness
2. 情感驱动核心有另一套 boredom/loneliness
3. 两边都会继续各自演化

这是高置信度结构 bug，不是单纯架构风格问题。

### 21.4 `heartFC_chat_enhanced` 同时消费多套情绪来源，进一步放大分叉

文件: `src/chat/heart_flow/heartFC_chat_enhanced.py`

当前主链同时会使用：

1. `emotion_driven_core` 快照
2. `emotion_feedback_loop` 报告
3. `adaptive_threshold_learner_v2` 的 boredom-driven willingness

这会导致：

1. “无聊值”不是一个真源
2. “主动想聊”不是一个真源
3. “疲惫但又无聊时该不该说”会取决于当前哪个模块先被看见

---

## 22. 资源链并不只是一主一辅，而是“双资源账本 + 主链二次加工版”

这一轮继续确认，资源系统不只是 `energy_manager` 和 `metabolism_engine` 重叠，主链还自己又做了一层再解释。

### 22.1 `EnergyChainDimension` 和 `MetabolismEngine` 都在维护聊天/思考/活跃/社交

文件:

1. `src/chat/heart_flow/energy_manager.py`
2. `src/core/metabolism_engine.py`

它们都在管理：

1. 聊天资源
2. 思考资源
3. 活跃度
4. 社交态势
5. 无聊/疲劳/摸鱼相关衍生量

但字段并不相同：

1. `EnergyChainDimension`: `chat_pool / thinking_value / activity_level / social_value / annoyance_level`
2. `MetabolismEngine`: `chat_fuel / thinking_fuel / activity_gauge / social_gauge / boredom_level / loafing_level / fatigue_accumulator`

结果：

1. 两套都像主账本
2. 没有谁是明确从属缓存
3. 下游只要随手读错一个，就会出现不同人格状态

### 22.2 `heartFC_chat_enhanced` 的 `_cached_metabolism_constraints` 实际是主链自己重算的

文件: `src/chat/heart_flow/heartFC_chat_enhanced.py`

主链在构建 `_cached_metabolism_constraints` 时，不是直接使用 `MetabolismEngine.behavior_constraint_summary()` 作为唯一来源，而是：

1. 从 `EnergyChainDimension` 读取 `combined_ratio / activity_level / social_value`
2. 自己再推导 `boredom_level / loafing_suppression / proactive_drive / reply_length_hint / energy_gate_open`

也就是说：

1. 很多地方名义上在读“代谢约束”
2. 实际读的是主链二次加工版
3. 这层二次加工甚至绕开了 `MetabolismEngine` 的额外资源维度

被绕开的包括：

1. `group_observation_fuel`
2. `pattern_analysis_fuel`
3. `impression_evolution_fuel`
4. `scene_adaptation_energy`

这会造成一种很隐蔽的问题：

1. 这些高级资源机制看上去实现了
2. 但主链很多时候根本没认真用它们来裁决行为
3. 表现出来就像“特性写了，但没真正激活”

### 22.3 `MetabolismEngine.behavior_constraint_summary()` 和主链代谢约束不是同一协议

文件:

1. `src/core/metabolism_engine.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`

`MetabolismEngine.behavior_constraint_summary()` 输出的是：

1. `thinking_budget_ok`
2. `reply_length_hint`
3. `activity_can_initiate`
4. `social_openness`
5. `boredom_watch_drive`
6. `loafing_suppression`
7. `energy_gate_open`
8. `is_perfunctory`

但主链缓存里同时保留了：

1. `boredom`
2. `boredom_level`
3. `loafing`
4. `loafing_level`
5. `chat_fuel`
6. `chat_value`
7. `energy_suppression`
8. `proactive_drive`

这说明：

1. 代谢系统没有形成统一输出 contract
2. 主链既在消费标准化约束，又在消费自己拼的衍生值
3. “硬约束”和“软提示”已经混在同一包里

---

## 23. 冷启动基线、默认值和规范化口径同时打架

### 23.1 新用户初始关系在不同系统里不是一个值

文件:

1. `src/modules/social_value/social_value_core.py`
2. `src/chat/heart_flow/fondness_trust.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`

目前至少有三套冷启动基线：

1. `social_value_core` 新用户初始值默认 `+5.0`
2. `FondnessTrustDimension` 的 `fondness/trust` 从 `0.0` 开始
3. 主链很多兜底仍然把 `affection/trust` 当 `50.0`

这意味着新用户会被同时视作：

1. 略偏友好
2. 完全中立
3. 普通熟人

这会直接让：

1. 冷启动语气
2. 回复概率
3. 风险评估
4. 社交距离感

全部出现不一致。

### 23.2 `social_value_core` 会把明确的 `severity=0.0` 洗成 `0.5`

文件: `src/modules/social_value/social_value_core.py`

当外部已提供 `behavior_signal` 时：

1. `severity` 读取用了 `external_behavior.get("severity", 0.5) or 0.5`

所以只要外部显式传入：

1. `severity=0.0`

最终也会变成：

1. `0.5`

这属于高置信度 bug。

后果：

1. 原本“零严重度 / 很轻微”的行为
2. 会被系统按中等严重度处理
3. 社交值变化会平白被放大

### 23.3 风险系统对 `social_value` 的标准化口径不统一

文件:

1. `src/chat/heart_flow/heartflow_decision.py`
2. `src/modules/brain/decision_brain.py`

这些模块直接把 `social_value` 当成 `0~1` 信号处理：

1. `max(0.0, min(1.0, social_value))`

但上游真实社交值很多时候来自：

1. `-100 ~ +100`

于是结果会变成：

1. 所有负值都被压成 `0`
2. 所有大于 `1` 的正值几乎都被压成 `1`
3. 中间细粒度几乎全部丢失

这不是单纯尺度不同，而是：

1. 风险侧拿到的是“被拍扁过的社交值”
2. 很难再正确区分“略不信任”和“极度敌对”

### 23.4 `UnderstandResult.activity_level` 强制钳在 `0~1`，容易吞掉 0~100 输出

文件: `src/modules/perception/understand.py`

`activity_level` 默认值是 `0.5`，并且解析时统一经过 `_safe_float()`：

1. 小于 0 按 0
2. 大于 1 按 1

这意味着如果模型或上游 prompt 输出：

1. `activity_level=50`

最终会被直接压成：

1. `1.0`

因此这个接口虽然看起来叫 `activity_level`，但实际上只接受：

1. 归一化小数

而系统其他地方的 `activity_level` / `activity_gauge` 却大量使用：

1. `0~100`
2. 字符串标签

这会进一步放大协议污染。

---

## 24. 已确认的“死字段 / 假激活机制”

### 24.1 `world_snapshot` 读取了 `emotion_driven_core` 根本不输出的字段

文件:

1. `src/core/world_snapshot.py`
2. `src/chat/heart_flow/emotion_driven_core.py`

`world_snapshot` 会尝试从 `emotion_driven_core.get_state_snapshot()` 读取：

1. `social_openness`
2. `quiet_desire`
3. `group_attention_level`
4. `self_consciousness`
5. `freshness`

但 `emotion_driven_core.get_state_snapshot()` 实际输出里并没有这些键。

结果：

1. `world_snapshot` 这些主观字段长期只能吃默认值
2. 相关判断看起来存在，实际上根本没被真实状态驱动

这类问题很隐蔽，因为：

1. 没报错
2. 也有默认值
3. 面板上还能显示

但机制实际上是“假激活”。

### 24.2 基于这些默认值派生出的 mood 判断也会长期偏假

文件: `src/core/world_snapshot.py`

后续 `overall_mood` 的一些分支依赖：

1. `quiet_desire`
2. `social_openness`

由于这两个字段并没有真实来源，很多“想安静 / 外向打开”的判断会长期只是默认值演绎，而不是真感受。

---

## 25. 下一阶段新增硬修建议

在前面四组硬修之外，这一轮建议再单列三组。

### 第五组: 情绪账本合并

文件:

1. `src/chat/heart_flow/emotion_driven_core.py`
2. `src/core/emotion_feedback_loop.py`
3. `src/core/adaptive_threshold_learner_v2.py`

目标:

1. 合并 boredom/loneliness/social_desire/proactive_willingness 真源
2. 只允许一个模块做主累计
3. 其他模块只能提交 delta 或读取快照

### 第六组: 资源总账本收敛

文件:

1. `src/chat/heart_flow/energy_manager.py`
2. `src/core/metabolism_engine.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`

目标:

1. 选出唯一资源真源
2. 取消主链自造“代谢约束二次加工版”
3. 让高级资源维度真正进入主决策

### 第七组: 冷启动与默认值协议冻结

文件:

1. `src/modules/social_value/social_value_core.py`
2. `src/chat/heart_flow/fondness_trust.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`
4. `src/chat/heart_flow/heartflow_decision.py`
5. `src/modules/brain/decision_brain.py`
6. `src/modules/perception/understand.py`

目标:

1. 统一 cold-start baseline
2. 冻结 `social_value / affection / trust / activity_level` 的单位
3. 禁止再用默认值偷偷把未知态伪装成中性态

### 第八组: 注意力 / 看群 / 面板协议修复

文件:

1. `src/core/subjective_attention_flow.py`
2. `src/core/state_dashboard.py`
3. `src/core/watch_state_machine.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

目标:

1. 冻结 `attention_snapshot` 的机器可读 schema
2. 禁止中文标签和英文枚举值混用
3. 统一 watch/attention 的桥接协议
4. 修掉 `deep_withdrawal / deep_withdrawn` 错拼

### 第九组: 关系出口与消费层收口

文件:

1. `src/core/world_snapshot.py`
2. `src/chat/heart_flow/inner_voice.py`
3. `src/chat/chat_core_base.py`
4. `src/modules/social_value/social_affect_fuser.py`

目标:

1. 冻结 `relation export contract`
2. 让 `world_snapshot` 成为唯一关系出口
3. 取消消费层继续手搓 `trust/annoyance/affection`
4. 明确 `trust_value / trust_score / favorability / affection` 的单向桥接规则

### 第十组: 检索 governor 与上下文 contract

文件:

1. `src/memory_system/memory_retrieval.py`
2. `src/memory_system/retrieval_tools/query_direct_memory.py`
3. `src/memory_system/retrieval_tools/query_chat_history.py`
4. `src/memory_system/memory_core.py`
5. `src/modules/context_manager.py`

目标:

1. 统一“是否检索”的总门控
2. 明确“先最近、再关键词、再语义”的升级顺序
3. 冻结 `context_manager` 的窗口 contract
4. 给记忆检索加统一 token / result budget

### 第十一组: 当前轮消息池 purity 与来源标记

目标:

1. 把 `decision_messages` 收口成真正的“当前轮判定样本”
2. 历史上下文和 bot 自我回顾不再伪装成普通当前消息

内容:

1. 给 `decision_messages` 增加强制来源标签: `incoming / historical_context / synthetic_self`
2. 把“当前轮目标提取、面板对象提取、旁白规划对象提取”统一限制在 `incoming` 样本集合
3. 历史上下文只作为辅助字段，不再直接参与“谁是当前对象”的倒序扫描
4. 自我回顾文本也不要混进主消息池，而应进入单独的 self-context 插槽

### 第十二组: 内心独白 governor 收口

目标:

1. 统一“什么时候允许触发第一层内心独白”
2. 让窥屏态、被动回复、主动空闲都走同一套频控协议

内容:

1. `_should_run_voice()` 升级成唯一总闸
2. `peek / reactive / proactive` 三条路径都只能通过这道总闸触发独白
3. 将“冷却时间”“夜间抑制”“资源不足”“quiet_preference”“群刷屏态”统一写成同一份 break-down
4. 禁止旁路路径只检查时间冷却就直接调用 `_invoke_inner_voice()`

---

## 26. 自我感知链没有成为主决策硬门

### 26.1 `SelfAwareness` 当前更像遥测账本，而不是自我文本真源

类型: `duplicate authority`

文件:

1. `src/modules/recall/self_awareness.py`
2. `src/chat/proactive/proactive_integration_hub.py`

producer:

1. `SelfAwareness.record_message()` 设计上可以记录自己发出的正文
2. `record_action()` 记录自己的行为动作
3. `record_user_event()` 记录外部用户事件

bridge / 回退:

1. 全仓未见 `SelfAwareness.record_message()` 被主链稳定调用
2. `ProactiveIntegrationHub` 只会在发送后写 `record_action()`
3. 在用户消息到达后写 `record_user_event()`
4. 这两个调用都把 `content` 固定写成空串
5. `_collect_self_awareness()` 只往外输出数量统计和整体 stats

consumer:

1. `ProactiveIntegrationHub.get_state_snapshot()` 暴露的是 `self_awareness_data`
2. 这个数据目前只包含 `recent_messages_count / recent_actions_count / stats`
3. 主决策并不会直接读取“我上一句具体说了什么”

结论:

1. 系统目前更擅长知道“自己做过几次动作”
2. 却不擅长知道“自己刚才具体说了什么”
3. 所以 `SelfAwareness` 现在更像行为遥测账本，不是内容级自我上下文真源

### 26.2 自我觉察和自我文本上下文已经分裂成两条链

类型: `duplicate authority`

文件:

1. `src/modules/recall/self_awareness.py`
2. `src/chat/heart_flow/reply_coordinator.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `SelfAwareness` 维护自我行为历史
2. `reply_coordinator` / `heartFC` 又维护一份最近 bot 文本

bridge / 回退:

1. `heartFC` 通过 `_recent_bot_utterances`、持久化最近回复文本等方式自己维护 bot 历史
2. 这条链并不依赖 `SelfAwareness`

consumer:

1. 防复读、引用保护、旁白规划读的是“最近 bot 文本”
2. 主动整合中心读的是 `SelfAwareness` 的统计

结论:

1. 自我文本和自我行为没有统一到一条链
2. “我说过什么”和“我做过什么”由不同账本各自维护
3. 这正是后续人格一致性和自我复读控制容易失真的根源之一

---

## 27. `SelfReplyRecognizer` 才是真正的自我消息硬门

### 27.1 自我消息识别链已经形成硬消费口

类型: `active bug`

文件:

1. `src/core/self_reply_recognizer.py`
2. `src/core/message_semantic_router.py`
3. `src/chat/message_receive/bot.py`
4. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `SelfReplyRecognizer.mark_bot_message()` 记录 bot 自消息
2. 内容哈希映射 `_content_hash_map` 负责近似自识别
3. `_reply_chains` 维护 bot reply chain

bridge / 回退:

1. `message_semantic_router` 会检查引用目标是不是 bot
2. `bot.py` 在消息进入时会先跑 `is_bot_message()`
3. `heartFC` 也会在筛消息时用 `_is_bot_message_obj()` 走 recognizer

consumer:

1. 自己的消息会被识别成 bot message
2. 引用 bot 的消息会进入不同的语义路径
3. reply chain 会影响“是不是在接自己前文”

结论:

1. 真正进入主决策硬门的是 `SelfReplyRecognizer`
2. 它解决的是“是不是 bot 自己 / 是不是在接 bot 的链”
3. 而不是“bot 对自己上一句内容的完整回顾”

### 27.2 自我识别链和自我回顾链并没有统一

类型: `duplicate authority`

文件:

1. `src/core/self_reply_recognizer.py`
2. `src/modules/recall/self_awareness.py`

producer:

1. 一个模块记录 message id、hash、reply chain
2. 另一个模块记录 sent message、action、user event

bridge / 回退:

1. 这两条链都在描述“bot 自己”
2. 但彼此没有形成统一 snapshot 或统一 contract

consumer:

1. 前者被真正用于“是不是自己”
2. 后者主要被用于“最近做了多少事”

结论:

1. 自我识别和自我回顾被拆成两套协议
2. 这会让系统看起来“知道自己是谁”，但不一定“知道自己刚说过什么”

---

## 28. `ContextManager` 存在静默截断与弱检索协议

### 28.1 上下文窗口 contract 已经静默错位

类型: `active bug`

文件:

1. `src/modules/context_manager.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `ContextManager` 默认只保留 `5` 条消息
2. `ConversationContext.max_messages` 的默认设计甚至允许更大窗口

bridge / 回退:

1. `heartFC` 请求 `get_recent_context(limit=8)`
2. `ContextManager.get_recent_context()` 又把 limit 强制压回 `_default_max_messages`

consumer:

1. `heartFC` 把 recent context 重新包装成 pseudo messages
2. 这些 pseudo messages 会参与当前轮决策消息池

结论:

1. 调用方以为自己拿到了 8 条上下文
2. 实际上最多只有 5 条
3. 这类静默截断会直接造成“上下文不完整但没有报错”的隐性污染

### 28.2 `search_context()` 的相关性协议对中文不稳

类型: `latent bug`

文件:

1. `src/modules/context_manager.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `search_relevant()` 按空格切词
2. 再用 `text in content` 和 term overlap 做评分

bridge / 回退:

1. 中文自然聊天文本很多时候并不靠空格分词
2. 长句、粘连短句、表情混排都会让 overlap 失真

consumer:

1. `heartFC` 会把 relevant context 也塞回决策消息池
2. 下游会把这部分结果当“命中的相关前情”

结论:

1. 这不是严格的中文上下文检索
2. 更像一个轻量关键词 / 子串碰撞器
3. 命中质量和调用方主观期待存在明显落差

---

## 29. 群场景链不是单一真源，而是双权威并存

### 29.1 `group_scene_state` 负责“能不能插话”的硬约束

类型: `duplicate authority`

文件:

1. `src/core/group_scene_state.py`
2. `src/chat/heart_flow/group_atmosphere.py`
3. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `GroupSceneState.snapshot()` 会产出 `suitable_to_join`
2. 同时给出 `join_unsuitable_reason / dominant_speaker / atmosphere`

bridge / 回退:

1. `GroupAtmosphereDimension` 基本上是对 `group_scene_state` 的包装和缓存
2. 它不是新的独立群态真源，而是 scene snapshot 的维度化外壳

consumer:

1. `heartFC._apply_scene_hard_constraints()` 把 `suitable_to_join` 当硬约束
2. `inner_narration_planner` 也接收 `scene_suitable_to_join`

结论:

1. “适不适合插话”这一项，主硬门主要来自 `group_scene_state`
2. `group_atmosphere` 更像对它的包装维度，不是第三个独立真源

### 29.2 `group_sense` 又单独维护一套“刷屏 / 争议 / 话题需要”判断

类型: `duplicate authority`

文件:

1. `src/modules/perception/group_sense.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `GroupSense.analyze()` 输出 `burst_detected`
2. 输出 `controversy_detected`
3. 输出 `topic_hints / needs_topic / activity_level`

bridge / 回退:

1. `heartFC` 把最近 50 条消息重组后喂给 `group_sense`
2. 这条链并不依赖 `group_scene_state.snapshot()`

consumer:

1. `heartFC` 会根据 `burst_detected / controversy_detected` 单独压 desire
2. 也会把这些结果作为额外群聊感知日志和上下文

结论:

1. 系统实际上存在两套“群里现在适不适合说话”的裁决器
2. 一套来自 `group_scene_state`
3. 一套来自 `group_sense`
4. 两者没有统一 arbitration table

---

## 30. 关系统一出口之后仍被消费层二次造值

### 30.1 `world_snapshot` 已收口为 canonical-first，alias 仅剩兼容镜像

类型: `latent bug`

文件:

1. `src/core/world_snapshot.py`

producer:

1. `social_affect_fuser` 提供 `social_value / trust_value / annoyance_value`
2. `emotion_tracker` 提供 `affection / trust_score / trauma_score / pressure`

bridge / 回退:

1. 第四轮硬修后，`world_snapshot._reconcile_overlapping_values()` 已改成 canonical 单向镜像：
   - `social_value -> favorability`
   - `trust_value -> trust_score`
2. `to_rapport_dict()` 已固定 canonical 输出：
   - `affection` 只读 `u.affection`
   - `trust` 与 `trust_value` 只读 `u.trust_value`
3. 兼容 alias 仍会输出，但只作为兼容透出，不再反向参与主语义回填
4. rapport contract 现已显式带：
   - `relation_contract_version = 2`
   - `unified_source = world_snapshot`

consumer:

1. `inner_voice / chat_core_base / group_generator / private_generator / replyer_manager / perception_engine` 主消费链已切到 canonical 字段
2. 低频兼容消费方仍可能继续读取 alias 输出

结论:

1. 这一节原先记录的“出口仍会反向污染 canonical”的结论已经过时
2. 当前真实状态是：
   - `world_snapshot` 已经成为 canonical-first 的统一关系出口
   - alias 仍保留输出一轮，用于减少外部兼容爆炸
3. 当前残余风险不在出口反向污染，而在“仓内是否还有低频消费者继续读 alias”

### 30.2 主消费链已迁到 canonical，残余风险转移到低频兼容口

类型: `latent bug`

文件:

1. `src/chat/heart_flow/inner_voice.py`
2. `src/chat/chat_core_base.py`
3. `src/chat/replyer/group_generator.py`
4. `src/chat/replyer/private_generator.py`
5. `src/chat/replyer/replyer_manager.py`
6. `src/chat/proactive/perception_engine.py`

producer:

1. `src/core/world_snapshot.py`
2. `build_relation_rapport_snapshot()`

bridge / 回退:

1. `chat_core_base` 现已直接读取 canonical：
   - `trust_value`
   - `annoyance_value`
   - `affection`
2. `inner_voice` 主路径已改成直接吃 `rapport_data["trust_value"]`
3. `group_generator / private_generator / replyer_manager / perception_engine` 本轮也统一切到 `world_snapshot` 的 canonical rapport dict
4. 老结论里提到的 “`inner_voice` / `chat_core_base` 继续用 `latest_score` 手搓代理值” 已不再符合当前代码真相

consumer:

1. 当前高频关系消费口都会直接吃 `world_snapshot` 导出的 canonical 值
2. 残余问题主要在低频兼容调用点是否还会读 `trust / trust_score / favorability`

结论:

1. 这一节原先的“高频主消费链仍在二次造值”结论已过时
2. 第四轮硬修后，高频主消费链基本完成 canonical 收口
3. 后续仍要继续全仓扫低频兼容口，但问题级别已从“主链双权威”下降为“边缘残留兼容口”

---

## 31. 记忆检索存在“跳过门局部化 + 上层强搜 + 下层粗搜”三段互搏

### 31.1 `plan_memory_retrieval()` 已成为统一前置 governor，残余风险在绕过入口

类型: `latent bug`

文件:

1. `src/memory_system/memory_retrieval.py`
2. `src/chat/replyer/group_generator.py`
3. `src/chat/replyer/private_generator.py`

producer:

1. `plan_memory_retrieval()` 现在负责统一产出：
   - `should_skip`
   - `reason`
   - `query_text`
   - `tool_order`
   - `max_tool_steps`
   - `max_result_chars`
   - `stop_after_first_hit`
2. `build_memory_retrieval_prompt()` 负责消费 governor 并驱动执行链

bridge / 回退:

1. `group_generator / private_generator` 当前都已改成只调用 `build_memory_retrieval_prompt()`
2. prompt 组装侧不再自造“该不该搜、先搜什么”的局部策略
3. 当前残余风险是未来若出现新的直接调用者，仍可能绕过 governor

consumer:

1. 群聊与私聊主回复链都走统一 governor
2. `_process_memory_retrieval()` 继续执行 governor 下发的工具顺序与预算

结论:

1. 这节原先“统一 governor 尚未形成”的结论已过时
2. 现在主链已经形成统一总闸
3. 当前剩余问题变成“是否还有旁路新入口绕开它”，而不是主链内部各判各的

### 31.2 上层 ReAct prompt 已改成服从 governor，而不是继续强推激进检索

类型: `latent bug`

文件:

1. `src/memory_system/memory_retrieval.py`

producer:

1. `memory_retrieval_react_prompt_head` 定义检索行为规则

bridge / 回退:

1. 第四轮硬修后，prompt 规则已改成：
   - 先服从 governor
   - 先最近、再关键词、再语义
   - 命中后立即停止升级
   - 当前上下文足够时直接返回空结果
2. `_react_agent_solve_question()` 也增加了 `allowed_tools / stop_after_first_hit`，避免模型绕开工具顺序

consumer:

1. 负责实际调用工具的 ReAct 检索链会直接吃这份策略

结论:

1. 这节原先“prompt 还在鼓励宁可多搜不可漏搜”的结论已经过时
2. 当前 prompt 已与 governor 对齐，主链不会再被提示词推向无限升级
3. 残余风险主要在单个工具命中质量，而不是 prompt 继续把策略拉偏

### 31.3 下层执行并不是真正的“上下文检索”，而更像粗粒度关键词检索

类型: `latent bug`

文件:

1. `src/memory_system/retrieval_tools/query_direct_memory.py`
2. `src/memory_system/memory_core.py`
3. `src/memory_system/retrieval_tools/query_chat_history.py`

producer:

1. `query_direct_memory` 先把 query 拆成若干 term
2. 但后续只取第一个 term 作为 keyword
3. `memory_core.query_memories()` 的长期库查询仍是 `content.contains(keyword)`

bridge / 回退:

1. `query_chat_history` 会先抓一个窗口再逐条筛
2. 第四轮硬修后，`query_direct_memory` 已不再退化成“只看第一关键词”：
   - 多 term 继续保留
   - 多 term 同时命中时会获得额外加权
3. `_process_memory_retrieval()` 已按 governor 支持：
   - `max_tool_steps`
   - `stop_after_first_hit`
   - `max_result_chars`
4. 上层检索结果也会统一截断，避免无上限塞进 prompt
5. 但长期记忆底层仍存在 `contains(keyword)` 级别的粗搜残余

consumer:

1. 群聊 / 私聊 prompt 会把返回内容直接当“相关前情”
2. token 消耗和命中质量都受这套粗协议控制

结论:

1. 当前主链已经稳定实现“先 governor、再按工具顺序、命中即停、结果截断”的分层协议
2. 这一节留下来的真实残余问题不再是“没有 governor”，而是“底层长期记忆粗搜仍不够语义化”
3. 所以后续要继续修的是工具质量，而不是再次在上层补一套检索策略

---

## 32. 自我文本回顾真正进主链的不是 `SelfAwareness`，而是 synthetic self message

### 32.1 bot 的“我刚说过什么”主要来自最近发言缓存

类型: `duplicate authority`

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/modules/recall/self_awareness.py`
3. `src/chat/proactive/proactive_integration_hub.py`

producer:

1. `heartFC` 维护 `_recent_bot_utterances`
2. 过期时还会退回 `_load_recent_persisted_bot_texts()`
3. `SelfAwareness` 这边虽然有 `record_message()` 能记正文，但主链没有稳定接它

bridge / 回退:

1. `_build_synthetic_self_messages()` 把最近 bot 发言转换成 `user_id="bot"` 的 pseudo message
2. `_prepare_decision_messages()` 再把这些 synthetic self message 直接并入主判定池
3. `ProactiveIntegrationHub` 的 `record_action / record_user_event` 仍然只记空内容账本

consumer:

1. `_build_decision_context_packet()` 会从 `decision_messages` 倒序找 `latest_bot`
2. `_has_targeted_bot_message()` 等针对 bot 的判断也吃这批消息
3. 后续的复读、续接、记忆提示都会间接受它影响

结论:

1. 自我文本回顾真正进主链的，不是 `SelfAwareness`
2. 而是 `recent_bot_utterances -> synthetic self message -> decision_messages`
3. 所以“自我识别”“自我回顾”“自我账本”目前还是三条链，没有单一真源

## 33. 当前轮 `decision_messages` 存在“当前消息 / 历史上下文 / 自我回顾”混池污染

### 33.1 `_prepare_decision_messages()` 是污染汇入口

类型: `active bug`

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/modules/context_manager.py`

producer:

1. `filtered_messages` 提供当前轮消息
2. `ContextManager.get_recent_context()` 提供最近上下文
3. `ContextManager.search_context()` 提供相关前情
4. `_build_synthetic_self_messages()` 提供 bot 自我回顾

bridge / 回退:

1. `_prepare_decision_messages()` 把三类来源全部包装成同类 message 对象
2. 这些对象最终只按 `timestamp` 排序
3. 下游已经无法判断哪条是真正刚收到的消息，哪条只是历史补充

consumer:

1. `_evaluate_content_state_signal()`、`_analyze_repetition_pressure()`、`_analyze_harassment_pressure()`
2. `_prefetch_memory_hint()`
3. `_integrate_narration_planning()`
4. `_build_decision_context_packet()`
5. `_emit_target_profile()` 前的目标对象提取

结论:

1. 当前主决策并不是围绕“incoming batch”纯样本在运行
2. 而是围绕一池混合样本在运行
3. 这会把上下文辅助信息误升级成主判定依据

## 34. `ContextManager` 的弱检索不只会截断，还会反向污染“当前对象是谁”

### 34.1 上下文弱检索 + pseudo message 桥接，会劫持对象识别

类型: `active bug`

文件:

1. `src/modules/context_manager.py`
2. `src/chat/heart_flow/heartFC_chat_enhanced.py`

producer:

1. `search_relevant()` 只做空格分词和子串匹配
2. `get_recent_context()` 还存在 `8 -> 5` 的静默截断

bridge / 回退:

1. 命中的上下文会被包装成普通 pseudo message
2. 这些 pseudo message 没有“历史上下文”硬标签可供目标提取时排除

consumer:

1. `_integrate_narration_planning()` 会据此读取用户负面情绪与印象标签
2. `_build_decision_context_packet()` 会据此决定 `target_user_id / target_name / latest_user_text`
3. `_emit_target_profile()` 也会因此把面板对象切到旧上下文用户

结论:

1. `ContextManager` 的问题不只是“少给了 3 条上下文”
2. 更严重的是，它会把弱相关历史命中伪装成“当前对象”
3. 这就是系统容易出现对象漂移、续接错人、面板看错人的根因之一

## 35. 第一层内心独白的触发频控没有统一 governor

### 35.1 主链有动态阈值，但窥屏态绕过了这套总闸

类型: `duplicate authority`

文件:

1. `src/chat/heart_flow/heartFC_chat_enhanced.py`
2. `src/chat/heart_flow/inner_voice.py`

producer:

1. `_compute_llm_call_level()` 会综合能量、夜间、场景、群模式、quiet_preference 等因素
2. `_should_run_voice()` 本来应作为主链“要不要跑独白”的统一判定口

bridge / 回退:

1. 主反应链和主动链会走 `_should_run_voice()`
2. 但窥屏态 `_run_peek_with_reflection()` 明确改成“只做冷却保护，消息来了就调独白”
3. 它直接调用 `_invoke_inner_voice()`，没有经过动态阈值 break-down
4. 触发后还会写回同一个 `_last_voice_ts`

consumer:

1. 窥屏态本身会更频繁地产生第一层内心想法
2. 后续主反应链又会被同一个 `_last_voice_ts` 压住，出现“该想时反而没想”的反向干扰
3. 用户主观体验就会变成“它有时老在想，有时又突然木掉”

结论:

1. 内心独白并不是单一 governor 在控频
2. 而是“主链动态阈值 + 窥屏态时间冷却”两套机制并存
3. 这正是第一层内心所想表现不稳定、看起来像左右脑互搏的一条核心链路
