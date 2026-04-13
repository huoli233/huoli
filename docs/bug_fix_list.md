# HuoLi 项目 Bug 修复清单

> 审查日期：2026-04-13
> 状态：审查已完成，停止分析，进入修复阶段
> 规则：不再分析新问题，只修以下已确认的 bug

---

## P0 - 必须修复（影响运行时行为）

### BUG-001: brain_planner 错误回退为 complete_talk
- **文件**: `src/chat/brain_chat/brain_planner.py:256-266, 281-292, 698-706, 743-762`
- **问题**: 所有错误（解析失败、无效动作、模型失败、空响应、请求异常）统一回退为 `complete_talk`，模型故障被误当对话自然收束
- **后果**: 过早停话、错误沉默、错过应答时机
- **修复方向**: 把错误回退和业务性 `complete_talk` 分开；故障场景返回显式 degraded/error action

### BUG-002: DB MessageRecv 降级对象伪装完整消息
- **状态**: fixed
- **文件**: `src/plugin_system/apis/send_api.py:140-181`、`src/chat/message_receive/message.py:157-199`、`src/common/database/database_model.py:182-185`
- **问题**: `db_message_to_message_recv()` 生成的对象伪装成 `MessageRecv`，但没有真实 `message_segment`，`additional_config/priority_info` 遇到非 dict 直接清空为 `{}`
- **后果**: 引用/提及/优先级信息静默丢失，下游误以为引用的是完整原消息
- **修复方向**: 显式定义 `ReconstructedMessage` 类型，打上 `reconstructed=true` 标记，禁止当完整消息使用

### BUG-003: 插件 config schema 把推断值冒充运行态
- **文件**: `src/webui/routers/plugin.py:1650-1825`
- **问题**: 插件未加载时，config-schema 接口从 config.toml 推断出伪 schema 并补充伪造的 `required/hidden/disabled/choices`，返回结构跟运行态完全一样
- **后果**: 前端生成错误表单，通过保存接口写回错误配置
- **修复方向**: 返回结构加 `schema_source=runtime|inferred`，未加载场景禁止驱动保存

---

## P1 - 应该修复（影响可维护性和稳定性）

### BUG-004: status_line 三来源混拼
- **文件**: `src/chat/heart_flow/heartFC_chat_enhanced.py:7156-7185`
- **问题**: `status_line` 先用缓存快照，失败用引擎方法，再失败用手工拼串，三个来源不同粒度不同完备度
- **后果**: 前端和监控拿到来源不明的字符串
- **修复方向**: 暴露 `status_line_source`，降级为纯展示文本，禁止跨边界直接消费

### BUG-005: plugin_id 推断修正被后续接口当真实主键
- **文件**: `src/webui/routers/plugin.py:1444-1508` 及后续管理接口
- **问题**: 列表接口根据 `author/repository_url/folder_name` 推断 `plugin_id` 并写回 manifest，后续接口以这个 ID 作为定位主键
- **后果**: 推断值错误时，读接口错误放大成管理误操作
- **修复方向**: 禁止列表接口自动修正主键，ID 修复做成显式迁移操作

### BUG-006: reasoning/action_reasoning 跨链语义不一致
- **文件**: `src/chat/planner_actions/planner.py:285-300`、`src/chat/chat_core_base.py:543-580`、`src/chat/brain_chat/brain_planner.py:268-278`、`src/chat/logger/plan_reply_logger.py:122-133`
- **问题**: 主 planner、统一规划回退链、brain planner 各自对 `reasoning`/`action_reasoning` 填不同语义内容，但 `PlanReplyLogger` 统一持久化
- **后果**: 日志分析、回放调试、数据训练把不同语义混为一谈
- **修复方向**: 统一动作日志 schema，显式拆分 `planner_reasoning`/`action_decision_reason`/`content_plan`

### BUG-007: can_reply_confidence 名字宣称置信度实际只是布尔值
- **文件**: `src/core/world_snapshot.py:752-770`
- **问题**: `can_reply_confidence` 名称暗示数值置信度，实际直接复用 `can_reply` 的布尔值
- **后果**: 调用方误当概率/评分使用
- **修复方向**: 改名为 `can_reply`/`reply_allowed`；若需置信度则单独提供数值字段

---

## P2 - 建议修复（命名规范和接口清晰度）

### BUG-008: UnifiedFlowSnapshot 临时诊断态伪装稳定执行契约
- **文件**: `src/chat/heart_flow/heartFC_chat_enhanced.py:5801-5872,6037-6064`、`src/common/data_models/heartflow_models.py:900-941`
- **问题**: `UnifiedFlowSnapshot` 把 `voice_should_reply`/`final_decision`/`blocker` 序列化成通用快照字段，直接回填 `watch_reason/phase_reason`
- **修复方向**: 快照限制为纯观测字段，仲裁态改入专用内部诊断结构

### BUG-009: ActionPlannerInfo 同名字段跨层语义漂移
- **文件**: `src/common/data_models/info_data_model.py:19-26`、`src/chat/logger/plan_reply_logger.py:122-133`、`src/chat/utils/chat_message_builder.py:88-102`、`src/webui/routers/annual_report.py:611-626`
- **问题**: `ActionPlannerInfo` 同时暴露 `reasoning` 与 `action_reasoning`，跨层后语义漂移；年报把 `action_reasoning` 长度当思考深度
- **修复方向**: 统一规划/执行原因字段，只保留一个 canonical source

---

## 修复规则

1. **停止分析新问题** - 只修以上 9 项
2. **按优先级修** - P0 > P1 > P2
3. **修完一项划掉一项** - 在本文件标记修复状态
4. **每个修复独立 commit** - 不要混合多个 bug 的修复
