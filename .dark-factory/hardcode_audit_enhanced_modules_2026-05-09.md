# enhanced_modules 硬编码审计 - 2026-05-09

## 结论

本轮确认 `proactive_reactive_flow_mixin.py` 的强制回复兜底链路之前确实还有硬编码：`刀盾`、固定 bot 名、问句触发词、招呼词仍写在 Python 判断里。当前已迁移到 `runtime_tuning` 配置和 `identity_bot` 身份配置读取。

审计范围是 `src/chat/heart_flow/enhanced_modules/*.py`。分类口径：

- P0：会进入聊天、模型 prompt、风格/长度/规划提示的中文文本。
- P1：策略触发词、状态原因、运行态文本、行为解释文本。
- P2：非平凡数值阈值、窗口、权重、冷却时间。
- log/doc/ascii_protocol 未列入下表，详见同名 JSON 明细。

## 模块统计

| 模块 | P0 聊天/Prompt | P1 策略/运行文本 | P2 数值 |
| --- | ---: | ---: | ---: |
| scene_planner_bridge_mixin.py | 86 | 51 | 123 |
| voice_pipeline_mixin.py | 84 | 41 | 142 |
| scene_context_analysis_mixin.py | 56 | 15 | 47 |
| strategy_relation_style_mixin.py | 52 | 21 | 218 |
| proactive_context_prompt_mixin.py | 40 | 66 | 119 |
| interaction_core_mixin.py | 36 | 56 | 167 |
| proactive_idle_reply_mixin.py | 23 | 20 | 26 |
| loop_reply_execution_mixin.py | 22 | 0 | 5 |
| loop_state_flow_mixin.py | 14 | 6 | 19 |
| scene_bot_lifecycle_mixin.py | 13 | 77 | 150 |
| proactive_reactive_flow_mixin.py | 11 | 54 | 11 |
| runtime_state_trace_mixin.py | 4 | 4 | 13 |
| runtime_integration_lifecycle_mixin.py | 3 | 28 | 118 |
| loop_main_driver_mixin.py | 2 | 73 | 97 |
| strategy_action_state_mixin.py | 2 | 82 | 174 |
| runtime_sensory_pipeline_mixin.py | 0 | 29 | 171 |
| runtime_subjective_signal_mixin.py | 0 | 12 | 167 |
| runtime_watch_governor_mixin.py | 0 | 22 | 93 |
| loop_phase_transition_mixin.py | 0 | 14 | 35 |
| loop_resource_feedback_mixin.py | 0 | 132 | 232 |
| strategy_proactive_decision_mixin.py | 0 | 10 | 189 |
| shared_runtime.py | 0 | 3 | 22 |
| loop_cycle_core_mixin.py | 0 | 0 | 0 |
| loop_flow_mixin.py | 0 | 0 | 0 |
| proactive_execution_mixin.py | 0 | 0 | 0 |
| runtime_governor_pipeline_mixin.py | 0 | 0 | 0 |
| scene_analysis_mixin.py | 0 | 0 | 0 |
| strategy_integration_mixin.py | 0 | 0 | 0 |
| subjective_runtime_mixin.py | 0 | 0 | 0 |

## 优先治理队列

1. P0 prompt/chat：`scene_planner_bridge_mixin.py`、`voice_pipeline_mixin.py`、`scene_context_analysis_mixin.py`、`strategy_relation_style_mixin.py`、`proactive_context_prompt_mixin.py`。
2. P1 策略触发词：`interaction_core_mixin.py`、`proactive_context_prompt_mixin.py`、`strategy_action_state_mixin.py`、`runtime_sensory_pipeline_mixin.py`。
3. P2 数值参数：`loop_resource_feedback_mixin.py`、`strategy_relation_style_mixin.py`、`strategy_proactive_decision_mixin.py`、`strategy_action_state_mixin.py`、`runtime_sensory_pipeline_mixin.py`。

## 本轮已处理

- `proactive_reactive_flow_mixin.py` 强制回复兜底触发词改为读取：
  - `heartfc_force_reply_fallback_daodun_markers`
  - `heartfc_force_reply_fallback_bot_name_markers`
  - `heartfc_force_reply_fallback_question_markers`
  - `heartfc_force_reply_fallback_greeting_markers`
- bot 名称触发额外读取 `identity_bot.nickname`、`identity_bot.alias_names` 和 `global_config.bot`，不再在模块里写死具体名字。
- `scripts/heartflow_state_regression.py` 增加源码契约，防止强制回复函数体重新出现配置中的触发词字面量。

机器可读明细：`.dark-factory/hardcode_audit_enhanced_modules_2026-05-09.json`。
