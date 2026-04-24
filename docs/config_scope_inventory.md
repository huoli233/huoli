# SRC 全量配置分级盘点

> 由 `scripts/config_scope_inventory.py` 基于当前工作区生成。

## 1. SRC 顶层盘点

| 区域 | Python文件数 | 二级子目录(含py数) |
| --- | --- | --- |
| bw_learner | 8 | - |
| chat | 149 | behavior(2), brain_chat(19), chatter(1), emoji_system(1), heart_flow(76), knowledge(14), logger(1), message_receive(5), planner_actions(3), proactive(6), prompts(2), replyer(8), utils(10) |
| common | 28 | config(2), data_models(7), database(2), message(3), message_types(5) |
| config | 8 | - |
| core | 48 | autonomous_core(3) |
| dream | 4 | configs(0), tools(1) |
| express | 3 | - |
| hippo_memorizer | 4 | configs(0) |
| llm_models | 10 | model_client(4), payload_content(3) |
| manager | 3 | - |
| memory_system | 37 | retrieval_tools(9) |
| modules | 66 | brain(2), modcore(16), perception(12), recall(9), safety(7), social_value(8), trauma(8), user_persona(0) |
| person_info | 4 | - |
| plugin_system | 29 | apis(15), base(8), core(5), utils(1) |
| plugins | 4 | built_in(4) |
| webui | 39 | api(2), core(3), middleware(1), routers(19), schemas(5), services(1), utils(0) |

## 2. 第一层：十大类型总览

| 一级类型 | 中文名 | 三级模块数 | system | mixed | user |
| --- | --- | --- | --- | --- | --- |
| dialogue_orchestration | 对话编排 | 15 | 14 | 1 | 0 |
| emotion_psychology | 情绪与心理 | 3 | 3 | 0 | 0 |
| identity_persona | 身份与人格 | 9 | 7 | 1 | 1 |
| interface_observability | 界面与可观测 | 4 | 4 | 0 | 0 |
| learning_adaptation | 学习与自适应 | 2 | 2 | 0 | 0 |
| memory_knowledge | 记忆与知识 | 12 | 12 | 0 | 0 |
| perception_context | 感知与上下文 | 13 | 13 | 0 | 0 |
| runtime_resources | 运行资源与开关 | 4 | 3 | 1 | 0 |
| safety_guard | 安全与防护 | 4 | 4 | 0 | 0 |
| trauma_relation | 创伤与关系 | 12 | 12 | 0 | 0 |

## 3. 第二层：二十到三十类型展开

| 二级类型 | 中文名 | 所属一级类型 | 三级模块数 | 代表模块 |
| --- | --- | --- | --- | --- |
| adaptive_learning | 学习与自适应 | learning_adaptation / 学习与自适应 | 1 | adaptive_learning |
| context_and_vision | 上下文与视觉 | perception_context / 感知与上下文 | 2 | context, vision |
| disposition_sliders | 性格因子 | identity_persona / 身份与人格 | 1 | personality_factors |
| emotion_stream | 情绪流 | emotion_psychology / 情绪与心理 | 2 | chat_emotion, heartfc_thresholds |
| energy_and_trigger | 能量与触发 | runtime_resources / 运行资源与开关 | 2 | energy_runtime, trigger_runtime |
| frequency_and_switches | 频控与开关 | runtime_resources / 运行资源与开关 | 2 | frequency_control, module_switches |
| group_perception | 群体感知 | perception_context / 感知与上下文 | 2 | perception_group_atmosphere, perception_group_sense |
| harassment_injection_guard | 风险检测 | safety_guard / 安全与防护 | 2 | harassment_detection, injection_detection |
| identity_profiles | 身份档案 | identity_persona / 身份与人格 | 3 | identity_anchor, identity_bot, identity_user_persistence |
| memory_capacity_decay | 容量与衰减 | memory_knowledge / 记忆与知识 | 4 | memory, memory_capacity, memory_decay, memory_dedup |
| memory_retrieval | 检索与召回 | memory_knowledge / 记忆与知识 | 1 | memory_retrieval |
| monitoring_thresholds | 阈值与状态展示 | interface_observability / 界面与可观测 | 1 | webui_state_monitor_thresholds |
| persona_dynamics | 人格动态 | identity_persona / 身份与人格 | 5 | affection_dynamics, persona_controller, persona_generator, persona_switcher ... |
| phase_scheduling | 阶段与调度 | dialogue_orchestration / 对话编排 | 2 | phase_timing, schedule |
| proactive_reply | 主动与回复 | dialogue_orchestration / 对话编排 | 4 | heartflow_decision, heartflow_runtime, inner_voice, proactive_decider |
| protection_governance | 保护治理 | safety_guard / 安全与防护 | 2 | global_shield, user_protection |
| psychological_state | 心理状态 | emotion_psychology / 情绪与心理 | 1 | psychological_core |
| recall_rewrite | 回想与纠错 | memory_knowledge / 记忆与知识 | 7 | recall_correction, recall_dimension, recall_post_send, recall_self_awareness ... |
| routing_generation | 路由与生成 | dialogue_orchestration / 对话编排 | 8 | brain_chat_runtime, brain_pfc_action, brain_pfc_goal, brain_pfc_reply ... |
| runtime_tuning | 运行调优 | learning_adaptation / 学习与自适应 | 1 | runtime_tuning |
| semantic_understanding | 语义理解 | perception_context / 感知与上下文 | 4 | perception_buffer, perception_message_preprocessor, perception_signal_detector, perception_understand |
| skill_dispatch | 技能调度 | dialogue_orchestration / 对话编排 | 1 | skill |
| social_relation | 社交关系 | trauma_relation / 创伤与关系 | 5 | social_affect_fuser, social_calculator, social_phase_tracker, social_settlement ... |
| trauma_runtime | 创伤运行 | trauma_relation / 创伤与关系 | 7 | trauma_complex, trauma_fragment, trauma_layers, trauma_system ... |
| user_perception | 用户感知 | perception_context / 感知与上下文 | 5 | perception_behavior, perception_interest, perception_self_sense, perception_user_relation ... |
| webui_runtime | WebUI运行 | interface_observability / 界面与可观测 | 3 | webui_git_mirror, webui_rate_limit, webui_websocket |

## 4. 第三层：五十到一百个模块类型明细

| 三级模块类型 | 一级/二级路径 | 编辑分级 | 四层映射状态 | 用户可改键 | 绑定语义域 |
| --- | --- | --- | --- | --- | --- |
| adaptive_learning | 学习与自适应 / 学习与自适应 | system | 已接入 | - | adaptive_learning |
| affection_dynamics | 身份与人格 / 人格动态 | system | 已接入 | - | affection_dynamics |
| brain_chat_runtime | 对话编排 / 路由与生成 | system | 已接入 | - | brain_chat_runtime |
| brain_pfc_action | 对话编排 / 路由与生成 | system | 已接入 | - | brain_pfc_action |
| brain_pfc_goal | 对话编排 / 路由与生成 | system | 已接入 | - | brain_pfc_goal |
| brain_pfc_reply | 对话编排 / 路由与生成 | system | 已接入 | - | brain_pfc_reply |
| brain_planner | 对话编排 / 路由与生成 | system | 已接入 | - | brain_planner |
| brain_waiter | 对话编排 / 路由与生成 | system | 已接入 | - | brain_waiter |
| chat_emotion | 情绪与心理 / 情绪流 | system | 已接入 | - | chat_emotion |
| context | 感知与上下文 / 上下文与视觉 | system | 已接入 | - | context |
| energy_runtime | 运行资源与开关 / 能量与触发 | system | 已接入 | - | energy_runtime |
| frequency_control | 运行资源与开关 / 频控与开关 | system | 已接入 | - | frequency_control |
| global_shield | 安全与防护 / 保护治理 | system | 已接入 | - | global_shield |
| harassment_detection | 安全与防护 / 风险检测 | system | 已接入 | - | harassment_detection |
| heartfc_thresholds | 情绪与心理 / 情绪流 | system | 已接入 | - | heartfc_thresholds |
| heartflow_decision | 对话编排 / 主动与回复 | system | 已接入 | - | heartflow_decision |
| heartflow_runtime | 对话编排 / 主动与回复 | mixed | 已接入 | focus_channels | heartflow_runtime |
| identity_anchor | 身份与人格 / 身份档案 | system | 已接入 | - | identity_anchor |
| identity_bot | 身份与人格 / 身份档案 | mixed | 已接入 | nickname, alias_names, identity_templates, role_visual_profiles, skill_prompt_templates | identity_bot |
| identity_user_persistence | 身份与人格 / 身份档案 | system | 已接入 | - | identity_user_persistence |
| injection_detection | 安全与防护 / 风险检测 | system | 已接入 | - | injection_detection |
| inner_voice | 对话编排 / 主动与回复 | system | 已接入 | - | inner_voice |
| memory | 记忆与知识 / 容量与衰减 | system | 已接入 | - | memory |
| memory_capacity | 记忆与知识 / 容量与衰减 | system | 已接入 | - | memory_capacity |
| memory_decay | 记忆与知识 / 容量与衰减 | system | 已接入 | - | memory_decay |
| memory_dedup | 记忆与知识 / 容量与衰减 | system | 已接入 | - | memory_dedup |
| memory_retrieval | 记忆与知识 / 检索与召回 | system | 已接入 | - | memory_retrieval |
| message_processor | 对话编排 / 路由与生成 | system | 已接入 | - | message_processor |
| model_routing | 对话编排 / 路由与生成 | system | 已接入 | - | model_routing |
| module_switches | 运行资源与开关 / 频控与开关 | mixed | 已接入 | heartflow_enabled | module_switches |
| perception_behavior | 感知与上下文 / 用户感知 | system | 已接入 | - | perception_behavior |
| perception_buffer | 感知与上下文 / 语义理解 | system | 已接入 | - | perception_buffer |
| perception_group_atmosphere | 感知与上下文 / 群体感知 | system | 已接入 | - | perception_group_atmosphere |
| perception_group_sense | 感知与上下文 / 群体感知 | system | 已接入 | - | perception_group_sense |
| perception_interest | 感知与上下文 / 用户感知 | system | 已接入 | - | perception_interest |
| perception_message_preprocessor | 感知与上下文 / 语义理解 | system | 已接入 | - | perception_message_preprocessor |
| perception_self_sense | 感知与上下文 / 用户感知 | system | 已接入 | - | perception_self_sense |
| perception_signal_detector | 感知与上下文 / 语义理解 | system | 已接入 | - | perception_signal_detector |
| perception_understand | 感知与上下文 / 语义理解 | system | 已接入 | - | perception_understand |
| perception_user_relation | 感知与上下文 / 用户感知 | system | 已接入 | - | perception_user_relation |
| perception_user_state | 感知与上下文 / 用户感知 | system | 已接入 | - | perception_user_state |
| persona_controller | 身份与人格 / 人格动态 | system | 已接入 | - | persona_controller |
| persona_generator | 身份与人格 / 人格动态 | system | 已接入 | - | persona_generator |
| persona_switcher | 身份与人格 / 人格动态 | system | 已接入 | - | persona_switcher |
| personality | 身份与人格 / 人格动态 | system | 已接入 | - | personality |
| personality_factors | 身份与人格 / 性格因子 | user | 已接入 | sensitivity, tolerance, reactiveness, recovery_speed, social_warmth, curiosity_drive, reply_eagerness | personality_factors |
| phase_timing | 对话编排 / 阶段与调度 | system | 已接入 | - | phase_timing |
| proactive_decider | 对话编排 / 主动与回复 | system | 已接入 | - | proactive_decider |
| psychological_core | 情绪与心理 / 心理状态 | system | 已接入 | - | psychological_core |
| recall_correction | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_correction |
| recall_dimension | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_dimension |
| recall_post_send | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_post_send |
| recall_self_awareness | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_self_awareness |
| recall_self_behavior | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_self_behavior |
| recall_shuffle | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_shuffle |
| recall_typo | 记忆与知识 / 回想与纠错 | system | 已接入 | - | recall_typo |
| runtime_tuning | 学习与自适应 / 运行调优 | system | 已接入 | - | runtime_tuning |
| schedule | 对话编排 / 阶段与调度 | system | 已接入 | - | schedule |
| skill | 对话编排 / 技能调度 | system | 已接入 | - | skill |
| social_affect_fuser | 创伤与关系 / 社交关系 | system | 已接入 | - | social_affect_fuser |
| social_calculator | 创伤与关系 / 社交关系 | system | 已接入 | - | social_calculator |
| social_phase_tracker | 创伤与关系 / 社交关系 | system | 已接入 | - | social_phase_tracker |
| social_settlement | 创伤与关系 / 社交关系 | system | 已接入 | - | social_settlement |
| social_value_core | 创伤与关系 / 社交关系 | system | 已接入 | - | social_value_core |
| trauma_complex | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_complex |
| trauma_fragment | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_fragment |
| trauma_layers | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_layers |
| trauma_system | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_system |
| trauma_timeline | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_timeline |
| trauma_triggers | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_triggers |
| trauma_worldview | 创伤与关系 / 创伤运行 | system | 已接入 | - | trauma_worldview |
| trigger_runtime | 运行资源与开关 / 能量与触发 | system | 已接入 | - | trigger_runtime |
| user_protection | 安全与防护 / 保护治理 | system | 已接入 | - | user_protection |
| vision | 感知与上下文 / 上下文与视觉 | system | 已接入 | - | vision |
| webui_git_mirror | 界面与可观测 / WebUI运行 | system | 已接入 | - | webui_git_mirror |
| webui_rate_limit | 界面与可观测 / WebUI运行 | system | 已接入 | - | webui_rate_limit |
| webui_state_monitor_thresholds | 界面与可观测 / 阈值与状态展示 | system | 已接入 | - | webui_state_monitor_thresholds |
| webui_websocket | 界面与可观测 / WebUI运行 | system | 已接入 | - | webui_websocket |

### 分级解释

- `system`：系统级，默认不开放给用户编辑。
- `mixed`：混合级，仅 `user_editable_keys` 中列出的键允许用户改。
- `user`：用户级，整组主要面向用户偏好与行为风格调节。

## 5. 第四层：模块关系层

| 模块 | 所属二级类型 | 主要关联模块 |
| --- | --- | --- |
| adaptive_learning | 学习与自适应 | skill, runtime_tuning, recall_self_behavior, personality_factors |
| affection_dynamics | 人格动态 | personality, persona_controller, persona_generator, persona_switcher |
| brain_chat_runtime | 路由与生成 | model_routing, message_processor, brain_planner, brain_pfc_action, brain_pfc_reply, brain_waiter |
| brain_pfc_action | 路由与生成 | model_routing, message_processor, brain_chat_runtime, brain_planner, brain_pfc_reply, brain_waiter |
| brain_pfc_goal | 路由与生成 | model_routing, message_processor, brain_chat_runtime, brain_planner, brain_pfc_action, brain_pfc_reply |
| brain_pfc_reply | 路由与生成 | model_routing, message_processor, brain_chat_runtime, brain_planner, brain_pfc_action, brain_waiter |
| brain_planner | 路由与生成 | model_routing, message_processor, brain_chat_runtime, brain_pfc_action, brain_pfc_reply, brain_waiter |
| brain_waiter | 路由与生成 | model_routing, message_processor, brain_chat_runtime, brain_planner, brain_pfc_action, brain_pfc_reply |
| chat_emotion | 情绪流 | heartfc_thresholds |
| context | 上下文与视觉 | vision, memory_retrieval, perception_understand, message_processor |
| energy_runtime | 能量与触发 | trigger_runtime, frequency_control, heartflow_runtime, chat_emotion |
| frequency_control | 频控与开关 | module_switches |
| global_shield | 保护治理 | user_protection |
| harassment_detection | 风险检测 | injection_detection |
| heartfc_thresholds | 情绪流 | chat_emotion |
| heartflow_decision | 主动与回复 | proactive_decider, heartflow_runtime, inner_voice |
| heartflow_runtime | 主动与回复 | schedule, phase_timing, proactive_decider, inner_voice, heartflow_decision |
| identity_anchor | 身份档案 | identity_bot, personality_factors, social_phase_tracker, identity_user_persistence |
| identity_bot | 身份档案 | identity_anchor, personality_factors, heartflow_runtime, skill, identity_user_persistence |
| identity_user_persistence | 身份档案 | identity_bot, identity_anchor |
| injection_detection | 风险检测 | harassment_detection |
| inner_voice | 主动与回复 | proactive_decider, heartflow_decision, heartflow_runtime |
| memory | 容量与衰减 | memory_retrieval, context, recall_post_send, social_phase_tracker, memory_capacity, memory_decay |
| memory_capacity | 容量与衰减 | memory, memory_decay, memory_dedup |
| memory_decay | 容量与衰减 | memory, memory_capacity, memory_dedup |
| memory_dedup | 容量与衰减 | memory, memory_capacity, memory_decay |
| memory_retrieval | 检索与召回 | - |
| message_processor | 路由与生成 | model_routing, brain_chat_runtime, brain_planner, brain_pfc_action, brain_pfc_reply, brain_waiter |
| model_routing | 路由与生成 | brain_chat_runtime, vision, adaptive_learning, runtime_tuning, message_processor, brain_planner |
| module_switches | 频控与开关 | heartflow_runtime, skill, vision, proactive_decider, frequency_control |
| perception_behavior | 用户感知 | perception_self_sense, perception_user_relation, perception_user_state, perception_interest |
| perception_buffer | 语义理解 | perception_message_preprocessor, perception_signal_detector, perception_understand |
| perception_group_atmosphere | 群体感知 | perception_group_sense |
| perception_group_sense | 群体感知 | perception_group_atmosphere |
| perception_interest | 用户感知 | perception_self_sense, perception_user_relation, perception_user_state, perception_behavior |
| perception_message_preprocessor | 语义理解 | perception_signal_detector, perception_understand, perception_buffer |
| perception_self_sense | 用户感知 | perception_user_relation, perception_user_state, perception_behavior, perception_interest |
| perception_signal_detector | 语义理解 | perception_message_preprocessor, perception_understand, perception_buffer |
| perception_understand | 语义理解 | perception_message_preprocessor, perception_signal_detector, perception_buffer |
| perception_user_relation | 用户感知 | perception_self_sense, perception_user_state, perception_behavior, perception_interest |
| perception_user_state | 用户感知 | perception_self_sense, perception_user_relation, perception_behavior, perception_interest |
| persona_controller | 人格动态 | personality, affection_dynamics, persona_generator, persona_switcher |
| persona_generator | 人格动态 | personality, affection_dynamics, persona_controller, persona_switcher |
| persona_switcher | 人格动态 | personality, affection_dynamics, persona_controller, persona_generator |
| personality | 人格动态 | affection_dynamics, persona_controller, persona_generator, persona_switcher |
| personality_factors | 性格因子 | personality, heartflow_decision, social_calculator, psychological_core |
| phase_timing | 阶段与调度 | schedule, heartflow_runtime, brain_waiter, heartflow_decision |
| proactive_decider | 主动与回复 | heartflow_decision, heartflow_runtime, inner_voice |
| psychological_core | 心理状态 | trauma_system, social_calculator, chat_emotion, heartflow_decision |
| recall_correction | 回想与纠错 | recall_post_send, recall_typo, recall_self_behavior, recall_self_awareness, recall_dimension, recall_shuffle |
| recall_dimension | 回想与纠错 | recall_post_send, recall_typo, recall_self_behavior, recall_self_awareness, recall_correction, recall_shuffle |
| recall_post_send | 回想与纠错 | recall_typo, recall_self_behavior, recall_self_awareness, recall_dimension, recall_correction, recall_shuffle |
| recall_self_awareness | 回想与纠错 | recall_post_send, recall_typo, recall_self_behavior, recall_dimension, recall_correction, recall_shuffle |
| recall_self_behavior | 回想与纠错 | recall_post_send, recall_typo, recall_self_awareness, recall_dimension, recall_correction, recall_shuffle |
| recall_shuffle | 回想与纠错 | recall_post_send, recall_typo, recall_self_behavior, recall_self_awareness, recall_dimension, recall_correction |
| recall_typo | 回想与纠错 | recall_post_send, recall_self_behavior, recall_self_awareness, recall_dimension, recall_correction, recall_shuffle |
| runtime_tuning | 运行调优 | schedule, model_routing, heartflow_runtime, adaptive_learning |
| schedule | 阶段与调度 | phase_timing, proactive_decider, heartflow_runtime, recall_post_send |
| skill | 技能调度 | - |
| social_affect_fuser | 社交关系 | social_calculator, social_phase_tracker, chat_emotion, heartflow_decision, social_settlement, social_value_core |
| social_calculator | 社交关系 | social_settlement, social_phase_tracker, personality_factors, psychological_core, social_affect_fuser, social_value_core |
| social_phase_tracker | 社交关系 | social_calculator, social_settlement, social_affect_fuser, social_value_core |
| social_settlement | 社交关系 | social_calculator, social_phase_tracker, social_affect_fuser, social_value_core |
| social_value_core | 社交关系 | social_calculator, social_settlement, social_phase_tracker, social_affect_fuser |
| trauma_complex | 创伤运行 | trauma_fragment, trauma_layers, trauma_system, trauma_timeline, trauma_triggers, trauma_worldview |
| trauma_fragment | 创伤运行 | trauma_complex, trauma_layers, trauma_system, trauma_timeline, trauma_triggers, trauma_worldview |
| trauma_layers | 创伤运行 | trauma_complex, trauma_fragment, trauma_system, trauma_timeline, trauma_triggers, trauma_worldview |
| trauma_system | 创伤运行 | trauma_layers, trauma_triggers, trauma_worldview, psychological_core, trauma_complex, trauma_fragment |
| trauma_timeline | 创伤运行 | trauma_complex, trauma_fragment, trauma_layers, trauma_system, trauma_triggers, trauma_worldview |
| trauma_triggers | 创伤运行 | trauma_complex, trauma_fragment, trauma_layers, trauma_system, trauma_timeline, trauma_worldview |
| trauma_worldview | 创伤运行 | trauma_complex, trauma_fragment, trauma_layers, trauma_system, trauma_timeline, trauma_triggers |
| trigger_runtime | 能量与触发 | energy_runtime, proactive_decider, heartflow_decision, module_switches |
| user_protection | 保护治理 | global_shield |
| vision | 上下文与视觉 | context, perception_understand, message_processor, memory |
| webui_git_mirror | WebUI运行 | webui_rate_limit, webui_websocket |
| webui_rate_limit | WebUI运行 | webui_websocket, webui_git_mirror |
| webui_state_monitor_thresholds | 阈值与状态展示 | chat_emotion, social_phase_tracker, trauma_system, energy_runtime |
| webui_websocket | WebUI运行 | webui_rate_limit, webui_git_mirror |

## 6. 已存在的运行时配置桥接文件

- `src/chat/brain_chat/runtime_config.py`
- `src/memory_system/runtime_config.py`
- `src/modules/modcore/dynamic_persona/runtime_config.py`
- `src/modules/perception/runtime_config.py`
- `src/modules/recall/runtime_config.py`
- `src/modules/safety/runtime_config.py`
- `src/modules/social_value/runtime_config.py`
- `src/modules/trauma/runtime_config.py`
- `src/person_info/runtime_config.py`
- `src/webui/runtime_config.py`

## 7. 仍有旧配置体系痕迹、可继续收口到四层映射的热点

### `chat`

- `src/chat/heart_flow/behavior_analyzer.py`

### `common`

- `src/common/config/config_engine.py`
- `src/common/config/prompt_manager.py`

### `memory_system`

- `src/memory_system/aging_processor.py`
- `src/memory_system/emotion_importance.py`
- `src/memory_system/output_formatter.py`
- `src/memory_system/overload_governor.py`
- `src/memory_system/vector_store.py`

### `modules`

- `src/modules/modcore/dynamic_persona/persona_config_parser.py`
- `src/modules/modcore/perception/perception_generator.py`
- `src/modules/modcore/social_cognition/relationship_controller.py`
- `src/modules/safety/harassment_detector.py`
- `src/modules/safety/injection/detector.py`
- `src/modules/safety/social_calculator.py`
- `src/modules/safety/user_protection.py`

## 8. 当前建议

- 已接入四层映射的 `78` 个模块，优先继续细化 `edit_scope` 与 `user_editable_keys`。
- 现阶段一级类型共 `10` 个，二级类型共 `26` 个，三级模块类型共 `78` 个，满足你要的四层盘点规模。
- 旧配置热点里，`memory_system` 和 `modules/safety` 是下一轮最适合继续动刀的目录。
- 基础设施目录（如 `common`、`llm_models`、`plugin_system`）若无真实配置热点，默认保持系统级。
