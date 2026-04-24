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

## 2. 已接入四层映射的 Module View 与编辑分级

| Module View | 编辑分级 | 绑定语义域 | 用户可改键 | 系统锁定键 |
| --- | --- | --- | --- | --- |
| adaptive_learning | system | adaptive_learning | - | - |
| affection_dynamics | system | affection_dynamics | - | - |
| brain_chat_runtime | system | brain_chat_runtime | - | - |
| brain_pfc_action | system | brain_pfc_action | - | - |
| brain_pfc_goal | system | brain_pfc_goal | - | - |
| brain_pfc_reply | system | brain_pfc_reply | - | - |
| brain_planner | system | brain_planner | - | - |
| brain_waiter | system | brain_waiter | - | - |
| chat_emotion | system | chat_emotion | - | - |
| context | system | context | - | - |
| energy_runtime | system | energy_runtime | - | - |
| frequency_control | system | frequency_control | - | - |
| global_shield | system | global_shield | - | - |
| harassment_detection | system | harassment_detection | - | - |
| heartfc_thresholds | system | heartfc_thresholds | - | - |
| heartflow_decision | system | heartflow_decision | - | - |
| heartflow_runtime | mixed | heartflow_runtime | focus_channels | initiative_probability |
| identity_anchor | system | identity_anchor | - | - |
| identity_bot | mixed | identity_bot | nickname, alias_names, identity_templates, role_visual_profiles, skill_prompt_templates | bot_id |
| identity_user_persistence | system | identity_user_persistence | - | - |
| injection_detection | system | injection_detection | - | - |
| inner_voice | system | inner_voice | - | - |
| memory | system | memory | - | - |
| memory_capacity | system | memory_capacity | - | - |
| memory_decay | system | memory_decay | - | - |
| memory_dedup | system | memory_dedup | - | - |
| memory_retrieval | system | memory_retrieval | - | - |
| message_processor | system | message_processor | - | - |
| model_routing | system | model_routing | - | - |
| module_switches | mixed | module_switches | heartflow_enabled | - |
| perception_behavior | system | perception_behavior | - | - |
| perception_buffer | system | perception_buffer | - | - |
| perception_group_atmosphere | system | perception_group_atmosphere | - | - |
| perception_group_sense | system | perception_group_sense | - | - |
| perception_interest | system | perception_interest | - | - |
| perception_message_preprocessor | system | perception_message_preprocessor | - | - |
| perception_self_sense | system | perception_self_sense | - | - |
| perception_signal_detector | system | perception_signal_detector | - | - |
| perception_understand | system | perception_understand | - | - |
| perception_user_relation | system | perception_user_relation | - | - |
| perception_user_state | system | perception_user_state | - | - |
| persona_controller | system | persona_controller | - | - |
| persona_generator | system | persona_generator | - | - |
| persona_switcher | system | persona_switcher | - | - |
| personality | system | personality | - | - |
| personality_factors | user | personality_factors | sensitivity, tolerance, reactiveness, recovery_speed, social_warmth, curiosity_drive, reply_eagerness | - |
| phase_timing | system | phase_timing | - | - |
| proactive_decider | system | proactive_decider | - | - |
| psychological_core | system | psychological_core | - | - |
| recall_correction | system | recall_correction | - | - |
| recall_dimension | system | recall_dimension | - | - |
| recall_post_send | system | recall_post_send | - | - |
| recall_self_awareness | system | recall_self_awareness | - | - |
| recall_self_behavior | system | recall_self_behavior | - | - |
| recall_shuffle | system | recall_shuffle | - | - |
| recall_typo | system | recall_typo | - | - |
| runtime_tuning | system | runtime_tuning | - | - |
| schedule | system | schedule | - | - |
| skill | system | skill | - | - |
| social_affect_fuser | system | social_affect_fuser | - | - |
| social_calculator | system | social_calculator | - | - |
| social_phase_tracker | system | social_phase_tracker | - | - |
| social_settlement | system | social_settlement | - | - |
| social_value_core | system | social_value_core | - | - |
| trauma_complex | system | trauma_complex | - | - |
| trauma_fragment | system | trauma_fragment | - | - |
| trauma_layers | system | trauma_layers | - | - |
| trauma_system | system | trauma_system | - | - |
| trauma_timeline | system | trauma_timeline | - | - |
| trauma_triggers | system | trauma_triggers | - | - |
| trauma_worldview | system | trauma_worldview | - | - |
| trigger_runtime | system | trigger_runtime | - | - |
| user_protection | system | user_protection | - | - |
| vision | system | vision | - | - |
| webui_git_mirror | system | webui_git_mirror | - | - |
| webui_rate_limit | system | webui_rate_limit | - | - |
| webui_state_monitor_thresholds | system | webui_state_monitor_thresholds | - | - |
| webui_websocket | system | webui_websocket | - | - |

### 分级解释

- `system`：系统级，默认不开放给用户编辑。
- `mixed`：混合级，仅 `user_editable_keys` 中列出的键允许用户改。
- `user`：用户级，整组主要面向用户偏好与行为风格调节。

## 3. 已存在的运行时配置桥接文件

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

## 4. 仍有旧配置体系痕迹、可继续收口到四层映射的热点

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

## 5. 当前建议

- 已接入四层映射的模块，继续细化 `edit_scope` 即可，不需要再回到旧配置引擎。
- 仍出现旧配置引擎调用的热点目录，优先作为下一轮四层映射改造候选。
- 基础设施目录（如 `common`、`llm_models`、`plugin_system`）若无真实配置热点，默认保持系统级。
