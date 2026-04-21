import sys
sys.path.insert(0, "src")

untested = [
    "webui.routers.model","webui.routers.plugin","webui.routers.system",
    "webui.routers.expression","webui.routers.emoji","webui.routers.config",
    "webui.routers.annual_report","webui.routers.jargon","webui.routers.knowledge",
    "webui.routers.statistics","webui.routers.person","webui.routers.chat",
    "webui.routers.logs","webui.routes","webui.app",
    "webui.schemas.auth","webui.schemas.plugin","webui.schemas.statistics",
    "webui.schemas.emoji","webui.schemas.chat","webui.config_schema",
    "webui.dependencies","webui.middleware.anti_crawler",
    "plugin_system.apis.tool_api","plugin_system.apis.send_api",
    "plugin_system.apis.message_api","plugin_system.apis.logging_api",
    "plugin_system.apis.person_api","plugin_system.apis.llm_api",
    "plugin_system.apis.frequency_api","plugin_system.apis.emoji_api",
    "plugin_system.apis.config_api","plugin_system.apis.chat_api",
    "core.emotion_feedback_loop","core.cross_engine_validator",
    "core.engine_orchestrator","core.state_dashboard",
    "core.deep_visibility_scorer","core.gossip_ritual_strategy",
    "core.group_scene_state","core.freshness_decay_engine",
    "core.dynamic_context_window","core.content_state_tracker",
    "core.huoli_core",
    "memory_system.user_cognitive_store",
    "memory_system.retrieval_tools.query_chat_history",
    "memory_system.retrieval_tools.query_embedding_memory",
    "memory_system.retrieval_tools.query_person_info",
    "memory_system.retrieval_tools.query_words",
    "memory_system.retrieval_tools.query_lpmm_knowledge",
    "memory_system.retrieval_tools.return_information",
    "memory_system.retrieval_tools.tool_loader",
    "chat.utils.utils_voice","chat.utils.message_preprocessor",
    "chat.utils.user_repeat_detector","chat.utils.utils_image",
    "chat.utils.typo_generator","chat.utils.timer_calculator",
    "chat.utils.chat_message_builder","chat.prompts.soul_config_loader",
    "person_info.person_info","person_info.user_persistence","person_info.bot_identity",
    "modules.trauma.trauma_triggers","modules.trauma.trauma_timeline",
    "modules.trauma.trauma_worldview","modules.trauma.trauma_layers",
    "modules.trauma.fragment_stimulus","modules.trauma.complex_psychology",
    "modules.safety.social_calculator","modules.safety.user_protection",
    "bw_learner.jargon_explainer","bw_learner.expression_selector",
    "bw_learner.learner_utils","bw_learner.expression_auto_check_task",
    "hippo_memorizer.summary_storage","hippo_memorizer.config_loader",
    "hippo_memorizer.action_recorder",
    "llm_models.utils","llm_models.model_client.openai_client",
    "llm_models.model_client.gemini_client","llm_models.model_client.base_client",
    "llm_models.model_client._registry",
    "dream.dream_generator","dream.tools.dream_tools","dream.config_loader",
]

ok = 0
fail = 0
fails = []
for m in untested:
    try:
        __import__(m)
        ok += 1
    except Exception as e:
        fail += 1
        fails.append(f"{m}: {type(e).__name__}: {e}")

print(f"\n=== UNTTESTED MODULES: {ok} OK, {fail} FAIL ===")
for f in fails:
    print(f"  FAIL: {f}")
