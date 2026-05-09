import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_MODEL = "gemini-2.5-flash"
BLOCKED_REPLYER_MODELS: set[str] = set()
BLOCKED_RUNTIME_MODEL_TEXT = ("gemini-3", "gemini-2.5-flash-lite")
LEGACY_COMPARE_PROVIDER_NAMES = {"AggAPI"}
LEGACY_COMPARE_MODEL_NAMES = {
    "gemma-3-1b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gemma-4-26b",
    "gemma-4-31b",
}
TOOL_MODELS = ("qwen3-30b",)
NON_TOOL_TASKS = (
    "utils",
    "model_monitor",
    "planner",
    "vlm",
    "focus_chat",
    "lightweight",
    "lpmm_entity_extract",
    "lpmm_rdf_build",
)


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _blocked_runtime_models(model_list: list[str]) -> list[str]:
    return sorted(
        model_name
        for model_name in model_list
        if any(blocked in model_name for blocked in BLOCKED_RUNTIME_MODEL_TEXT)
    )


def _assert_no_blocked_runtime_model_text(rel: str, path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    blocked = [token for token in BLOCKED_RUNTIME_MODEL_TEXT if token in source]
    assert not blocked, f"{rel} 仍包含非指定 Gemini 配置: {blocked}"


def _assert_target_model_client_contract(rel: str, data: dict[str, Any]) -> None:
    providers = {
        str(provider.get("name")): provider
        for provider in data.get("api_providers", [])
        if isinstance(provider, dict) and provider.get("name")
    }
    models = [item for item in data.get("models", []) if isinstance(item, dict)]
    for model in models:
        if model.get("name") != TARGET_MODEL:
            continue
        provider_name = str(model.get("api_provider", "") or "")
        provider = providers.get(provider_name)
        assert provider, f"{rel}:{TARGET_MODEL} 引用了未定义 provider {provider_name}"
        provider_client_type = str(provider.get("client_type", "") or "").strip().lower()
        model_client_type = str(model.get("client_type", "") or "").strip().lower()
        if provider_client_type == "openai":
            assert model_client_type in {"", "openai"}, (
                f"{rel}:{TARGET_MODEL} 使用 OpenAI 兼容 provider={provider_name}，"
                f"模型级 client_type 不能覆盖为 {model_client_type!r}"
            )
        if rel == "config/model_config.toml" and provider_name == "GeminiProxy":
            base_url = str(provider.get("base_url", "") or "").rstrip("/")
            assert base_url.endswith("/v1"), (
                f"{rel}:GeminiProxy 是 OpenAI 兼容地址，base_url 必须带 /v1: {base_url}"
            )


def _assert_compare_template_not_legacy(rel: str, data: dict[str, Any]) -> dict[str, list[str]]:
    provider_names = sorted(
        str(provider.get("name", ""))
        for provider in data.get("api_providers", [])
        if isinstance(provider, dict) and provider.get("name")
    )
    model_names = sorted(
        str(model.get("name", ""))
        for model in data.get("models", [])
        if isinstance(model, dict) and model.get("name")
    )
    legacy_providers = sorted(set(provider_names) & LEGACY_COMPARE_PROVIDER_NAMES)
    legacy_models = sorted(set(model_names) & LEGACY_COMPARE_MODEL_NAMES)
    assert not legacy_providers, f"{rel} 仍包含旧默认供应商: {legacy_providers}"
    assert not legacy_models, f"{rel} 仍包含旧默认模型: {legacy_models}"
    return {
        "providers": provider_names,
        "models": model_names,
    }


def check_runtime_model_config() -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for rel in (
        "config/model_config.toml",
        "config/model_config_optimized.toml",
        "template/model_config_template.toml",
    ):
        path = ROOT / rel
        if not path.exists():
            continue
        _assert_no_blocked_runtime_model_text(rel, path)
        data = _load_toml(path)
        _assert_target_model_client_contract(rel, data)
        tasks = data.get("model_task_config", {})
        route_map: dict[str, list[str]] = {}
        for task_name, task_config in tasks.items():
            if not isinstance(task_config, dict):
                continue
            model_list = list(task_config.get("model_list", []))
            blocked = _blocked_runtime_models(model_list)
            assert not blocked, f"{rel}:{task_name} 不能再路由到 Gemini 3/非指定 Gemini 变体: {blocked}"
        for task_name in NON_TOOL_TASKS:
            if task_name not in tasks:
                continue
            model_list = list(tasks[task_name].get("model_list", []))
            route_map[task_name] = model_list
            assert model_list == [TARGET_MODEL], f"{rel}:{task_name} 仍未统一到 {TARGET_MODEL}: {model_list}"
        tool_models = list(tasks.get("tool_use", {}).get("model_list", []))
        if tool_models:
            assert tool_models != [TARGET_MODEL], f"{rel}:tool_use 不能只配置不支持工具调用的 {TARGET_MODEL}"
            assert tool_models == list(TOOL_MODELS), f"{rel}:tool_use 应使用远端工具模型 {TOOL_MODELS}: {tool_models}"
            models_by_name = {item.get("name"): item for item in data.get("models", []) if isinstance(item, dict)}
            for model_name in tool_models:
                model_info = models_by_name.get(model_name)
                assert model_info, f"{rel}:tool_use 引用了未定义模型 {model_name}"
                assert model_info.get("supports_tool_calling") is True, (
                    f"{rel}:tool_use 模型 {model_name} 未声明 supports_tool_calling=true"
                )
            route_map["tool_use"] = tool_models
        models_by_name = {item.get("name"): item for item in data.get("models", []) if isinstance(item, dict)}
        replyer_models = list(tasks.get("replyer", {}).get("model_list", []))
        if replyer_models:
            blocked = sorted(set(replyer_models) & BLOCKED_REPLYER_MODELS)
            assert not blocked, f"{rel}:replyer 不应再包含鉴权失败模型: {blocked}"
            assert TARGET_MODEL in replyer_models, f"{rel}:replyer 必须包含 {TARGET_MODEL}: {replyer_models}"
            for model_name in replyer_models:
                model_info = models_by_name.get(model_name)
                assert model_info, f"{rel}:replyer 引用了未定义模型 {model_name}"
                provider = str(model_info.get("api_provider", "") or "")
                assert provider, f"{rel}:replyer 模型 {model_name} 缺少 api_provider"
            route_map["replyer"] = replyer_models
        checked[rel] = route_map
    compare_path = ROOT / "template/compare/model_config_template.toml"
    if compare_path.exists():
        _assert_no_blocked_runtime_model_text("template/compare/model_config_template.toml", compare_path)
        data = _load_toml(compare_path)
        compare_identity = _assert_compare_template_not_legacy(
            "template/compare/model_config_template.toml",
            data,
        )
        tasks = data.get("model_task_config", {})
        compare_routes: dict[str, list[str]] = {}
        for task_name, task_config in tasks.items():
            if not isinstance(task_config, dict):
                continue
            model_list = list(task_config.get("model_list", []))
            blocked = _blocked_runtime_models(model_list)
            assert not blocked, (
                "template/compare/model_config_template.toml:"
                f"{task_name} 不能再路由到 Gemini 3/非指定 Gemini 变体: {blocked}"
            )
            if task_name in {"vlm", "replyer", "focus_chat"}:
                compare_routes[task_name] = model_list
        replyer_models = list(tasks.get("replyer", {}).get("model_list", []))
        if replyer_models:
            blocked = sorted(set(replyer_models) & BLOCKED_REPLYER_MODELS)
            assert not blocked, f"template/compare/model_config_template.toml:replyer 不应再包含鉴权失败模型: {blocked}"
            compare_routes["replyer"] = replyer_models
        compare_routes["providers"] = compare_identity["providers"]
        checked["template/compare/model_config_template.toml"] = compare_routes
    return checked


def audit_data_json_sources() -> dict[str, Any]:
    data_dir = ROOT / "data"
    json_files = sorted(data_dir.rglob("*.json")) if data_dir.exists() else []
    assert not json_files, "data 目录不允许再保留或生成运行态 JSON 文件"
    grouped: dict[str, int] = {}
    for path in json_files:
        try:
            key = str(path.parent.relative_to(data_dir))
        except ValueError:
            key = str(path.parent)
        grouped[key] = grouped.get(key, 0) + 1
    night_cycle_source = (ROOT / "src/core/night_cycle_system.py").read_text(encoding="utf-8")
    assert "_night_state_slot_key" in night_cycle_source
    assert "_ensure_persistence_slot_table" in night_cycle_source
    assert "Slot.insert(" in night_cycle_source
    assert ".write_text(" not in night_cycle_source
    hippo_summary_source = (ROOT / "src/hippo_memorizer/summary_storage.py").read_text(encoding="utf-8")
    hippo_topic_source = (ROOT / "src/memory_system/chat_history_summarizer.py").read_text(encoding="utf-8")
    social_source = (ROOT / "src/modules/social_value/social_storage.py").read_text(encoding="utf-8")
    assert "_save_to_db" in hippo_summary_source
    assert "atomic_json_dump" not in hippo_summary_source
    assert "_save_topic_cache_payload" in hippo_topic_source
    assert "json.dump(data" not in hippo_topic_source
    assert "_persist_sync" in social_source
    assert "json.dump(data" not in social_source
    return {
        "json_file_count": len(json_files),
        "groups": grouped,
        "replyer_action_json_count": len(list((data_dir / "temp").glob("replyer_action_*.json")))
        if (data_dir / "temp").exists()
        else 0,
        "night_cycle_state_backend": "Huoli.db:persistence_slot",
        "night_cycle_state_json_write_removed": True,
        "hippo_summary_backend": "Huoli.db:persistence_slot",
        "hippo_topic_cache_backend": "Huoli.db:persistence_slot",
        "social_value_backend": "Huoli.db:persistence_slot",
        "legacy_json_compat_removed": True,
    }


def _tool_names_from_definitions(definitions: list[dict[str, Any]]) -> list[str]:
    return sorted(str(item.get("name", "")) for item in definitions if item.get("name"))


def audit_registered_tools() -> dict[str, Any]:
    try:
        from src.plugin_system.apis.tool_api import get_llm_available_tool_definitions
        from src.plugin_system.core.component_registry import component_registry
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    plugin_tools = get_llm_available_tool_definitions()
    plugin_names = [str(name) for name, _definition in plugin_tools]

    memory_names: list[str] = []
    memory_error = ""
    try:
        from src.memory_system.retrieval_tools.tool_loader import init_all_tools
        from src.memory_system.retrieval_tools.tool_registry import get_tool_registry

        init_all_tools()
        memory_names = sorted(
            name
            for name in get_tool_registry().get_all_tools()
            if name != "return_information"
        )
    except Exception as exc:
        memory_error = str(exc)

    effective_names: list[str] = []
    executor_error = ""
    try:
        from src.plugin_system.core.tool_use import ToolExecutor

        executor = ToolExecutor(chat_id="audit")
        effective_names = _tool_names_from_definitions(executor._get_tool_definitions())
    except Exception as exc:
        executor_error = str(exc)

    all_names = [*plugin_names, *memory_names]
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    stats = component_registry.get_registry_stats()
    return {
        "available": True,
        "plugin_tool_count": len(plugin_names),
        "plugin_tool_names": sorted(plugin_names),
        "memory_tool_count": len(memory_names),
        "memory_tool_names": memory_names,
        "memory_tool_error": memory_error,
        "tool_executor_effective_count": len(effective_names),
        "tool_executor_effective_names": effective_names,
        "tool_executor_error": executor_error,
        "duplicates": duplicates,
        "component_registry": {
            "tool_components": stats.get("tool_components", 0),
            "total_components": stats.get("total_components", 0),
            "total_plugins": stats.get("total_plugins", 0),
        },
    }


def audit_reasoning_effort_policy() -> dict[str, Any]:
    from src.config.api_ada_configs import ModelInfo, TaskConfig
    from src.llm_models.utils_model import LLMRequest

    request = LLMRequest(TaskConfig(model_list=[TARGET_MODEL]), request_type="audit")
    default_model = ModelInfo(
        name=TARGET_MODEL,
        model_identifier="gemini-2.5-flash",
        api_provider="Google",
        client_type="gemini",
        suppress_reasoning=True,
    )
    default_params = request._effective_extra_params(default_model)
    assert default_params["enable_thinking"] is False
    assert default_params["include_thoughts"] is False
    assert default_params["thinking_budget"] == 0
    assert "reasoning_effort" not in default_params

    explicit_model = ModelInfo(
        name=TARGET_MODEL,
        model_identifier="deepseek-ai/DeepSeek-V4-Flash",
        api_provider="SiliconFlow",
        client_type="openai",
        extra_params={"reasoning_effort": "minimal"},
    )
    explicit_params = request._effective_extra_params(explicit_model)
    assert explicit_params["reasoning_effort"] == "low"
    utils_source = (ROOT / "src/llm_models/utils_model.py").read_text(encoding="utf-8")
    assert '"minimal": "low"' in utils_source
    return {
        "default_suppress_reasoning_uses_prompt_guard": True,
        "minimal_effort_mapped_to": explicit_params["reasoning_effort"],
        "minimal_effort_not_sent_raw": True,
    }


def main() -> None:
    result = {
        "model_config": check_runtime_model_config(),
        "data_json_sources": audit_data_json_sources(),
        "registered_tools": audit_registered_tools(),
        "reasoning_effort_policy": audit_reasoning_effort_policy(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
