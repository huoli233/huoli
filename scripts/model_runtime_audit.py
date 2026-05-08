import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_MODEL = "siliconflow-deepseek-v4-flash"
NON_TOOL_TASKS = (
    "utils",
    "model_monitor",
    "replyer",
    "planner",
    "focus_chat",
    "lightweight",
    "lpmm_entity_extract",
    "lpmm_rdf_build",
)


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


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
        data = _load_toml(path)
        tasks = data.get("model_task_config", {})
        route_map: dict[str, list[str]] = {}
        for task_name in NON_TOOL_TASKS:
            if task_name not in tasks:
                continue
            model_list = list(tasks[task_name].get("model_list", []))
            route_map[task_name] = model_list
            assert model_list == [TARGET_MODEL], f"{rel}:{task_name} 仍未统一到 {TARGET_MODEL}: {model_list}"
        tool_models = list(tasks.get("tool_use", {}).get("model_list", []))
        if tool_models:
            assert tool_models != [TARGET_MODEL], f"{rel}:tool_use 不能只配置不支持工具调用的 {TARGET_MODEL}"
            route_map["tool_use"] = tool_models
        checked[rel] = route_map
    return checked


def audit_data_json_sources() -> dict[str, Any]:
    data_dir = ROOT / "data"
    json_files = sorted(data_dir.rglob("*.json")) if data_dir.exists() else []
    grouped: dict[str, int] = {}
    for path in json_files:
        try:
            key = str(path.parent.relative_to(data_dir))
        except ValueError:
            key = str(path.parent)
        grouped[key] = grouped.get(key, 0) + 1
    return {
        "json_file_count": len(json_files),
        "groups": grouped,
        "replyer_action_json_count": len(list((data_dir / "temp").glob("replyer_action_*.json")))
        if (data_dir / "temp").exists()
        else 0,
    }


def main() -> None:
    result = {
        "model_config": check_runtime_model_config(),
        "data_json_sources": audit_data_json_sources(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
