import time
from typing import Any, Dict, List


def build_config_scope_snapshot() -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        hub = get_core_config()
        profile = hub.profile_mapping_block()
        module_views = profile.get("module_views", {})
        if not isinstance(module_views, dict):
            module_views = {}
    except Exception:
        return {
            "updated_at": time.time(),
            "summary": {
                "total": 0,
                "system": 0,
                "mixed": 0,
                "user": 0,
                "editable": 0,
            },
            "modules": [],
        }

    modules: List[Dict[str, Any]] = []
    counts = {"system": 0, "mixed": 0, "user": 0}

    for module_name in sorted(module_views):
        raw_view = module_views.get(module_name, {})
        if not isinstance(raw_view, dict):
            raw_view = {}
        resolved = hub.resolve_module_view(module_name)
        scope = str(resolved.edit_scope or "system")
        if scope not in counts:
            scope = "system"
        counts[scope] += 1
        semantic_domains = raw_view.get("semantic_domains", [])
        if not isinstance(semantic_domains, list):
            semantic_domains = []
        user_keys = list(resolved.user_editable_keys)
        system_keys = list(resolved.system_only_keys)
        modules.append(
            {
                "module": module_name,
                "scope": scope,
                "semantic_domains": [str(item) for item in semantic_domains],
                "user_editable_keys": user_keys,
                "system_only_keys": system_keys,
                "editable_key_count": len(user_keys),
                "system_key_count": len(system_keys),
            }
        )

    return {
        "updated_at": time.time(),
        "summary": {
            "total": len(modules),
            "system": counts["system"],
            "mixed": counts["mixed"],
            "user": counts["user"],
            "editable": counts["mixed"] + counts["user"],
        },
        "modules": modules,
    }
