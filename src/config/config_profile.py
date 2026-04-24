from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


PROFILE_LAYER_NAMES = (
    "base_parameters",
    "semantic_domains",
    "scenario_profiles",
    "module_views",
)

_META_KEYS = {
    "base_refs",
    "semantic_domains",
    "legacy_blocks",
    "description",
    "note",
    "file_refs",
    "edit_scope",
    "user_editable_keys",
    "system_only_keys",
}


@dataclass(frozen=True)
class ConfigTrace:
    """Single layer contribution in a resolved module config view."""

    layer: str
    source: str
    keys: Tuple[str, ...]
    fallback: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "source": self.source,
            "keys": list(self.keys),
            "fallback": self.fallback,
            "note": self.note,
        }


@dataclass(frozen=True)
class ModuleConfigView:
    """Resolved configuration view for one module and optional scenario."""

    module: str
    scenario: Optional[str]
    values: Dict[str, Any]
    trace: Tuple[ConfigTrace, ...]
    fallback_used: bool
    edit_scope: str = "system"
    user_editable_keys: Tuple[str, ...] = ()
    system_only_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "scenario": self.scenario,
            "values": copy.deepcopy(self.values),
            "trace": [item.to_dict() for item in self.trace],
            "fallback_used": self.fallback_used,
            "edit_scope": self.edit_scope,
            "user_editable_keys": list(self.user_editable_keys),
            "system_only_keys": list(self.system_only_keys),
        }


class ConfigProfileResolver:
    """Resolve the four-layer profile mapping into module-facing views."""

    def __init__(
        self,
        profile_mapping: Optional[Dict[str, Any]] = None,
        legacy_blocks: Optional[Dict[str, Dict[str, Any]]] = None,
        profile_root: Optional[Path] = None,
    ) -> None:
        self.profile_mapping = copy.deepcopy(profile_mapping or {})
        self.legacy_blocks = copy.deepcopy(legacy_blocks or {})
        self.profile_root = Path(profile_root) if profile_root is not None else None
        self._external_cache: Dict[str, Dict[str, Any]] = {}

    def resolve(self, module_name: str, scenario: Optional[str] = None) -> ModuleConfigView:
        module = str(module_name).strip()
        if not module:
            raise ValueError("module_name cannot be empty")

        values: Dict[str, Any] = {}
        trace: list[ConfigTrace] = []
        profile_specific_contributed = False
        legacy_contributed = self._merge_legacy(values, trace)

        base_layer = self._layer("base_parameters")
        module_view = self._mapping_at("module_views", module)
        base_refs = self._list_from(module_view.get("base_refs") if isinstance(module_view, dict) else None)
        for ref in ("defaults", *base_refs, module):
            part = self._sub_mapping(base_layer, ref)
            if part:
                contributed = self._merge_profile_source(
                    values,
                    part,
                    trace,
                    "base_parameters",
                    f"base_parameters.{ref}",
                    payload_mode="mapping",
                )
                if contributed and ref != "defaults":
                    profile_specific_contributed = True

        domain_layer = self._layer("semantic_domains")
        domain_names = [module]
        if isinstance(module_view, dict):
            domain_names.extend(self._list_from(module_view.get("semantic_domains")))
        for domain in self._unique(domain_names):
            part = self._sub_mapping(domain_layer, domain)
            if part:
                if self._merge_profile_source(
                    values,
                    part,
                    trace,
                    "semantic_domains",
                    f"semantic_domains.{domain}",
                    payload_mode="mapping",
                ):
                    profile_specific_contributed = True

        if scenario:
            scenario_root = self._sub_mapping(self._layer("scenario_profiles"), scenario)
            scenario_defaults = self._payload_from(scenario_root)
            if scenario_defaults or self._file_refs_from(scenario_root):
                self._merge_profile_source(
                    values,
                    scenario_root,
                    trace,
                    "scenario_profiles",
                    f"scenario_profiles.{scenario}",
                    payload_mode="payload",
                )
                profile_specific_contributed = True
            scenario_module = self._sub_mapping(scenario_root, module)
            if scenario_module:
                if self._merge_profile_source(
                    values,
                    scenario_module,
                    trace,
                    "scenario_profiles",
                    f"scenario_profiles.{scenario}.{module}",
                    payload_mode="mapping",
                ):
                    profile_specific_contributed = True

        if isinstance(module_view, dict):
            if self._payload_from(module_view) or self._file_refs_from(module_view):
                self._merge_profile_source(
                    values,
                    module_view,
                    trace,
                    "module_views",
                    f"module_views.{module}",
                    payload_mode="payload",
                )
                profile_specific_contributed = True
        edit_scope, user_editable_keys, system_only_keys = self._resolve_edit_policy(
            module_view if isinstance(module_view, dict) else {}
        )

        return ModuleConfigView(
            module=module,
            scenario=scenario,
            values=copy.deepcopy(values),
            trace=tuple(trace),
            fallback_used=legacy_contributed and not profile_specific_contributed,
            edit_scope=edit_scope,
            user_editable_keys=user_editable_keys,
            system_only_keys=system_only_keys,
        )

    def _merge_legacy(self, values: Dict[str, Any], trace: list[ConfigTrace]) -> bool:
        used = False
        for block_name, block_values in self.legacy_blocks.items():
            if not isinstance(block_values, dict) or not block_values:
                continue
            self._merge_with_trace(
                values,
                block_values,
                trace,
                "legacy_fallback",
                block_name,
                fallback=True,
                note="old config block fallback",
            )
            used = True
        return used

    def _layer(self, name: str) -> Dict[str, Any]:
        layer = self.profile_mapping.get(name)
        return copy.deepcopy(layer) if isinstance(layer, dict) else {}

    def _mapping_at(self, layer_name: str, key: str) -> Dict[str, Any]:
        layer = self._layer(layer_name)
        return self._sub_mapping(layer, key)

    def _merge_profile_source(
        self,
        values: Dict[str, Any],
        raw: Dict[str, Any],
        trace: list[ConfigTrace],
        layer: str,
        source: str,
        payload_mode: str,
    ) -> bool:
        contributed = False
        for file_ref in self._file_refs_from(raw):
            patch = self._load_file_patch(file_ref)
            if not patch:
                continue
            self._merge_with_trace(
                values,
                patch,
                trace,
                layer,
                f"{source}@{file_ref}",
                note="external profile file",
            )
            contributed = True
        if payload_mode == "payload":
            local_patch = self._payload_from(raw)
        else:
            local_patch = {
                key: copy.deepcopy(value)
                for key, value in raw.items()
                if key not in _META_KEYS
            }
        if local_patch:
            self._merge_with_trace(values, local_patch, trace, layer, source)
            contributed = True
        return contributed

    @staticmethod
    def _sub_mapping(root: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = root.get(key) if isinstance(root, dict) else None
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    @staticmethod
    def _payload_from(raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        payload = raw.get("values")
        if isinstance(payload, dict):
            return copy.deepcopy(payload)
        return copy.deepcopy({k: v for k, v in raw.items() if k not in _META_KEYS and not isinstance(v, dict)})

    def _file_refs_from(self, raw: Dict[str, Any]) -> Tuple[str, ...]:
        if not isinstance(raw, dict):
            return ()
        return self._list_from(raw.get("file_refs"))

    def _load_file_patch(self, file_ref: str) -> Dict[str, Any]:
        ref = str(file_ref).strip()
        if not ref:
            return {}
        cached = self._external_cache.get(ref)
        if cached is not None:
            return copy.deepcopy(cached)
        if self.profile_root is None:
            return {}
        try:
            path = (self.profile_root / ref).resolve()
            profile_root = self.profile_root.resolve()
            path.relative_to(profile_root)
        except Exception:
            return {}
        if not path.exists() or not path.is_file():
            return {}
        try:
            import tomlkit

            with open(path, "r", encoding="utf-8") as fh:
                parsed = tomlkit.load(fh)
            data = dict(parsed)
        except Exception:
            return {}
        self._external_cache[ref] = copy.deepcopy(data)
        return copy.deepcopy(data)

    def _resolve_edit_policy(
        self, raw: Dict[str, Any]
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        scope = "system"
        user_keys: Tuple[str, ...] = ()
        system_keys: Tuple[str, ...] = ()
        for file_ref in self._file_refs_from(raw):
            patch = self._load_file_patch(file_ref)
            scope, user_keys, system_keys = self._overlay_edit_policy(
                scope, user_keys, system_keys, patch
            )
        return self._overlay_edit_policy(scope, user_keys, system_keys, raw)

    def _overlay_edit_policy(
        self,
        scope: str,
        user_keys: Tuple[str, ...],
        system_keys: Tuple[str, ...],
        raw: Dict[str, Any],
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        if not isinstance(raw, dict):
            return scope, user_keys, system_keys
        next_scope = str(raw.get("edit_scope", scope) or scope).strip() or "system"
        next_user_keys = user_keys
        if "user_editable_keys" in raw:
            next_user_keys = self._list_from(raw.get("user_editable_keys"))
        next_system_keys = system_keys
        if "system_only_keys" in raw:
            next_system_keys = self._list_from(raw.get("system_only_keys"))
        return next_scope, next_user_keys, next_system_keys

    @staticmethod
    def _merge_with_trace(
        values: Dict[str, Any],
        patch: Dict[str, Any],
        trace: list[ConfigTrace],
        layer: str,
        source: str,
        fallback: bool = False,
        note: str = "",
    ) -> None:
        if not patch:
            return
        values.update(ConfigProfileResolver._overlay_dicts(values, patch))
        trace.append(
            ConfigTrace(
                layer=layer,
                source=source,
                keys=tuple(sorted(str(key) for key in patch.keys())),
                fallback=fallback,
                note=note,
            )
        )

    @staticmethod
    def _overlay_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in patch.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigProfileResolver._overlay_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @staticmethod
    def _list_from(raw: Any) -> Tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, Iterable):
            return tuple(str(item) for item in raw if str(item).strip())
        return ()

    @staticmethod
    def _unique(items: Iterable[str]) -> Tuple[str, ...]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return tuple(result)
