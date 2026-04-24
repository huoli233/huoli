from __future__ import annotations

import argparse
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
CORE_CONFIG = ROOT / "config" / "core_config.toml"
SKIP_PARTS = {".git", ".venv", "__pycache__", "dist", "node_modules"}
LEGACY_PATTERNS = (
    re.compile(r"\bget_default_config_engine\b"),
    re.compile(r"\bConfigEngine\b"),
    re.compile(r"\.get_(?:social|relationship|memory|model|prompt|developer|user)_[a-z_]*config\b"),
)

TOP_LEVEL = {
    "identity_persona": "身份与人格",
    "dialogue_orchestration": "对话编排",
    "perception_context": "感知与上下文",
    "memory_knowledge": "记忆与知识",
    "emotion_psychology": "情绪与心理",
    "trauma_relation": "创伤与关系",
    "safety_guard": "安全与防护",
    "learning_adaptation": "学习与自适应",
    "runtime_resources": "运行资源与开关",
    "interface_observability": "界面与可观测",
}

SECOND_LEVEL = {
    "identity_profiles": ("identity_persona", "身份档案"),
    "persona_dynamics": ("identity_persona", "人格动态"),
    "disposition_sliders": ("identity_persona", "性格因子"),
    "routing_generation": ("dialogue_orchestration", "路由与生成"),
    "phase_scheduling": ("dialogue_orchestration", "阶段与调度"),
    "proactive_reply": ("dialogue_orchestration", "主动与回复"),
    "skill_dispatch": ("dialogue_orchestration", "技能调度"),
    "context_and_vision": ("perception_context", "上下文与视觉"),
    "user_perception": ("perception_context", "用户感知"),
    "group_perception": ("perception_context", "群体感知"),
    "semantic_understanding": ("perception_context", "语义理解"),
    "memory_capacity_decay": ("memory_knowledge", "容量与衰减"),
    "memory_retrieval": ("memory_knowledge", "检索与召回"),
    "recall_rewrite": ("memory_knowledge", "回想与纠错"),
    "emotion_stream": ("emotion_psychology", "情绪流"),
    "psychological_state": ("emotion_psychology", "心理状态"),
    "trauma_runtime": ("trauma_relation", "创伤运行"),
    "social_relation": ("trauma_relation", "社交关系"),
    "harassment_injection_guard": ("safety_guard", "风险检测"),
    "protection_governance": ("safety_guard", "保护治理"),
    "adaptive_learning": ("learning_adaptation", "学习与自适应"),
    "runtime_tuning": ("learning_adaptation", "运行调优"),
    "energy_and_trigger": ("runtime_resources", "能量与触发"),
    "frequency_and_switches": ("runtime_resources", "频控与开关"),
    "webui_runtime": ("interface_observability", "WebUI运行"),
    "monitoring_thresholds": ("interface_observability", "阈值与状态展示"),
}

CATEGORY_MODULES = {
    "identity_profiles": ["identity_bot", "identity_anchor", "identity_user_persistence"],
    "persona_dynamics": ["personality", "affection_dynamics", "persona_controller", "persona_generator", "persona_switcher"],
    "disposition_sliders": ["personality_factors"],
    "routing_generation": ["model_routing", "message_processor", "brain_chat_runtime", "brain_planner", "brain_pfc_action", "brain_pfc_reply", "brain_waiter", "brain_pfc_goal"],
    "phase_scheduling": ["phase_timing", "schedule"],
    "proactive_reply": ["proactive_decider", "heartflow_decision", "heartflow_runtime", "inner_voice"],
    "skill_dispatch": ["skill"],
    "context_and_vision": ["context", "vision"],
    "user_perception": ["perception_self_sense", "perception_user_relation", "perception_user_state", "perception_behavior", "perception_interest"],
    "group_perception": ["perception_group_sense", "perception_group_atmosphere"],
    "semantic_understanding": ["perception_message_preprocessor", "perception_signal_detector", "perception_understand", "perception_buffer"],
    "memory_capacity_decay": ["memory", "memory_capacity", "memory_decay", "memory_dedup"],
    "memory_retrieval": ["memory_retrieval"],
    "recall_rewrite": ["recall_post_send", "recall_typo", "recall_self_behavior", "recall_self_awareness", "recall_dimension", "recall_correction", "recall_shuffle"],
    "emotion_stream": ["chat_emotion", "heartfc_thresholds"],
    "psychological_state": ["psychological_core"],
    "trauma_runtime": ["trauma_complex", "trauma_fragment", "trauma_layers", "trauma_system", "trauma_timeline", "trauma_triggers", "trauma_worldview"],
    "social_relation": ["social_calculator", "social_settlement", "social_phase_tracker", "social_affect_fuser", "social_value_core"],
    "harassment_injection_guard": ["harassment_detection", "injection_detection"],
    "protection_governance": ["global_shield", "user_protection"],
    "adaptive_learning": ["adaptive_learning"],
    "runtime_tuning": ["runtime_tuning"],
    "energy_and_trigger": ["energy_runtime", "trigger_runtime"],
    "frequency_and_switches": ["frequency_control", "module_switches"],
    "webui_runtime": ["webui_rate_limit", "webui_websocket", "webui_git_mirror"],
    "monitoring_thresholds": ["webui_state_monitor_thresholds"],
}

RELATION_OVERRIDES = {
    "identity_bot": ["identity_anchor", "personality_factors", "heartflow_runtime", "skill"],
    "identity_anchor": ["identity_bot", "personality_factors", "social_phase_tracker"],
    "personality_factors": ["personality", "heartflow_decision", "social_calculator", "psychological_core"],
    "schedule": ["phase_timing", "proactive_decider", "heartflow_runtime", "recall_post_send"],
    "phase_timing": ["schedule", "heartflow_runtime", "brain_waiter", "heartflow_decision"],
    "heartflow_runtime": ["schedule", "phase_timing", "proactive_decider", "inner_voice"],
    "context": ["vision", "memory_retrieval", "perception_understand", "message_processor"],
    "vision": ["context", "perception_understand", "message_processor", "memory"],
    "memory": ["memory_retrieval", "context", "recall_post_send", "social_phase_tracker"],
    "adaptive_learning": ["skill", "runtime_tuning", "recall_self_behavior", "personality_factors"],
    "runtime_tuning": ["schedule", "model_routing", "heartflow_runtime", "adaptive_learning"],
    "social_calculator": ["social_settlement", "social_phase_tracker", "personality_factors", "psychological_core"],
    "social_affect_fuser": ["social_calculator", "social_phase_tracker", "chat_emotion", "heartflow_decision"],
    "psychological_core": ["trauma_system", "social_calculator", "chat_emotion", "heartflow_decision"],
    "trauma_system": ["trauma_layers", "trauma_triggers", "trauma_worldview", "psychological_core"],
    "module_switches": ["heartflow_runtime", "skill", "vision", "proactive_decider"],
    "energy_runtime": ["trigger_runtime", "frequency_control", "heartflow_runtime", "chat_emotion"],
    "trigger_runtime": ["energy_runtime", "proactive_decider", "heartflow_decision", "module_switches"],
    "webui_state_monitor_thresholds": ["chat_emotion", "social_phase_tracker", "trauma_system", "energy_runtime"],
    "model_routing": ["brain_chat_runtime", "vision", "adaptive_learning", "runtime_tuning"],
}


def _iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        yield path


def _load_core_config() -> Dict:
    return tomllib.loads(CORE_CONFIG.read_text(encoding="utf-8"))


def _src_inventory() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for child in sorted([p for p in SRC_ROOT.iterdir() if p.is_dir() and p.name not in SKIP_PARTS]):
        py_count = sum(1 for _ in child.rglob("*.py"))
        subdirs = []
        for sub in sorted([p for p in child.iterdir() if p.is_dir() and p.name not in SKIP_PARTS]):
            sub_py = sum(1 for _ in sub.rglob("*.py"))
            subdirs.append(f"{sub.name}({sub_py})")
        rows.append(
            {
                "area": child.name,
                "py_count": py_count,
                "subdirs": ", ".join(subdirs) if subdirs else "-",
            }
        )
    return rows


def _module_view_inventory(core_cfg: Dict) -> List[Dict[str, object]]:
    module_views = core_cfg["profile_mapping"]["module_views"]
    rows: List[Dict[str, object]] = []
    for name in sorted(module_views):
        view = module_views[name]
        rows.append(
            {
                "module": name,
                "scope": view.get("edit_scope", "system"),
                "domains": ", ".join(view.get("semantic_domains", [])) or "-",
                "user_keys": ", ".join(view.get("user_editable_keys", [])) or "-",
                "system_keys": ", ".join(view.get("system_only_keys", [])) or "-",
            }
        )
    return rows


def _module_taxonomy_rows(core_cfg: Dict) -> List[Dict[str, object]]:
    module_views = core_cfg["profile_mapping"]["module_views"]
    module_to_l2: Dict[str, str] = {}
    for l2_key, modules in CATEGORY_MODULES.items():
        for module in modules:
            module_to_l2[module] = l2_key

    missing = sorted(set(module_views) - set(module_to_l2))
    extra = sorted(set(module_to_l2) - set(module_views))
    if missing:
        raise RuntimeError(f"taxonomy missing module_views: {missing}")
    if extra:
        raise RuntimeError(f"taxonomy has unknown module_views: {extra}")

    rows: List[Dict[str, object]] = []
    for module in sorted(module_views):
        view = module_views[module]
        l2_key = module_to_l2[module]
        l1_key, l2_label = SECOND_LEVEL[l2_key]
        related = _related_modules(module, l2_key)
        rows.append(
            {
                "module": module,
                "l1_key": l1_key,
                "l1_label": TOP_LEVEL[l1_key],
                "l2_key": l2_key,
                "l2_label": l2_label,
                "scope": view.get("edit_scope", "system"),
                "domains": ", ".join(view.get("semantic_domains", [])) or "-",
                "user_keys": ", ".join(view.get("user_editable_keys", [])) or "-",
                "system_keys": ", ".join(view.get("system_only_keys", [])) or "-",
                "related_modules": ", ".join(related) if related else "-",
                "four_layer_status": "已接入",
            }
        )
    return rows


def _related_modules(module: str, l2_key: str) -> List[str]:
    related = []
    for item in RELATION_OVERRIDES.get(module, []):
        if item != module and item not in related:
            related.append(item)
    for sibling in CATEGORY_MODULES.get(l2_key, []):
        if sibling != module and sibling not in related:
            related.append(sibling)
    return related[:6]


def _runtime_config_files() -> List[str]:
    paths = []
    for path in SRC_ROOT.rglob("runtime_config.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        paths.append(path.relative_to(ROOT).as_posix())
    return sorted(paths)


def _legacy_hotspots() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for path in _iter_py_files(SRC_ROOT):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in LEGACY_PATTERNS):
            rel = path.relative_to(ROOT).as_posix()
            top = rel.split("/")[1] if rel.startswith("src/") and len(rel.split("/")) > 1 else rel
            grouped[top].append(rel)
    return dict(sorted((key, sorted(values)) for key, values in grouped.items()))


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_markdown() -> str:
    core_cfg = _load_core_config()
    src_rows = _src_inventory()
    taxonomy_rows = _module_taxonomy_rows(core_cfg)
    runtime_files = _runtime_config_files()
    legacy_hotspots = _legacy_hotspots()

    l1_summary: Dict[str, Dict[str, int]] = {}
    l2_summary: Dict[str, Dict[str, object]] = {}
    for row in taxonomy_rows:
        bucket = l1_summary.setdefault(
            row["l1_key"],
            {
                "label": row["l1_label"],
                "count": 0,
                "system": 0,
                "mixed": 0,
                "user": 0,
            },
        )
        bucket["count"] += 1
        bucket[str(row["scope"])] += 1
        l2_bucket = l2_summary.setdefault(
            row["l2_key"],
            {
                "label": row["l2_label"],
                "parent": row["l1_key"],
                "parent_label": row["l1_label"],
                "modules": [],
            },
        )
        l2_bucket["modules"].append(str(row["module"]))

    lines: List[str] = []
    lines.append("# SRC 全量配置分级盘点")
    lines.append("")
    lines.append("> 由 `scripts/config_scope_inventory.py` 基于当前工作区生成。")
    lines.append("")
    lines.append("## 1. SRC 顶层盘点")
    lines.append("")
    lines.append(
        _render_table(
            ["区域", "Python文件数", "二级子目录(含py数)"],
            [[str(row["area"]), str(row["py_count"]), str(row["subdirs"])] for row in src_rows],
        )
    )
    lines.append("")
    lines.append("## 2. 第一层：十大类型总览")
    lines.append("")
    lines.append(
        _render_table(
            ["一级类型", "中文名", "三级模块数", "system", "mixed", "user"],
            [
                [
                    key,
                    str(item["label"]),
                    str(item["count"]),
                    str(item["system"]),
                    str(item["mixed"]),
                    str(item["user"]),
                ]
                for key, item in sorted(l1_summary.items())
            ],
        )
    )
    lines.append("")
    lines.append("## 3. 第二层：二十到三十类型展开")
    lines.append("")
    lines.append(
        _render_table(
            ["二级类型", "中文名", "所属一级类型", "三级模块数", "代表模块"],
            [
                [
                    key,
                    str(item["label"]),
                    f"{item['parent']} / {item['parent_label']}",
                    str(len(item["modules"])),
                    ", ".join(item["modules"][:4]) + (" ..." if len(item["modules"]) > 4 else ""),
                ]
                for key, item in sorted(l2_summary.items())
            ],
        )
    )
    lines.append("")
    lines.append("## 4. 第三层：五十到一百个模块类型明细")
    lines.append("")
    lines.append(
        _render_table(
            ["三级模块类型", "一级/二级路径", "编辑分级", "四层映射状态", "用户可改键", "绑定语义域"],
            [
                [
                    str(row["module"]),
                    f"{row['l1_label']} / {row['l2_label']}",
                    str(row["scope"]),
                    str(row["four_layer_status"]),
                    str(row["user_keys"]),
                    str(row["domains"]),
                ]
                for row in taxonomy_rows
            ],
        )
    )
    lines.append("")
    lines.append("### 分级解释")
    lines.append("")
    lines.append("- `system`：系统级，默认不开放给用户编辑。")
    lines.append("- `mixed`：混合级，仅 `user_editable_keys` 中列出的键允许用户改。")
    lines.append("- `user`：用户级，整组主要面向用户偏好与行为风格调节。")
    lines.append("")
    lines.append("## 5. 第四层：模块关系层")
    lines.append("")
    lines.append(
        _render_table(
            ["模块", "所属二级类型", "主要关联模块"],
            [
                [
                    str(row["module"]),
                    str(row["l2_label"]),
                    str(row["related_modules"]),
                ]
                for row in taxonomy_rows
            ],
        )
    )
    lines.append("")
    lines.append("## 6. 已存在的运行时配置桥接文件")
    lines.append("")
    for path in runtime_files:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 7. 仍有旧配置体系痕迹、可继续收口到四层映射的热点")
    lines.append("")
    if not legacy_hotspots:
        lines.append("- 无")
    else:
        for area, files in legacy_hotspots.items():
            lines.append(f"### `{area}`")
            lines.append("")
            for rel in files[:40]:
                lines.append(f"- `{rel}`")
            if len(files) > 40:
                lines.append(f"- 其余 {len(files) - 40} 个文件略")
            lines.append("")
    lines.append("## 8. 当前建议")
    lines.append("")
    lines.append("- 已接入四层映射的 `78` 个模块，优先继续细化 `edit_scope` 与 `user_editable_keys`。")
    lines.append("- 现阶段一级类型共 `10` 个，二级类型共 `26` 个，三级模块类型共 `78` 个，满足你要的四层盘点规模。")
    lines.append("- 旧配置热点里，`memory_system` 和 `modules/safety` 是下一轮最适合继续动刀的目录。")
    lines.append("- 基础设施目录（如 `common`、`llm_models`、`plugin_system`）若无真实配置热点，默认保持系统级。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full src config scope inventory.")
    parser.add_argument(
        "--output",
        help="Optional output file path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    content = _render_markdown()
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")
        return
    print(content)


if __name__ == "__main__":
    main()
