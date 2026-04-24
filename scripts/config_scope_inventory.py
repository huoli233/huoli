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
    view_rows = _module_view_inventory(core_cfg)
    runtime_files = _runtime_config_files()
    legacy_hotspots = _legacy_hotspots()

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
    lines.append("## 2. 已接入四层映射的 Module View 与编辑分级")
    lines.append("")
    lines.append(
        _render_table(
            ["Module View", "编辑分级", "绑定语义域", "用户可改键", "系统锁定键"],
            [
                [
                    str(row["module"]),
                    str(row["scope"]),
                    str(row["domains"]),
                    str(row["user_keys"]),
                    str(row["system_keys"]),
                ]
                for row in view_rows
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
    lines.append("## 3. 已存在的运行时配置桥接文件")
    lines.append("")
    for path in runtime_files:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 4. 仍有旧配置体系痕迹、可继续收口到四层映射的热点")
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
    lines.append("## 5. 当前建议")
    lines.append("")
    lines.append("- 已接入四层映射的模块，继续细化 `edit_scope` 即可，不需要再回到旧配置引擎。")
    lines.append("- 仍出现旧配置引擎调用的热点目录，优先作为下一轮四层映射改造候选。")
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
