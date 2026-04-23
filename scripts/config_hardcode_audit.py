import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIRS = ("src", "scripts")
SKIP_PARTS = {
    ".git",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}
SKIP_FILENAMES = {
    "config_hardcode_audit.py",
    "config_profile_regression.py",
}

CATEGORY_KEYWORDS = {
    "threshold": ("threshold", "bar", "floor", "ceiling", "limit", "cap", "score", "gate"),
    "cooldown": ("cooldown", "interval", "timeout", "ttl", "delay", "frequency", "period", "gap"),
    "probability": ("probability", "chance", "ratio", "rate", "weight", "boost", "penalty"),
    "model_task": ("model", "task", "llm", "vlm", "small", "large", "thinking"),
    "prompt_param": ("prompt", "temperature", "top_p", "max_tokens", "system", "instruction"),
}

CALL_KEYWORDS = {
    "model",
    "model_name",
    "model_task",
    "task",
    "temperature",
    "top_p",
    "max_tokens",
    "timeout",
    "prompt",
    "system_prompt",
}


@dataclass
class Finding:
    module: str
    category: str
    path: str
    line: int
    name: str
    value: str
    context: str


def iter_files(scan_roots: Iterable[Path]) -> Iterable[Path]:
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.name in SKIP_FILENAMES:
                continue
            yield path


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if parts[0] == "src" and len(parts) >= 3:
        return "/".join(parts[:3])
    if parts[0] == "src" and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def categorize(name: str, value: Any, context: str) -> str | None:
    haystack = f"{name} {context}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            if (
                category == "probability"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not 0 <= float(value) <= 1
            ):
                continue
            return category
    return None


def value_text(value: Any) -> str:
    raw = repr(value)
    return raw if len(raw) <= 120 else f"{raw[:117]}..."


class HardcodeVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: list[Finding] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [self._target_name(target) for target in node.targets]
        self._record_from_value(node.value, ",".join(name for name in names if name), node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_from_value(node.value, self._target_name(node.target), node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg in CALL_KEYWORDS:
                self._record_from_value(keyword.value, keyword.arg or "", getattr(keyword.value, "lineno", node.lineno))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        context = self._line(node.lineno)
        for comparator in node.comparators:
            value = self._literal(comparator)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                category = categorize("", value, context)
                if category:
                    self._add(category, node.lineno, "compare", value, context)
        self.generic_visit(node)

    def _record_from_value(self, node: ast.AST, name: str, line: int) -> None:
        value = self._literal(node)
        if value is None:
            return
        context = self._line(line)
        if isinstance(value, list):
            for item in value:
                category = categorize(name, item, context)
                if category:
                    self._add(category, line, name, item, context)
            return
        category = categorize(name, value, context)
        if category:
            self._add(category, line, name, value, context)

    def _add(self, category: str, line: int, name: str, value: Any, context: str) -> None:
        self.findings.append(
            Finding(
                module=module_name(self.path),
                category=category,
                path=str(self.path.relative_to(ROOT)),
                line=line,
                name=name or "<literal>",
                value=value_text(value),
                context=context.strip(),
            )
        )

    def _line(self, line: int) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1]
        return ""

    @staticmethod
    def _target_name(target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        if isinstance(target, ast.Tuple):
            return ",".join(HardcodeVisitor._target_name(item) for item in target.elts)
        return ""

    @staticmethod
    def _literal(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = []
            for item in node.elts:
                literal = HardcodeVisitor._literal(item)
                if literal is not None:
                    values.append(literal)
            return values
        return None


def audit(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_files(paths):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        try:
            tree = ast.parse("\n".join(lines), filename=str(path))
        except SyntaxError:
            continue
        visitor = HardcodeVisitor(path, lines)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


def grouped(findings: list[Finding]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for finding in findings:
        result.setdefault(finding.module, {}).setdefault(finding.category, []).append(asdict(finding))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit hardcoded config-like values by module.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    parser.add_argument("paths", nargs="*", help="Optional project-relative paths to scan.")
    args = parser.parse_args()

    scan_roots = [ROOT / path for path in args.paths] if args.paths else [ROOT / path for path in DEFAULT_SCAN_DIRS]
    findings = sorted(audit(scan_roots), key=lambda item: (item.module, item.category, item.path, item.line))
    if args.json:
        print(json.dumps(grouped(findings), ensure_ascii=False, indent=2))
        return

    print("# Config Hardcode Audit")
    print()
    print(f"Total findings: {len(findings)}")
    for module, categories in grouped(findings).items():
        print()
        print(f"## {module}")
        for category, items in categories.items():
            print(f"- {category}: {len(items)}")
            for item in items[:12]:
                print(f"  - {item['path']}:{item['line']} {item['name']} = {item['value']}")
            if len(items) > 12:
                print(f"  - ... {len(items) - 12} more")


if __name__ == "__main__":
    main()
