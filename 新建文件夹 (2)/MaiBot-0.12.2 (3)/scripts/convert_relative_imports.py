"""
将所有相对导入转为绝对导入（插件系统除外）
运行方式: python scripts/convert_relative_imports.py
"""
import os
import re

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

SKIP_DIRS = {"plugin_system", "plugins"}

IMPORT_RE = re.compile(r'^(\s*)(from\s+)(\.+)(\S*?)(\s+import\s+.*)$')


def get_package_path(filepath: str) -> list:
    rel = os.path.relpath(filepath, PROJECT_ROOT)
    parts = rel.replace("\\", "/").split("/")
    return parts[:-1]


def convert_line(line: str, file_package: list) -> str:
    m = IMPORT_RE.match(line)
    if not m:
        return line
    indent = m.group(1)
    from_kw = m.group(2)
    dots = m.group(3)
    module_part = m.group(4)
    import_part = m.group(5)
    dot_count = len(dots)
    if dot_count > len(file_package):
        return line
    base_parts = file_package[:len(file_package) - (dot_count - 1)]
    if module_part:
        abs_parts = base_parts + module_part.split(".")
    else:
        abs_parts = base_parts
    abs_module = ".".join(abs_parts)
    return f"{indent}from {abs_module}{import_part}\n"


def process_file(filepath: str) -> int:
    file_package = get_package_path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = 0
    new_lines = []
    for line in lines:
        new_line = convert_line(line, file_package)
        if new_line != line:
            changed += 1
        new_lines.append(new_line)
    if changed > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed


def should_skip(dirpath: str) -> bool:
    rel = os.path.relpath(dirpath, SRC_ROOT).replace("\\", "/")
    parts = rel.split("/")
    return any(p in SKIP_DIRS for p in parts)


def main():
    total_files = 0
    total_changes = 0
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        if should_skip(dirpath):
            continue
        for fn in filenames:
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            fp = os.path.join(dirpath, fn)
            count = process_file(fp)
            if count > 0:
                rel = os.path.relpath(fp, PROJECT_ROOT)
                print(f"  [{count:2d}处] {rel}")
                total_files += 1
                total_changes += count
    print(f"\n完成: {total_files} 个文件, {total_changes} 处相对导入已转为绝对导入")


if __name__ == "__main__":
    main()
