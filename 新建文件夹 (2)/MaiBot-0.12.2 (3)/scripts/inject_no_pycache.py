"""
批量注入 __pycache__ 禁止代码到所有 __init__.py 文件
运行方式: python scripts/inject_no_pycache.py
"""
import os
import sys

INJECT_LINES = [
    "import sys",
    "import os",
    "sys.dont_write_bytecode = True",
    'os.environ["PYTHONDONTWRITEBYTECODE"] = "1"',
]

MARKER = "sys.dont_write_bytecode = True"

def inject_file(filepath: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if MARKER in content:
        return False
    lines = content.split("\n")
    insert_idx = 0
    has_sys_import = False
    has_os_import = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if i == insert_idx:
                quote = stripped[:3]
                if content.count(quote) >= 2:
                    end_pos = content.find(quote, content.find(quote) + 3)
                    if end_pos >= 0:
                        insert_idx = content[:end_pos + 3].count("\n") + 1
            continue
        if stripped == "import sys":
            has_sys_import = True
        if stripped == "import os":
            has_os_import = True
    inject_block = []
    if not has_sys_import:
        inject_block.append("import sys")
    if not has_os_import:
        inject_block.append("import os")
    inject_block.append("sys.dont_write_bytecode = True")
    inject_block.append('os.environ["PYTHONDONTWRITEBYTECODE"] = "1"')
    if insert_idx > 0 and insert_idx < len(lines):
        if lines[insert_idx].strip():
            inject_block.append("")
    lines.insert(insert_idx, "\n".join(inject_block))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True

def main():
    src_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    injected = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        for fn in filenames:
            if fn == "__init__.py":
                fp = os.path.join(dirpath, fn)
                if inject_file(fp):
                    rel = os.path.relpath(fp, os.path.dirname(src_root))
                    print(f"  [注入] {rel}")
                    injected += 1
                else:
                    skipped += 1
    print(f"\n完成: {injected} 个文件已注入, {skipped} 个已跳过")

if __name__ == "__main__":
    main()
