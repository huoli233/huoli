"""
清理所有 __init__.py 文件，仅保留 __pycache__ 禁止代码
运行方式: python scripts/strip_init_files.py
"""
import os

CLEAN_CONTENT = """import sys
import os
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
"""

def main():
    src_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    cleaned = 0
    already_clean = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        for fn in filenames:
            if fn == "__init__.py":
                fp = os.path.join(dirpath, fn)
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip() == CLEAN_CONTENT.strip():
                    already_clean += 1
                    continue
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(CLEAN_CONTENT)
                rel = os.path.relpath(fp, os.path.dirname(src_root))
                print(f"  [清理] {rel}")
                cleaned += 1
    print(f"\n完成: {cleaned} 个文件已清理, {already_clean} 个已是干净状态")

if __name__ == "__main__":
    main()
