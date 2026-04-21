import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_json_dump(data: Any, file_path: Path | str, **kwargs) -> None:
    """原子化 JSON 写入：先写临时文件再 rename，防止中途崩溃导致数据损坏"""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        suffix=".tmp",
        prefix=".atm_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
