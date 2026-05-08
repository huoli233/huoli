from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from src.common.logger import get_logger

logger = get_logger("记忆健康")


@dataclass
class DatabaseHealthStatus:
    checked_at: float = 0.0
    healthy: bool = False
    degraded: bool = True
    quick_check: str = "not_checked"
    schema_version: int = 0
    table_count: int = 0
    db_file: str = ""
    error: str = ""
    degraded_reason: str = "not_checked"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_latest_status = DatabaseHealthStatus()


def get_memory_health_status() -> DatabaseHealthStatus:
    return _latest_status


def assess_database_health(*, raise_on_error: bool = False) -> DatabaseHealthStatus:
    """执行非破坏性 SQLite 健康检查。

    只读取 quick_check、user_version 和表数量；不迁移、不 VACUUM、不写入。
    """
    global _latest_status
    from src.common.database.database import db

    checked_at = time.time()
    was_closed = db.is_closed()
    db_file = str(getattr(db, "database", "") or "")
    try:
        db.connect(reuse_if_open=True)
        quick_row = db.execute_sql("PRAGMA quick_check").fetchone()
        quick_check = str(quick_row[0] if quick_row else "unknown")
        version_row = db.execute_sql("PRAGMA user_version").fetchone()
        schema_version = int(version_row[0] if version_row else 0)
        table_row = db.execute_sql(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()
        table_count = int(table_row[0] if table_row else 0)
        healthy = quick_check.lower() == "ok"
        _latest_status = DatabaseHealthStatus(
            checked_at=checked_at,
            healthy=healthy,
            degraded=not healthy,
            quick_check=quick_check,
            schema_version=schema_version,
            table_count=table_count,
            db_file=db_file,
            error="",
            degraded_reason="" if healthy else f"quick_check={quick_check}",
        )
        if healthy:
            logger.info(
                f"记忆库健康检查通过: tables={table_count}, user_version={schema_version}"
            )
        else:
            logger.error(f"记忆库 quick_check 异常: {quick_check}")
            if raise_on_error:
                raise RuntimeError(f"记忆库 quick_check 异常: {quick_check}")
        return _latest_status
    except Exception as exc:
        _latest_status = DatabaseHealthStatus(
            checked_at=checked_at,
            healthy=False,
            degraded=True,
            quick_check="error",
            schema_version=0,
            table_count=0,
            db_file=db_file,
            error=str(exc),
            degraded_reason="database_health_check_failed",
        )
        logger.error(f"记忆库健康检查失败: {exc}")
        if raise_on_error:
            raise
        return _latest_status
    finally:
        if was_closed and not db.is_closed():
            db.close()