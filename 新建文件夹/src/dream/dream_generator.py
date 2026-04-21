import time
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger
from src.dream.config_loader import (
    get_goals,
    get_prompt_template,
    get_output_instructions,
    get_instruction_template,
)

logger = get_logger("梦境生成")


async def generate_dream_summary(
    chat_id: str,
    records: List[Dict],
    llm_bridge=None,
) -> Dict[str, Any]:
    if not records:
        return {
            "success": False,
            "reason": "没有记录需要生成摘要",
        }
    if not llm_bridge:
        return {
            "success": False,
            "reason": "LLM 桥接未初始化",
        }
    records_text = _format_records_for_summary(records)
    goals = get_goals()
    goals_text = "\n".join([f"- {g}" for g in goals])
    output_inst = get_output_instructions("summary")
    output_text = "\n".join(
        [f"{i + 1}. {inst}" for i, inst in enumerate(output_inst)]
    )
    analyze_inst = get_instruction_template("analyze_records")
    template = get_prompt_template("summary_generate")
    output_label = get_instruction_template("output_label") or "请输出"
    if template:
        prompt = template.format(
            goals_block=f"工作目标:\n{goals_text}",
            records_block=records_text,
            output_instruction=(
                f"{output_label}:\n{output_text}" if output_text else ""
            ),
        )
    else:
        prompt = _build_summary_prompt_fallback(
            goals_text, records_text, output_text
        )
    try:
        response = await llm_bridge.chat(
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "success": True,
            "chat_id": chat_id,
            "summary": response.get("content", ""),
            "record_count": len(records),
            "generated_at": time.time(),
        }
    except Exception as e:
        logger.error(f"生成摘要失败: {e}")
        return {
            "success": False,
            "reason": str(e),
        }


def _build_summary_prompt_fallback(
    goals_text: str, records_text: str, output_text: str
) -> str:
    parts = []
    if goals_text:
        parts.append(f"工作目标:\n{goals_text}")
    parts.append(f"聊天记录:\n{records_text}")
    if output_text:
        parts.append(f"请输出:\n{output_text}")
    return "\n\n".join(parts)


def _format_records_for_summary(records: List[Dict]) -> str:
    lines = []
    for i, record in enumerate(records[:20], 1):
        theme = record.get("theme", "无主题")
        summary = record.get("summary", "无摘要")
        clues = record.get("clues", record.get("keywords", []))
        keywords_str = (
            ", ".join(clues) if isinstance(clues, list) else str(clues)
        )
        lines.append(f"[{i}] 主题: {theme}")
        lines.append(f"    摘要: {summary[:200]}...")
        if keywords_str:
            lines.append(f"    线索标签: {keywords_str}")
        lines.append("")
    return "\n".join(lines)


async def generate_maintenance_report(
    chat_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    report = {
        "chat_id": chat_id,
        "generated_at": time.time(),
        "iterations": context.get("iterations", 0),
        "tools_used": context.get("tools_used", []),
        "records_modified": context.get("records_modified", 0),
        "duration_seconds": context.get("duration_seconds", 0),
        "finish_reason": context.get("finish_reason", ""),
    }
    return report


async def analyze_record_quality(
    records: List[Dict],
    llm_bridge=None,
) -> Dict[str, Any]:
    if not records:
        return {
            "success": True,
            "quality_score": 1.0,
            "issues": [],
        }
    issues = []
    for record in records:
        summary = record.get("summary", "")
        theme = record.get("theme", "")
        clues = record.get("clues", record.get("keywords", []))
        if not summary or len(summary) < 10:
            issues.append(
                {
                    "memory_id": record.get("id"),
                    "issue": "摘要过短或为空",
                    "severity": "medium",
                }
            )
        if not theme or len(theme) < 2:
            issues.append(
                {
                    "memory_id": record.get("id"),
                    "issue": "主题缺失或过短",
                    "severity": "high",
                }
            )
        if not clues or len(clues) == 0:
            issues.append(
                {
                    "memory_id": record.get("id"),
                    "issue": "缺少线索标签",
                    "severity": "low",
                }
            )
    quality_score = max(0, 1.0 - len(issues) / (len(records) * 3))
    return {
        "success": True,
        "quality_score": round(quality_score, 2),
        "total_records": len(records),
        "issues_count": len(issues),
        "issues": issues[:10],
    }
