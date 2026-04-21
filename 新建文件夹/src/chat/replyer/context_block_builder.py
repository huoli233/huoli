import re
from typing import List


def sanitize_extra_info(extra_info: str) -> str:
    text = str(extra_info or "").strip()
    if not text:
        return ""

    allowed_headers = (
        "[前情执行块]",
        "[决策上下文包]",
        "[避免复读]",
        "[回复形式]",
        "[主人格摘要]",
        "[身份边界]",
        "[接管意图]",
        "[内心独白执行约束]",
        "[关系阶段]",
        "[关系信号]",
        "[心理信号]",
        "[回复边界]",
        "[表达节奏]",
        "[情绪保护]",
        "[分层回复模式]",
        "[表达长度]",
        "[表达细节]",
        "[避免模式]",
        "[自我风格学习]",
        "[时段调节]",
        "[★冷拒模式★]",
        "[★烦躁模式★]",
        "[灵魂指令]",
        "[防御模式]",
        "[当前情感状态]",
        "[当前心理状态]",
        "[内心思考]",
        "[当前情绪]",
        "[人格约束]",
        "[当前人格片段]",
        "[连续对话参考]",
        "[回复长度策略]",
        "[精力状态]",
        "[主动行为意图]",
        "[内容规划]",
        "[当前感受]",
        "[最近观察]",
        "[投递指令]",
    )
    kept: List[str] = []
    current: List[str] = []
    current_allowed = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current and current_allowed:
                kept.append("\n".join(current).strip())
            current = []
            current_allowed = False
            continue

        is_header = line.startswith("[") and line.endswith("]")
        if is_header:
            if current and current_allowed:
                kept.append("\n".join(current).strip())
            current = [line]
            current_allowed = line.startswith(allowed_headers)
            continue

        if not current:
            if any(
                marker in line
                for marker in (
                    "关系=",
                    "续接提示:",
                    "重复短句判定:",
                    "当前对象:",
                    "最新用户话头:",
                    "你上一句:",
                )
            ):
                kept.append(line)
            continue

        if current_allowed:
            current.append(line)

    if current and current_allowed:
        kept.append("\n".join(current).strip())

    deduped: List[str] = []
    seen = set()
    for block in kept:
        normalized = re.sub(r"\s+", " ", block).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(block)
    return "\n\n".join(deduped[:6])


def build_reply_context_block(
    recent_context: str,
    relevant_context: str,
    extra_info: str,
    recent_reply_guard: str = "",
) -> str:
    parts: List[str] = []

    recent_text = str(recent_context or "").strip()
    if recent_text:
        parts.append(f"[最近对话上下文]\n{recent_text}")

    relevant_text = str(relevant_context or "").strip()
    if relevant_text:
        parts.append(f"[相关前情]\n{relevant_text}")

    sanitized_extra = sanitize_extra_info(extra_info)
    if sanitized_extra:
        parts.append(sanitized_extra)

    guard = str(recent_reply_guard or "").strip()
    if guard:
        parts.append(guard)

    if not parts:
        return ""
    merged = "\n\n".join(part for part in parts if part)
    return f"以下是你在回复时需要参考的上下文，请先消化再回复\n{merged}\n以上是回复时必须参考的上下文"
