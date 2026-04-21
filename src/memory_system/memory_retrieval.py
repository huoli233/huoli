import time
import json
import asyncio
import inspect
import re
from typing import List, Dict, Any, Optional, Tuple
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.apis import llm_api
from src.llm_models.utils_model import _is_tool_calling_unsupported, _resolve_model_client_type
from src.llm_models.model_client._registry import client_registry
from src.common.database.database_model import ThinkingBack
from src.memory_system.retrieval_tools.tool_registry import get_tool_registry
from src.memory_system.retrieval_tools.tool_loader import init_all_tools
from src.llm_models.payload_content.message import (
    MessageBuilder,
    RoleType,
    Message,
)
from src.llm_models.payload_content.tool_option import ToolCall
from src.chat.message_receive.chat_stream import get_chat_manager
from src.bw_learner.jargon_explainer import retrieve_concepts_with_jargon

logger = get_logger("记忆检索")

THINKING_BACK_NOT_FOUND_RETENTION_SECONDS = 36000  # 未找到答案记录保留时长
THINKING_BACK_CLEANUP_INTERVAL_SECONDS = 3000  # 清理频率
_last_not_found_cleanup_ts: float = 0.0

_INVALID_MEMORY_MARKERS = frozenset(
    [
        "未找到",
        "没有找到",
        "not found",
        "无匹配",
        "不存在",
        "查询文本为空",
        "未指定查询参数",
        "查询失败",
        "参数错误",
        "用户名称为空",
        "查询线索为空",
        "LPMM知识库未启用",
        "LPMM知识库未初始化",
        "无法获取查询向量",
        "没有可检索的记忆记录",
        "未提供有效的词语",
        "未提供要查询的词语",
        "工具执行失败",
        "missing 1 required positional argument",
        "return_information",
        "search_chat_history",
        "query_direct_memory",
        "query_words",
        "probe_chat_history",
        "token的限制",
        "配置文件",
        "max_token",
        "8096",
        "调到最大",
        "读起来有点配置文件的",
        "把这个token的限制给我",
        "最大token限制",
        "配置文件里面的",
        "token限制给我改成",
        "模型配置",
        "max_tokens",
        "token限制",
    ]
)


def _has_memory_react_tool_models() -> bool:
    """检测 memory.react 是否存在明确兼容工具调用的模型。"""
    try:
        task_cfg = getattr(model_config.model_task_config, "tool_use", None)
        model_list = list(getattr(task_cfg, "model_list", []) or [])
        if not task_cfg or not model_list:
            return False
        for model_name in model_list:
            if _is_tool_calling_unsupported(model_name):
                continue
            try:
                model_info = model_config.get_model_info(model_name)
                declared_support = getattr(model_info, "supports_tool_calling", None)
                if declared_support is False:
                    continue
                api_provider = model_config.get_provider(model_info.api_provider)
                resolved_client_type = _resolve_model_client_type(model_info, api_provider)
                client = client_registry.get_client_class_instance(
                    resolved_client_type,
                    api_provider=api_provider,
                    force_new=False,
                )
                if not client.supports_function_calling():
                    continue
                return True
            except Exception as exc:
                logger.debug(f"[记忆检索] 工具模型探测跳过 {model_name}: {exc}")
                continue
        return False
    except Exception as exc:
        logger.debug(f"[记忆检索] 工具模型可用性探测失败，按可用处理: {exc}")
        return True


def _filter_config_meta_talk(text: str) -> str:
    """过滤检索结果中关于token限制、配置文件、模型配置的元讨论
    确保只保留与用户查询（如刀盾）直接相关的记忆内容
    使用多层模式匹配和相关性评分，避免无关配置讨论污染回复
    """
    if not text or not text.strip():
        return ""
    original = str(text).strip()
    lines = original.splitlines()
    filtered_lines = []
    config_patterns = [
        re.compile(r"token.*限制|限制.*token", re.IGNORECASE),
        re.compile(r"配置文件|config|toml", re.IGNORECASE),
        re.compile(r"max_tokens|max_token|8096|8192|4096", re.IGNORECASE),
        re.compile(r"调到最大|改成|读起来有点|把这个.*给我", re.IGNORECASE),
        re.compile(r"模型.*配置|配置.*模型|qwen3|gpt-5", re.IGNORECASE),
        re.compile(r"知识库未|未启用|初始化|向量|probe|return_information", re.IGNORECASE),
    ]
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        is_config_meta = any(p.search(line_stripped) for p in config_patterns)
        if is_config_meta:
            continue
        if any(marker in line_stripped.lower() for marker in _INVALID_MEMORY_MARKERS):
            continue
        if _looks_like_low_value_short_reply(line_stripped):
            continue
        filtered_lines.append(line_stripped)
    cleaned = "\n".join(filtered_lines).strip()
    if not cleaned or len(cleaned) < 8:
        return ""
    if not _is_high_value_memory_text(cleaned):
        return ""
    return cleaned


def _score_memory_relevance(query: str, memory_text: str) -> float:
    """计算记忆文本与查询的相关性分数
    使用关键词重叠、语义近似和长度加权，避免无关配置文本得分高
    """
    if not query or not memory_text:
        return 0.0
    query_norm = _squash_text_for_match(query)
    memory_norm = _squash_text_for_match(memory_text)
    if not query_norm or not memory_norm:
        return 0.0
    query_words = set(query_norm[i : i + 2] for i in range(len(query_norm) - 1))
    memory_words = set(memory_norm[i : i + 2] for i in range(len(memory_norm) - 1))
    overlap = len(query_words & memory_words)
    if len(query_words) == 0:
        return 0.0
    overlap_ratio = overlap / len(query_words)
    length_bonus = min(len(memory_norm) / 50.0, 2.0)
    if any(bad in memory_norm for bad in ["token", "config", "配置文件", "max", "限制"]):
        length_bonus *= 0.1
    score = overlap_ratio * 0.6 + length_bonus * 0.4
    return min(max(score, 0.0), 1.0)


def _classify_memory_hit_quality(query: str, raw_text: Optional[str]) -> Tuple[str, str, float]:
    """把工具检索结果归类为 strong/weak/invalid，供 governor 执行层统一停机策略。"""
    sanitized = _sanitize_memory_payload(raw_text)
    if not sanitized:
        return "", "invalid", 0.0
    if _looks_like_self_recall_memory_query(query):
        self_recall_markers = ("自己说过", "我说过", "上一句", "上句", "分钟前", "刚刚说")
        if any(marker in sanitized for marker in self_recall_markers):
            return sanitized, "strong", 0.95
    relevance = _score_memory_relevance(query, sanitized)
    payload_len = len(_squash_text_for_match(sanitized))
    if relevance >= 0.42 and payload_len >= 16:
        return sanitized, "strong", relevance
    if relevance >= 0.22 and payload_len >= 8:
        return sanitized, "weak", relevance
    return "", "invalid", relevance


def _advanced_sanitize_memory(text: str, query: str) -> str:
    """高级记忆清理
    结合过滤元讨论、相关性评分、提取答案负载
    返回高价值干净记忆文本
    """
    if not text:
        return ""
    filtered = _filter_config_meta_talk(text)
    if not filtered:
        return ""
    score = _score_memory_relevance(query, filtered)
    if score < 0.25:
        return ""
    payload = _extract_answer_payload(filtered)
    if not _is_high_value_memory_text(payload):
        return ""
    if len(payload) > 800:
        payload = payload[:800]
    return payload.strip()


def _build_reflection_prompt_for_meta(query: str, raw_response: str) -> str:
    """构建反射prompt用于纠正元配置讨论
    如果检测到配置相关，返回强化指令要求聚焦查询
    """
    if any(bad in raw_response.lower() for bad in ["token", "config", "配置文件", "max_token"]):
        return f"""查询是关于“{query}”的记忆检索。
检测到无关的配置讨论（token限制、配置文件等），这是错误。
**严格规则**：只返回与“{query}”直接相关的过往记忆或知识，不要提及任何模型配置、token限制、文件路径或调试信息。
如果没有相关记忆，简洁返回“未找到相关记忆”。
总结只包含有用记忆内容。"""
    return ""


class MemoryReActReflector:
    """记忆ReAct反射器
    独立类封装反射逻辑，与原ReAct函数完全不同结构
    使用评分+多轮过滤+反射prompt生成，避免模型漂移到配置讨论
    """

    def __init__(self):
        self._reflection_cache: Dict[str, float] = {}
        self._meta_penalty_patterns = [
            re.compile(r"token.*限|限.*token|配置文件|max_tokens", re.IGNORECASE),
            re.compile(r"调.*最大|改成.*8096|知识库未|工具执行失败", re.IGNORECASE),
        ]

    def detect_meta_drift(self, response: str) -> bool:
        """检测是否漂移到元配置讨论"""
        if not response:
            return False
        lower = response.lower()
        return any(p.search(lower) for p in self._meta_penalty_patterns)

    def generate_reflection_prompt(self, query: str, raw_response: str, iteration: int) -> str:
        """生成反射prompt"""
        if self.detect_meta_drift(raw_response):
            return _build_reflection_prompt_for_meta(query, raw_response)
        return f"查询：{query}\n当前响应：{raw_response[:300]}\n请聚焦查询本身，提供相关记忆总结，不要讨论配置。"

    def reflect_and_rescore(self, query: str, response: str, iteration: int) -> Tuple[str, float]:
        """反射并重新评分"""
        if not response:
            return "", 0.0
        key = f"{query}:{iteration}"
        if key in self._reflection_cache:
            return response, self._reflection_cache[key]
        if self.detect_meta_drift(response):
            reflected = self.generate_reflection_prompt(query, response, iteration)
            score = 0.15
        else:
            reflected = response
            score = _score_memory_relevance(query, response)
        self._reflection_cache[key] = score
        if len(self._reflection_cache) > 50:
            def _safe_get_iteration(cache_key: str) -> int:
                """安全提取缓存key中的迭代次数，防止格式异常导致IndexError"""
                if ":" not in cache_key:
                    return 0
                parts = cache_key.split(":")
                if len(parts) < 2:
                    return 0
                try:
                    return int(parts[1])
                except (ValueError, TypeError):
                    return 0

            oldest = sorted(self._reflection_cache.keys(), key=_safe_get_iteration)[0]
            del self._reflection_cache[oldest]
        return reflected, score


_reactor = MemoryReActReflector()


def _apply_reflection_filter(query: str, raw_info: str, iteration: int = 0) -> str:
    """应用反射过滤器"""
    cleaned, score = _reactor.reflect_and_rescore(query, raw_info, iteration)
    if score < 0.35:
        return ""
    return _advanced_sanitize_memory(cleaned, query)


_INTERNAL_MEMORY_PATTERNS = (
    re.compile(r"第\s*\d+\s*次迭代.*?工具执行失败", re.IGNORECASE),
    re.compile(r"missing\s+\d+\s+required\s+positional\s+argument", re.IGNORECASE),
    re.compile(
        r"\b(?:return_information|search_chat_history|query_direct_memory|query_words|probe_chat_history)\b",
        re.IGNORECASE,
    ),
    re.compile(r"工具执行失败", re.IGNORECASE),
)

_LOW_VALUE_SHORT_REPLIES = frozenset(
    [
        "咋了",
        "咋啦",
        "干嘛",
        "然后",
        "活力",
        "嗯",
        "啊",
        "哦",
        "在吗",
        "啥",
        "什么",
    ]
)

_NO_RESULT_GUIDANCE_PATTERNS = (
    re.compile(r'^当前未找到与["“].+?["”]相关的聊天记录', re.IGNORECASE),
    re.compile(r'^未找到与["“]?.+?["”]?相关的(?:历史)?聊天记录', re.IGNORECASE),
    re.compile(r"^根据现有信息[，,].*未发现与", re.IGNORECASE),
    re.compile(r"^若需(?:进一步)?查找[，,].*请提供", re.IGNORECASE),
    re.compile(r"^请补充具体关键词或参与人名称", re.IGNORECASE),
    re.compile(r"^例如[:：].*(最近在做什么|刀盾)", re.IGNORECASE),
    re.compile(r"可能的解决方法是[:：]", re.IGNORECASE),
    re.compile(r"如果需要[，,].*提供更详细的信息", re.IGNORECASE),
    re.compile(r"重新确认参与人/时间信息", re.IGNORECASE),
    re.compile(r"按时间范围查询", re.IGNORECASE),
)

_QUERY_PATH_RE = re.compile(r"[A-Za-z]:\\|[A-Za-z]:/|[/\\].+[/\\]|\.py\b|\.md\b", re.IGNORECASE)


def _squash_text_for_match(text: str) -> str:
    """压缩文本中的空白和常见标点，便于做低价值判定。"""
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def _extract_answer_payload(text: str) -> str:
    """从“问题/答案”格式中提取答案正文。"""
    if not text:
        return ""
    matched = re.search(r"答案[:：]\s*(.+)", text, re.S)
    if matched:
        return matched.group(1).strip()
    return text.strip()


def _strip_pseudo_tool_call_markers(text: str) -> str:
    """移除模型返回中的伪工具调用标记，仅保留自然语言思考文本。"""
    if not text:
        return ""
    cleaned = re.sub(
        r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>",
        "",
        text,
        flags=re.S,
    )
    return cleaned.strip()


def _parse_pseudo_tool_calls(response_text: str) -> List[ToolCall]:
    """把 `<|tool_call_*|>` 伪格式解析成标准 ToolCall。"""
    if not response_text or "<|tool_call_begin|>" not in response_text:
        return []
    pattern = re.compile(
        r"<\|tool_call_begin\|>\s*functions\.([a-zA-Z0-9_]+):\d+\s*"
        r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>",
        re.S,
    )
    parsed_calls: List[ToolCall] = []
    for idx, match in enumerate(pattern.finditer(response_text)):
        func_name = str(match.group(1) or "").strip()
        args_text = str(match.group(2) or "{}").strip()
        if not func_name:
            continue
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError:
            args = {}
        parsed_calls.append(ToolCall(call_id=f"pseudo-call-{idx}", func_name=func_name, args=args))
    return parsed_calls


def _looks_like_low_value_short_reply(text: str) -> bool:
    """判断文本是否像“咋了/干嘛”这类低信息量短答。"""
    normalized = _squash_text_for_match(text)
    if not normalized:
        return True
    if normalized in _LOW_VALUE_SHORT_REPLIES:
        return True
    return len(normalized) <= 3


def _looks_like_no_result_guidance(text: Optional[str]) -> bool:
    """判断文本是否只是“没查到，建议换关键词”的兜底提示。"""
    payload = str(text or "").strip()
    if not payload:
        return True
    return any(pattern.search(payload) for pattern in _NO_RESULT_GUIDANCE_PATTERNS)


def _build_compound_query_context(message: str, target: str) -> str:
    """为短句或重复句构造带前情的检索上下文，避免只拿一句低信息文本裸查。"""
    history = str(message or "").strip()
    latest = str(target or "").strip()
    if not latest:
        return history

    normalized_latest = _squash_text_for_match(latest)
    if len(normalized_latest) > 8:
        return history

    history_lines = [line.strip() for line in history.splitlines() if line.strip()]
    recent_lines = history_lines[-8:]
    if not recent_lines:
        return latest

    recent_context = "\n".join(recent_lines)
    return (
        f"以下是最近对话前情，请重点理解最后这句短消息可能在延续什么上下文：\n"
        f"{recent_context}\n"
        f"当前最新短消息：{latest}"
    )


def _is_valid_memory_query_text(text: str) -> bool:
    payload = str(text or "").strip()
    if not payload:
        return False
    if len(payload) > 120:
        return False
    if _QUERY_PATH_RE.search(payload):
        return False
    if payload.count("\n") >= 3:
        return False
    return True


def _compress_memory_query_text(message: str, target: str) -> str:
    """压缩检索问题，优先保留最新用户句和必要前情，避免整段元消息污染检索。"""
    latest = str(target or "").strip()
    if latest and len(_squash_text_for_match(latest)) <= 12:
        history_lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
        tail = history_lines[-4:]
        if tail:
            compact_tail = " | ".join(line[:40] for line in tail)
            candidate = f"前情: {compact_tail} | 最新: {latest}"
            if _is_valid_memory_query_text(candidate):
                return candidate
        return latest

    candidate = str(message or "").strip()
    if len(candidate) > 120:
        candidate = candidate[:120]
    return candidate


_MEMORY_SELF_RECALL_HINTS = (
    "刚才我说了什么",
    "我刚刚说了什么",
    "上一句",
    "上句",
    "前一句",
    "你刚才说了什么",
    "你上一句",
    "刚刚那句",
)

_MEMORY_AMBIGUOUS_HINTS = (
    "上次",
    "之前",
    "刚才",
    "刚刚",
    "那个",
    "这个",
    "那句",
    "这句",
    "那条",
    "这个事",
    "继续",
    "然后呢",
    "后来",
    "记得吗",
)


def _looks_like_self_recall_memory_query(text: str) -> bool:
    payload = str(text or "")
    return any(hint in payload for hint in _MEMORY_SELF_RECALL_HINTS)


def _looks_like_ambiguous_memory_reference(
    text: str,
    split_terms: Optional[List[str]] = None,
) -> bool:
    payload = str(text or "").strip()
    if not payload:
        return False
    if any(hint in payload for hint in _MEMORY_AMBIGUOUS_HINTS):
        return True
    # 已经抽取出明确关键词时，按关键词检索路径处理，避免过早升到 embedding。
    if split_terms:
        return False
    return len(_squash_text_for_match(payload)) <= 6


def plan_memory_retrieval(message: str, target: str, extra_context: str = "") -> Dict[str, Any]:
    """统一记忆检索 governor：决定是否跳过、查询文本和工具升级顺序。"""
    latest = str(target or "").strip()
    history = str(message or "").strip()
    context = str(extra_context or "").strip()

    normalized_latest = _squash_text_for_match(latest)
    plan: Dict[str, Any] = {
        "should_skip": False,
        "reason": "",
        "query_text": "",
        "tool_order": [],
        "max_tool_steps": 0,
        "max_result_chars": 0,
        "stop_after_first_hit": True,
    }
    if not normalized_latest:
        plan["should_skip"] = True
        plan["reason"] = "latest_empty"
        return plan

    # 只在 extra_context 中识别结构化前情，避免把普通聊天历史中的标签文本误判为高质量前情包。
    compact_extra_context = context.strip()
    structured_context_markers = (
        "[前情执行块]",
        "[决策上下文包]",
        "[连续对话参考]",
        "前情执行块:",
        "前情执行块：",
        "决策上下文包:",
        "决策上下文包：",
        "连续对话参考:",
        "连续对话参考：",
        "续接提示:",
        "续接提示：",
        "重复短句判定:",
        "重复短句判定：",
    )
    if any(marker in compact_extra_context for marker in structured_context_markers):
        plan["should_skip"] = True
        plan["reason"] = "context_already_explains"
        return plan

    import re as _re

    _clean_latest = _re.sub(r"[\s\u200b-\u200f\ufeff]", "", normalized_latest)
    if len(_clean_latest) <= 2:
        plan["should_skip"] = True
        plan["reason"] = "too_short"
        return plan
    _emoji_check = _re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\u2600-\u26FF\u2700-\u27BF]",
        "",
        _clean_latest,
    )
    if len(_emoji_check) == 0:
        plan["should_skip"] = True
        plan["reason"] = "emoji_only"
        return plan

    _recent_cache = getattr(should_skip_memory_retrieval, "_recent_queries", None)
    if _recent_cache is None:
        _recent_cache = {}
        should_skip_memory_retrieval._recent_queries = _recent_cache
    _now_ts = time.time()
    _cache_key = (normalized_latest[:60], str(target or "")[:20])
    if _cache_key in _recent_cache and (_now_ts - _recent_cache[_cache_key]) < 45.0:
        plan["should_skip"] = True
        plan["reason"] = "duplicate_recent_query"
        return plan
    _recent_cache[_cache_key] = _now_ts
    if len(_recent_cache) > 50:
        _old_keys = sorted(_recent_cache.keys(), key=lambda k: _recent_cache[k])[:15]
        for _ok in _old_keys:
            del _recent_cache[_ok]

    query_text = _build_compound_query_context(history, latest)
    compressed_query = _compress_memory_query_text(history, latest)
    if _is_valid_memory_query_text(compressed_query):
        query_text = compressed_query
    plan["query_text"] = query_text

    split_terms = [
        token.strip()
        for token in _re.split(r"[\s,，。！？!?;；:：|/]+", latest)
        if len(token.strip()) >= 2
    ]
    if _looks_like_self_recall_memory_query(latest):
        tool_order = ["query_direct_memory", "search_chat_history"]
        plan["reason"] = "self_recall"
        plan["max_result_chars"] = 720
    elif _looks_like_ambiguous_memory_reference(latest, split_terms):
        tool_order = [
            "query_direct_memory",
            "search_chat_history",
            "search_memory_by_embedding",
        ]
        plan["reason"] = "ambiguous_short_reference"
        plan["max_result_chars"] = 880
    elif len(split_terms) >= 1:
        tool_order = ["search_chat_history", "query_direct_memory"]
        plan["reason"] = "keyword_history_first"
        plan["max_result_chars"] = 960
    else:
        tool_order = ["query_direct_memory", "search_chat_history", "search_memory_by_embedding"]
        plan["reason"] = "ambiguous_short_reference"
        plan["max_result_chars"] = 880
    plan["tool_order"] = list(dict.fromkeys(tool_order))
    plan["max_tool_steps"] = len(plan["tool_order"])
    return plan


def should_skip_memory_retrieval(message: str, target: str, extra_context: str = "") -> bool:
    """当当前上下文已足够解释话头时，跳过记忆检索，避免再次污染 prompt。"""
    return bool(plan_memory_retrieval(message, target, extra_context).get("should_skip", False))


def _sanitize_unknown_concepts(concepts: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for word in concepts or []:
        if not isinstance(word, str):
            continue
        item = word.strip()
        if not item:
            continue
        if len(item) > 20:
            continue
        if _QUERY_PATH_RE.search(item):
            continue
        if any(sep in item for sep in ("\n", "\r", "\t")):
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _is_high_value_memory_text(text: Optional[str]) -> bool:
    """判断一段检索结果是否值得进入记忆缓存或回复提示词。"""
    if not text or not text.strip():
        return False
    payload = _extract_answer_payload(text)
    if not payload:
        return False
    if _looks_like_no_result_guidance(payload):
        return False
    if any(marker in payload for marker in _INVALID_MEMORY_MARKERS):
        return False
    normalized = _squash_text_for_match(payload)
    if not normalized:
        return False
    if _looks_like_low_value_short_reply(payload):
        return False
    if "问题：" in text and "答案：" in text and len(normalized) < 24:
        return False
    return True


def _sanitize_memory_payload(text: Optional[str]) -> str:
    """清理工具层元信息和失败文本，只保留可给回复模型看的结果。"""
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in _INTERNAL_MEMORY_PATTERNS):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text:
        return ""
    if not _is_high_value_memory_text(cleaned_text):
        return ""
    # 使用记忆渲染器管道深度净化（剥离媒体标签/URL/工具调用等冗余元数据）
    try:
        from src.memory_system.memory_formatter import ContentSanitizer

        cleaned_text = ContentSanitizer.purify(cleaned_text)
    except ImportError:
        pass
    return cleaned_text


def _cleanup_stale_not_found_thinking_back() -> None:
    """定期清理过期的未找到答案记录

    时间格式注意：ThinkingBack.update_time 为 Unix秒（DoubleField），
    与 time.time() 返回值格式一致，可直接比较。
    """

    global _last_not_found_cleanup_ts

    now = time.time()
    if now - _last_not_found_cleanup_ts < THINKING_BACK_CLEANUP_INTERVAL_SECONDS:
        return

    threshold_time = now - THINKING_BACK_NOT_FOUND_RETENTION_SECONDS
    try:
        deleted_rows = (
            ThinkingBack.delete()
            .where((ThinkingBack.found_answer == 0) & (ThinkingBack.update_time < threshold_time))
            .execute()
        )
        if deleted_rows:
            logger.info(f"清理过期的未找到答案thinking_back记录 {deleted_rows} 条")
        _last_not_found_cleanup_ts = now
    except Exception as e:
        logger.error(f"清理未找到答案的thinking_back记录失败: {e}")


def invalidate_memory_cache(stream_id: str) -> int:
    """
    使指定聊天流的记忆缓存失效

    在回复成功后调用，防止模型重复回复相同内容

    Args:
        stream_id: 聊天流ID（等同于 ThinkingBack.chat_id，两者语义完全相同）

    Returns:
        删除的记录数
    """
    try:
        deleted_rows = ThinkingBack.delete().where(ThinkingBack.chat_id == stream_id).execute()
        logger.info(f"[记忆缓存] 清除 {stream_id[:16]} 的缓存，删除 {deleted_rows} 条记录")
        return deleted_rows
    except Exception as e:
        logger.error(f"清除记忆缓存失败: {e}")
        return 0


def init_memory_retrieval_prompt():
    """初始化记忆检索相关的 prompt 模板和工具"""
    # 首先注册所有工具
    init_all_tools()

    # ReAct代理提示词（使用函数调用，要求先思考再行动）
    Prompt(
        """你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要搜集信息来帮助你进行回复。
当前聊天记录：
{chat_history}

已收集的信息：
{collected_info}

**工具说明：**
- search_chat_history：查询过往对话概括、历史事件、之前提到过的概念，也支持时间范围过滤
- query_direct_memory：直接查询长期记忆库，特别适合找机器人自己刚说过的话、最近几分钟/几小时内的短句或上下文
- search_memory_by_embedding：通过语义相似度检索记忆。关键词不明确、用户在模糊指代、只记得大意但不记得原话时，优先尝试这个工具
- query_words：查询不熟悉词语/缩写/黑话的含义

**重要检索规则（必须遵守）：**
1. **先服从检索 governor**：如果系统已经提供足够的最近上下文/前情摘要，或当前查询明显低价值，
   不要为了检索而检索，不要重复空搜。
2. **按升级顺序检索**：
   - 查机器人自己刚说过的话、最近短句、上一句原话 → 优先 `query_direct_memory`
   - 查明确关键词、历史事件、之前聊过的主题 → 优先 `search_chat_history`
   - 说法模糊、只记得大意、关键词不稳定 → 再升级到 `search_memory_by_embedding`
3. **避免重复同类搜索**：同一轮已经查过同类线索且没有新线索时，不要再次调用同类工具。
4. **短句不是一律强搜**：短句/简称/代词只有在缺少当前轮前情时才需要检索；
   如果最近上下文已经足够解释，就直接结束并返回空信息。
5. **命中后及时停止升级**：一旦已有足够可用的结果，就立刻停止，不要继续把所有工具都打一遍。
6. **严格服从工具顺序和预算**：只允许使用 governor 给出的工具顺序；结果够用就停，结果太长就截断。

**思考流程：**
- 先判断：当前是否真的需要检索，还是最近上下文已经足够
- 如果是在找“机器人自己刚说过的话”或最近短句 → 优先 `query_direct_memory`
- 如果是明确关键词/事件/人物/主题 → 优先 `search_chat_history`
- 如果前两步没命中，且用户表达模糊、像是在凭印象回忆 → 调用 `search_memory_by_embedding`
- 同时可以用 `query_words` 查询不确定的词汇
- 任何一步命中后都不要继续升级，直接 `return_information`
""",
        name="memory_retrieval_react_prompt_head",
    )

    # 额外，如果最后一轮迭代：ReAct Agent prompt（使用function calling，要求先思考再行动）
    Prompt(
        """你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要根据搜集到的信息总结信息。
如果搜集到的信息对于参与聊天，回答问题有帮助，请加入总结，如果无关，请不要加入到总结。

当前聊天记录：
{chat_history}

已收集的信息：
{collected_info}


分析：
- 基于已收集的信息，总结出对当前聊天有帮助的相关信息
- **如果收集的信息对当前聊天有帮助**，在思考中直接给出总结信息，格式为：return_information(information="你的总结信息")
- **如果信息无关或没有帮助**，在思考中给出：return_information(information="")

**重要规则：**
- 必须严格使用检索到的信息回答问题，不要编造信息
- 答案必须精简，不要过多解释
""",
        name="memory_retrieval_react_final_prompt",
    )


def _log_conversation_messages(
    conversation_messages: List[Message],
    head_prompt: Optional[str] = None,
    final_status: Optional[str] = None,
) -> None:
    """输出对话消息列表的日志

    Args:
        conversation_messages: 对话消息列表
        head_prompt: 第一条系统消息（head_prompt）的内容，可选
        final_status: 最终结果状态描述（例如：找到答案/未找到答案），可选
    """
    if not global_config.debug.show_memory_prompt:
        return

    log_lines: List[str] = []

    # 如果有head_prompt，先添加为第一条消息
    if head_prompt:
        msg_info = "========================================\n[消息 1] 角色: System\n-----------------------------"
        msg_info += f"\n{head_prompt}"
        log_lines.append(msg_info)
        start_idx = 2
    else:
        start_idx = 1

    if not conversation_messages and not head_prompt:
        return

    for idx, msg in enumerate(conversation_messages, start_idx):
        role_name = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

        # 构建单条消息的日志信息
        # msg_info = f"\n========================================\n[消息 {idx}] 角色: {role_name} 内容类型: {content_type}\n-----------------------------"
        msg_info = (
            f"\n========================================\n[消息 {idx}] 角色: {role_name}\n-----------------------------"
        )

        # if full_content:
        #     msg_info += f"\n{full_content}"
        if msg.content:
            msg_info += f"\n{msg.content}"

        if msg.tool_calls:
            msg_info += f"\n  工具调用: {len(msg.tool_calls)}个"
            for tool_call in msg.tool_calls:
                msg_info += f"\n    - {tool_call.func_name}: {json.dumps(tool_call.args, ensure_ascii=False)}"

        # if msg.tool_call_id:
        # msg_info += f"\n  工具调用ID: {msg.tool_call_id}"

        log_lines.append(msg_info)

    total_count = len(conversation_messages) + (1 if head_prompt else 0)
    log_text = f"消息列表 (共{total_count}条):{''.join(log_lines)}"
    if final_status:
        log_text += f"\n\n[最终结果] {final_status}"
    logger.info(log_text)


async def _react_agent_solve_question(
    chat_id: str,
    max_iterations: int = 5,
    timeout: float = 30.0,
    initial_info: str = "",
    chat_history: str = "",
    allowed_tools: Optional[List[str]] = None,
    max_tool_steps: Optional[int] = None,
    stop_after_first_hit: bool = False,
) -> Tuple[bool, str, List[Dict[str, Any]], bool]:
    """使用ReAct架构的Agent来解决问题

    Args:
        chat_id: 聊天ID（等同于 ChatStream.stream_id，数据库 ThinkingBack 表
                 使用 chat_id 字段名存储此值，两者语义完全相同）
        max_iterations: 最大迭代次数
        timeout: 超时时间（秒）
        initial_info: 初始信息，将作为collected_info的初始值
        chat_history: 聊天记录，将传递给 ReAct Agent prompt

    Returns:
        Tuple[bool, str, List[Dict[str, Any]], bool]: (是否找到答案, 答案内容, 思考步骤列表, 是否超时)
    """
    start_time = time.time()
    collected_info = initial_info if initial_info else ""
    # 构造日志前缀：[聊天流名称]，用于在日志中标识聊天流
    try:
        chat_name = get_chat_manager().get_stream_name(chat_id) or chat_id
    except Exception:
        chat_name = chat_id
    react_log_prefix = f"[{chat_name}] "
    thinking_steps = []
    is_timeout = False
    conversation_messages: List[Message] = []
    first_head_prompt: Optional[str] = None  # 保存第一次使用的head_prompt（用于日志显示）
    # 连续未调用工具的次数计数，用于空转早退
    _consecutive_no_tool_calls = 0
    _consecutive_empty_rounds = 0
    _NO_TOOL_EARLY_EXIT_THRESHOLD = 2  # 连续2次思考但不调用工具则提前结束
    _EMPTY_ROUND_EARLY_EXIT_THRESHOLD = 2  # 连续2次无工具且无可见响应则提前结束
    allowed_tool_names = list(dict.fromkeys(allowed_tools or []))
    if allowed_tool_names:
        allowed_tool_names.extend(["return_information", "query_words"])
        allowed_tool_names = list(dict.fromkeys(allowed_tool_names))
    remaining_tool_steps = max(0, int(max_tool_steps or 0)) if max_tool_steps is not None else None
    _budget_exhausted = False
    _llm_call_failed = False

    iteration = 0
    while iteration < max_iterations:
        # 检查超时
        if time.time() - start_time > timeout:
            logger.warning(f"ReAct Agent超时，已迭代{iteration}次")
            is_timeout = True
            break

        # 获取工具注册器
        tool_registry = get_tool_registry()

        # 获取bot_name
        bot_name = global_config.bot.nickname

        # 获取当前时间
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 计算剩余迭代次数
        current_iteration = iteration + 1
        remaining_iterations = max_iterations - current_iteration

        # 提取函数调用中参数的值，支持单引号和双引号
        def _extract_quoted_content_v2(text, func_name, param_name):
            """从文本中提取函数调用中参数的值，支持单引号和双引号

            Args:
                text: 要搜索的文本
                func_name: 函数名，如 'return_information'
                param_name: 参数名，如 'information'

            Returns:
                提取的参数值，如果未找到则返回None
            """
            if not text:
                return None

            # 查找函数调用位置（不区分大小写）
            func_pattern = func_name.lower()
            text_lower = text.lower()
            func_pos = text_lower.find(func_pattern)
            if func_pos == -1:
                return None

            # 查找参数名和等号
            param_pattern = f"{param_name}="
            param_pos = text_lower.find(param_pattern, func_pos)
            if param_pos == -1:
                return None

            # 跳过参数名、等号和空白
            start_pos = param_pos + len(param_pattern)
            while start_pos < len(text) and text[start_pos] in " \t\n":
                start_pos += 1

            if start_pos >= len(text):
                return None

            # 确定引号类型
            quote_char = text[start_pos]
            if quote_char not in ['"', "'"]:
                return None

            # 查找匹配的结束引号（考虑转义）
            end_pos = start_pos + 1
            while end_pos < len(text):
                if text[end_pos] == quote_char:
                    # 检查是否是转义的引号
                    if end_pos > start_pos + 1 and text[end_pos - 1] == "\\":
                        end_pos += 1
                        continue
                    # 找到匹配的引号
                    content = text[start_pos + 1 : end_pos]
                    # 处理转义字符
                    content = content.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
                    return content
                end_pos += 1

            return None

        # 正常迭代：使用head_prompt决定调用哪些工具（包含return_information工具）
        tool_definitions = tool_registry.get_tool_definitions()
        if allowed_tool_names:
            tool_definitions = [tool_def for tool_def in tool_definitions if tool_def["name"] in allowed_tool_names]
        if remaining_tool_steps is not None and remaining_tool_steps <= 0:
            tool_definitions = [td for td in tool_definitions if td["name"] == "return_information"]
            if not tool_definitions or not collected_info or not collected_info.strip():
                logger.info(f"{react_log_prefix}工具预算已用尽且无可用工具或无收集信息，跳过LLM调用直接进入最终评估")
                _budget_exhausted = True
                break
        # tool_names = [tool_def["name"] for tool_def in tool_definitions]
        # logger.debug(f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具: {', '.join(tool_names)} (共{len(tool_definitions)}个)")

        # head_prompt应该只构建一次，使用初始的collected_info，后续迭代都复用同一个
        if first_head_prompt is None:
            # 第一次构建，使用初始的collected_info（即initial_info）
            initial_collected_info = initial_info if initial_info else ""
            first_head_prompt = await global_prompt_manager.format_prompt(
                "memory_retrieval_react_prompt_head",
                bot_name=bot_name,
                time_now=time_now,
                chat_history=chat_history,
                collected_info=initial_collected_info,
                current_iteration=current_iteration,
                remaining_iterations=remaining_iterations,
                max_iterations=max_iterations,
            )

        # 后续迭代都复用第一次构建的head_prompt
        head_prompt = first_head_prompt

        def message_factory(
            _client,
            *,
            _head_prompt: str = head_prompt,
            _conversation_messages: List[Message] = conversation_messages,
        ) -> List[Message]:
            messages: List[Message] = []

            system_builder = MessageBuilder()
            system_builder.set_role(RoleType.System)
            system_builder.add_text_content(_head_prompt)
            messages.append(system_builder.build())

            messages.extend(_conversation_messages)

            # 要求 contents 至少有一条 user 消息，避免 fallback 到部分模型时报
            # “No user query found in messages.”
            has_user_message = any(getattr(msg, "role", None) == RoleType.User for msg in messages)
            if not has_user_message:
                kickoff_builder = MessageBuilder()
                kickoff_builder.set_role(RoleType.User)
                kickoff_builder.add_text_content("请根据上述指令开始分析，检索相关记忆并返回结果。")
                messages.append(kickoff_builder.build())

            return messages

        (
            success,
            response,
            reasoning_content,
            model_name,
            tool_calls,
        ) = await llm_api.generate_with_model_with_message_factory(
            message_factory,
            model_set=model_config.model_task_config.tool_use,
            tool_options=tool_definitions,
            request_type="memory.react",
        )

        # logger.info(
        # f"ReAct Agent 第 {iteration + 1} 次迭代 模型: {model_name} ，调用工具数量: {len(tool_calls) if tool_calls else 0} ，调用工具响应: {response}"
        # )

        if not success:
            logger.error(f"ReAct Agent LLM调用失败: {response}")
            _llm_call_failed = True
            break

        response_text = str(response or "").strip()
        reasoning_text = str(reasoning_content or "").strip()
        has_visible_response = bool(response_text or reasoning_text)

        # 兼容部分模型仅输出伪工具调用标记而不填充 tool_calls 的情况
        if not tool_calls and response_text:
            pseudo_tool_calls = _parse_pseudo_tool_calls(response_text)
            if pseudo_tool_calls:
                tool_calls = pseudo_tool_calls
                response_text = _strip_pseudo_tool_call_markers(response_text)
                has_visible_response = bool(response_text or reasoning_text)
                logger.info(
                    f"{react_log_prefix}第 {iteration + 1} 次迭代 模型={model_name} "
                    f"识别到伪工具调用标记 {len(pseudo_tool_calls)} 个，已转换为标准工具调用"
                )

        # 注意：这里会检查return_information工具调用，如果检测到return_information工具，会根据information参数决定返回信息或退出查询

        assistant_message: Optional[Message] = None
        if tool_calls:
            if allowed_tool_names:
                tool_calls = [call for call in tool_calls if call.func_name in allowed_tool_names]
            return_calls = [call for call in tool_calls if call.func_name == "return_information"]
            non_terminal_calls = [call for call in tool_calls if call.func_name != "return_information"]
            if remaining_tool_steps is not None:
                non_terminal_calls = non_terminal_calls[:remaining_tool_steps]
            if stop_after_first_hit and non_terminal_calls:
                non_terminal_calls = non_terminal_calls[:1]
            tool_calls = return_calls + non_terminal_calls
        if tool_calls:
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            if response_text:
                assistant_builder.add_text_content(response_text)
            assistant_builder.set_tool_calls(tool_calls)
            assistant_message = assistant_builder.build()
        elif response_text:
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            assistant_builder.add_text_content(response_text)
            assistant_message = assistant_builder.build()

        # 记录思考步骤
        step = {
            "iteration": iteration + 1,
            "thought": response_text,
            "actions": [],
            "observations": [],
        }

        if assistant_message:
            conversation_messages.append(assistant_message)

        # 记录思考过程到collected_info中
        if has_visible_response:
            thought_summary = reasoning_text or (response_text[:200] if response_text else "")
            if thought_summary:
                collected_info += f"\n[思考] {thought_summary}\n"

        if remaining_tool_steps is not None and remaining_tool_steps <= 0 and not tool_calls:
            logger.info(f"{react_log_prefix}工具预算已用尽，提前进入最终评估")
            _budget_exhausted = True
            break

        # 处理工具调用
        if not tool_calls:
            # 如果没有工具调用，检查响应文本中是否包含return_information函数调用格式或JSON格式
            if response_text:
                # 首先尝试解析JSON格式的return_information
                def parse_json_return_information(text: str):
                    """从文本中解析JSON格式的return_information，返回information字符串，如果未找到则返回None"""
                    if not text:
                        return None

                    try:
                        # 尝试提取JSON对象（可能包含在代码块中或直接是JSON）
                        json_text = text.strip()

                        # 如果包含代码块标记，提取JSON部分
                        if "```json" in json_text:
                            start = json_text.find("```json") + 7
                            end = json_text.find("```", start)
                            if end != -1:
                                json_text = json_text[start:end].strip()
                        elif "```" in json_text:
                            start = json_text.find("```") + 3
                            end = json_text.find("```", start)
                            if end != -1:
                                json_text = json_text[start:end].strip()

                        # 尝试解析JSON
                        data = json.loads(json_text)

                        # 检查是否包含return_information字段
                        if isinstance(data, dict) and "return_information" in data:
                            information = data.get("information", "")
                            return information
                    except (json.JSONDecodeError, ValueError, TypeError):
                        # 如果JSON解析失败，尝试在文本中查找JSON对象
                        try:
                            # 查找第一个 { 和最后一个 } 之间的内容（更健壮的JSON提取）
                            first_brace = text.find("{")
                            if first_brace != -1:
                                # 从第一个 { 开始，找到匹配的 }
                                brace_count = 0
                                json_end = -1
                                for i in range(first_brace, len(text)):
                                    if text[i] == "{":
                                        brace_count += 1
                                    elif text[i] == "}":
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i + 1
                                            break

                                if json_end != -1:
                                    json_text = text[first_brace:json_end]
                                    data = json.loads(json_text)
                                    if isinstance(data, dict) and "return_information" in data:
                                        information = data.get("information", "")
                                        return information
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass

                    return None

                # 尝试从文本中解析return_information函数调用
                def parse_return_information_from_text(text: str):
                    """从文本中解析return_information函数调用，返回information字符串，如果未找到则返回None"""
                    if not text:
                        return None

                    # 查找return_information函数调用位置（不区分大小写）
                    func_pattern = "return_information"
                    text_lower = text.lower()
                    func_pos = text_lower.find(func_pattern)
                    if func_pos == -1:
                        return None

                    # 使用同作用域已定义的_extract_quoted_content_v2解析information参数
                    information = _extract_quoted_content_v2(text, "return_information", "information")

                    # 如果information存在（即使是空字符串），也返回它
                    return information

                # 首先尝试解析JSON格式
                parsed_information_json = parse_json_return_information(response_text)
                is_json_format = parsed_information_json is not None

                # 如果JSON解析成功，使用JSON结果
                if is_json_format:
                    parsed_information = parsed_information_json
                else:
                    # 如果JSON解析失败，尝试解析函数调用格式
                    parsed_information = parse_return_information_from_text(response_text)

                if parsed_information is not None or is_json_format:
                    # 检测到return_information格式（可能是JSON格式或函数调用格式）
                    format_type = "JSON格式" if is_json_format else "函数调用格式"
                    normalized_information = _sanitize_memory_payload(parsed_information or "")
                    # 返回信息（即使为空字符串也返回）
                    step["actions"].append(
                        {
                            "action_type": "return_information",
                            "action_params": {"information": parsed_information or ""},
                        }
                    )
                    if normalized_information:
                        step["observations"] = [f"检测到return_information{format_type}调用，返回信息"]
                        thinking_steps.append(step)
                        logger.info(
                            f"{react_log_prefix}第 {iteration + 1} 次迭代 通过return_information{format_type}返回信息: {
                                normalized_information[:100]
                            }..."
                        )

                        _log_conversation_messages(
                            conversation_messages,
                            head_prompt=first_head_prompt,
                            final_status=f"返回信息：{normalized_information}",
                        )

                        return True, normalized_information, thinking_steps, False
                    else:
                        # 信息为空，直接退出查询
                        step["observations"] = [f"检测到return_information{format_type}调用，信息为空"]
                        thinking_steps.append(step)
                        logger.info(
                            f"{react_log_prefix}第 {iteration + 1} 次迭代 通过return_information{
                                format_type
                            }判断信息为空"
                        )

                        _log_conversation_messages(
                            conversation_messages,
                            head_prompt=first_head_prompt,
                            final_status="信息为空：通过return_information文本格式判断信息为空",
                        )

                        return False, "", thinking_steps, False

                # 如果没有检测到return_information格式，记录思考过程，继续下一轮迭代
                step["observations"] = [f"思考完成，但未调用工具。响应: {response_text}"]
                logger.info(
                    f"{react_log_prefix}第 {iteration + 1} 次迭代 模型={model_name} 思考完成但未调用工具: {response_text}"
                )
                collected_info += f"思考: {response_text}"
                _consecutive_no_tool_calls += 1
                _consecutive_empty_rounds = 0
            else:
                logger.warning(f"{react_log_prefix}第 {iteration + 1} 次迭代 模型={model_name} 无工具调用且无可见响应")
                step["observations"] = ["无响应且无工具调用"]
                _consecutive_no_tool_calls += 1
                _consecutive_empty_rounds += 1
            thinking_steps.append(step)
            iteration += 1  # 在continue之前增加迭代计数，避免跳过iteration += 1
            if _consecutive_empty_rounds >= _EMPTY_ROUND_EARLY_EXIT_THRESHOLD:
                logger.info(f"{react_log_prefix}连续{_consecutive_empty_rounds}次无工具且无可见响应，提前进入最终评估")
                break
            # 连续多次未调用工具，提前退出避免空转浪费
            if _consecutive_no_tool_calls >= _NO_TOOL_EARLY_EXIT_THRESHOLD:
                logger.info(f"{react_log_prefix}连续{_consecutive_no_tool_calls}次未调用工具，提前进入最终评估")
                break
            continue

        # 成功调用了工具，重置空转计数
        _consecutive_no_tool_calls = 0
        _consecutive_empty_rounds = 0
        # 处理工具调用
        # 首先检查是否有return_information工具调用，如果有则立即返回，不再处理其他工具
        return_information_info = None
        for tool_call in tool_calls:
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            if tool_name == "return_information":
                return_information_info = _sanitize_memory_payload(tool_args.get("information", ""))

                # 返回信息（即使为空也返回）
                step["actions"].append(
                    {
                        "action_type": "return_information",
                        "action_params": {"information": return_information_info},
                    }
                )
                if return_information_info:
                    # 有信息，返回
                    step["observations"] = ["检测到return_information工具调用，返回信息"]
                    thinking_steps.append(step)
                    logger.info(
                        f"{react_log_prefix}第 {iteration + 1} 次迭代 通过return_information工具返回信息: {
                            return_information_info
                        }"
                    )

                    _log_conversation_messages(
                        conversation_messages,
                        head_prompt=first_head_prompt,
                        final_status=f"返回信息：{return_information_info}",
                    )

                    return True, return_information_info, thinking_steps, False
                else:
                    # 信息为空，直接退出查询
                    step["observations"] = ["检测到return_information工具调用，信息为空"]
                    thinking_steps.append(step)
                    logger.info(f"{react_log_prefix}第 {iteration + 1} 次迭代 通过return_information工具判断信息为空")

                    _log_conversation_messages(
                        conversation_messages,
                        head_prompt=first_head_prompt,
                        final_status="信息为空：通过return_information工具判断信息为空",
                    )

                    return False, "", thinking_steps, False

        # 如果没有return_information工具调用，继续处理其他工具
        tool_tasks = []
        executed_non_terminal_steps = 0
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            logger.debug(
                f"{react_log_prefix}第 {iteration + 1} 次迭代 工具调用 {i + 1}/{len(tool_calls)}: {tool_name}({tool_args})"
            )

            # 跳过return_information工具调用（已经在上面处理过了）
            if tool_name == "return_information":
                continue

            # 普通工具调用
            tool = tool_registry.get_tool(tool_name)
            if tool:
                # 准备工具参数（需要添加chat_id如果工具需要）
                import inspect

                sig = inspect.signature(tool.execute_func)
                tool_params = tool_args.copy()
                if "chat_id" in sig.parameters:
                    tool_params["chat_id"] = chat_id

                # 创建异步任务
                async def execute_single_tool(tool_instance, params, tool_name_str, iter_num):
                    try:
                        observation = await tool_instance.execute(**params)
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != "chat_id"])
                        return f"查询{tool_name_str}({param_str})的结果：{observation}"
                    except Exception as e:
                        error_msg = f"工具执行失败: {str(e)}"
                        logger.error(f"{react_log_prefix}第 {iter_num + 1} 次迭代 工具 {tool_name_str} {error_msg}")
                        return f"查询{tool_name_str}失败: {error_msg}"

                tool_tasks.append(execute_single_tool(tool, tool_params, tool_name, iteration))
                step["actions"].append({"action_type": tool_name, "action_params": tool_args})
                executed_non_terminal_steps += 1
            else:
                error_msg = f"未知的工具类型: {tool_name}"
                logger.warning(
                    f"{react_log_prefix}第 {iteration + 1} 次迭代 工具 {i + 1}/{len(tool_calls)} {error_msg}"
                )
                tool_tasks.append(asyncio.create_task(asyncio.sleep(0, result=f"查询{tool_name}失败: {error_msg}")))

        # 并行执行所有工具
        if tool_tasks:
            try:
                observations = await asyncio.wait_for(
                    asyncio.gather(*tool_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"{react_log_prefix}工具并行执行超时(30s)，跳过本轮工具结果")
                observations = [f"工具执行超时" for _ in tool_tasks]
            high_value_observation = ""

            # 处理执行结果
            for i, (tool_call_item, observation) in enumerate(zip(tool_calls, observations, strict=False)):
                if isinstance(observation, Exception):
                    observation = f"工具执行异常: {str(observation)}"
                    logger.error(f"{react_log_prefix}第 {iteration + 1} 次迭代 工具 {i + 1} 执行异常: {observation}")

                observation_text = observation if isinstance(observation, str) else str(observation)
                stripped_observation = observation_text.strip()
                step["observations"].append(observation_text)
                collected_info += f"\n{observation_text}\n"
                normalized_observation, hit_quality, _ = _classify_memory_hit_quality(chat_history, stripped_observation)
                if not high_value_observation and hit_quality == "strong":
                    high_value_observation = normalized_observation
                if stripped_observation:
                    # 不再自动检测工具输出中的jargon，改为通过 query_words 工具主动查询
                    tool_builder = MessageBuilder()
                    tool_builder.set_role(RoleType.Tool)
                    tool_builder.add_text_content(observation_text)
                    tool_builder.add_tool_call(tool_call_item.call_id)
                    conversation_messages.append(tool_builder.build())

            if stop_after_first_hit and high_value_observation:
                thinking_steps.append(step)
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status=f"返回信息：{high_value_observation}",
                )
                return True, high_value_observation, thinking_steps, False
        if remaining_tool_steps is not None and executed_non_terminal_steps > 0:
            remaining_tool_steps = max(0, remaining_tool_steps - executed_non_terminal_steps)

        thinking_steps.append(step)

        iteration += 1

    # 正常迭代结束后，如果达到最大迭代次数、超时或空转早退，执行最终评估
    # 最终评估单独处理，不算在迭代中
    should_do_final_evaluation = False
    _early_exit_no_tool = _consecutive_no_tool_calls >= _NO_TOOL_EARLY_EXIT_THRESHOLD
    _early_exit_empty = _consecutive_empty_rounds >= _EMPTY_ROUND_EARLY_EXIT_THRESHOLD
    if is_timeout:
        should_do_final_evaluation = True
        logger.warning(f"{react_log_prefix}超时，已迭代{iteration}次，进入最终评估")
    elif _budget_exhausted:
        should_do_final_evaluation = True
        logger.info(f"{react_log_prefix}工具预算已用尽，已迭代{iteration}次，进入最终评估")
    elif _early_exit_empty:
        should_do_final_evaluation = True
        logger.info(
            f"{react_log_prefix}连续{_consecutive_empty_rounds}次无工具且无可见响应，已迭代{iteration}次，进入最终评估"
        )
    elif _early_exit_no_tool:
        should_do_final_evaluation = True
        logger.info(
            f"{react_log_prefix}连续{_consecutive_no_tool_calls}次未调用工具，已迭代{iteration}次，进入最终评估"
        )
    elif iteration >= max_iterations:
        should_do_final_evaluation = True
        logger.info(f"{react_log_prefix}达到最大迭代次数（已迭代{iteration}次），进入最终评估")
    elif _llm_call_failed:
        should_do_final_evaluation = True
        logger.info(f"{react_log_prefix}LLM调用失败，已迭代{iteration}次，进入最终评估")

    if should_do_final_evaluation:
        if _budget_exhausted and collected_info and collected_info.strip():
            sanitized_collected = _sanitize_memory_payload(collected_info.strip())
            if sanitized_collected:
                logger.info(f"{react_log_prefix}工具预算已用尽，已收集到有价值信息，跳过最终评估LLM调用直接返回")
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status=f"预算耗尽直接返回：{sanitized_collected[:100]}",
                )
                return True, sanitized_collected, thinking_steps, False
        # 获取必要变量用于最终评估
        tool_registry = get_tool_registry()
        bot_name = global_config.bot.nickname
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        current_iteration = iteration + 1
        remaining_iterations = 0

        # 提取函数调用中参数的值，支持单引号和双引号
        def extract_quoted_content(text, func_name, param_name):
            """从文本中提取函数调用中参数的值，支持单引号和双引号

            Args:
                text: 要搜索的文本
                func_name: 函数名，如 'return_information'
                param_name: 参数名，如 'information'

            Returns:
                提取的参数值，如果未找到则返回None
            """
            if not text:
                return None

            # 查找函数调用位置（不区分大小写）
            func_pattern = func_name.lower()
            text_lower = text.lower()
            func_pos = text_lower.find(func_pattern)
            if func_pos == -1:
                return None

            # 查找参数名和等号
            param_pattern = f"{param_name}="
            param_pos = text_lower.find(param_pattern, func_pos)
            if param_pos == -1:
                return None

            # 跳过参数名、等号和空白
            start_pos = param_pos + len(param_pattern)
            while start_pos < len(text) and text[start_pos] in " \t\n":
                start_pos += 1

            if start_pos >= len(text):
                return None

            # 确定引号类型
            quote_char = text[start_pos]
            if quote_char not in ['"', "'"]:
                return None

            # 查找匹配的结束引号（考虑转义）
            end_pos = start_pos + 1
            while end_pos < len(text):
                if text[end_pos] == quote_char:
                    # 检查是否是转义的引号
                    if end_pos > start_pos + 1 and text[end_pos - 1] == "\\":
                        end_pos += 1
                        continue
                    # 找到匹配的引号
                    content = text[start_pos + 1 : end_pos]
                    # 处理转义字符
                    content = content.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
                    return content
                end_pos += 1

            return None

        # 执行最终评估
        evaluation_prompt = await global_prompt_manager.format_prompt(
            "memory_retrieval_react_final_prompt",
            bot_name=bot_name,
            time_now=time_now,
            chat_history=chat_history,
            collected_info=collected_info if collected_info else "暂无信息",
            current_iteration=current_iteration,
            remaining_iterations=remaining_iterations,
            max_iterations=max_iterations,
        )

        (
            eval_success,
            eval_response,
            eval_reasoning_content,
            eval_model_name,
            eval_tool_calls,
        ) = await llm_api.generate_with_model_with_tools(
            evaluation_prompt,
            model_set=model_config.model_task_config.utils,
            tool_options=[],  # 最终评估阶段不提供工具
            request_type="memory.react.final",
        )

        if not eval_success:
            logger.error(f"ReAct Agent 最终评估阶段 LLM调用失败: {eval_response}")
            _log_conversation_messages(
                conversation_messages,
                head_prompt=first_head_prompt,
                final_status="未找到答案：最终评估阶段LLM调用失败",
            )
            return False, "最终评估阶段LLM调用失败", thinking_steps, is_timeout

        if global_config.debug.show_memory_prompt:
            logger.info(f"{react_log_prefix}最终评估Prompt: {evaluation_prompt}")
            logger.info(f"{react_log_prefix}最终评估响应: {eval_response}")

        # 从最终评估响应中提取return_information
        return_information_content = None

        if eval_response:
            return_information_content = extract_quoted_content(eval_response, "return_information", "information")
            return_information_content = _sanitize_memory_payload(return_information_content)
            if return_information_content is None or return_information_content == "":
                natural_eval_payload = _sanitize_memory_payload(eval_response)
                if natural_eval_payload:
                    return_information_content = natural_eval_payload

        # 如果提取到信息，返回（无论是否超时，都视为成功完成）
        if return_information_content is not None:
            eval_step = {
                "iteration": current_iteration,
                "thought": f"[最终评估] {eval_response}",
                "actions": [
                    {
                        "action_type": "return_information",
                        "action_params": {"information": return_information_content},
                    }
                ],
                "observations": ["最终评估阶段检测到return_information"],
            }
            thinking_steps.append(eval_step)
            if return_information_content and return_information_content.strip():
                logger.info(f"ReAct Agent 最终评估阶段返回信息: {return_information_content}")
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status=f"返回信息：{return_information_content}",
                )
                return True, return_information_content, thinking_steps, False
            else:
                logger.info("ReAct Agent 最终评估阶段判断信息为空")
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status="信息为空：最终评估阶段判断信息为空",
                )
                return False, "", thinking_steps, False

        # 如果没有明确判断，视为not_enough_info，返回空字符串（不返回任何信息）
        eval_step = {
            "iteration": current_iteration,
            "thought": f"[最终评估] {eval_response}",
            "actions": [
                {
                    "action_type": "return_information",
                    "action_params": {"information": ""},
                }
            ],
            "observations": ["已到达最大迭代次数，信息为空"],
        }
        thinking_steps.append(eval_step)
        logger.info("ReAct Agent 已到达最大迭代次数，信息为空")

        _log_conversation_messages(
            conversation_messages,
            head_prompt=first_head_prompt,
            final_status="未找到答案：已到达最大迭代次数，无法找到答案",
        )

        return False, "", thinking_steps, is_timeout

    # 如果正常迭代过程中提前找到答案返回，不会到达这里
    # 如果正常迭代结束但没有触发最终评估（理论上不应该发生），直接返回
    logger.warning("ReAct Agent正常迭代结束，但未触发最终评估")
    _log_conversation_messages(
        conversation_messages,
        head_prompt=first_head_prompt,
        final_status="未找到答案：正常迭代结束",
    )

    return False, "", thinking_steps, is_timeout


def _get_recent_query_history(chat_id: str, time_window_seconds: float = 600.0) -> str:
    """获取最近一段时间内的查询历史（用于避免重复查询）

    Args:
        chat_id: 聊天ID（等同于 ChatStream.stream_id，数据库 ThinkingBack 表
                 使用 chat_id 字段名存储此值）
        time_window_seconds: 时间窗口（秒），默认10分钟

    Returns:
        str: 格式化的查询历史字符串
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.update_time >= start_time))
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return ""

        history_lines = []
        history_lines.append("最近已查询的问题和结果：")

        for record in records:
            status = "✓ 已找到答案" if record.found_answer else "✗ 未找到答案"
            answer_preview = ""
            # 只有找到答案时才显示答案内容
            if record.found_answer and record.answer:
                # 截取答案前100字符
                answer_preview = record.answer[:100]
                if len(record.answer) > 100:
                    answer_preview += "..."

            history_lines.append(f"- 问题：{record.question}")
            history_lines.append(f"  状态：{status}")
            if answer_preview:
                history_lines.append(f"  答案：{answer_preview}")
            history_lines.append("")  # 空行分隔

        return "\n".join(history_lines)

    except Exception as e:
        logger.error(f"获取查询历史失败: {e}")
        return ""


def _get_recent_found_answers(chat_id: str, time_window_seconds: float = 600.0) -> List[str]:
    """获取最近一段时间内已找到答案的查询记录（用于返回给 replyer）

    Args:
        chat_id: 聊天ID（等同于 ChatStream.stream_id，数据库 ThinkingBack 表
                 使用 chat_id 字段名存储此值）
        time_window_seconds: 时间窗口（秒），默认10分钟

    Returns:
        List[str]: 格式化的答案列表，每个元素格式为 "问题：xxx\n答案：xxx"
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内已找到答案的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.update_time >= start_time)
                & (ThinkingBack.found_answer == 1)
                & (ThinkingBack.answer.is_null(False))
                & (ThinkingBack.answer != "")
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(3)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return []

        found_answers = []
        for record in records:
            if record.answer and _is_high_value_memory_text(record.answer):
                found_answers.append(f"问题：{record.question}\n答案：{record.answer}")

        return found_answers

    except Exception as e:
        logger.error(f"获取最近已找到答案的记录失败: {e}")
        return []


def _store_thinking_back(
    chat_id: str,
    question: str,
    context: str,
    found_answer: bool,
    answer: str,
    thinking_steps: List[Dict[str, Any]],
) -> None:
    """存储或更新思考过程到数据库（如果已存在则更新，否则创建）

    Args:
        chat_id: 聊天ID（等同于 ChatStream.stream_id，数据库 ThinkingBack 表
                 使用 chat_id 字段名存储此值）
        question: 问题
        context: 上下文信息
        found_answer: 是否找到答案
        answer: 答案内容
        thinking_steps: 思考步骤列表
    """
    try:
        now = time.time()
        if found_answer and not _is_high_value_memory_text(answer):
            logger.debug(f"过滤低价值记忆结果，不写入有效缓存: {question[:50]}...")
            found_answer = False
            answer = ""

        # 先查询是否已存在相同chat_id和问题的记录
        existing = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.question == question))
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if existing.exists():
            # 更新现有记录
            record = existing.get()
            record.context = context
            record.found_answer = found_answer
            record.answer = answer
            record.thinking_steps = json.dumps(thinking_steps, ensure_ascii=False)
            record.update_time = now
            record.save()
            logger.info(f"已更新思考过程到数据库，问题: {question[:50]}...")
        else:
            # 创建新记录
            ThinkingBack.create(
                chat_id=chat_id,
                question=question,
                context=context,
                found_answer=found_answer,
                answer=answer,
                thinking_steps=json.dumps(thinking_steps, ensure_ascii=False),
                create_time=now,
                update_time=now,
            )
            # logger.info(f"已创建思考过程到数据库，问题: {question[:50]}...")
    except Exception as e:
        logger.error(f"存储思考过程失败: {e}")


# ==================== 缓存探测器 ====================


class ThinkingRecordCache:
    """ThinkingBack缓存探测器 —— 在发起LLM调用前先行检查历史结果。

    查找同一聊天流中、时间窗口内、已找到答案的记录，
    通过词粒度重叠率判断问题相似性。命中时直接复用历史答案。
    融合XBcore的混合检索思想（词级+语义级），但只保留轻量词级匹配。
    """

    def __init__(
        self,
        window_seconds: float = 3600.0,
        overlap_bar: float = 0.6,
        scan_limit: int = 20,
    ):
        self._window = window_seconds
        self._overlap_bar = overlap_bar
        self._scan_limit = scan_limit

    def probe(self, question: str, stream_id: str) -> Optional[str]:
        """在ThinkingBack表中搜索可复用的历史答案。

        检索接口关系说明：
        - ThinkingRecordCache.probe(question, stream_id): 本方法，ThinkingBack 缓存探测
        - HippoMemoryBuffer.probe(query, ceiling): 海马体缓冲区关键词检索
        - RecollectionHub.query_memories(stream_id, query, ...): 长期存储+海马体分层检索
        注意：各 probe/query_memories 方法签名不同，参数名和含义各异，调用时需注意区分。

        返回:
            命中的答案字符串，或None
        """
        try:
            cutoff = time.time() - self._window
            rows = (
                ThinkingBack.select()
                .where(
                    (ThinkingBack.chat_id == stream_id)
                    & (ThinkingBack.found_answer)
                    & (ThinkingBack.update_time >= cutoff)
                )
                .order_by(ThinkingBack.update_time.desc())
                .limit(self._scan_limit)
            )
            q_lower = question.lower().strip()
            q_tokens = set(q_lower.split())
            for row in rows:
                # 防御性检查：row.answer可能为None或空字符串
                row_answer = row.answer if row.answer else ""
                if not row_answer or not _is_high_value_memory_text(row_answer):
                    continue
                r_lower = (row.question or "").lower().strip()
                # 精确匹配（大小写不敏感）
                if r_lower == q_lower:
                    logger.info(f"[缓存命中] 精确匹配: {question[:40]}")
                    return row_answer
                # 词重叠度匹配
                r_tokens = set(r_lower.split())
                if q_tokens and r_tokens:
                    overlap = len(q_tokens & r_tokens) / max(len(q_tokens), len(r_tokens))
                    if overlap >= self._overlap_bar:
                        logger.info(f"[缓存命中] 重叠度{overlap:.2f}: {question[:40]} ~ {row.question[:40]}")
                        return row_answer
        except Exception as exc:
            logger.debug(f"[缓存探测] 失败: {exc}")
        return None


_thinking_cache = ThinkingRecordCache()


# ==================== 关键词直驱工具探针 ====================


class DirectToolProbe:
    """直接工具探针 —— 绕过LLM，用关键词驱动注册工具快速扫描。

    从问题中提取中英文关键词，直接注入工具的第一个参数执行查询。
    命中时跳过昂贵的多轮ReAct推理。
    融合MIMiaoCore的HistoryQueryTool直查概念。
    """

    # 判定"没找到"的标志词
    _MISS_MARKERS = _INVALID_MEMORY_MARKERS
    _SKIP_TOOL_NAMES = frozenset(
        [
            "get_chat_history_detail",
            "query_person_info",
            "query_words",
        ]
    )

    @staticmethod
    def _parse_probe_result(raw_result: Any) -> Tuple[Optional[str], int]:
        """兼容 direct probe 新旧返回格式。"""
        if (
            isinstance(raw_result, tuple)
            and len(raw_result) == 2
            and isinstance(raw_result[1], int)
        ):
            return raw_result[0], int(raw_result[1])
        return raw_result, 0

    @staticmethod
    def _prepare_probe_text(question: str) -> str:
        """保留自然语言原貌，只做最轻量的压缩清洗。"""
        if not question:
            return ""
        lines = [line.strip() for line in question.splitlines() if line.strip()]
        compact = " ".join(lines)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact[:200]

    @staticmethod
    def _build_probe_kwargs(
        tool_name: str,
        signature: inspect.Signature,
        query_text: str,
        stream_id: str,
    ) -> Optional[Dict[str, Any]]:
        """按照工具签名构造参数，避免把查询词错误塞进 stream_id 一类字段。"""
        if tool_name in DirectToolProbe._SKIP_TOOL_NAMES:
            return None

        parameters = signature.parameters
        kwargs: Dict[str, Any] = {}
        if "chat_id" in parameters:
            kwargs["chat_id"] = stream_id
        if "stream_id" in parameters:
            kwargs["stream_id"] = stream_id

        if tool_name in ("search_memory_by_embedding", "search_knowledge"):
            kwargs["query"] = query_text
            if "top_k" in parameters:
                kwargs["top_k"] = 3
            if "limit" in parameters:
                kwargs["limit"] = 3
            return kwargs

        remaining_required = [
            name for name, param in parameters.items() if name not in kwargs and param.default is inspect._empty
        ]
        if len(remaining_required) != 1:
            return None
        target_name = remaining_required[0]
        if target_name not in {"query", "keyword", "words"}:
            return None
        kwargs[target_name] = query_text
        return kwargs

    @staticmethod
    async def attempt(
        question: str,
        stream_id: str,
        preferred_tools: Optional[List[str]] = None,
        max_tool_steps: Optional[int] = None,
        stop_after_first_hit: bool = True,
        return_meta: bool = False,
    ) -> Any:
        """尝试直接用关键词驱动工具获取答案。

        返回:
            汇总结果字符串（命中时），或None（未命中）。
        """
        try:
            registry = get_tool_registry()
            query_str = DirectToolProbe._prepare_probe_text(question)
            if not query_str:
                return None
            tool_sequence = list(
                dict.fromkeys(
                    preferred_tools
                    or ["query_direct_memory", "search_chat_history", "search_memory_by_embedding"]
                )
            )
            if max_tool_steps is not None and max_tool_steps > 0:
                tool_sequence = tool_sequence[: max(1, int(max_tool_steps))]
            best_hit_text = ""
            best_hit_score = -1.0
            attempted_steps = 0
            for tool_tag in tool_sequence:
                if tool_tag in ("found_answer", "return_information"):
                    continue
                tool_obj = registry.get_tool(tool_tag)
                if tool_obj is None:
                    continue
                try:
                    sig = inspect.signature(tool_obj.execute_func)
                    kwargs = DirectToolProbe._build_probe_kwargs(
                        tool_name=tool_tag,
                        signature=sig,
                        query_text=query_str,
                        stream_id=stream_id,
                    )
                    if not kwargs:
                        continue
                    attempted_steps += 1
                    observation = await tool_obj.execute(**kwargs)
                    obs_text = str(observation).strip() if observation else ""
                    sanitized_text, hit_quality, relevance_score = _classify_memory_hit_quality(query_str, obs_text)
                    if hit_quality == "strong":
                        if stop_after_first_hit:
                            logger.info(f"[直驱探针] {tool_tag} 强命中: {question[:40]}")
                            payload = sanitized_text[:800]
                            if return_meta:
                                return payload, attempted_steps
                            return payload
                        if relevance_score > best_hit_score:
                            best_hit_score = relevance_score
                            best_hit_text = sanitized_text[:800]
                            logger.info(f"[直驱探针] {tool_tag} 强命中并缓存候选: {question[:40]}")
                    elif hit_quality == "weak" and relevance_score > best_hit_score:
                        best_hit_score = relevance_score
                        best_hit_text = sanitized_text[:800]
                        logger.info(f"[直驱探针] {tool_tag} 弱命中，继续探测: {question[:40]}")
                except Exception:
                    continue
            if best_hit_text:
                if return_meta:
                    return best_hit_text, attempted_steps
                return best_hit_text
            if return_meta:
                return None, attempted_steps
        except Exception as exc:
            logger.debug(f"[直驱探针] 失败: {exc}")
            if return_meta:
                return None, 0
        return None


# ==================== 近期发现收集器 ====================


def _collect_recent_findings(
    stream_id: str,
    current_answer: Optional[str],
    window_seconds: float = 600.0,
    ceiling: int = 3,
) -> List[str]:
    """收集当前结果 + 近期已找到答案的缓存记录，去重合并。

    返回格式化的答案字符串列表。
    """
    all_items: List[str] = []
    # 先加入当前结果
    sanitized_current = _sanitize_memory_payload(current_answer)
    if sanitized_current:
        all_items.append(sanitized_current)
    # 拉取近期缓存
    recent = _get_recent_found_answers(stream_id, window_seconds)
    for cached_item in recent:
        sanitized_cached = _sanitize_memory_payload(cached_item)
        if not sanitized_cached:
            continue
        # 去重：跳过与当前结果相同问题的缓存
        if sanitized_current and sanitized_cached == sanitized_current:
            continue
        all_items.append(sanitized_cached)
        if len(all_items) >= ceiling + 1:
            break
    return all_items


def _trim_retrieval_findings(findings: List[str], max_result_chars: int) -> str:
    """统一裁剪记忆检索结果，避免不同消费口重复猜预算。"""
    merged = "\n\n".join(str(item).strip() for item in findings if str(item).strip())
    if not merged:
        return ""
    if max_result_chars > 0 and len(merged) > max_result_chars:
        trimmed = merged[: max(0, max_result_chars)].rstrip()
        return f"{trimmed}\n...(记忆检索已截断)"
    return merged


# ==================== 上下文智能适配器 ====================


def _estimate_tokens(text: str) -> int:
    """估算文本token数（中文友好：1中文字≈1.5-2token，英文≈0.25 token/词）"""
    if not text:
        return 0
    _cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    _non_cn = len(text) - _cn_chars
    return int(_cn_chars * 1.8 + max(0, _non_cn) * 0.3)


def _analyze_text_metadata(text: str) -> Dict[str, Any]:
    """分析文本元信息，返回结构化字典"""
    if not text:
        return {"lines": 0, "chars": 0, "kb": 0.0, "tokens": 0, "cn_ratio": 0.0, "dialogue_turns": 0}
    _lines = [line for line in text.splitlines() if line.strip()]
    _chars = len(text)
    _cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    _tokens = _estimate_tokens(text)
    _turns = len([line for line in _lines if ":" in line or "：" in line])
    return {
        "lines": len(_lines),
        "chars": _chars,
        "kb": round(_chars / 1024, 2),
        "tokens": _tokens,
        "cn_ratio": round(_cn / max(1, _chars), 2),
        "dialogue_turns": _turns,
    }


def _extract_key_dialogue_turns(lines: List[str], keep_n: int) -> List[str]:
    """从对话行中提取关键轮次（保留首尾+去重连续重复）"""
    if len(lines) <= keep_n:
        return lines
    _kept: List[str] = []
    _prev_hash: Optional[int] = None
    _dup_streak = 0
    for line in lines:
        _h = hash(line.strip()) & 0x7FFFFFFF
        if _h == _prev_hash:
            _dup_streak += 1
            if _dup_streak <= 1:
                _kept.append(line)
            continue
        _prev_hash = _h
        _dup_streak = 0
        _kept.append(line)
    if len(_kept) <= keep_n:
        return _kept
    _first = _kept[: max(1, keep_n // 4)]
    _last = _kept[-max(1, keep_n // 2) :]
    _mid_start = len(_kept) // 4
    _mid_end = _mid_start + (keep_n - len(_first) - len(_last))
    _middle = _kept[_mid_start : max(_mid_start, _mid_end)]
    return list(dict.fromkeys(_first + _middle + _last))


def _analyze_content_quality(lines: List[str]) -> Dict[str, Any]:
    """内容质量分析器 — 扫描文本检测刷屏/复读/无意义行为，返回结构化标签"""
    if not lines:
        return {
            "total": 0,
            "meaningful": 0,
            "spam_count": 0,
            "repeat_groups": [],
            "spam_tags": [],
            "flood_score": 0.0,
            "content_profile": "",
            "speaker_activity": {},
        }
    _n = len(lines)
    _clean = [line.strip() for line in lines if line.strip()]
    if not _clean:
        return {
            "total": _n,
            "meaningful": 0,
            "spam_count": _n,
            "repeat_groups": [],
            "spam_tags": ["全空行"],
            "flood_score": 1.0,
            "content_profile": "空内容",
            "speaker_activity": {},
        }
    _line_hashes: List[int] = [hash(line) & 0x7FFFFFFF for line in _clean]
    _hash_freq: Dict[int, int] = {}
    for _h in _line_hashes:
        _hash_freq[_h] = _hash_freq.get(_h, 0) + 1
    _repeat_groups: List[Dict[str, Any]] = []
    for _h, _cnt in sorted(_hash_freq.items(), key=lambda x: -x[1]):
        if _cnt < 2:
            break
        _idx_list = [i for i, v in enumerate(_line_hashes) if v == _h]
        # 防御性检查：确保索引列表非空，避免IndexError
        if not _idx_list:
            continue
        _first_content = _clean[_idx_list[0]]
        _truncated = (_first_content[:40] + "...") if len(_first_content) > 40 else _first_content
        # 安全计算是否连续重复：只有在有多个索引时才判断连续性
        _is_consecutive_flag = False
        if len(_idx_list) > 1:
            _is_consecutive_flag = all((_idx_list[i + 1] - _idx_list[i] <= 2) for i in range(len(_idx_list) - 1))
        _repeat_groups.append(
            {
                "content": _truncated,
                "count": _cnt,
                "indices": _idx_list,
                "is_consecutive": _is_consecutive_flag,
            }
        )
    _spam_tags: List[str] = []
    _consecutive_repeat_max = 0
    for _rg in _repeat_groups:
        if _rg["count"] >= 4 and _rg["is_consecutive"]:
            _consecutive_repeat_max = max(_consecutive_repeat_max, _rg["count"])
            _spam_tags.append(f"连续复读×{_rg['count']}: {_rg['content']}")
        elif _rg["count"] >= 6:
            _spam_tags.append(f"分散重复×{_rg['count']}: {_rg['content']}")
    _short_line_count = sum(1 for line in _clean if len(line) <= 3)
    _very_short_ratio = _short_line_count / max(1, len(_clean))
    if _very_short_ratio > 0.5 and len(_clean) >= 8:
        _spam_tags.append(f"碎片化内容(≤3字占{_very_short_ratio * 100:.0f}%)")
    _forward_marker_count = sum(1 for line in _clean if "转发" in line or "【转发" in line or "========== 转发消息")
    if _forward_marker_count >= 2:
        _spam_tags.append(f"转发堆叠({_forward_marker_count}处)")
    _same_speaker_burst = 0
    _prev_speaker: Optional[str] = None
    _burst_streak = 0
    for ln in _clean:
        _c = max(ln.find(":"), ln.find("："))
        if 0 < _c < 20:
            _spk = ln[:_c].strip()
        else:
            _spk = None
        if _spk and _spk == _prev_speaker:
            _burst_streak += 1
            _same_speaker_burst = max(_same_speaker_burst, _burst_streak)
        else:
            _burst_streak = 1
            _prev_speaker = _spk
    if _same_speaker_burst >= 5:
        _spam_tags.append(f"单方连续发言×{_same_speaker_burst}")
    _speaker_activity: Dict[str, Dict[str, Any]] = {}
    for ln in _clean:
        _c = max(ln.find(":"), ln.find("："))
        if 0 < _c < 20:
            _spk = ln[:_c].strip()
            if not _spk:
                continue
            if _spk not in _speaker_activity:
                _speaker_activity[_spk] = {"count": 0, "chars": 0, "is_dominant": False}
            _speaker_activity[_spk]["count"] += 1
            _speaker_activity[_spk]["chars"] += len(ln)
    if _speaker_activity:
        _sorted_spk = sorted(_speaker_activity.items(), key=lambda x: -x[1]["count"])
        _top_count = _sorted_spk[0][1]["count"] if _sorted_spk else 0
        if _top_count >= 3 and len(_clean) > 0 and _top_count / len(_clean) > 0.65:
            _dominant = _sorted_spk[0][0]
            _speaker_activity[_dominant]["is_dominant"] = True
            _pct = int(_top_count / len(_clean) * 100)
            _spam_tags.append(f"单人主导({_dominant}占{_pct}%/{_top_count}条)")
    _repeat_total = sum(rg["count"] - 1 for rg in _repeat_groups)
    _spam_count = _repeat_total
    _meaningful = max(0, _n - _spam_count)
    _flood_parts = []
    if _consecutive_repeat_max >= 4:
        _flood_parts.append(f"连续复读={_consecutive_repeat_max}")
    if _very_short_ratio > 0.5:
        _flood_parts.append("碎片化高")
    if _same_speaker_burst >= 5:
        _flood_parts.append(f"单方刷屏={_same_speaker_burst}")
    if _forward_marker_count >= 2:
        _flood_parts.append("转发堆叠")
    _flood_score = min(
        1.0,
        (
            0.3 * min(1.0, _consecutive_repeat_max / 8.0)
            + 0.25 * _very_short_ratio
            + 0.25 * min(1.0, _same_speaker_burst / 15.0)
            + 0.2 * min(1.0, _forward_marker_count / 5.0)
        ),
    )
    if _flood_score < 0.15 and not _spam_tags:
        _profile = "正常对话"
    elif _flood_score < 0.35:
        _profile = "轻度噪点"
    elif _flood_score < 0.55:
        _profile = "中度刷屏"
    elif _flood_score < 0.75:
        _profile = "重度刷屏"
    else:
        _profile = "严重刷屏"
    return {
        "total": _n,
        "meaningful": _meaningful,
        "spam_count": _spam_count,
        "repeat_groups": _repeat_groups[:5],
        "spam_tags": _spam_tags[:8],
        "flood_score": round(_flood_score, 2),
        "content_profile": _profile,
        "speaker_activity": _speaker_activity,
    }


def _format_quality_report(quality: Dict[str, Any]) -> str:
    """将质量分析结果格式化为模型可读的声明"""
    if not quality or quality.get("total", 0) == 0:
        return ""
    _parts: List[str] = []
    _parts.append(
        f"[🔍内容质检] 总{quality['total']}条 | "
        f"有效≈{quality['meaningful']} | "
        f"噪声≈{quality['spam_count']} | "
        f"质量等级={quality['content_profile']} | "
        f"刷屏指数={quality['flood_score']:.2f}"
    )
    if quality.get("spam_tags"):
        _tag_str = "；".join(quality["spam_tags"][:5])
        _parts.append(f"[⚠️行为标签] {_tag_str}")
    if quality.get("repeat_groups"):
        _top_rg = quality["repeat_groups"][0]
        if _top_rg["count"] >= 3:
            _parts.append(f'[🔄最频重复] "{_top_rg["content"]}" ×{_top_rg["count"]}')
    _dominant = [spk for spk, info in quality.get("speaker_activity", {}).items() if info.get("is_dominant")]
    if _dominant:
        _d_info = quality["speaker_activity"][_dominant[0]]
        _parts.append(f"[📢主导者] {_dominant[0]} 发言{_d_info['count']}次/{_d_info['chars']}字")
    return "\n".join(_parts)


def _build_context_declaration(
    original_meta: Dict[str, Any],
    strategy: str,
    adapted_meta: Dict[str, Any],
    label: str = "对话上下文",
) -> str:
    """生成上下文声明，让模型知道它面对的文本量级和处理方式"""
    _strategy_desc = {
        "pass_through": "完整传入",
        "smart_compress": "智能压缩（已去除冗余重复）",
        "summary_index": "摘要索引（保留关键轮次+结构化概览）",
        "metadata_only": "仅核心内容（原文过长，已提取最相关部分）",
    }
    _orig = original_meta
    _adap = adapted_meta
    return (
        f"[📋{label}元信息] 原文={_orig['lines']}行/{_orig['chars']}字/"
        f"~{_orig['tokens']}tokens/{_orig['kb']}KB | "
        f"中文占比{_orig['cn_ratio'] * 100:.0f}% | "
        f"对话轮次≈{_orig['dialogue_turns']}\n"
        f"[📐适配策略] {_strategy_desc.get(strategy, strategy)} | "
        f"当前={_adap['lines']}行/~{_adap['tokens']}tokens"
    )


_STRATEGY_PASS = "pass_through"
_STRATEGY_COMPRESS = "smart_compress"
_STRATEGY_SUMMARY = "summary_index"
_STRATEGY_META = "metadata_only"


def _adapt_chat_context_for_model(
    raw_text: str,
    max_context_tokens: int = 3200,
    label: str = "对话上下文",
) -> str:
    """智能上下文适配器入口

    分析文本 → 选择策略 → 适配 → 注入声明

    策略分级:
      pass_through:   文本token < 阈值60% → 完整传入
      smart_compress:  文本token 60%-100% → 去重+提取关键轮次
      summary_index:  文本token 100%-200% → 摘要+最新N句
      metadata_only:  文本token > 200%     → 仅元信息+最近3句
    """
    if not raw_text or not isinstance(raw_text, str):
        return raw_text or ""
    _meta = _analyze_text_metadata(raw_text)
    _raw_lines = [line for line in raw_text.splitlines() if line.strip()]
    _quality = _analyze_content_quality(_raw_lines) if _raw_lines else None
    _threshold_60 = int(max_context_tokens * 0.60)
    _threshold_100 = int(max_context_tokens * 0.95)
    _threshold_200 = int(max_context_tokens * 1.80)
    _raw_tokens = _meta["tokens"]
    if _raw_tokens < _threshold_60:
        _strategy = _STRATEGY_PASS
        _result = raw_text
    elif _raw_tokens < _threshold_100:
        _strategy = _STRATEGY_COMPRESS
        _all_lines = [line for line in raw_text.splitlines() if line.strip()]
        _target_lines = max(6, len(_all_lines) * 2 // 3)
        _picked = _extract_key_dialogue_turns(_all_lines, _target_lines)
        _result = "\n".join(_picked)
    elif _raw_tokens < _threshold_200:
        _strategy = _STRATEGY_SUMMARY
        _all_lines = [line for line in raw_text.splitlines() if line.strip()]
        _latest = _extract_key_dialogue_turns(_all_lines[-max(8, len(_all_lines) // 3) :], 12)
        _early = _extract_key_dialogue_turns(_all_lines[: min(10, len(_all_lines) // 4)], 5)
        _summary_parts = []
        _speakers_seen = set()
        for ln in _all_lines:
            _colon_idx = max(ln.find(":"), ln.find("："))
            if _colon_idx > 0 and _colon_idx < 20:
                _spk = ln[:_colon_idx].strip()
                if _spk and _spk not in _speakers_seen:
                    _speakers_seen.add(_spk)
        if _speakers_seen:
            _summary_parts.append(f"参与角色: {', '.join(list(_speakers_seen)[:8])}")
        _summary_parts.append(f"总消息数: {len(_all_lines)}条")
        _header = "[📝上下文摘要] " + " | ".join(_summary_parts)
        _body = "\n".join(_early + (["---"] if _early and _latest else []) + _latest)
        _result = f"{_header}\n{_body}"
    else:
        _strategy = _STRATEGY_META
        _all_lines = [line for line in raw_text.splitlines() if line.strip()]
        _latest = _all_lines[-5:] if len(_all_lines) > 5 else _all_lines
        _speakers = set()
        for ln in _all_lines:
            _c = max(ln.find(":"), ln.find("："))
            if 0 < _c < 20:
                _spk = ln[:_c].strip()
                if _spk:
                    _speakers.add(_spk)
        _header = f"[⚠️上下文超长] 原文{_meta['lines']}行/~{_meta['tokens']}tokens，超出模型承载能力。以下为最新片段。"
        if _speakers:
            _header += f" 参与者: {', '.join(list(_speakers)[:6])}"
        _result = f"{_header}\n" + "\n".join(_latest)
    _adapted_meta = _analyze_text_metadata(_result)
    _declaration = _build_context_declaration(_meta, _strategy, _adapted_meta, label)
    _quality_report = _format_quality_report(_quality) if _quality else ""
    logger.info(
        f"[上下文适配] {label}: {_strategy} | "
        f"{_meta['chars']}字→{_adapted_meta['chars']}字 "
        f"({_meta['tokens']}→{_adapted_meta['tokens']}tokens)"
        + (f" | 质量={_quality['content_profile']}" if _quality else "")
    )
    return f"{_declaration}\n{_quality_report}\n{_result}"


# ==================== 三阶段检索管线 ====================


async def _process_memory_retrieval(
    chat_id: str,
    context: str,
    initial_info: str = "",
    max_iterations: Optional[int] = None,
    chat_history: str = "",
    preferred_tools: Optional[List[str]] = None,
    max_tool_steps: Optional[int] = None,
    stop_after_first_hit: bool = False,
) -> Optional[str]:
    """三阶段记忆检索管线: 1)缓存探测 → 2)直驱工具扫描 → 3)ReAct Agent推理。

    任何一阶段命中即短路返回，避免不必要的LLM调用开销。

    参数:
        chat_id: 聊天流ID（等同于 ChatStream.stream_id，数据库 ThinkingBack 表
                 使用 chat_id 字段名存储此值，两者语义完全相同）
        context: 上下文信息
        initial_info: 初始信息（jargon解释等），传递给ReAct Agent
        max_iterations: 最大迭代次数
        chat_history: 聊天记录文本
    返回:
        找到的答案字符串，或None
    """
    _cleanup_stale_not_found_thinking_back()

    # —— 第1阶段：缓存探测 ——
    # 从ThinkingBack中查找近1小时内、同一聊天流的相似问题答案
    cached_hit = _thinking_cache.probe(chat_history[:200], chat_id)
    if cached_hit:
        return cached_hit

    preferred_tool_order = list(dict.fromkeys(preferred_tools or []))
    if max_tool_steps is not None and max_tool_steps > 0:
        preferred_tool_order = preferred_tool_order[: max(1, int(max_tool_steps))]

    # —— 第2阶段：直驱工具扫描 ——
    # 绕过LLM，用关键词直接驱动注册工具查询
    probe_result = await DirectToolProbe.attempt(
        chat_history[:200],
        chat_id,
        preferred_tools=preferred_tool_order,
        max_tool_steps=max_tool_steps,
        stop_after_first_hit=stop_after_first_hit,
        return_meta=True,
    )
    probe_hit, probe_steps = DirectToolProbe._parse_probe_result(probe_result)
    react_tool_budget = max_tool_steps
    if max_tool_steps is not None and max_tool_steps > 0:
        react_tool_budget = max(0, int(max_tool_steps) - max(0, int(probe_steps)))
    if probe_hit:
        # 命中时存储到ThinkingBack以便后续缓存复用
        _store_thinking_back(
            chat_id=chat_id,
            question=chat_history[:200],
            context=context,
            found_answer=True,
            answer=probe_hit,
            thinking_steps=[{"iteration": 0, "method": "direct_tool_probe"}],
        )
        return probe_hit
    if react_tool_budget is not None and react_tool_budget <= 0:
        return None
    if not _has_memory_react_tool_models():
        logger.info("[记忆检索] memory.react 无兼容工具模型，跳过ReAct迭代，避免无工具空转。")
        return None

    # —— 第3阶段：ReAct Agent推理 ——
    question_initial_info = initial_info or ""
    if max_iterations is None:
        max_iterations = global_config.memory.max_agent_iterations
    _safe_chat_history = _adapt_chat_context_for_model(chat_history)
    found_answer, answer, thinking_steps, is_timeout = await _react_agent_solve_question(
        chat_id=chat_id,
        max_iterations=max_iterations,
        timeout=global_config.memory.agent_timeout_seconds,
        initial_info=question_initial_info,
        chat_history=_safe_chat_history,
        allowed_tools=preferred_tool_order,
        max_tool_steps=react_tool_budget,
        stop_after_first_hit=stop_after_first_hit,
    )

    # 存储推理结果到ThinkingBack（超时时除外）
    if not is_timeout:
        _store_thinking_back(
            chat_id=chat_id,
            question=chat_history[:200],
            context=context,
            found_answer=found_answer,
            answer=answer if found_answer else "",
            thinking_steps=thinking_steps,
        )
    else:
        logger.info("ReAct Agent超时，不存储结果")

    sanitized_answer = _sanitize_memory_payload(answer)
    if found_answer and sanitized_answer:
        return sanitized_answer
    return None


async def build_memory_retrieval_prompt(
    message: str,
    sender: str,
    target: str,
    chat_stream,
    think_level: int = 1,
    unknown_words: Optional[List[str]] = None,
    relation_context: str = "",
    extra_context: str = "",
) -> str:
    """构建记忆检索提示
    Args:
        message: 聊天历史记录
        sender: 发送者名称
        target: 目标消息内容
        chat_stream: 聊天流对象
        think_level: 思考深度等级
        unknown_words: Planner 提供的未知词语列表，优先使用此列表而不是从聊天记录匹配
        relation_context: 关系优先级提示（F14：高好感用户优先检索相关记忆）

    Returns:
        str: 记忆检索结果字符串
    """
    start_time = time.time()

    # 构造日志前缀：[聊天流名称]，用于在日志中标识聊天流（优先群名称/用户昵称）
    try:
        group_info = chat_stream.group_info
        user_info = chat_stream.user_info
        # 群聊优先使用群名称
        if group_info is not None and getattr(group_info, "group_name", None):
            stream_name = group_info.group_name.strip() or str(group_info.group_id)
        # 私聊使用用户昵称
        elif user_info is not None and getattr(user_info, "user_nickname", None):
            stream_name = user_info.user_nickname.strip() or str(user_info.user_id)
        # 兜底使用 stream_id
        else:
            stream_name = chat_stream.stream_id
    except Exception:
        stream_name = chat_stream.stream_id
    log_prefix = f"[{stream_name}] " if stream_name else ""

    logger.info(f"{log_prefix}检测是否需要回忆，元消息：{message[:30]}...，消息长度: {len(message)}")
    try:
        chat_id = chat_stream.stream_id
        retrieval_plan = plan_memory_retrieval(message, target, extra_context=extra_context)
        if retrieval_plan.get("should_skip"):
            logger.info(
                f"{log_prefix}记忆检索已跳过: {retrieval_plan.get('reason', 'skip')} | target={target[:40]}"
            )
            return ""

        retrieval_context = str(retrieval_plan.get("query_text", "") or "").strip()
        if not retrieval_context:
            logger.debug(f"{log_prefix}记忆检索计划未生成有效查询文本，直接跳过")
            return ""
        # F14：注入关系优先级提示，影响记忆检索方向
        if relation_context:
            retrieval_context = f"{relation_context}\n{retrieval_context}"

        # 初始阶段：使用 Planner 提供的 unknown_words 进行检索（如果提供）
        initial_info = ""
        if unknown_words and len(unknown_words) > 0:
            # 清理和去重 unknown_words
            cleaned_concepts = _sanitize_unknown_concepts(unknown_words)
            if cleaned_concepts:
                # 对匹配到的概念进行jargon检索，作为初始信息
                concept_info = await retrieve_concepts_with_jargon(cleaned_concepts, chat_id)
                if concept_info:
                    initial_info += concept_info
                    logger.info(
                        f"{log_prefix}使用 Planner 提供的 unknown_words，共 {len(cleaned_concepts)} 个概念，检索结果: {concept_info[:100]}..."
                    )
                else:
                    logger.debug(f"{log_prefix}unknown_words 检索未找到任何结果")

        # 直接使用 ReAct Agent 进行记忆检索（跳过问题生成步骤）
        base_max_iterations = global_config.memory.max_agent_iterations
        max_tool_steps = max(1, int(retrieval_plan.get("max_tool_steps", 1) or 1))
        # 根据think_level调整迭代次数：think_level=1时不变，think_level=0时减半
        if think_level == 0:
            max_iterations = max(1, base_max_iterations // 2)  # 至少为1
        else:
            max_iterations = base_max_iterations
        max_iterations = min(max_iterations, max(2, max_tool_steps + 1))
        timeout_seconds = global_config.memory.agent_timeout_seconds
        logger.debug(
            f"{log_prefix}直接使用 ReAct Agent 进行记忆检索，think_level={think_level}，设置最大迭代次数: {max_iterations}（基础值: {base_max_iterations}），超时时间: {timeout_seconds}秒"
        )

        # 直接调用 ReAct Agent 处理记忆检索
        try:
            result = await _process_memory_retrieval(
                chat_id=chat_id,
                context=retrieval_context,
                initial_info=initial_info,
                max_iterations=max_iterations,
                chat_history=retrieval_context,
                preferred_tools=list(retrieval_plan.get("tool_order", []) or []),
                max_tool_steps=max_tool_steps,
                stop_after_first_hit=bool(retrieval_plan.get("stop_after_first_hit", True)),
            )
        except Exception as e:
            logger.error(f"{log_prefix}处理记忆检索时发生异常: {e}")
            result = None

        end_time = time.time()

        sanitized_result = _sanitize_memory_payload(result)
        max_result_chars = int(retrieval_plan.get("max_result_chars", 0) or 0)
        if sanitized_result:
            # 合并当前结果与近期缓存答案（去重）
            all_findings = _collect_recent_findings(chat_id, sanitized_result, window_seconds=600.0, ceiling=3)
            if all_findings:
                retrieved_memory = _trim_retrieval_findings(all_findings, max_result_chars)
                current_cnt = 1
                cached_cnt = len(all_findings) - current_cnt
                logger.info(
                    f"{log_prefix}记忆检索成功，耗时: {(end_time - start_time):.3f}秒，"
                    f"当前查询 {current_cnt} 条，缓存 {cached_cnt} 条，共 {len(all_findings)} 条"
                )
                return f"你回忆起了以下信息：\n{retrieved_memory}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
            logger.info(f"{log_prefix}记忆检索成功，耗时: {(end_time - start_time):.3f}秒")
            return f"你回忆起了以下信息：\n{sanitized_result}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
        else:
            # 即使当次没查到，也尝试返回近期缓存答案
            cached_only = _collect_recent_findings(chat_id, None, window_seconds=600.0, ceiling=3)
            if cached_only:
                retrieved_memory = _trim_retrieval_findings(cached_only, max_result_chars)
                logger.info(f"{log_prefix}当次未找到新结果，但有 {len(cached_only)} 条近期缓存答案")
                return f"你回忆起了以下信息：\n{retrieved_memory}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
            logger.debug(f"{log_prefix}记忆检索未找到相关信息")
            return ""

    except Exception as e:
        logger.error(f"{log_prefix}记忆检索时发生异常: {str(e)}")
        return ""
