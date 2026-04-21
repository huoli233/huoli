import difflib
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("学习工具")


def filter_message_content(content: Optional[str]) -> str:
    """过滤消息内容，移除回复、@、图片等格式"""
    if not content:
        return ""
    content = re.sub(r"\[回复.*?\]，说：\s*", "", content)
    content = re.sub(r"@<[^>]*>", "", content)
    content = re.sub(r"\[picid:[^\]]*\]", "", content)
    content = re.sub(r"\[表情包：[^\]]*\]", "", content)
    return content.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度，返回0-1之间的值"""
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def calculate_style_similarity(style1: str, style2: str) -> float:
    """计算两个 style 的相似度，返回0-1之间的值"""
    if not style1 or not style2:
        return 0.0

    def remove_ignored_words(text: str) -> str:
        text = text.replace("使用", "")
        text = text.replace("句式", "")
        return text.strip()

    cleaned_style1 = remove_ignored_words(style1)
    cleaned_style2 = remove_ignored_words(style2)
    if not cleaned_style1 or not cleaned_style2:
        return 0.0
    return difflib.SequenceMatcher(
        None, cleaned_style1, cleaned_style2
    ).ratio()


def _compute_weights(population: List[Dict]) -> List[float]:
    """根据表达的count计算权重，范围限定在1~5之间"""
    if not population:
        return []
    counts = []
    for item in population:
        count = item.get("count", 1)
        try:
            count_value = float(count)
        except (TypeError, ValueError):
            count_value = 1.0
        counts.append(max(count_value, 0.0))
    min_count = min(counts)
    max_count = max(counts)
    if max_count == min_count:
        return [1.0 for _ in counts]
    weights = []
    for count_value in counts:
        normalized = (count_value - min_count) / (max_count - min_count)
        weights.append(1.0 + normalized * 4.0)
    return weights


def weighted_sample(population: List[Dict], k: int) -> List[Dict]:
    """加权随机抽样函数"""
    if not population or k <= 0:
        return []
    if len(population) <= k:
        return population.copy()
    selected: List[Dict] = []
    population_copy = population.copy()
    for _ in range(min(k, len(population_copy))):
        weights = _compute_weights(population_copy)
        total_weight = sum(weights)
        if total_weight <= 0:
            idx = random.randint(0, len(population_copy) - 1)
            selected.append(population_copy.pop(idx))
            continue
        threshold = random.uniform(0, total_weight)
        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                selected.append(population_copy.pop(idx))
                break
    return selected


def parse_chat_id_list(chat_id_value: Any) -> List[List[Any]]:
    """解析chat_id字段，兼容旧格式（字符串）和新格式（JSON列表）"""
    if not chat_id_value:
        return []
    if isinstance(chat_id_value, str):
        try:
            parsed = json.loads(chat_id_value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, str):
                return [[parsed, 1]]
            else:
                return [[str(chat_id_value), 1]]
        except (json.JSONDecodeError, TypeError):
            return [[str(chat_id_value), 1]]
    elif isinstance(chat_id_value, list):
        return chat_id_value
    else:
        return [[str(chat_id_value), 1]]


def update_chat_id_list(
    chat_id_list: List[List[Any]],
    target_chat_id: str,
    increment: int = 1,
) -> List[List[Any]]:
    """更新chat_id列表，如果target_chat_id已存在则增加计数，否则添加新条目"""
    item = _find_chat_id_item(chat_id_list, target_chat_id)
    if item is not None:
        if len(item) >= 2:
            item[1] = (
                item[1] if isinstance(item[1], (int, float)) else 0
            ) + increment
        else:
            item.append(increment)
    else:
        chat_id_list.append([target_chat_id, increment])
    return chat_id_list


def _find_chat_id_item(
    chat_id_list: List[List[Any]],
    target_chat_id: str,
) -> Optional[List[Any]]:
    """在chat_id列表中查找匹配的项"""
    for item in chat_id_list:
        if (
            isinstance(item, list)
            and len(item) >= 1
            and str(item[0]) == str(target_chat_id)
        ):
            return item
    return None


def chat_id_list_contains(
    chat_id_list: List[List[Any]], target_chat_id: str
) -> bool:
    """检查chat_id列表中是否包含指定的chat_id"""
    return _find_chat_id_item(chat_id_list, target_chat_id) is not None


def contains_bot_self_name(
    content: str, bot_nickname: str = "", bot_alias_names: List[str] = None
) -> bool:
    """判断词条是否包含机器人的昵称或别名"""
    if not content:
        return False
    target = content.strip().lower()
    nickname = str(bot_nickname or "").strip().lower()
    alias_names = [
        str(alias or "").strip().lower() for alias in (bot_alias_names or [])
    ]
    candidates = [name for name in [nickname, *alias_names] if name]
    return any(name in target for name in candidates)


def build_context_paragraph(
    messages: List[Any], center_index: int
) -> Optional[str]:
    """构建包含中心消息上下文的段落（前3条+后3条）"""
    if not messages or center_index < 0 or center_index >= len(messages):
        return None
    context_start = max(0, center_index - 3)
    context_end = min(len(messages), center_index + 1 + 3)
    context_messages = messages[context_start:context_end]
    if not context_messages:
        return None
    try:
        from src.chat.utils.chat_message_builder import build_readable_messages

        paragraph = build_readable_messages(
            messages=context_messages,
            read_mark=0.0,
            truncate=False,
            show_actions=False,
            show_pic=True,
            remove_emoji_stickers=False,
            pic_single=True,
        )
    except Exception as e:
        logger.warning(f"构建上下文段落失败: {e}")
        return None
    paragraph = paragraph.strip()
    return paragraph or None


def is_bot_message(msg: Any, bot_platforms: Dict[str, str] = None) -> bool:
    """判断消息是否来自机器人自身"""
    if msg is None:
        return False
    if not bot_platforms:
        return False
    platform = str(getattr(msg, "user_platform", "") or "").strip().lower()
    user_id = str(getattr(msg, "user_id", "") or "").strip()
    if not platform or not user_id:
        return False
    bot_account = bot_platforms.get(platform)
    return bool(bot_account and user_id == bot_account)


def parse_expression_response(
    response: str,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """解析 LLM 返回的表达风格总结和黑话 JSON，提取两个列表"""
    if not response:
        return [], []
    raw = response.strip()
    json_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_block_pattern, raw, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()
    parsed = None
    expressions: List[Tuple[str, str, str]] = []
    jargon_entries: List[Tuple[str, str]] = []
    try:
        if raw.startswith("[") and raw.endswith("]"):
            parsed = json.loads(raw)
        else:
            parsed = _repair_json(raw)
    except Exception as parse_error:
        try:
            fixed_raw = _fix_chinese_quotes_in_json(raw)
            if fixed_raw.startswith("[") and fixed_raw.endswith("]"):
                parsed = json.loads(fixed_raw)
            else:
                parsed = _repair_json(fixed_raw)
        except Exception as fix_error:
            logger.error(f"解析表达风格 JSON 失败: {parse_error}")
            logger.error(f"修复后仍失败: {fix_error}")
            return [], []
    if isinstance(parsed, dict):
        parsed_list = [parsed]
    elif isinstance(parsed, list):
        parsed_list = parsed
    else:
        logger.error(f"表达风格解析结果类型异常: {type(parsed)}")
        return [], []
    for item in parsed_list:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        source_id = str(item.get("source_id", "")).strip()
        if situation and style and source_id:
            expressions.append((situation, style, source_id))
        elif item.get("content"):
            content = str(item.get("content", "")).strip()
            source_id = str(item.get("source_id", "")).strip()
            if content and source_id:
                jargon_entries.append((content, source_id))
    return expressions, jargon_entries


def _repair_json(text: str) -> Any:
    """修复并解析 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("JSON修复后仍解析失败，返回空字典")
    text = text.replace("'", '"')
    text = text.replace(",]", "]")
    text = text.replace(",}", "}")
    brace_count = text.count("{") - text.count("}")
    bracket_count = text.count("[") - text.count("]")
    if brace_count > 0:
        text += "}" * brace_count
    if bracket_count > 0:
        text += "]" * bracket_count
    return json.loads(text)


def _fix_chinese_quotes_in_json(text: str) -> str:
    """使用状态机修复 JSON 字符串值中的中文引号"""
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        char = text[i]
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        if char == "\\":
            result.append(char)
            escape_next = True
            i += 1
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        if in_string:
            if char == '"':
                result.append('\\"')
            elif char == '"':
                result.append('\\"')
            else:
                result.append(char)
        else:
            result.append(char)
        i += 1
    return "".join(result)


def parse_platform_accounts(platforms: List[Dict]) -> Dict[str, str]:
    """解析平台账号配置"""
    result = {}
    if not platforms:
        return result
    for platform_config in platforms:
        if isinstance(platform_config, dict):
            platform = str(platform_config.get("platform", "")).lower()
            account = str(platform_config.get("account", ""))
            if platform and account:
                result[platform] = account
    return result
