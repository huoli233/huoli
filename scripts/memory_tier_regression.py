# ruff: noqa: E402
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_system.memory_retrieval import (  # noqa: E402
    _filter_memory_payload_for_tier,
    plan_memory_retrieval,
)


def check_memory_tier_contract() -> dict:
    normal = plan_memory_retrieval(
        "用户A：昨天说过刀盾配置\n用户B：换个话题",
        "刀盾怎么弄",
    )
    assert normal["should_skip"] is True
    assert normal["memory_tier"] == "current_context"
    assert normal["reason"] == "current_context_only"
    assert normal["tool_order"] == []

    temporary = plan_memory_retrieval(
        "用户A：我刚刚发了一张图\n机器人：看到了",
        "刚才那个呢",
    )
    assert temporary["should_skip"] is False
    assert temporary["memory_tier"] == "temporary"
    assert temporary["recall_intent"] == "temporary_context_recall"
    assert temporary["allow_long_term"] is False
    assert temporary["tool_order"] == ["query_direct_memory"]

    self_recall = plan_memory_retrieval(
        "机器人：我刚才说了一句短话",
        "你上一句是什么",
    )
    assert self_recall["memory_tier"] == "temporary"
    assert self_recall["recall_intent"] == "self_recall"
    assert self_recall["tool_order"] == ["query_direct_memory"]

    explicit_long_term = plan_memory_retrieval(
        "用户A：今天聊点别的",
        "你还记得我们之前聊过的刀盾吗",
    )
    assert explicit_long_term["should_skip"] is False
    assert explicit_long_term["memory_tier"] == "long_term"
    assert explicit_long_term["recall_intent"] == "explicit_long_term_recall"
    assert explicit_long_term["allow_long_term"] is True
    assert explicit_long_term["tool_order"] == [
        "search_chat_history",
        "query_direct_memory",
        "search_memory_by_embedding",
    ]

    associative = plan_memory_retrieval(
        "用户A：今天状态怎么样",
        "你怎么看",
        extra_context="[关系优先] 用户A是高信任对象，优先检索与TA相关的记忆。",
    )
    assert associative["memory_tier"] == "long_term"
    assert associative["recall_intent"] == "internal_associative_recall"
    assert associative["allow_long_term"] is True

    burned = plan_memory_retrieval(
        "用户A：重复短句",
        "刚才那个呢",
        extra_context="避免复读: 这类内容刚处理过且回应质量足够",
    )
    assert burned["should_skip"] is True
    assert burned["reason"] == "context_burned"

    temporary_payload = _filter_memory_payload_for_tier(
        "30秒前，聊天里提到过：继续这个话题\n"
        "记忆ID：42\n"
        "主题：很久以前的旧主题\n"
        "05-01 12:00，聊天里提到过：旧长期摘要\n"
        "3小时前，聊天里提到过：过时上下文",
        "temporary",
    )
    assert "30秒前" in temporary_payload
    assert "记忆ID" not in temporary_payload
    assert "旧长期摘要" not in temporary_payload
    assert "3小时前" not in temporary_payload

    return {
        "normal_context_skip": True,
        "temporary_only_direct_memory": True,
        "explicit_long_term_enabled": True,
        "internal_associative_recall": True,
        "burned_context_blocks_recall": True,
        "temporary_payload_filtered": True,
    }


def main() -> None:
    print(json.dumps({"memory_tier": check_memory_tier_contract()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()