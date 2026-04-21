from typing import Any, Dict, List, Optional


def classify_behavior_signal(
    text: str,
    user_id: str = "",
    interaction_count: int = 0,
    repetition_signal: Optional[Dict[str, Any]] = None,
    harassment_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "category": "neutral",
        "behavior_type": "casual_chat",
        "intent": "chat",
        "severity": 0.25,
        "is_new_user": interaction_count < 10,
        "reason": "默认中性互动",
    }

    lowered = (text or "").lower()
    if result["is_new_user"]:
        result["reason"] = "新用户默认宽松观察"

    insult_markers: List[str] = [
        "傻逼",
        "滚",
        "闭嘴",
        "恶心",
        "去死",
        "脑残",
        "废物",
    ]
    hostile_markers: List[str] = [
        "烦",
        "别说了",
        "讨厌",
        "无聊",
        "差劲",
        "有病",
    ]
    friendly_markers: List[str] = [
        "谢谢",
        "喜欢",
        "棒",
        "厉害",
        "可爱",
        "辛苦",
        "感谢",
    ]
    question_markers: List[str] = [
        "?",
        "？",
        "怎么",
        "为什么",
        "什么",
        "吗",
        "是不是",
    ]

    if harassment_signal and harassment_signal.get("detected"):
        severity = 0.95 if harassment_signal.get("severity") == "high" else 0.8
        result.update(
            {
                "category": "harassing",
                "behavior_type": "repetitive_harassment",
                "intent": "harass",
                "severity": severity,
                "reason": harassment_signal.get("reason", "持续骚扰/冒犯输入"),
            }
        )
        return result

    if any(marker in lowered for marker in insult_markers):
        result.update(
            {
                "category": "hostile",
                "behavior_type": "direct_insult",
                "intent": "insult",
                "severity": 0.9,
                "reason": "检测到直接攻击性表达",
            }
        )
        return result

    if (
        repetition_signal
        and repetition_signal.get("detected")
        and repetition_signal.get("low_info_cluster")
    ):
        _exact_repeats = int(repetition_signal.get("exact_repeat_count", 0) or 0)
        # 2次重复仅标记为中性警告，3次及以上才判定为不友好
        if _exact_repeats >= 3:
            result.update(
                {
                    "category": "unfriendly",
                    "behavior_type": "low_info_repeat",
                    "intent": "pressure",
                    "severity": 0.65,
                    "reason": repetition_signal.get("reason", "低信息重复施压"),
                }
            )
            return result
        else:
            result.update(
                {
                    "category": "neutral",
                    "behavior_type": "mild_repeat",
                    "intent": "chat",
                    "severity": 0.35,
                    "reason": f"低信息重复{_exact_repeats}次，暂不升级为不友好",
                }
            )
            return result

    if any(marker in lowered for marker in hostile_markers):
        result.update(
            {
                "category": "unfriendly",
                "behavior_type": "hostile_remark",
                "intent": "hostile",
                "severity": 0.7,
                "reason": "检测到明显负向表达",
            }
        )
        return result

    if any(marker in lowered for marker in friendly_markers):
        result.update(
            {
                "category": "friendly",
                "behavior_type": "friendly_chat",
                "intent": "friendly",
                "severity": 0.55,
                "reason": "检测到正向友好表达",
            }
        )
        return result

    if any(marker in lowered for marker in question_markers):
        result.update(
            {
                "category": "neutral",
                "behavior_type": "casual_inquiry",
                "intent": "inquiry",
                "severity": 0.4,
                "reason": "以提问为主的普通互动",
            }
        )
        return result

    return result
