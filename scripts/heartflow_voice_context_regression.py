# ruff: noqa: E402
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.core_config_engine import CoreSettingsHub
from src.chat.heart_flow.enhanced_modules.scene_planner_bridge_mixin import ScenePlannerBridgeMixin
from src.chat.heart_flow.inner_voice import SelfDialogueEngine
from src.chat.proactive import perception_engine
from src.chat.proactive.perception_engine import AlertGrade, AwarenessEngine, AwarenessVerdict, CrowdVibe
from src.common.data_models.heartflow_models import VoiceVerdict


def check_inner_voice_structured_parse() -> dict:
    engine = SelfDialogueEngine("voice-context-regression")

    field_verdict = engine._extract_structured_reply(
        "思考=这家伙又在玩什么新梗...不过还挺好玩的\n欲望=6\n行动=reply",
        "刀盾",
        "tester",
    )
    assert "新梗" in field_verdict.thinking
    assert "好玩" in field_verdict.thinking
    assert field_verdict.reply_desire_level == 6
    assert field_verdict.next_action == "reply"
    assert field_verdict.thinking_source == "llm_field"
    assert "欲望=" not in field_verdict.thinking
    assert "行动=" not in field_verdict.thinking

    inline_verdict = engine._extract_structured_reply(
        "思考=这家伙又在玩什么新梗...不过还挺好玩的 欲望=6 行动=reply",
        "刀盾",
        "tester",
    )
    assert inline_verdict.thinking == field_verdict.thinking
    assert inline_verdict.next_action == "reply"
    assert inline_verdict.thinking_source == "llm_field"

    leading_number_verdict = engine._extract_structured_reply(
        "6, 思考=这家伙又在玩什么新梗...不过还挺好玩的",
        "刀盾",
        "tester",
    )
    assert "新梗" in leading_number_verdict.thinking
    assert "好玩" in leading_number_verdict.thinking
    assert leading_number_verdict.reply_desire_level == 6
    assert leading_number_verdict.thinking_source == "llm_field"
    assert "思考=" not in leading_number_verdict.thinking

    json_verdict = engine._extract_structured_reply(
        '{"思考":"这句像在试探我","欲望":4,"行动":"wait"}',
        "测试",
        "tester",
    )
    assert json_verdict.thinking == "这句像在试探我"
    assert json_verdict.reply_desire_level == 4
    assert json_verdict.next_action == "wait"
    assert json_verdict.thinking_source == "llm_json"

    fallback_verdict = engine._extract_structured_reply("6", "嗯", "tester")
    assert fallback_verdict.thinking_source == "local_fallback"
    assert fallback_verdict.thinking

    return {
        "field_source": field_verdict.thinking_source,
        "json_source": json_verdict.thinking_source,
        "fallback_source": fallback_verdict.thinking_source,
    }


def check_voice_verdict_source_roundtrip() -> dict:
    verdict = VoiceVerdict(
        thinking="这句像在试探我",
        reply_desire_level=4,
        next_action="wait",
        thinking_source="llm_json",
    )
    payload = verdict.to_dict()
    restored = VoiceVerdict.from_dict(payload)
    assert payload["thinking_source"] == "llm_json"
    assert restored.thinking_source == "llm_json"
    return {"thinking_source": restored.thinking_source}


def check_soul_context_dedupe() -> dict:
    bridge = ScenePlannerBridgeMixin()
    bridge.stream_id = "voice-context-regression"
    bridge.log_prefix = "[regression]"
    duplicated = "\n".join(
        [
            "[当前心理状态] 烦躁度0，回复平静。",
            "[当前心理状态] 烦躁度0，回复平静。",
            "[当前情感状态] 感到平静",
            "[当前情感状态] 感到平静",
        ]
    )
    deduped = bridge._ensure_soul_data_in_extra_info(duplicated)
    summary = bridge._summarize_soul_data_lines(deduped, limit=8)
    assert deduped.count("[当前心理状态] 烦躁度0，回复平静。") == 1
    assert deduped.count("[当前情感状态] 感到平静") == 1
    assert summary == [
        "[当前心理状态] 烦躁度0，回复平静。",
        "[当前情感状态] 感到平静",
    ]
    return {"soul_lines": len(summary)}


def check_awareness_log_dedupe() -> dict:
    if CoreSettingsHub._sole_ref is None:
        CoreSettingsHub.boot(ROOT / "config")
    engine = AwarenessEngine()
    verdict = AwarenessVerdict(
        alert_grade=AlertGrade.FAINT,
        engagement_pull=0.29,
        inference_source="rule",
        crowd_vibe=CrowdVibe.SPARSE,
    )
    info_messages = []
    debug_messages = []
    original_info = perception_engine.logger.info
    original_debug = perception_engine.logger.debug
    perception_engine.logger.info = info_messages.append
    perception_engine.logger.debug = debug_messages.append
    try:
        engine._log_awareness_verdict("a7807ea1-channel", "规则判定", verdict, CrowdVibe.SPARSE)
        engine._log_awareness_verdict("a7807ea1-channel", "规则判定", verdict, CrowdVibe.SPARSE)
        engine._last_log_ts -= 31.0
        engine._log_awareness_verdict("a7807ea1-channel", "规则判定", verdict, CrowdVibe.SPARSE)
    finally:
        perception_engine.logger.info = original_info
        perception_engine.logger.debug = original_debug

    assert len(info_messages) == 1
    assert len(debug_messages) == 2
    assert "重复快照已合并" in debug_messages[0]
    return {"info_logs": len(info_messages), "debug_logs": len(debug_messages)}


def main() -> None:
    results = {
        "inner_voice_structured_parse": check_inner_voice_structured_parse(),
        "voice_verdict_source_roundtrip": check_voice_verdict_source_roundtrip(),
        "soul_context_dedupe": check_soul_context_dedupe(),
        "awareness_log_dedupe": check_awareness_log_dedupe(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
