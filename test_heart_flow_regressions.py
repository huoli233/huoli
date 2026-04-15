import asyncio
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from src.chat.heart_flow.heartFC_chat_enhanced import (
    BehaviorGovernorVerdict,
    EnhancedHeartFChatting,
    ModelGovernorVerdict,
    RestGovernorVerdict,
)
from src.chat.heart_flow.persona_engine import CharacterFoundry, SlotRecord
from src.common.data_models.heartflow_models import (
    BehaviorIntent,
    DormantSetup,
    FlowPhase,
    FlowStateContainer,
    UnifiedFlowSnapshot,
    VoiceVerdict,
)
from src.common.data_models.message_data_model import (
    ForwardNode,
    ReplyContent,
    ReplyContentType,
    ReplySetModel,
)
from src.common.data_models.proactive_models import (
    DeliveryPayload,
    DialogueStage,
    ExecutionReport,
    JournalKind,
    MindJournalEntry,
    SessionContext,
    ThinkOutcome,
    TriggerSignal,
)
from src.chat.heart_flow.reply_coordinator import ReplyCoordinator
from src.chat.heart_flow.heartflow_decision import ThreatAssessor
from src.chat.heart_flow.inner_voice import SelfDialogueEngine
from src.chat.proactive.proactive_integration_hub import ProactiveIntegrationHub
from src.chat.proactive.session_tracker import DialogueMemoir
from src.chat.proactive.silence_watcher import QuietPeriodMonitor
from src.chat.heart_flow.fondness_trust import FondnessTrustDimension, UserRelationState
from src.core.self_reply_recognizer import MessageSource, SelfReplyRecognizer
from src.core.subjective_attention_flow import (
    AttentionFlowSnapshot,
    AttentionState,
    AttentionTransitionReason,
)
from src.core.impression_evolution_hub import ImpressionEvolutionHub
from src.core.state_dashboard import StateDashboardEngine
from src.core.planner_prompt_injection import PlannerPromptInjectionLayer
from src.core.state_coupling_matrix import StateCouplingMatrixEngine
from src.core.unified_planner import UnifiedPlanner
from src.core.world_snapshot import (
    TargetUserState,
    WorldSnapshot,
    _collect_emotion_state_sync,
    _reconcile_overlapping_values,
)
from src.core.adaptive_threshold_learner_v2 import AdaptiveThresholdLearnerV2
from src.config.config import global_config
from src.modules.brain.decision_brain import DecisionBrain, DecisionContext, GroupDecision, RiskPredictionSystem
from src.modules.modcore.dynamic_persona.emotion_database import EmotionDatabase
from src.modules.modcore.dynamic_persona.emotion_tracker import EmotionTracker, UserEmotionState
from src.modules.modcore.psychological_core import PsychologicalCore
from src.modules.modcore.social_cognition.relationship_controller import RelationshipController
from src.memory_system.memory_retrieval import (
    DirectToolProbe,
    build_memory_retrieval_prompt,
    plan_memory_retrieval,
    _process_memory_retrieval,
)
from src.memory_system.memory_core import _score_memory_query_match
from src.memory_system.retrieval_tools.query_chat_history import _score_record_relevance
from src.memory_system.retrieval_tools.query_direct_memory import query_direct_memory


class DummyOrch:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, key):
        return self.mapping[key]


class FakeLearningHub:
    def __init__(self):
        self.observed = []
        self.used = []

    def observe(self, event):
        self.observed.append(event)

    def record_usage(self, name, success=True):
        self.used.append((name, success))


class FakeSkillHub:
    def __init__(self):
        self.calls = []

    def record_invocation(self, name, success=True):
        self.calls.append((name, success))


class FakeImpressionHub:
    def __init__(self):
        self.calls = []

    def record_interaction(self, user_id, **kwargs):
        self.calls.append(("interaction", user_id, kwargs))
        truth = SimpleNamespace(
            positive_interactions=2 if kwargs.get("is_positive") else 0,
            shared_experience_count=0,
        )
        return SimpleNamespace(truth=truth)

    def record_conflict(self, user_id, reason=""):
        self.calls.append(("conflict", user_id, {"reason": reason}))

    def evolve_with_group_context(self, user_id, atmo, density, pat_type):
        self.calls.append(
            (
                "evolve",
                user_id,
                {"atmo": atmo, "density": density, "pat_type": pat_type},
            )
        )


class FakeEmotionTracker:
    def __init__(self):
        self.affection_updates = []

    def get_user_state(self, user_id):
        return SimpleNamespace(affection=0.0)

    def update_affection(self, user_id, delta, reason):
        self.affection_updates.append((user_id, delta, reason))

    def get_layered_response_mode(self, user_id):
        return {"tone": "neutral", "response_length": "normal"}


class FakeLedger:
    def __init__(self, ack_value=False):
        self.ack_value = ack_value
        self.ack_calls = []
        self.sweep_calls = 0
        self.register_calls = []

    def acknowledge_user_reply(self, *args, **kwargs):
        self.ack_calls.append((args, kwargs))
        return self.ack_value

    def sweep_expired(self):
        self.sweep_calls += 1
        return 0

    def register_proactive_fire(self, channel_id, source_intent_id=""):
        self.register_calls.append((channel_id, source_intent_id))


class FakeDecider:
    def __init__(self, ledger):
        self.ledger = ledger


class FakeIntegrationHub:
    def __init__(self):
        self.user_calls = []
        self.outcome_calls = []
        self.bot_calls = []

    def on_user_message_received(self, channel_id, user_id="", content=""):
        self.user_calls.append((channel_id, user_id, content))

    def on_proactive_outcome(self, **kwargs):
        self.outcome_calls.append(kwargs)
        return SimpleNamespace(outcome_tag="reply", reward_delta=1.0)

    def on_bot_message_sent(self, channel_id, user_id="", was_proactive=False, content=""):
        self.bot_calls.append((channel_id, user_id, was_proactive, content))


class FakeSelfAwareness:
    def __init__(self):
        self.message_calls = []
        self.action_calls = []
        self.event_calls = []
        self.recent_messages = []
        self.recent_actions = []
        self.recent_events = []
        self.stats = {"stream_id": "demo-stream", "total_messages": 0}

    def record_message(self, **kwargs):
        self.message_calls.append(kwargs)

    def record_action(self, **kwargs):
        self.action_calls.append(kwargs)

    def record_user_event(self, **kwargs):
        self.event_calls.append(kwargs)

    def get_recent_messages(self, stream_id, limit=10):
        return list(self.recent_messages[:limit])

    def get_recent_actions(self, stream_id, limit=10):
        return list(self.recent_actions[:limit])

    def get_recent_user_events(self, stream_id, limit=10):
        return list(self.recent_events[:limit])

    def get_interaction_stats(self, stream_id):
        return dict(self.stats)


class FakeDecisionMaker:
    def __init__(self):
        self.calls = []

    async def evaluate_restraint_from_state(self, stream_id, **kwargs):
        self.calls.append((stream_id, kwargs))
        return {
            "recent_reply_burst": kwargs.get("recent_reply_burst"),
            "source": kwargs.get("source"),
        }


class FakeOutcomeLedger:
    def __init__(self):
        self.resolve_pending_calls = []
        self.resolve_with_consequence_calls = []

    def resolve_pending(self, *args, **kwargs):
        self.resolve_pending_calls.append((args, kwargs))
        return 0

    def resolve_pending_with_consequence(self, channel_id, consequence, target_user=""):
        self.resolve_with_consequence_calls.append((channel_id, consequence, target_user))
        return 1


class FakeMemoirCabinet:
    def __init__(self):
        self.lookup_sync_calls = []
        self.lookup_by_channel_calls = []

    def lookup_sync(self, key):
        self.lookup_sync_calls.append(key)
        return None

    def lookup_by_channel(self, channel_id):
        self.lookup_by_channel_calls.append(channel_id)
        return SimpleNamespace(
            consecutive_timeouts=3,
            phase="open",
            exchange_tally=7,
            last_user_spoke_at=123.0,
            is_waiting=lambda: True,
            born_at=99.0,
        )


def make_chat_for_learning(impression_hub):
    chat = EnhancedHeartFChatting.__new__(EnhancedHeartFChatting)
    chat.stream_id = "demo-stream"
    chat.log_prefix = "[test]"
    chat._last_user_id = "user-1"
    chat._engine_orch = DummyOrch(
        {
            "impression_hub": impression_hub,
            "learning_hub": FakeLearningHub(),
            "skill_hub": FakeSkillHub(),
        }
    )
    chat._cached_scene_snapshot = None
    chat._cached_participant_summary = None
    chat._cached_pattern_evidence = []
    chat._cached_learning_summary = {"items": ["学习项"]}
    return chat


def make_minimal_chat():
    chat = EnhancedHeartFChatting.__new__(EnhancedHeartFChatting)
    chat.stream_id = "demo-stream"
    chat.log_prefix = "[test]"
    chat._last_voice_ts = 0.0
    chat._last_proactive_voice_ts = 0.0
    chat._bot_reply_counter = 0
    chat._bot_reply_timeline = []
    chat._proactive_reply_timeline = []
    chat._reactive_reply_timeline = []
    chat._unanswered_bot_turns = 0
    chat._last_bot_reply_ts = 0.0
    chat.last_active_time = 0.0
    chat._legacy_constraint_hits = 2
    chat._hourly_window_start = 0.0
    chat._hourly_reply_count = 0
    chat._hourly_proactive_reply_count = 0
    chat._night_soft_wake_info = {"action": "soft_wake"}
    chat._last_user_id = ""
    chat._last_proactive_intent_id = "intent-1"
    chat._recent_bot_utterances = []
    chat._last_decision_incoming_messages = []
    chat._last_decision_context_messages = []
    chat._last_decision_self_messages = []
    chat._freshness_engine_initialized = False
    chat._cached_deep_visibility_results = []
    chat._cached_scene_snapshot = None
    chat._last_group_context_signal = {}
    chat._last_group_sense = None
    chat._cached_pattern_evidence = []
    chat._cached_presence_state = None
    chat._cached_metabolism_state = None
    chat._cached_awareness = None
    chat._last_model_governor_verdict = ModelGovernorVerdict()
    chat._model_large_last_ts = 0.0
    chat._model_large_hour_window_start = 0.0
    chat._model_large_proactive_hour_calls = 0
    chat._update_dynamic_ratio = lambda now: None
    chat._spawn = lambda coro: coro.close()
    chat._resolve_relation_view = lambda: {}
    chat._chatterbox_penalty = 0.0
    chat._consecutive_speaks = 0.0
    chat._chat_energy = 88.0
    chat._thinking_energy = 77.0
    return chat


class HeartFlowRegressionTests(unittest.TestCase):
    def test_world_snapshot_profile_summary_keeps_negative_metrics(self):
        user = TargetUserState(
            custom_label="冷场用户",
            relationship_level=1,
            social_value=-12.0,
            trust_value=-5.0,
            annoyance_value=8.0,
            mood="压抑",
        )
        summary = user.profile_summary()
        self.assertIn("群社交度=-12.0", summary)
        self.assertIn("群信任度=-5.0", summary)
        self.assertIn("群烦躁度=8.0", summary)

    def test_world_snapshot_reconcile_does_not_drop_negative_trust(self):
        snap = WorldSnapshot()
        snap.target_user.social_value = -10.0
        snap.target_user.favorability = 0.0
        snap.target_user.trust_score = -7.0
        snap.target_user.trust_value = 0.0
        _reconcile_overlapping_values(snap)
        self.assertEqual(snap.target_user.favorability, -10.0)
        self.assertEqual(snap.target_user.trust_value, 0.0)
        self.assertEqual(snap.target_user.trust_score, -7.0)

    def test_world_snapshot_to_rapport_dict_uses_canonical_relation_fields(self):
        snap = WorldSnapshot(
            target_user=TargetUserState(
                favorability=42.0,
                affection=-12.0,
                trust_score=75.0,
                trust_value=-8.0,
                social_value=5.0,
            )
        )
        rapport = snap.to_rapport_dict()
        self.assertEqual(rapport["relation_contract_version"], 2)
        self.assertEqual(rapport["unified_source"], "world_snapshot")
        self.assertEqual(rapport["affection"], -12.0)
        self.assertEqual(rapport["trust"], -8.0)
        self.assertEqual(rapport["favorability"], 42.0)
        self.assertEqual(rapport["trust_score"], 75.0)

    def test_base_data_model_model_copy_supports_update(self):
        snapshot = UnifiedFlowSnapshot(
            channel_id="demo-stream",
            phase=FlowPhase.STANDBY.value,
            watch_state="peek",
            blocker="旧状态",
        )
        updated = snapshot.model_copy(
            update={
                "phase": FlowPhase.ENGAGED.value,
                "watch_state": "engaged",
                "blocker": "",
            }
        )
        self.assertEqual(snapshot.phase, FlowPhase.STANDBY.value)
        self.assertEqual(updated.phase, FlowPhase.ENGAGED.value)
        self.assertEqual(updated.watch_state, "engaged")
        self.assertEqual(updated.blocker, "")

    def test_base_data_model_copy_supports_deep_update(self):
        snapshot = UnifiedFlowSnapshot(
            channel_id="demo-stream",
            phase=FlowPhase.STANDBY.value,
            watch_state="peek",
            blocker="旧状态",
        )
        updated = snapshot.copy(update={"final_decision": "reply"}, deep=True)
        self.assertEqual(snapshot.final_decision, "")
        self.assertEqual(updated.final_decision, "reply")

    def test_base_data_model_model_dump_supports_common_filters(self):
        snapshot = UnifiedFlowSnapshot(
            channel_id="demo-stream",
            phase=FlowPhase.STANDBY.value,
            watch_state="peek",
            blocker="旧状态",
            voice_should_reply=None,
        )
        payload = snapshot.model_dump(
            include={"channel_id", "watch_state", "voice_should_reply", "blocker"},
            exclude={"blocker"},
            exclude_none=True,
        )
        self.assertEqual(
            payload,
            {
                "channel_id": "demo-stream",
                "watch_state": "peek",
            },
        )

    def test_base_data_model_dict_alias_matches_model_dump(self):
        snapshot = UnifiedFlowSnapshot(
            channel_id="demo-stream",
            phase=FlowPhase.STANDBY.value,
            watch_state="peek",
            blocker="旧状态",
            voice_should_reply=None,
        )
        self.assertEqual(
            snapshot.dict(exclude={"blocker"}, exclude_none=True),
            snapshot.model_dump(exclude={"blocker"}, exclude_none=True),
        )

    def test_base_data_model_from_dict_restores_enum_and_nested_list_models(self):
        context = SessionContext.from_dict(
            {
                "channel_id": "demo-stream",
                "stage": "anticipating",
                "journal": [
                    {
                        "kind": "bot_verdict",
                        "written_at": 12.0,
                        "body_text": "继续观察",
                        "speaker_id": "bot",
                    }
                ],
            }
        )
        self.assertEqual(context.stage, DialogueStage.ANTICIPATING)
        self.assertEqual(len(context.journal), 1)
        self.assertIsInstance(context.journal[0], MindJournalEntry)
        self.assertEqual(context.journal[0].kind, JournalKind.BOT_VERDICT)

    def test_base_data_model_from_dict_restores_optional_and_list_models(self):
        report = ExecutionReport.from_dict(
            {
                "cycle_id": "cycle-1",
                "channel_id": "demo-stream",
                "trigger": {
                    "signal_tag": "mention",
                    "confidence": 0.8,
                    "source_module": "detector",
                    "created_at": 1.0,
                },
                "think_result": {
                    "task_id": "task-1",
                    "success": True,
                    "action_tag": "reply",
                    "parsed_thought": "可以接话",
                },
                "deliveries": [
                    {
                        "channel_id": "demo-stream",
                        "content_text": "收到",
                        "is_proactive": False,
                    }
                ],
            }
        )
        self.assertIsInstance(report.trigger, TriggerSignal)
        self.assertEqual(report.trigger.signal_tag, "mention")
        self.assertIsInstance(report.think_result, ThinkOutcome)
        self.assertEqual(report.think_result.action_tag, "reply")
        self.assertEqual(len(report.deliveries), 1)
        self.assertIsInstance(report.deliveries[0], DeliveryPayload)
        self.assertEqual(report.deliveries[0].content_text, "收到")

    def test_reply_set_model_from_dict_restores_nested_reply_objects(self):
        payload = {
            "reply_data": [
                {"content_type": "text", "content": "hi"},
                {
                    "content_type": "hybrid",
                    "content": [
                        {"content_type": "text", "content": "a"},
                        {"content_type": "image", "content": "b"},
                    ],
                },
                {
                    "content_type": "forward",
                    "content": [
                        {
                            "user_id": "1",
                            "user_nickname": "Alice",
                            "content": [
                                {
                                    "content_type": "text",
                                    "content": "nested",
                                }
                            ],
                        },
                        {
                            "user_id": "",
                            "user_nickname": "",
                            "content": "msg-123",
                        },
                    ],
                },
            ]
        }

        reply_set = ReplySetModel.from_dict(payload)

        self.assertEqual(len(reply_set.reply_data), 3)
        self.assertIsInstance(reply_set.reply_data[0], ReplyContent)
        self.assertEqual(reply_set.reply_data[0].content_type, ReplyContentType.TEXT)
        self.assertIsInstance(reply_set.reply_data[1].content[0], ReplyContent)
        self.assertEqual(
            reply_set.reply_data[1].content[1].content_type,
            ReplyContentType.IMAGE,
        )
        self.assertIsInstance(reply_set.reply_data[2].content[0], ForwardNode)
        self.assertIsInstance(
            reply_set.reply_data[2].content[0].content[0],
            ReplyContent,
        )
        self.assertEqual(reply_set.reply_data[2].content[1].content, "msg-123")

    def test_reply_content_roundtrip_preserves_forward_and_hybrid_shapes(self):
        reply = ReplyContent.construct_as_forward(
            [
                ForwardNode.construct_as_created_node(
                    "1",
                    "Alice",
                    [ReplyContent.construct_as_text("nested")],
                ),
                ForwardNode.construct_as_id_reference("msg-456"),
            ]
        )
        hybrid = ReplyContent.construct_as_hybrid(
            [
                (ReplyContentType.TEXT, "hello"),
                (ReplyContentType.EMOJI, ":)"),
            ]
        )

        restored_reply = ReplyContent.from_dict(reply.to_dict())
        restored_hybrid = ReplyContent.from_dict(hybrid.to_dict())

        self.assertEqual(restored_reply.content_type, ReplyContentType.FORWARD)
        self.assertIsInstance(restored_reply.content[0], ForwardNode)
        self.assertEqual(
            restored_reply.content[0].content[0].content_type,
            ReplyContentType.TEXT,
        )
        self.assertEqual(restored_reply.content[1].content, "msg-456")
        self.assertEqual(restored_hybrid.content_type, ReplyContentType.HYBRID)
        self.assertTrue(
            all(isinstance(item, ReplyContent) for item in restored_hybrid.content)
        )
        self.assertEqual(
            [item.content_type for item in restored_hybrid.content],
            [ReplyContentType.TEXT, ReplyContentType.EMOJI],
        )

    def test_flow_state_container_roundtrip_preserves_full_dormant_setup(self):
        state = FlowStateContainer(
            channel_ref="demo-stream",
            phase=FlowPhase.DORMANT,
            dormant_setup=DormantSetup(
                cause="late_night",
                duration_sec=180,
                entered_at=12.0,
                skip_streak=3,
                glance_probability=0.42,
                allow_awakening=True,
                rest_intensity=0.87,
                wake_drive=0.16,
                regret_chance=0.22,
                min_rest_sec=45,
                last_reconsider_at=18.0,
            ),
        )

        restored = FlowStateContainer.from_dict(state.to_dict())

        self.assertEqual(restored.phase, FlowPhase.DORMANT)
        self.assertEqual(restored.dormant_setup.cause, "late_night")
        self.assertEqual(restored.dormant_setup.skip_streak, 3)
        self.assertAlmostEqual(restored.dormant_setup.glance_probability, 0.42)
        self.assertTrue(restored.dormant_setup.allow_awakening)
        self.assertAlmostEqual(restored.dormant_setup.rest_intensity, 0.87)
        self.assertAlmostEqual(restored.dormant_setup.wake_drive, 0.16)
        self.assertAlmostEqual(restored.dormant_setup.regret_chance, 0.22)
        self.assertEqual(restored.dormant_setup.min_rest_sec, 45)
        self.assertEqual(restored.dormant_setup.last_reconsider_at, 18.0)

    def test_voice_verdict_roundtrip_preserves_invalid_state_and_error(self):
        verdict = VoiceVerdict(
            thinking="解析失败",
            should_reply=False,
            intents=[BehaviorIntent(intent_type="observe", priority=1)],
            is_valid=False,
            error_detail="llm_empty",
            dominant_unfinished_intent={"type": "observe"},
            comprehension_confidence=0.2,
        )

        restored = VoiceVerdict.from_dict(verdict.to_dict())

        self.assertFalse(restored.is_valid)
        self.assertEqual(restored.error_detail, "llm_empty")
        self.assertEqual(restored.dominant_unfinished_intent, {"type": "observe"})
        self.assertEqual(len(restored.intents), 1)
        self.assertIsInstance(restored.intents[0], BehaviorIntent)

    def test_slot_record_from_dict_coerces_dirty_snapshot_values(self):
        record = SlotRecord.from_dict(
            {
                "slot_id": "slot-1",
                "prompt_shard": "hello",
                "source_tag": None,
                "trait_labels": "warm",
                "birth_ts": "broken",
                "last_touch_ts": "",
                "visit_count": "oops",
            }
        )

        self.assertEqual(record.slot_id, "slot-1")
        self.assertEqual(record.source_tag, "dynamic")
        self.assertEqual(record.trait_labels, ["warm"])
        self.assertEqual(record.visit_count, 0)
        self.assertGreater(record.birth_ts, 0.0)
        self.assertGreater(record.last_touch_ts, 0.0)

    def test_character_foundry_restore_snapshot_tolerates_dirty_entries(self):
        foundry = CharacterFoundry.__new__(CharacterFoundry)
        foundry._roster = {}
        foundry._active_slot_id = "baseline"
        foundry._notes = []
        foundry._conf_cache = {}
        foundry._conf_ts = 0.0
        foundry._last_nonbaseline_ts = 0.0
        foundry._gradient = SimpleNamespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = os.path.join(tmpdir, "persona_pool.json")
            foundry._snapshot_dir = snapshot_path
            with open(snapshot_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "active_id": "slot-good",
                        "roster": [
                            {
                                "slot_id": "slot-good",
                                "prompt_shard": "good",
                                "trait_labels": "gentle",
                                "visit_count": "bad",
                            },
                            "broken-row",
                        ],
                        "notes": [
                            {
                                "note_id": "note-1",
                                "slot_id": "slot-good",
                                "user_ref": "u1",
                                "scene_desc": "ok",
                                "effect_rating": "bad",
                                "retention_strength": "0.8",
                                "boost_tally": "oops",
                            }
                        ],
                    },
                    fh,
                    ensure_ascii=False,
                )

            foundry._restore_snapshot()

        self.assertIn("slot-good", foundry._roster)
        self.assertEqual(foundry._active_slot_id, "slot-good")
        self.assertEqual(foundry._roster["slot-good"].trait_labels, ["gentle"])
        self.assertEqual(foundry._roster["slot-good"].visit_count, 0)
        self.assertEqual(len(foundry._notes), 1)
        self.assertAlmostEqual(foundry._notes[0].effect_rating, 0.5)
        self.assertEqual(foundry._notes[0].boost_tally, 0)

    def test_dialogue_memoir_deserialize_tolerates_dirty_payloads(self):
        memoir = DialogueMemoir.deserialize(
            {
                "user_id": "user-1",
                "channel_id": "demo-stream",
                "phase": "anticipating",
                "journal": [
                    {
                        "kind": "user_utterance",
                        "written_at": 5.0,
                        "body_text": "hello",
                    },
                    "broken-entry",
                ],
                "anticipation": {
                    "expected_reply_text": "等回复",
                    "max_wait_sec": "bad",
                    "commenced_at": None,
                    "latest_ponder_at": "oops",
                    "ponder_tally": "",
                },
                "born_at": "bad",
                "touched_at": None,
                "exchange_tally": "oops",
                "last_initiative_at": "bad",
                "consecutive_timeouts": None,
                "last_user_spoke_at": "",
                "last_topic": "topic",
                "last_mood": "calm",
            }
        )

        self.assertEqual(memoir.phase, DialogueStage.ANTICIPATING)
        self.assertEqual(len(memoir.latest_entries()), 1)
        self.assertEqual(memoir.latest_entries()[0].body_text, "hello")
        self.assertEqual(memoir.anticipation.expected_reply_text, "等回复")
        self.assertEqual(memoir.anticipation.max_wait_sec, 0)
        self.assertEqual(memoir.exchange_tally, 0)
        self.assertEqual(memoir.last_initiative_at, 0.0)
        self.assertEqual(memoir.last_topic, "topic")

    def test_resolve_latest_human_user_id_ignores_self_messages(self):
        chat = make_minimal_chat()
        chat._last_user_id = "stale-user"
        human = SimpleNamespace(user_id="user-1", processed_plain_text="你好")
        bot_echo = SimpleNamespace(
            user_id="bot-qq-id",
            processed_plain_text="我自己刚说的话",
            _is_bot=True,
        )
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))

        self.assertEqual(
            chat._resolve_latest_human_user_id(
                [bot_echo, human], allow_cached_fallback=False
            ),
            "user-1",
        )
        self.assertEqual(
            chat._resolve_latest_human_user_id(
                [bot_echo], allow_cached_fallback=False
            ),
            "",
        )

    def test_analyze_group_sense_excludes_bot_self_messages(self):
        chat = make_minimal_chat()
        chat._merge_group_context_signal = lambda payload: payload
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        captured = {}

        class FakeGroupSense:
            def analyze(self, stream_id, recent_messages, last_reply_time=0.0):
                captured["recent_messages"] = list(recent_messages)
                return SimpleNamespace(
                    activity_level="冷清",
                    message_count_5min=len(recent_messages),
                    active_user_count=len(
                        {item.get("sender_id") for item in recent_messages}
                    ),
                    silence_duration_seconds=120.0,
                    burst_detected=False,
                    burst_density=0.0,
                    controversy_detected=False,
                    dominant_users=[],
                    needs_topic=False,
                    topic_hints=[],
                    description="ok",
                )

        bot_msg = SimpleNamespace(
            user_id="bot-qq-id",
            processed_plain_text="我还在说",
            timestamp=10.0,
            _is_bot=True,
        )
        human_msg = SimpleNamespace(
            user_id="user-1",
            processed_plain_text="有人吗",
            timestamp=12.0,
            _is_bot=False,
        )

        with patch(
            "src.modules.perception.group_sense.get_group_sense",
            return_value=FakeGroupSense(),
        ):
            result = asyncio.run(chat._analyze_group_sense([bot_msg, human_msg]))

        self.assertEqual(
            captured["recent_messages"],
            [{"sender_id": "user-1", "timestamp": 12.0, "text": "有人吗"}],
        )
        self.assertEqual(result["active_user_count"], 1)

    def test_build_environment_snapshot_separates_self_messages(self):
        chat = make_minimal_chat()
        chat.chat_stream = SimpleNamespace(stream_name="测试群")
        chat._sample_channel_ambient = lambda: {"category": "平静"}
        chat._extract_current_topics = lambda messages: []
        chat._cached_narration_plan = None
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))

        human_msg = SimpleNamespace(
            user_id="user-1",
            user_nickname="Alice",
            processed_plain_text="你在吗",
            timestamp=20.0,
            _is_bot=False,
        )
        bot_msg = SimpleNamespace(
            user_id="bot-qq-id",
            user_nickname="Mai",
            processed_plain_text="我刚刚已经回过了",
            timestamp=25.0,
            _is_bot=True,
        )

        fake_channel = SimpleNamespace(
            thinking_value=80.0,
            thinking_ceiling=100.0,
            chat_pool=90.0,
            chat_ceiling=100.0,
            activity_level=45.0,
            social_value=0.0,
        )
        fake_d6 = SimpleNamespace(_ensure_channel=lambda _sid: fake_channel)
        fake_quiet = SimpleNamespace(measure_silence_sec=lambda _sid: 180.0)
        fake_memoir = SimpleNamespace(retrieve=lambda _uid: None)
        fake_trauma = SimpleNamespace(get_state=lambda: SimpleNamespace(stress_accumulation=0.0))
        fake_pool = SimpleNamespace(get_active_intentions=lambda _sid, limit=3: [])
        fake_impression_hub = SimpleNamespace(get_impression_summary=lambda _uid: {"exists": False})

        with patch(
            "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
            return_value=fake_d6,
        ), patch(
            "src.chat.proactive.silence_watcher.get_quiet_monitor",
            return_value=fake_quiet,
        ), patch(
            "src.chat.proactive.session_tracker.get_memoir_cabinet",
            return_value=fake_memoir,
        ), patch(
            "src.modules.trauma.trauma_system.get_trauma_system",
            return_value=fake_trauma,
        ), patch(
            "src.chat.proactive.intention_pool.get_intention_pool",
            return_value=fake_pool,
        ), patch(
            "src.core.impression_evolution_hub.get_impression_hub",
            return_value=fake_impression_hub,
        ):
            snap = chat._build_environment_snapshot(
                now=time.time(),
                relation_result={},
                messages=[human_msg, bot_msg],
            )

        self.assertEqual(
            snap.recent_messages,
            [{"speaker": "Alice", "content": "你在吗"}],
        )
        self.assertEqual(snap.active_users, ["Alice"])
        self.assertEqual(snap.my_last_message, "我刚刚已经回过了")

    def test_should_back_off_idle_proactive_when_only_self_is_talking(self):
        chat = make_minimal_chat()
        now = time.time()
        chat._unanswered_bot_turns = 1
        chat._proactive_reply_timeline = [now - 15.0]
        chat._user_msg_timeline = []
        chat._last_user_msg_time = 0.0
        chat._cached_voice = SimpleNamespace(reply_desire_level=5)

        fake_emotion_core = SimpleNamespace(
            get_state_snapshot=lambda _sid: {"boredom": 0.2, "loneliness": 0.2}
        )
        fake_rest_timing = SimpleNamespace(
            estimate_reply_cooldown=lambda **_kwargs: 60.0
        )

        with patch(
            "src.chat.heart_flow.emotion_driven_core.get_emotion_driven_core",
            return_value=fake_emotion_core,
        ), patch(
            "src.chat.heart_flow.skills.rest_timing.acquire_rest_timing_engine",
            return_value=fake_rest_timing,
        ):
            self.assertTrue(chat._should_back_off_idle_proactive())

    def test_should_short_circuit_self_echo_batch_only_for_bot_messages(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        bot_msg = SimpleNamespace(user_id="bot-qq-id", _is_bot=True)
        human_msg = SimpleNamespace(user_id="user-1", _is_bot=False)

        self.assertTrue(chat._should_short_circuit_self_echo_batch([bot_msg]))
        self.assertFalse(chat._should_short_circuit_self_echo_batch([bot_msg, human_msg]))
        self.assertFalse(chat._should_short_circuit_self_echo_batch([]))

    def test_integrate_message_routing_skips_bot_self_messages_in_roster(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))

        class FakeRouter:
            def classify_batch(self, batch):
                return [
                    SimpleNamespace(category="plain_text", is_at_bot=False, is_quote_to_bot=False)
                    for _ in batch
                ]

            def summarize_batch(self, routes):
                return {"count": len(routes)}

        class FakeSubjectiveTime:
            def __init__(self):
                self.batches = []

            def register_batch(self, batch):
                self.batches.append(list(batch))

        class FakeRoster:
            def __init__(self):
                self.reset_calls = []
                self.touch_calls = []

            def reset_consecutive(self, user_id):
                self.reset_calls.append(user_id)

            def touch(self, user_id, **kwargs):
                self.touch_calls.append((user_id, kwargs))

            def summary(self):
                return {"count": len(self.touch_calls)}

        roster = FakeRoster()
        subj_time = FakeSubjectiveTime()
        chat._engine_orch = DummyOrch(
            {
                "semantic_router": FakeRouter(),
                "subjective_time": subj_time,
                "participant_roster": roster,
            }
        )

        bot_msg = SimpleNamespace(
            user_id="bot-qq-id",
            processed_plain_text="我自己刚说的话",
            user_name="Mai",
            _is_bot=True,
        )
        human_msg = SimpleNamespace(
            user_id="user-1",
            processed_plain_text="有人吗",
            user_name="Alice",
            _is_bot=False,
        )

        chat._integrate_message_routing([bot_msg, human_msg])

        self.assertEqual(subj_time.batches, [[bot_msg, human_msg]])
        self.assertEqual(roster.reset_calls, ["user-1"])
        self.assertEqual(len(roster.touch_calls), 1)
        self.assertEqual(roster.touch_calls[0][0], "user-1")
        self.assertEqual(roster.touch_calls[0][1]["text_hint"], "有人吗")

    def test_run_peek_observe_loop_ignores_bot_self_messages(self):
        chat = make_minimal_chat()
        chat._cached_watch_level = SimpleNamespace(value="peek")
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        captured = {}
        fake_channel = SimpleNamespace(chat_pool=10.0, activity_level=5.0)

        class FakeD6:
            def on_event(self, event_name, payload):
                captured["event_name"] = event_name
                captured["payload"] = dict(payload)

            def _ensure_channel(self, _stream_id):
                return fake_channel

        bot_msg = SimpleNamespace(content="我自己刚说的话", _is_bot=True, user_id="bot-qq-id")
        human_msg = SimpleNamespace(content="有人吗", _is_bot=False, user_id="user-1")

        with patch(
            "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
            return_value=FakeD6(),
        ):
            asyncio.run(chat._run_peek_observe_loop([bot_msg, human_msg]))

        self.assertEqual(captured["event_name"], "user_message")
        self.assertEqual(captured["payload"]["content_length"], len("有人吗"))
        self.assertFalse(captured["payload"]["is_repeat"])

    def test_background_proactive_blocks_after_long_ignored_chain(self):
        chat = make_minimal_chat()
        chat._proactive_startup_grace_until = 0.0
        chat._last_idle_proactive_ts = 0.0
        chat._cached_night_phase = None
        chat._unanswered_bot_turns = 4
        chat._chatterbox_penalty = 2.0
        chat._consecutive_speaks = 6.0
        chat._last_user_msg_time = 0.0
        chat._user_msg_timeline = []

        with patch.object(chat, "_should_back_off_idle_proactive", return_value=False):
            should_act, reason = asyncio.run(
                chat._evaluate_proactive_opportunity(
                    now=time.time(),
                    silence_sec=360.0,
                    is_background=True,
                )
            )

        self.assertFalse(should_act)
        self.assertIn("长期无人回应", reason)

    def test_build_proactive_no_target_guard_only_when_no_recent_human_target(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        bot_msg = SimpleNamespace(user_id="bot-qq-id", _is_bot=True)
        human_msg = SimpleNamespace(user_id="user-1", _is_bot=False)

        guard = chat._build_proactive_no_target_guard([bot_msg])
        self.assertIn("不要像在催人", guard)

        chat._last_user_msg_time = time.time()
        self.assertEqual(chat._build_proactive_no_target_guard([human_msg]), "")

    def test_apply_pre_send_reply_guard_rewrites_high_risk_plain_text(self):
        chat = make_minimal_chat()
        reply_set = ReplySetModel(
            reply_data=[
                ReplyContent(
                    content_type=ReplyContentType.TEXT,
                    content="看见了，说事啊",
                )
            ]
        )

        class FakePatrol:
            async def review_before_send(self, content, risk_note=""):
                return "慢慢说，什么事？", True, "语气太冲"

            async def audit_high_risk_reply(self, content, risk_note=""):
                return False, ""

        with patch(
            "src.chat.heart_flow.skills.focus_patrol.acquire_focus_patrol",
            return_value=FakePatrol(),
        ):
            reviewed_text, changed, reason, blocked = asyncio.run(
                chat._apply_pre_send_reply_guard(
                    reply_set,
                    audit_label="proactive",
                    risk_note="",
                )
            )

        self.assertTrue(changed)
        self.assertFalse(blocked)
        self.assertEqual(reviewed_text, "慢慢说，什么事？")
        self.assertEqual(reply_set.reply_data[0].content, "慢慢说，什么事？")
        self.assertIn("语气太冲", reason)

    def test_apply_pre_send_reply_guard_blocks_when_rewrite_still_risky(self):
        chat = make_minimal_chat()
        reply_set = ReplySetModel(
            reply_data=[
                ReplyContent(
                    content_type=ReplyContentType.TEXT,
                    content="看见了，说事啊",
                )
            ]
        )

        class FakePatrol:
            async def review_before_send(self, content, risk_note=""):
                return "你到底想说啥", True, "语气太冲"

            async def audit_high_risk_reply(self, content, risk_note=""):
                return True, "仍有冒犯"

        with patch(
            "src.chat.heart_flow.skills.focus_patrol.acquire_focus_patrol",
            return_value=FakePatrol(),
        ):
            reviewed_text, changed, reason, blocked = asyncio.run(
                chat._apply_pre_send_reply_guard(
                    reply_set,
                    audit_label="proactive",
                    risk_note="",
                )
            )

        self.assertTrue(changed)
        self.assertTrue(blocked)
        self.assertEqual(reviewed_text, "")
        self.assertIn("仍有冒犯", reason)

    def test_behavior_governor_blocks_proactive_after_ignored_chain(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._cached_presence_state = SimpleNamespace(
            watch_state_rank=1,
            quiet_preference=0.35,
            avoidance_tendency=0.20,
            social_willingness=0.55,
        )
        chat._cached_emotion_state = SimpleNamespace(boredom=0.62, loneliness=0.66, social_desire=0.70)
        chat._cached_metabolism_constraints = {"loafing_level": 0.18, "chat_value": 82.0, "thinking_value": 71.0}
        chat._unanswered_bot_turns = 4
        chat._chatterbox_penalty = 2.4
        chat._consecutive_speaks = 6.0
        chat._last_user_msg_time = 0.0
        chat._user_msg_timeline = []
        verdict = chat._evaluate_behavior_governor(
            incoming_batch=[],
            silence_sec=360.0,
            requested_mode="proactive",
            is_background=True,
        )
        self.assertIn(verdict.reply_mode, {"defer", "rest"})
        self.assertEqual(verdict.silence_policy, "silent")
        self.assertTrue(
            any(code in verdict.reason_codes for code in ("ignored_chain", "chatterbox_guard", "idle_backoff"))
        )

    def test_behavior_governor_keeps_weak_message_in_observe(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._cached_presence_state = SimpleNamespace(
            watch_state_rank=1,
            quiet_preference=0.60,
            avoidance_tendency=0.15,
            social_willingness=0.58,
        )
        chat._cached_emotion_state = SimpleNamespace(boredom=0.41, loneliness=0.30, social_desire=0.44)
        chat._cached_metabolism_constraints = {"loafing_level": 0.22, "chat_value": 76.0, "thinking_value": 68.0}
        weak_msg = SimpleNamespace(user_id="user-1", content="嗯", timestamp=time.time(), _is_bot=False)
        verdict = chat._evaluate_behavior_governor(
            incoming_batch=[weak_msg],
            silence_sec=35.0,
            requested_mode="reactive",
            target_message=weak_msg,
        )
        self.assertEqual(verdict.reply_mode, "observe")
        self.assertIn(verdict.interrupt_level, {"peek", "skim"})
        self.assertEqual(verdict.quote_policy, "soft_reference")
        self.assertFalse(verdict.allow_generation)
        self.assertLessEqual(verdict.max_watch_rank, 2)

    def test_reply_coordinator_downgrades_stale_quote_to_soft_reference(self):
        coordinator = ReplyCoordinator()
        stale_msg = SimpleNamespace(
            user_id="user-1",
            content="你怎么看",
            timestamp=time.time() - 900.0,
            is_quote_reply=False,
        )
        route = coordinator.resolve_style_route(
            delivery_form="",
            target_message=stale_msg,
            reference_user_name="",
            mention_user_name="",
            fallback_selector=lambda msg, rel, hs: {
                "reply_style": "quote",
                "quote_message": True,
                "reason": "问题句",
            },
            relation_view={},
            harassment_signal=None,
            is_bot_message=lambda msg: False,
        )
        self.assertEqual(route["quote_policy"], "soft_reference")
        self.assertFalse(route["quote_message"])
        self.assertEqual(route["reply_style"], "soft_reference")

    def test_reply_coordinator_rejects_bot_anchor_quote(self):
        coordinator = ReplyCoordinator()
        bot_msg = SimpleNamespace(user_id="bot-qq-id", content="我自己刚说的话", timestamp=time.time())
        route = coordinator.resolve_style_route(
            delivery_form="quote",
            target_message=bot_msg,
            reference_user_name="爱丽丝",
            mention_user_name="",
            fallback_selector=lambda msg, rel, hs: {
                "reply_style": "quote",
                "quote_message": True,
                "reason": "规划器要求引用",
            },
            relation_view={},
            harassment_signal=None,
            is_bot_message=lambda msg: True,
        )
        self.assertEqual(route["quote_policy"], "none")
        self.assertFalse(route["quote_message"])

    def test_reply_coordinator_drops_quote_when_anchor_too_old(self):
        coordinator = ReplyCoordinator()
        old_msg = SimpleNamespace(user_id="user-1", content="前情", timestamp=time.time() - 2400.0)
        route = coordinator.resolve_style_route(
            delivery_form="quote",
            target_message=old_msg,
            reference_user_name="爱丽丝",
            mention_user_name="",
            fallback_selector=lambda msg, rel, hs: {
                "reply_style": "quote",
                "quote_message": True,
                "reason": "规划器要求引用",
            },
            relation_view={},
            harassment_signal=None,
            is_bot_message=lambda msg: False,
        )
        self.assertEqual(route["quote_policy"], "none")
        self.assertFalse(route["quote_message"])

    def test_model_governor_rate_limits_proactive_large_within_120s(self):
        chat = make_minimal_chat()
        now = time.time()
        chat._model_large_last_ts = now - 30.0
        behavior = BehaviorGovernorVerdict(
            reply_mode="proactive",
            interrupt_level="engage",
            allow_generation=True,
            model_tier="small",
        )
        verdict = chat._evaluate_model_governor(
            now=now,
            desired_level=2,
            incoming_batch=[],
            pinged_msg=None,
            is_proactive=True,
            source="background_proactive",
            behavior_verdict=behavior,
            high_risk=True,
        )
        self.assertEqual(verdict.tier, "small")
        self.assertTrue(verdict.rate_limited)
        self.assertTrue(verdict.fallback_to_small)

    def test_model_governor_demotes_noncritical_large_to_small(self):
        chat = make_minimal_chat()
        verdict = chat._evaluate_model_governor(
            now=time.time(),
            desired_level=2,
            incoming_batch=[SimpleNamespace(user_id="user-1", content="嗯", _is_bot=False)],
            pinged_msg=None,
            is_proactive=False,
            source="reactive_decision",
            behavior_verdict=BehaviorGovernorVerdict(
                reply_mode="observe",
                interrupt_level="peek",
                allow_generation=False,
                model_tier="small",
            ),
            high_risk=False,
        )
        self.assertEqual(verdict.tier, "skip")

    def test_execute_proactive_reply_stops_when_behavior_governor_blocks(self):
        chat = make_minimal_chat()
        chat._proactive_send_lock = None
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._run_self_restraint_check = AsyncMock(return_value={"allow": True, "mode": "allow"})
        chat._analyze_repetition_pressure = lambda incoming_batch: {}
        chat._analyze_harassment_pressure = lambda incoming_batch: {}
        chat._mark_message_content_deferred = lambda *args, **kwargs: None
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict(
            reply_mode="rest",
            interrupt_level="ignore",
            quote_policy="none",
            silence_policy="silent",
            reason_codes=["default_silence"],
        )

        class FakeCoordinator:
            def select_reply_target(self, **kwargs):
                return None

        llm_decision = SimpleNamespace(
            social_intention="想说点什么",
            content_plan="",
            emotional_state="",
        )

        with patch(
            "src.chat.heart_flow.reply_coordinator.acquire_reply_coordinator",
            return_value=FakeCoordinator(),
        ):
            result = asyncio.run(chat._execute_proactive_reply(llm_decision=llm_decision, incoming_batch=[]))

        self.assertFalse(result)

    def test_rest_governor_blocks_weak_messages_when_high_loafing(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._cached_presence_state = SimpleNamespace(
            watch_state_rank=1,
            quiet_preference=0.72,
            avoidance_tendency=0.52,
            social_willingness=0.28,
        )
        chat._cached_metabolism_constraints = {
            "loafing_level": 0.84,
            "chat_value": 63.0,
            "thinking_value": 59.0,
        }
        chat._cached_emotion_state = SimpleNamespace(boredom=0.18, loneliness=0.16)
        chat._last_user_msg_time = 0.0
        chat._user_msg_timeline = []
        verdict = chat._evaluate_rest_governor(incoming_batch=[], source="reactive")
        self.assertEqual(verdict.posture, "loaf")
        self.assertEqual(verdict.interruption_policy, "block")
        self.assertTrue(verdict.should_loaf)

    def test_rest_governor_allows_direct_target_override(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._message_targets_bot = lambda msg: True
        chat._cached_presence_state = SimpleNamespace(
            watch_state_rank=1,
            quiet_preference=0.75,
            avoidance_tendency=0.60,
            social_willingness=0.20,
        )
        chat._cached_metabolism_constraints = {
            "loafing_level": 0.80,
            "chat_value": 58.0,
            "thinking_value": 55.0,
        }
        weak_target = SimpleNamespace(user_id="user-1", content="@你 在吗", timestamp=time.time(), _is_bot=False)
        verdict = chat._evaluate_rest_governor(
            incoming_batch=[weak_target],
            source="reactive",
            targeted_to_bot=True,
        )
        self.assertEqual(verdict.interruption_policy, "allow")

    def test_run_self_restraint_check_returns_short_only_when_rest_governor_allows_targeted_override(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict(
            posture="peek_only",
            interruption_policy="peek_only",
            should_rest=False,
            should_loaf=True,
            reason_codes=["soft_loafing_guard"],
        )

        class FakeDecisionMaker:
            async def evaluate_restraint_from_state(self, *args, **kwargs):
                raise AssertionError("should not reach decision maker when rest governor already downgrades")

        with patch(
            "src.chat.heart_flow.heartflow_decision.acquire_decision_maker",
            return_value=FakeDecisionMaker(),
        ):
            result = asyncio.run(
                chat._run_self_restraint_check(
                    decision_messages=[SimpleNamespace(user_id="user-1", content="@你 在吗")],
                    source="reactive",
                    targeted_to_bot=True,
                )
            )

        self.assertTrue(result["allow"])
        self.assertEqual(result["mode"], "short_only")
        self.assertIn("休息Governor", result["reason"])

    def test_apply_sensory_gates_uses_rest_governor_block_reason(self):
        chat = make_minimal_chat()
        chat._is_bot_message_obj = lambda msg: bool(getattr(msg, "_is_bot", False))
        chat._cached_visibility_decisions = []
        chat._cached_understanding_results = []
        chat._cached_metabolism_constraints = {}
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict(
            posture="rest",
            interruption_policy="block",
            should_rest=True,
            should_loaf=False,
            reason_codes=["resource_exhausted"],
        )
        weak_msg = SimpleNamespace(user_id="user-1", content="嗯", timestamp=time.time(), _is_bot=False)
        reason = chat._apply_sensory_gates([weak_msg], pinged_msg=None)
        self.assertIn("休息门控", reason)
        self.assertIn("resource_exhausted", reason)

    def test_apply_governor_watch_transition_drops_engaged_to_blackout_on_rest(self):
        chat = make_minimal_chat()
        chat._has_targeted_bot_message = lambda batch: False
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict(
            posture="rest",
            interruption_policy="block",
            should_rest=True,
            should_loaf=False,
            reason_codes=["resource_exhausted"],
        )
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict()
        captured = {}

        def _build_snapshot(_source):
            return SimpleNamespace(watch_state="engaged", phase="", blocker="")

        def _apply_snapshot(snapshot, watch_reason="", phase_reason=""):
            captured["watch_state"] = snapshot.watch_state
            captured["phase"] = snapshot.phase
            captured["blocker"] = snapshot.blocker
            captured["watch_reason"] = watch_reason
            captured["phase_reason"] = phase_reason

        chat._build_unified_flow_snapshot = _build_snapshot
        chat._apply_unified_flow_snapshot = _apply_snapshot

        transition = chat._apply_governor_watch_transition(
            current_watch="engaged",
            incoming_batch=[],
            has_ping=False,
        )

        self.assertIn("engaged->blackout", transition)
        self.assertEqual(captured["watch_state"], "blackout")
        self.assertEqual(captured["phase"], FlowPhase.DORMANT.value)
        self.assertIn("RestGovernor", captured["phase_reason"])

    def test_apply_governor_watch_transition_promotes_peek_to_active_on_engage(self):
        chat = make_minimal_chat()
        msg = SimpleNamespace(user_id="user-1", content="@你 在吗", timestamp=time.time(), _is_bot=False)
        chat._has_targeted_bot_message = lambda batch: True
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict()
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict(
            reply_mode="reply",
            interrupt_level="engage",
            quote_policy="quote_reply",
            target_user_id="user-1",
            silence_policy="ambient",
            reason_codes=["direct_relevance"],
        )
        chat._get_latest_human_message = lambda batch: batch[-1] if batch else None
        captured = {}

        def _build_snapshot(_source):
            return SimpleNamespace(watch_state="peek", phase="", blocker="")

        def _apply_snapshot(snapshot, watch_reason="", phase_reason=""):
            captured["watch_state"] = snapshot.watch_state
            captured["phase"] = snapshot.phase
            captured["watch_reason"] = watch_reason
            captured["phase_reason"] = phase_reason

        chat._build_unified_flow_snapshot = _build_snapshot
        chat._apply_unified_flow_snapshot = _apply_snapshot

        transition = chat._apply_governor_watch_transition(
            current_watch="peek",
            incoming_batch=[msg],
            has_ping=True,
        )

        self.assertIn("peek->active_watch", transition)
        self.assertEqual(captured["watch_state"], "active_watch")
        self.assertEqual(captured["phase"], FlowPhase.STANDBY.value)
        self.assertIn("BehaviorGovernor", captured["phase_reason"])

    def test_apply_governor_watch_cap_downgrades_active_watch_to_skim_window(self):
        chat = make_minimal_chat()
        captured = {}

        def _build_snapshot(_source):
            return SimpleNamespace(watch_state="active_watch", phase="", blocker="")

        def _apply_snapshot(snapshot, watch_reason="", phase_reason=""):
            captured["watch_state"] = snapshot.watch_state
            captured["phase"] = snapshot.phase
            captured["watch_reason"] = watch_reason
            captured["phase_reason"] = phase_reason

        chat._build_unified_flow_snapshot = _build_snapshot
        chat._apply_unified_flow_snapshot = _apply_snapshot

        transition = chat._apply_governor_watch_cap(
            current_watch="active_watch",
            cap_rank=2,
            reason="BehaviorGovernor:skim:test",
        )

        self.assertIn("active_watch->skim_window", transition)
        self.assertEqual(captured["watch_state"], "skim_window")
        self.assertEqual(captured["phase"], FlowPhase.STANDBY.value)
        self.assertIn("Governor限幅", captured["phase_reason"])

    def test_evaluate_watch_full_transitions_respects_governor_cap_for_peek(self):
        chat = make_minimal_chat()
        chat._apply_governor_watch_transition = lambda **kwargs: None
        chat._has_targeted_bot_message = lambda batch: False
        chat._get_latest_human_message = lambda batch: batch[-1] if batch else None
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict()
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict(
            reply_mode="observe",
            interrupt_level="peek",
            quote_policy="soft_reference",
            target_user_id="",
            silence_policy="ambient",
            reason_codes=["weak_human_signal"],
        )
        chat._is_human_message_obj = lambda msg: not bool(getattr(msg, "_is_bot", False))
        chat._tick_world_snapshot = SimpleNamespace(
            self_resources=SimpleNamespace(
                boredom=70.0,
                activity_level=50.0,
                silence_seconds=0.0,
            )
        )
        chat._cached_interruption_tiers = {"demo": 0.8}
        chat._cached_self_references = []
        chat._cached_pattern_evidence = []
        chat._cached_participant_summary = {"hot_count": 2, "warm_count": 1}
        chat._cached_presence_state = SimpleNamespace(
            social_willingness=0.5,
            watch_willingness=0.6,
            avoidance_tendency=0.0,
        )
        applied = {"called": False}
        chat._build_unified_flow_snapshot = lambda _source: SimpleNamespace(watch_state="peek", phase="", blocker="")

        def _apply_snapshot(snapshot, watch_reason="", phase_reason=""):
            applied["called"] = True

        chat._apply_unified_flow_snapshot = _apply_snapshot
        msg = SimpleNamespace(user_id="user-1", content="看看这个", timestamp=time.time(), _is_bot=False)

        with patch(
            "src.core.watch_state_machine.get_watch_machine",
            return_value=SimpleNamespace(current_level=SimpleNamespace(value="peek")),
        ):
            transition = chat._evaluate_watch_full_transitions(
                now=time.time(),
                has_ping=False,
                incoming_batch=[msg],
            )

        self.assertIsNone(transition)
        self.assertFalse(applied["called"])

    def test_should_promote_peek_reflection_blocks_reply_without_governor_engage(self):
        chat = make_minimal_chat()
        chat._has_targeted_bot_message = lambda batch: False
        chat._get_latest_human_message = lambda batch: batch[-1] if batch else None
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict()
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict(
            reply_mode="observe",
            interrupt_level="peek",
            quote_policy="soft_reference",
            target_user_id="",
            silence_policy="ambient",
            reason_codes=["weak_human_signal"],
        )

        allow, reason = chat._should_promote_peek_reflection(
            incoming_batch=[SimpleNamespace(user_id="user-1", content="路过看看", timestamp=time.time(), _is_bot=False)],
            peek_verdict=SimpleNamespace(is_valid=True, next_action="reply", reply_desire_level=7),
            pinged_msg=None,
        )

        self.assertFalse(allow)
        self.assertIn("peek_governor_hold", reason)

    def test_should_promote_peek_reflection_allows_engage_reply(self):
        chat = make_minimal_chat()
        chat._has_targeted_bot_message = lambda batch: True
        chat._get_latest_human_message = lambda batch: batch[-1] if batch else None
        chat._evaluate_rest_governor = lambda **kwargs: RestGovernorVerdict()
        chat._evaluate_behavior_governor = lambda **kwargs: BehaviorGovernorVerdict(
            reply_mode="reply",
            interrupt_level="engage",
            quote_policy="quote_reply",
            target_user_id="user-1",
            silence_policy="ambient",
            reason_codes=["direct_relevance"],
        )

        allow, reason = chat._should_promote_peek_reflection(
            incoming_batch=[SimpleNamespace(user_id="user-1", content="@你 在吗", timestamp=time.time(), _is_bot=False)],
            peek_verdict=SimpleNamespace(is_valid=True, next_action="reply", reply_desire_level=5),
            pinged_msg=None,
        )

        self.assertTrue(allow)
        self.assertIn("direct_target", reason)

    def test_run_peek_with_reflection_does_not_cache_voice_before_upgrade(self):
        chat = make_minimal_chat()
        chat._should_run_voice = lambda *args, **kwargs: True

        async def _noop_observe(_incoming_batch):
            return None

        async def _fake_invoke(*args, **kwargs):
            return SimpleNamespace(
                is_valid=True,
                next_action="reply",
                reply_desire_level=8,
                thinking="想接一句",
            )

        chat._run_peek_observe_loop = _noop_observe
        chat._sample_channel_ambient = lambda: {}
        chat._invoke_inner_voice = _fake_invoke
        result = asyncio.run(
            chat._run_peek_with_reflection(
                [SimpleNamespace(user_id="user-1", processed_plain_text="看看这个", content="看看这个")]
            )
        )

        self.assertIsNotNone(result)
        self.assertGreater(chat._last_voice_ts, 0.0)
        self.assertIsNone(getattr(chat, "_cached_voice", None))

    def test_world_snapshot_to_rapport_dict_includes_memory_and_psych_fields(self):
        snap = WorldSnapshot(
            target_user=TargetUserState(
                affection=3.0,
                trust_value=4.0,
                psychological_pressure=12.0,
                trauma_score=7.0,
                mood="紧绷",
            )
        )
        snap.scene.session_phase = "open"
        snap.scene.last_topic = "补丁"
        snap.scene.last_mood = "试探"
        rapport = snap.to_rapport_dict()
        self.assertEqual(rapport["psychological_pressure"], 12.0)
        self.assertEqual(rapport["trauma_score"], 7.0)
        self.assertEqual(rapport["mood"], "紧绷")
        self.assertIn("会话阶段=open", rapport["memory_summary"])
        self.assertIn("上次话题=补丁", rapport["memory_summary"])
        self.assertIn("上次气氛=试探", rapport["memory_summary"])

    def test_planner_raw_state_block_preserves_zero_relation_values(self):
        block = PlannerPromptInjectionLayer._build_raw_state_block(
            metabolism={"energy_ratio": 0.3, "loafing_level": 0.2, "boredom_level": 0.1},
            night_phase="night",
            relation={"affection": 0.0, "trust_value": 0.0, "annoyance_value": 0.0},
            watch="peek",
        )
        self.assertIn("好感=0", block)
        self.assertIn("信任=0", block)
        self.assertIn("厌烦=0", block)

    def test_planner_packet_preserves_zero_situation_values(self):
        layer = PlannerPromptInjectionLayer()
        packet = layer.build_packet(
            "demo-channel",
            situation={
                "energy": 0.0,
                "social_comfort": 0.0,
                "intensity": 0.0,
                "risk_score": 0.0,
            },
        )
        self.assertIsNotNone(packet.situation)
        self.assertEqual(packet.situation.internal_energy, 0.0)
        self.assertEqual(packet.situation.social_comfort, 0.0)
        self.assertEqual(packet.situation.engagement_intensity, 0.0)
        self.assertEqual(packet.situation.risk_assessment_score, 0.0)

    def test_state_coupling_matrix_preserves_zero_relation_values(self):
        engine = StateCouplingMatrixEngine()
        captured = {}

        def fake_evaluate(snap):
            captured["snap"] = snap
            return SimpleNamespace(
                boredom_level=snap.boredom_level,
                activity_drive=0.0,
                reply_style_tier="flat",
                loafing_block_strength=0.0,
                energy_ratio=snap.energy_ratio,
            )

        engine.evaluate = fake_evaluate
        engine.build_full_snapshot(
            relation={"affection": 0.0, "trust_value": 0.0, "annoyance_value": 0.0, "trauma_score": 0.0},
            watch_state="peek",
        )
        snap = captured["snap"]
        self.assertEqual(snap.affection, 0.0)
        self.assertEqual(snap.trust_value, 0.0)
        self.assertEqual(snap.annoyance_value, 0.0)
        self.assertEqual(snap.trauma_score, 0.0)

    def test_state_coupling_matrix_preserves_zero_presence_and_scene_values(self):
        engine = StateCouplingMatrixEngine()
        captured = {}

        def fake_evaluate(snap):
            captured["snap"] = snap
            return SimpleNamespace(
                boredom_level=snap.boredom_level,
                activity_drive=0.0,
                reply_style_tier="flat",
                loafing_block_strength=0.0,
                energy_ratio=snap.energy_ratio,
            )

        engine.evaluate = fake_evaluate
        engine.build_full_snapshot(
            presence=SimpleNamespace(
                social_willingness=0.0,
                watch_state_rank=0,
                avoidance_tendency=0.0,
                curiosity_level=0.0,
                outward_attention=0.0,
                quiet_preference=0.0,
            ),
            group_scene={
                "group_atmosphere_score": 0.0,
                "confidence": 0.9,
                "active_user_count": 0,
                "unique_speakers_5min": 6,
                "is_high_value_group": False,
                "high_value_group": True,
            },
            watch_state="",
        )
        snap = captured["snap"]
        self.assertEqual(snap.social_willingness, 0.0)
        self.assertEqual(snap.watch_state_rank, 0)
        self.assertEqual(snap.curiosity_level, 0.0)
        self.assertEqual(snap.outward_attention, 0.0)
        self.assertEqual(snap.quiet_preference, 0.0)
        self.assertEqual(snap.group_atmosphere_score, 0.0)
        self.assertEqual(snap.active_user_count, 0)
        self.assertFalse(snap.is_high_value_group)

    def test_group_generator_metric_helper_preserves_zero_and_negative_values(self):
        from src.chat.replyer.group_generator import DefaultReplyer

        self.assertEqual(DefaultReplyer._metric_or_default(0.0, 50.0), 0.0)
        self.assertEqual(DefaultReplyer._metric_or_default(-12.5, 50.0), -12.5)
        self.assertEqual(DefaultReplyer._metric_or_default(None, 50.0), 50.0)

    def test_threat_assessor_preserves_zero_social_value(self):
        with patch(
            "src.chat.heart_flow.heartflow_decision._load_decision_cfg",
            return_value={
                "threat_weights": {
                    "toxicity": 0.45,
                    "turbulence": 0.20,
                    "fragility": 0.15,
                    "antisocial": 0.20,
                }
            },
        ):
            score = ThreatAssessor().assess({"social_value": 0.0})
        self.assertAlmostEqual(score, 0.20)

    def test_risk_prediction_preserves_zero_social_value(self):
        score = RiskPredictionSystem().predict({"social_value": 0.0})
        self.assertAlmostEqual(score, 0.20)

    def test_proactive_collect_energy_data_preserves_zero_activity_level(self):
        hub = ProactiveIntegrationHub()
        hub._vitality_pool = SimpleNamespace(
            _ensure_channel=lambda _channel_id: SimpleNamespace(
                chat_pool=10.0,
                thinking_value=5.0,
                combined_ratio=lambda: 0.25,
                is_exhausted=lambda: False,
                social_value=-7.0,
                activity_level=0.0,
            )
        )
        result = hub._collect_energy_data("channel-z")
        self.assertEqual(result["social_value"], -7.0)
        self.assertEqual(result["activity_level"], 0.0)

    def test_state_dashboard_preserves_zero_metrics_in_bars(self):
        from src.core.state_dashboard import StateDashboardEngine

        engine = StateDashboardEngine("demo-channel")
        snap = engine.build_dashboard(
            force=True,
            raw_sources={
                "energy_ratio": 0.0,
                "fatigue": 0.0,
                "recovery_rate": 0.0,
                "debt_modifier": 0.0,
                "social_willingness": 0.0,
                "openness": 0.0,
                "interrupt_tolerance": 0.0,
                "avoidance": 0.0,
                "loneliness": 0.0,
                "social_hunger": 0.0,
                "group_engagement_score": 0.0,
                "mood": 0.0,
                "curiosity": 0.0,
                "boredom": 0.0,
                "loafing": 0.0,
                "annoyance": 0.0,
                "proactive_drive": 0.0,
                "visibility_threshold": 0.0,
                "process_ratio": 0.0,
                "peek_desire": 0.0,
                "withdrawal_depth": 0.0,
                "last_look_ago": 0.0,
            },
        )
        self.assertEqual(snap.vitality.recovery_rate, 0.0)
        self.assertEqual(snap.vitality.debt_modifier, 0.0)
        self.assertEqual(snap.social.social_willingness, 0.0)
        self.assertEqual(snap.social.group_engagement.value, "ghost")
        self.assertEqual(snap.mood.mood_value, 0.0)
        self.assertEqual(snap.attention.peek_desire, 0.0)
        self.assertEqual(snap.attention.last_look_ago_sec, 0.0)

    def test_state_dashboard_quick_attention_cache_preserves_zero_values(self):
        from src.core.state_dashboard import StateDashboardEngine

        engine = StateDashboardEngine("demo-channel")
        engine._cached_attention_snapshot = {
            "peek_desire": 0.0,
            "since_last_look_sec": 0.0,
        }
        self.assertEqual(engine._get_quick_peek_desire(), 0.0)
        self.assertEqual(engine._get_quick_last_look(), 0.0)

    def test_attention_flow_snapshot_to_dict_exposes_machine_readable_fields(self):
        snap = AttentionFlowSnapshot(
            state=AttentionState.DEEP_WITHDRAWAL,
            previous_state=AttentionState.WITHDRAWN,
            transition_reason=AttentionTransitionReason.EMOTIONAL_OVERLOAD,
            peek_desire=0.0,
            withdrawal_depth=0.85,
            time_since_last_look=0.0,
            consecutive_peeks_without_action=3,
        )
        payload = snap.to_dict()
        self.assertEqual(payload["state"], "deep_withdrawal")
        self.assertEqual(payload["state_label"], "深度退出")
        self.assertEqual(payload["previous_state"], "withdrawn")
        self.assertEqual(payload["reason"], "emotional_overload")
        self.assertIn("visibility_threshold", payload)
        self.assertIn("process_ratio", payload)
        self.assertEqual(payload["consecutive_empty_peeks"], 3)

    def test_state_dashboard_attention_mode_maps_deep_withdrawal_to_sleeping(self):
        from src.core.state_dashboard import StateDashboardEngine

        engine = StateDashboardEngine("demo-channel")
        engine._cached_attention_snapshot = {"state": "deep_withdrawal"}
        self.assertEqual(engine._get_quick_attention_mode(), "sleeping")
        snap = engine.build_dashboard(
            force=True,
            raw_sources={
                "attention_mode": "deep_withdrawal",
                "visibility_threshold": 0.92,
                "process_ratio": 0.0,
                "peek_desire": 0.0,
                "withdrawal_depth": 0.95,
                "consecutive_empty_peeks": 4,
                "since_last_look_sec": 1200.0,
            },
        )
        self.assertEqual(snap.attention.mode.value, "sleeping")
        self.assertEqual(snap.attention.consecutive_empty_peeks, 4)
        self.assertEqual(snap.attention.last_look_ago_sec, 1200.0)

    def test_inner_voice_retrieve_rapport_data_uses_world_snapshot_contract(self):
        engine = SelfDialogueEngine("demo-channel")
        with patch(
            "src.core.world_snapshot.build_relation_rapport_snapshot",
            return_value={
                "relationship": "谨慎观察",
                "affection": -6.0,
                "trust": 88.0,
                "social_value": -9.0,
                "positive_dim": 1.0,
                "negative_dim": 4.0,
                "trust_value": -3.0,
                "annoyance_value": 11.0,
                "psychological_pressure": 15.0,
                "trauma_score": 5.0,
                "interaction_count": 2,
                "relationship_level": 1,
                "custom_label": "谨慎观察",
                "trend_direction": "下降",
                "profile_summary": "群信任度=-3.0",
                "memory_summary": "上次话题=旧梗",
                "execution_hint": "先稳住。",
            },
        ):
            result = engine._retrieve_rapport_data("Alice", "user-1")
        self.assertEqual(result["trust_value"], -3.0)
        self.assertEqual(result["annoyance_value"], 11.0)
        self.assertEqual(result["memory_summary"], "上次话题=旧梗")
        self.assertEqual(result["execution_hint"], "先稳住。")

    def test_relation_style_hint_uses_world_snapshot_contract(self):
        chat = make_minimal_chat()
        action_message = SimpleNamespace(user_id="user-1")
        with patch(
            "src.core.world_snapshot.build_relation_rapport_snapshot",
            return_value={
                "relationship": "冷淡期",
                "affection": -7.0,
                "trust": 95.0,
                "trust_value": -4.0,
                "annoyance_value": 18.0,
                "psychological_pressure": 22.0,
                "trauma_score": 9.0,
            },
        ):
            hint = chat._build_relation_style_hint(action_message)
        self.assertIn("[关系阶段] 冷淡期", hint)
        self.assertIn("信任=-4.0", hint)
        self.assertIn("厌烦=18.0", hint)
        self.assertIn("好感=-7.0", hint)

    def test_psychological_core_exposes_canonical_trust_value(self):
        core = PsychologicalCore()
        state = SimpleNamespace(
            affection=12.0,
            trust_score=-9.0,
            annoyance=5.0,
            trauma_score=3.0,
            relationship="谨慎观察",
            impression="旧印象",
        )
        tracker = SimpleNamespace(get_user_state=lambda *_args, **_kwargs: state)
        with patch.object(core, "_get_emotion_tracker", return_value=tracker):
            result = core.get_user_psychological_state("demo-stream", "user-1")
        self.assertEqual(result["trust"], -9.0)
        self.assertEqual(result["trust_value"], -9.0)

    def test_psychological_core_prefers_state_trust_value_over_legacy_alias(self):
        core = PsychologicalCore()
        state = SimpleNamespace(
            affection=12.0,
            trust_value=-15.0,
            trust_score=70.0,
            annoyance=5.0,
            trauma_score=3.0,
            relationship="谨慎观察",
            impression="旧印象",
        )
        tracker = SimpleNamespace(get_user_state=lambda *_args, **_kwargs: state)
        with patch.object(core, "_get_emotion_tracker", return_value=tracker):
            result = core.get_user_psychological_state("demo-stream", "user-1")
        self.assertEqual(result["trust"], -15.0)
        self.assertEqual(result["trust_value"], -15.0)

    def test_decision_brain_prefers_canonical_trust_value(self):
        brain = DecisionBrain()
        group = GroupDecision()
        context = DecisionContext(
            emotion_state={
                "affection": 0.0,
                "trust": 88.0,
                "trust_value": -30.0,
            }
        )
        interest = group._compute_interest_score(context)
        factors = brain._extract_factors(context)
        self.assertAlmostEqual(interest, 0.35, places=4)
        self.assertAlmostEqual(factors["confidence"].value, 0.2, places=4)

    def test_adaptive_threshold_context_prefers_canonical_trust_value(self):
        learner = AdaptiveThresholdLearnerV2("demo-stream")
        ctx = learner.compute_context(
            messages=[],
            social={"affection": 0.0, "trust": 85.0, "trust_value": -12.0},
            energy={"energy_ratio": 0.7},
            group={"atmosphere_tension": 0.0},
            time_metrics={},
        )
        self.assertEqual(ctx.trust_level, -12.0)

    def test_compute_llm_call_level_passes_canonical_trust_value(self):
        chat = make_minimal_chat()
        chat._cached_affection_value = 5.0
        chat._cached_trust_value = -17.0
        chat._cached_annoyance_value = 0.0
        chat._cached_pressure_value = 0.0
        chat._cached_relationship_level = 2
        chat._cached_group_support = 0.0
        chat._cached_group_attack = 0.0
        chat._cached_familiarity = 0.0
        chat._cached_dominance = 0.0

        captured = {}

        class FakeLearner:
            def _detect_burst_mode(self, _now):
                return False, 0, 0.0

            def _calc_emotion(self, _texts):
                return 0.0

            def compute_integrated_decision(self, **kwargs):
                captured["trust"] = kwargs.get("trust")
                return "join", 0.1, 0.6

        with patch(
            "src.core.adaptive_threshold_learner_v2.get_adaptive_threshold_learner_v2",
            return_value=FakeLearner(),
        ):
            level, _, _, _ = chat._compute_llm_call_level(
                now=time.time(),
                messages=[],
                pinged=None,
                desire_level=5.0,
            )
        self.assertEqual(level, 1)
        self.assertEqual(captured["trust"], -17.0)

    def test_relationship_controller_summary_exposes_canonical_trust_value(self):
        controller = RelationshipController("demo-stream")
        state = controller.get_state("user-1")
        state.trust = -23.0
        state.affection = 8.0
        summary = controller.get_relationship_summary("user-1")
        self.assertEqual(summary["trust"], -23.0)
        self.assertEqual(summary["trust_value"], -23.0)

    def test_unified_planner_context_mirrors_canonical_trust_into_compat_field(self):
        planner = UnifiedPlanner()
        snapshot = WorldSnapshot(
            target_user=TargetUserState(
                trust_value=-6.0,
                trust_score=77.0,
            )
        )
        ctx = asyncio.run(
            planner._perceive_environment(
                "demo-stream",
                "user-1",
                hints={},
                world_snapshot=snapshot,
            )
        )
        self.assertEqual(ctx.trust_value, -6.0)
        self.assertEqual(ctx.trust_score, -6.0)

    def test_emotion_tracker_state_dict_mirrors_canonical_trust(self):
        state = UserEmotionState(user_id="u", trust_score=-12.0, trust=88.0)
        payload = state.to_dict()
        self.assertEqual(payload["trust_value"], -12.0)
        self.assertEqual(payload["trust_score"], -12.0)
        self.assertEqual(payload["trust"], -12.0)

    def test_emotion_tracker_dict_to_state_prefers_canonical_trust_value(self):
        tracker = EmotionTracker.__new__(EmotionTracker)
        tracker.stream_id = "demo-stream"
        state = tracker._dict_to_state(
            {
                "user_id": "u",
                "stream_id": "demo-stream",
                "trust_value": -8.0,
                "trust_score": 55.0,
                "trust": 99.0,
            }
        )
        self.assertEqual(state.trust_value, -8.0)
        self.assertEqual(state.trust_score, -8.0)
        self.assertEqual(state.trust, -8.0)

    def test_emotion_database_normalizes_dict_backed_trust_state(self):
        db = EmotionDatabase("demo-stream")
        db._initialized = True
        db._persist_user_states = lambda: True
        self.assertTrue(db.save_user_state("u", {"trust_score": -6.0}))
        stored = db.load_user_state("u")
        self.assertEqual(stored["trust_value"], -6.0)
        self.assertEqual(stored["trust_score"], -6.0)
        self.assertEqual(stored["trust"], -6.0)
        self.assertTrue(db.update_trust("u", 4.0, "recover"))
        stored = db.load_user_state("u")
        self.assertEqual(stored["trust_value"], -2.0)
        self.assertEqual(stored["trust_score"], -2.0)
        self.assertEqual(stored["trust"], -2.0)

    def test_emotion_database_dict_backed_updates_and_statistics(self):
        db = EmotionDatabase("demo-stream")
        db._initialized = True
        db._persist_user_states = lambda: True
        db._user_states = {
            "u1": {
                "user_id": "u1",
                "affection": 3.0,
                "relationship": "朋友",
                "interaction_count": 2,
                "is_blocked": False,
                "baseline_affection": 1.0,
                "created_at": time.time() - 7200,
            },
            "u2": {
                "user_id": "u2",
                "affection": -4.0,
                "relationship": "陌生人",
                "interaction_count": 1,
                "is_blocked": True,
                "baseline_affection": -2.0,
                "created_at": time.time() - 7200,
            },
        }
        self.assertTrue(db.update_affection("u1", 2.0, "good"))
        stats = db.get_statistics()
        ranking = db.get_user_affection_ranking(1)
        self.assertEqual(db._user_states["u1"]["affection"], 5.0)
        self.assertAlmostEqual(stats["avg_affection"], 0.5, places=4)
        self.assertEqual(stats["total_interactions"], 4)
        self.assertEqual(stats["blocked_users"], 1)
        self.assertEqual(ranking[0]["user_id"], "u1")

    def test_fondness_trust_serialization_uses_canonical_trust_value(self):
        dim = FondnessTrustDimension()
        dim._states["u@c"] = UserRelationState(  # pylint: disable=protected-access
            fondness_value=12.0,
            trust_value=23.0,
            positive_count=2,
            negative_count=1,
            last_interaction_ts=1.0,
            total_interactions=3,
        )
        payload = dim.serialize()
        self.assertEqual(payload["u@c"]["trust_value"], 23.0)
        self.assertEqual(payload["u@c"]["trust"], 23.0)

        dim2 = FondnessTrustDimension()
        dim2.deserialize(
            {
                "u@c": {
                    "fondness": 1.0,
                    "trust": 5.0,
                    "trust_value": 9.0,
                    "positive": 0,
                    "negative": 0,
                    "last_ts": 0.0,
                    "total": 0,
                }
            }
        )
        self.assertEqual(dim2._states["u@c"].trust_value, 9.0)  # pylint: disable=protected-access

    def test_state_dashboard_mood_bar_prefers_annoyance_value(self):
        engine = StateDashboardEngine("demo-stream")
        snapshot = engine.build_dashboard(
            force=True,
            raw_sources={
                "mood": 0.5,
                "curiosity": 0.3,
                "boredom": 0.0,
                "loafing": 0.0,
                "annoyance": 0.1,
                "annoyance_value": 0.9,
                "proactive_drive": 0.3,
            },
        )
        self.assertAlmostEqual(snapshot.mood.annoyance, 0.9)

    def test_quiet_patrol_note_activity_silently_degrades_without_event_loop(self):
        monitor = QuietPeriodMonitor()
        with patch(
            "src.chat.proactive.silence_watcher.asyncio.get_running_loop",
            side_effect=RuntimeError("no loop"),
        ), patch("src.chat.proactive.silence_watcher.logger.warning") as mock_warn:
            monitor.note_activity("demo-stream", idle_minutes=1)
        rec = monitor._pulse_table["demo-stream"]  # pylint: disable=protected-access
        self.assertGreater(rec.latest_any_ts, 0.0)
        self.assertGreater(rec.countdown_reset_ts, 0.0)
        self.assertIsNone(rec.countdown_handle)
        self.assertFalse(mock_warn.called)
        self.assertGreaterEqual(monitor.measure_silence_sec("demo-stream"), 0.0)

    def test_group_generator_length_guide_uses_world_snapshot_contract(self):
        from src.chat.replyer.group_generator import DefaultReplyer

        replyer = DefaultReplyer.__new__(DefaultReplyer)
        replyer.log_prefix = "[test]"
        replyer._build_recent_reply_guard = lambda *_args, **_kwargs: ""
        replyer._load_relation_rapport = lambda *_args, **_kwargs: {
            "social_value": 62.0,
            "trust_value": 41.0,
            "relationship_level": 4,
        }
        guide = replyer._build_length_guide(
            user_msg_length=18,
            recent_messages=[],
            chat_id="demo-stream",
            user_id="user-1",
            target_text="继续那个补丁",
        )
        self.assertIn("印象不错", guide)
        self.assertIn("有一定信任基础", guide)

    def test_private_generator_emotional_context_prefers_world_snapshot_relation_values(self):
        from src.chat.replyer.private_generator import PrivateReplyer

        replyer = PrivateReplyer.__new__(PrivateReplyer)
        with patch(
            "src.core.world_snapshot.build_relation_rapport_snapshot",
            return_value={
                "relationship": "冷淡期",
                "affection": -25.0,
                "trust_value": -4.0,
                "annoyance_value": 66.0,
                "psychological_pressure": 18.0,
                "trauma_score": 7.0,
            },
        ), patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=SimpleNamespace(
                get_user_state=lambda *_args, **_kwargs: SimpleNamespace(
                    nickname="对方",
                    mood="平静",
                    attitude="中立",
                    affection=88.0,
                    annoyance=3.0,
                    trauma_score=0.0,
                    intimacy=0.0,
                    psychological_pressure=0.0,
                    surface_mask=0.0,
                    inner_chaos=0.0,
                    mental_fatigue=0.0,
                    relationship="熟人",
                    is_blocked=False,
                )
            ),
        ), patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.AffectionStageMapper",
            new=SimpleNamespace(build_affinity_context=lambda _emo: ""),
            create=True,
        ), patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.AffectionStageMapper.build_affinity_context",
            return_value="",
            create=True,
        ), patch(
            "src.modules.modcore.psychological_core.get_response_mode",
            return_value="normal",
        ):
            context = asyncio.run(replyer._build_private_emotional_context("demo-stream", "user-1"))
        self.assertIn("冷淡期", context)
        self.assertIn("好感-25", context)
        self.assertIn("烦躁", context)

    def test_build_memory_retrieval_prompt_respects_internal_governor(self):
        fake_stream = SimpleNamespace(
            stream_id="demo-stream",
            group_info=None,
            user_info=None,
        )
        with patch(
            "src.memory_system.memory_retrieval._process_memory_retrieval",
            new=AsyncMock(return_value="不该被调用"),
        ) as mock_process:
            result = asyncio.run(
                build_memory_retrieval_prompt(
                    message="A: 我们刚才已经给了完整前情",
                    sender="A",
                    target="继续那个修复可以吗",
                    chat_stream=fake_stream,
                    extra_context="前情执行块: 已包含完整上下文",
                )
            )
        self.assertEqual(result, "")
        mock_process.assert_not_called()

    def test_plan_memory_retrieval_exposes_budget_and_stop_policy(self):
        plan = plan_memory_retrieval("A: 前情", "刚才我说了什么")
        self.assertEqual(plan["tool_order"], ["query_direct_memory", "search_chat_history"])
        self.assertTrue(plan["stop_after_first_hit"])
        self.assertEqual(plan["max_tool_steps"], 2)
        self.assertGreater(plan["max_result_chars"], 0)

        keyword_plan = plan_memory_retrieval("A: 前情", "刀盾attention修复")
        self.assertEqual(keyword_plan["tool_order"], ["search_chat_history", "query_direct_memory"])
        self.assertNotIn("search_memory_by_embedding", keyword_plan["tool_order"])

        ambiguous_plan = plan_memory_retrieval("A: 前情", "上次那个事继续")
        self.assertEqual(
            ambiguous_plan["tool_order"],
            ["query_direct_memory", "search_chat_history", "search_memory_by_embedding"],
        )

    def test_process_memory_retrieval_respects_governor_tool_budget(self):
        captured = {}

        async def fake_react(**kwargs):
            captured["allowed_tools"] = kwargs.get("allowed_tools")
            captured["stop_after_first_hit"] = kwargs.get("stop_after_first_hit")
            captured["max_tool_steps"] = kwargs.get("max_tool_steps")
            return True, "2分钟前，机器人自己说过：我们刚修了 attention snapshot。", [], False

        with patch(
            "src.memory_system.memory_retrieval._thinking_cache.probe",
            return_value=None,
        ), patch(
            "src.memory_system.memory_retrieval.DirectToolProbe.attempt",
            new=AsyncMock(return_value=None),
        ) as mock_probe, patch(
            "src.memory_system.memory_retrieval._react_agent_solve_question",
            new=AsyncMock(side_effect=fake_react),
        ), patch(
            "src.memory_system.memory_retrieval._store_thinking_back",
            return_value=None,
        ):
            result = asyncio.run(
                _process_memory_retrieval(
                    chat_id="demo-stream",
                    context="前情",
                    chat_history="刚才我说了什么",
                    preferred_tools=["query_direct_memory", "search_chat_history", "search_memory_by_embedding"],
                    max_tool_steps=2,
                    stop_after_first_hit=True,
                )
            )
        self.assertIn("attention snapshot", result)
        self.assertEqual(captured["allowed_tools"], ["query_direct_memory", "search_chat_history"])
        self.assertTrue(captured["stop_after_first_hit"])
        self.assertEqual(captured["max_tool_steps"], 2)
        self.assertEqual(mock_probe.await_args.kwargs.get("max_tool_steps"), 2)
        self.assertTrue(mock_probe.await_args.kwargs.get("stop_after_first_hit"))

    def test_build_memory_retrieval_prompt_trims_result_to_plan_budget(self):
        fake_stream = SimpleNamespace(
            stream_id="demo-stream",
            group_info=None,
            user_info=None,
        )
        long_answer = "答案：" + ("刀盾修复 " * 200)
        with patch(
            "src.memory_system.memory_retrieval.plan_memory_retrieval",
            return_value={
                "should_skip": False,
                "reason": "keyword_history_first",
                "query_text": "刀盾修复",
                "tool_order": ["search_chat_history", "query_direct_memory"],
                "max_tool_steps": 2,
                "max_result_chars": 120,
                "stop_after_first_hit": True,
            },
        ), patch(
            "src.memory_system.memory_retrieval._process_memory_retrieval",
            new=AsyncMock(return_value=long_answer),
        ):
            result = asyncio.run(
                build_memory_retrieval_prompt(
                    message="A: 刚才我们在修刀盾",
                    sender="A",
                    target="继续那个修复",
                    chat_stream=fake_stream,
                )
            )
        self.assertIn("记忆检索已截断", result)
        self.assertLess(len(result), 220)

    def test_direct_tool_probe_stops_after_first_governed_hit(self):
        calls = []

        async def fake_direct(query, chat_id=""):
            calls.append(("query_direct_memory", query, chat_id))
            return "2分钟前，机器人自己说过：我们刚修了 attention snapshot。"

        async def fake_history(keyword, stream_id=""):
            calls.append(("search_chat_history", keyword, stream_id))
            return "不该继续升级"

        async def fake_embed(query, stream_id="", top_k=3):
            calls.append(("search_memory_by_embedding", query, stream_id))
            return "不该继续升级"

        class _Tool:
            def __init__(self, func):
                self.execute_func = func

            async def execute(self, **kwargs):
                return await self.execute_func(**kwargs)

        class _Registry:
            def __init__(self):
                self._tools = {
                    "query_direct_memory": _Tool(fake_direct),
                    "search_chat_history": _Tool(fake_history),
                    "search_memory_by_embedding": _Tool(fake_embed),
                }

            def get_tool(self, name):
                return self._tools.get(name)

        with patch(
            "src.memory_system.memory_retrieval.get_tool_registry",
            return_value=_Registry(),
        ):
            result = asyncio.run(
                DirectToolProbe.attempt(
                    "刚才我说了什么",
                    "demo-stream",
                    preferred_tools=[
                        "query_direct_memory",
                        "search_chat_history",
                        "search_memory_by_embedding",
                    ],
                )
            )
        self.assertIn("attention snapshot", result)
        self.assertEqual([item[0] for item in calls], ["query_direct_memory"])

    def test_direct_tool_probe_weak_hit_keeps_searching_until_strong_hit(self):
        calls = []

        async def fake_direct(query, chat_id=""):
            calls.append(("query_direct_memory", query, chat_id))
            return "之前提到过修复方向，但细节没有记录。"

        async def fake_history(keyword, stream_id=""):
            calls.append(("search_chat_history", keyword, stream_id))
            return "2分钟前，机器人自己说过：我们刚修了刀盾 attention 修复。"

        async def fake_embed(query, stream_id="", top_k=3):
            calls.append(("search_memory_by_embedding", query, stream_id))
            return "不该继续升级"

        class _Tool:
            def __init__(self, func):
                self.execute_func = func

            async def execute(self, **kwargs):
                return await self.execute_func(**kwargs)

        class _Registry:
            def __init__(self):
                self._tools = {
                    "query_direct_memory": _Tool(fake_direct),
                    "search_chat_history": _Tool(fake_history),
                    "search_memory_by_embedding": _Tool(fake_embed),
                }

            def get_tool(self, name):
                return self._tools.get(name)

        with patch(
            "src.memory_system.memory_retrieval.get_tool_registry",
            return_value=_Registry(),
        ):
            result = asyncio.run(
                DirectToolProbe.attempt(
                    "刀盾 attention 修复",
                    "demo-stream",
                    preferred_tools=[
                        "query_direct_memory",
                        "search_chat_history",
                        "search_memory_by_embedding",
                    ],
                    stop_after_first_hit=True,
                )
            )
        self.assertIn("刀盾 attention 修复", result)
        self.assertEqual([item[0] for item in calls], ["query_direct_memory", "search_chat_history"])

    def test_query_direct_memory_passes_multiple_terms_to_memory_core(self):
        captured = []

        class _Hub:
            def query_memories(self, **kwargs):
                captured.append(kwargs)
                return [
                    SimpleNamespace(
                        record_id="rec-1",
                        birth_ts=time.time() - 120,
                        entry_category="conversation",
                        content="上次我们讨论了刀盾和 attention 的修复策略。",
                    )
                ]

        with patch(
            "src.memory_system.memory_core.acquire_recollection_hub",
            return_value=_Hub(),
        ):
            result = asyncio.run(
                query_direct_memory(
                    chat_id="demo-stream",
                    query="刀盾 attention 修复",
                    category="conversation",
                    limit=3,
                )
            )
        self.assertIn("刀盾", result)
        self.assertTrue(captured)
        self.assertEqual(captured[0]["keyword"], "刀盾 attention 修复")
        self.assertEqual(captured[0]["keyword_terms"], ["刀盾", "attention", "修复"])

    def test_query_direct_memory_prefers_exact_phrase_over_single_term_edge_hits(self):
        now = time.time()

        class _Hub:
            def query_memories(self, **kwargs):
                return [
                    SimpleNamespace(
                        record_id="old-phrase",
                        birth_ts=now - 7200,
                        entry_category="conversation",
                        content="上次我们讨论了刀盾 attention 修复策略和回归要点。",
                    ),
                    SimpleNamespace(
                        record_id="new-single",
                        birth_ts=now - 60,
                        entry_category="conversation",
                        content="今天又提了一下修复。",
                    ),
                ]

        with patch(
            "src.memory_system.memory_core.acquire_recollection_hub",
            return_value=_Hub(),
        ):
            result = asyncio.run(
                query_direct_memory(
                    chat_id="demo-stream",
                    query="刀盾 attention 修复",
                    category="conversation",
                    limit=2,
                )
            )

        first_line = result.splitlines()[0] if result else ""
        self.assertIn("刀盾 attention 修复策略", first_line)

    def test_query_chat_history_relevance_prefers_theme_keyword_hits(self):
        strong = SimpleNamespace(
            theme="刀盾修复记录",
            summary="昨天修了 attention 和刀盾相关问题",
            original_text="用户继续追问刀盾补丁",
            keywords='["刀盾","attention"]',
            participants='["alice"]',
            start_time=time.time() - 3600,
        )
        weak = SimpleNamespace(
            theme="杂谈",
            summary="只模糊提到修复",
            original_text="后来随口说了点别的",
            keywords='["修复"]',
            participants='["alice"]',
            start_time=time.time() - 3600,
        )
        strong_score = _score_record_relevance(strong, ["刀盾", "attention"], "alice")
        weak_score = _score_record_relevance(weak, ["刀盾", "attention"], "alice")
        self.assertGreater(strong_score, weak_score)

    def test_query_chat_history_relevance_keeps_recent_bonus_as_tie_break(self):
        older_precise = SimpleNamespace(
            theme="刀盾attention修复总结",
            summary="完整记录了刀盾 attention 修复的回归要点",
            original_text="关键补丁都在这条记录里",
            keywords='["刀盾","attention","修复"]',
            participants='["alice"]',
            start_time=time.time() - 3600 * 48,
        )
        recent_wide = SimpleNamespace(
            theme="闲聊",
            summary="提了一句修复",
            original_text="今天先这样",
            keywords='["修复"]',
            participants='["alice"]',
            start_time=time.time() - 600,
        )
        precise_score = _score_record_relevance(older_precise, ["刀盾", "attention", "修复"], "alice")
        wide_score = _score_record_relevance(recent_wide, ["刀盾", "attention", "修复"], "alice")
        self.assertGreater(precise_score, wide_score)

    def test_process_memory_retrieval_stops_when_direct_probe_exhausts_tool_budget(self):
        with patch(
            "src.memory_system.memory_retrieval._thinking_cache.probe",
            return_value=None,
        ), patch(
            "src.memory_system.memory_retrieval.DirectToolProbe.attempt",
            new=AsyncMock(return_value=(None, 2)),
        ), patch(
            "src.memory_system.memory_retrieval._react_agent_solve_question",
            new=AsyncMock(return_value=(True, "不该触发", [], False)),
        ) as react_mock:
            result = asyncio.run(
                _process_memory_retrieval(
                    chat_id="demo-stream",
                    context="前情",
                    chat_history="刀盾 attention 修复",
                    preferred_tools=["query_direct_memory", "search_chat_history"],
                    max_tool_steps=2,
                    stop_after_first_hit=True,
                )
            )
        self.assertIsNone(result)
        react_mock.assert_not_called()

    def test_memory_core_match_score_prefers_multi_term_overlap(self):
        precise = _score_memory_query_match(
            "上次我们讨论了刀盾 attention 修复策略。",
            keyword_terms=["刀盾", "attention", "修复"],
        )
        vague = _score_memory_query_match(
            "后来提过一点优化想法。",
            keyword_terms=["刀盾", "attention", "修复"],
        )
        self.assertGreater(precise, vague)

    def test_context_manager_search_context_matches_compact_multilingual_query(self):
        from src.modules.context_manager import ContextManager, create_context_message

        manager = ContextManager()
        channel_id = "ctx-search"
        now = time.time()
        manager.add_message(
            channel_id,
            create_context_message(
                content="我们刚才在修刀盾 attention 修复和上下文污染。",
                user_id="alice",
                message_id="match",
                timestamp=now,
            ),
        )
        manager.add_message(
            channel_id,
            create_context_message(
                content="先休息一下，晚点再聊。",
                user_id="bob",
                message_id="weak",
                timestamp=now + 1,
            ),
        )

        result = manager.search_context(channel_id, "刀盾attention修复", limit=2)
        self.assertTrue(result)
        self.assertEqual(result[0].message_id, "match")
        self.assertGreater(result[0].relevance_score, 0.0)

    def test_metric_has_signal_keeps_negative_relationship_values(self):
        chat = make_minimal_chat()
        self.assertTrue(chat._metric_has_signal(-12.5))
        self.assertTrue(chat._metric_has_signal(8.0))
        self.assertFalse(chat._metric_has_signal(0.0))

    def test_normalize_relation_snapshot_maps_legacy_keys_without_leaking_aliases(self):
        chat = make_minimal_chat()
        normalized = chat._normalize_relation_snapshot(
            {
                "favorability": 42.0,
                "trust_score": -6.0,
                "annoyance": 11.0,
                "pressure": 7.0,
            }
        )
        self.assertEqual(normalized["social_value"], 42.0)
        self.assertEqual(normalized["trust_value"], -6.0)
        self.assertEqual(normalized["annoyance_value"], 11.0)
        self.assertEqual(normalized["psychological_pressure"], 7.0)
        self.assertNotIn("favorability", normalized)
        self.assertNotIn("trust", normalized)
        self.assertNotIn("trust_score", normalized)
        self.assertNotIn("annoyance", normalized)
        self.assertNotIn("pressure", normalized)

        compat = chat._normalize_relation_snapshot(
            {
                "favorability": 42.0,
                "trust_score": -6.0,
                "annoyance": 11.0,
                "pressure": 7.0,
            },
            include_compat=True,
        )
        self.assertEqual(compat["favorability"], 42.0)
        self.assertEqual(compat["trust"], -6.0)
        self.assertEqual(compat["trust_score"], -6.0)
        self.assertEqual(compat["annoyance"], 11.0)
        self.assertEqual(compat["pressure"], 7.0)

    def test_sync_interest_level_uses_current_desire(self):
        chat = make_minimal_chat()
        chat._last_desire_level = 9
        chat._last_interest_level = 0.2
        synced = chat._sync_interest_level_from_desire()
        self.assertAlmostEqual(synced, 0.9)
        self.assertAlmostEqual(chat._last_interest_level, 0.9)

    def test_compute_relation_metrics_normalizes_legacy_relation_aliases(self):
        chat = make_minimal_chat()
        chat._tick_world_snapshot = None
        chat._cached_relation_result = None
        chat._cached_presence_state = None
        chat._cached_metabolism_constraints = None
        chat._inject_realtime_emotion = lambda payload: payload
        chat._engine_orch = DummyOrch(
            {
                "impression_hub": SimpleNamespace(
                    assemble_relation_snapshot=AsyncMock(
                        return_value={
                            "favorability": 20.0,
                            "trust_score": -8.0,
                            "annoyance": 13.0,
                        }
                    )
                )
            }
        )

        result = asyncio.run(chat._compute_relation_metrics([SimpleNamespace(user_id="user-1")]))

        self.assertEqual(result["social_value"], 20.0)
        self.assertEqual(result["trust_value"], -8.0)
        self.assertEqual(result["annoyance_value"], 13.0)
        self.assertNotIn("favorability", result)
        self.assertNotIn("trust_score", result)
        self.assertNotIn("annoyance", result)

    def test_post_reply_learning_mapping(self):
        msg = SimpleNamespace(user_id="user-1", processed_plain_text="你好")
        cases = [
            {
                "name": "friendly",
                "ctx": {
                    "behavior_signal": {
                        "category": "friendly",
                        "behavior_type": "friendly_chat",
                        "reason": "友好交流",
                    }
                },
                "expect_interaction": {"is_positive": True, "quality": 0.85},
                "expect_conflict": 0,
                "expect_affection": True,
            },
            {
                "name": "casual_inquiry",
                "ctx": {
                    "behavior_signal": {
                        "category": "neutral",
                        "behavior_type": "casual_inquiry",
                        "reason": "普通提问",
                    }
                },
                "expect_interaction": {"is_positive": True, "quality": 0.60},
                "expect_conflict": 0,
                "expect_affection": True,
            },
            {
                "name": "mild_repeat",
                "ctx": {
                    "behavior_signal": {
                        "category": "neutral",
                        "behavior_type": "mild_repeat",
                        "reason": "轻度重复",
                    }
                },
                "expect_interaction": None,
                "expect_conflict": 0,
                "expect_affection": False,
            },
            {
                "name": "low_info_repeat",
                "ctx": {
                    "behavior_signal": {
                        "category": "unfriendly",
                        "behavior_type": "low_info_repeat",
                        "reason": "低信息重复",
                    }
                },
                "expect_interaction": {"is_positive": False, "quality": 0.30},
                "expect_conflict": 0,
                "expect_affection": False,
            },
            {
                "name": "harassing",
                "ctx": {
                    "behavior_signal": {
                        "category": "harassing",
                        "behavior_type": "repetitive_harassment",
                        "reason": "持续辱骂",
                    }
                },
                "expect_interaction": None,
                "expect_conflict": 1,
                "expect_affection": False,
            },
        ]
        for case in cases:
            with self.subTest(case=case["name"]):
                tracker = FakeEmotionTracker()
                imp_hub = FakeImpressionHub()
                chat = make_chat_for_learning(imp_hub)
                with patch(
                    "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
                    return_value=tracker,
                ):
                    chat._integrate_post_reply_learning([msg], learning_context=case["ctx"])
                interaction_calls = [call for call in imp_hub.calls if call[0] == "interaction"]
                conflict_calls = [call for call in imp_hub.calls if call[0] == "conflict"]
                if case["expect_interaction"] is None:
                    self.assertFalse(interaction_calls)
                else:
                    self.assertEqual(len(interaction_calls), 1)
                    self.assertEqual(
                        interaction_calls[0][2]["is_positive"],
                        case["expect_interaction"]["is_positive"],
                    )
                    self.assertAlmostEqual(
                        interaction_calls[0][2]["quality"],
                        case["expect_interaction"]["quality"],
                    )
                self.assertEqual(len(conflict_calls), case["expect_conflict"])
                if case["expect_affection"]:
                    self.assertTrue(tracker.affection_updates)
                else:
                    self.assertFalse(tracker.affection_updates)

    def test_negative_interaction_does_not_update_subjective_dimensions(self):
        hub = ImpressionEvolutionHub("demo")
        flag = {"count": 0}
        hub._update_subjective_dimensions = lambda *args, **kwargs: flag.__setitem__("count", flag["count"] + 1)
        hub.record_interaction("u", is_positive=False, quality=0.2)
        self.assertEqual(flag["count"], 0)
        hub.record_interaction("u", is_positive=True, quality=0.9)
        self.assertEqual(flag["count"], 1)

    def test_impression_snapshot_preserves_external_negative_affection(self):
        hub = ImpressionEvolutionHub("demo")
        imp = hub.get_or_create("u", "user")
        imp.truth.positive_interactions = 20
        imp.truth.shared_experience_count = 5
        imp.truth.mutual_understanding_score = 0.9
        result = {
            "social_value": 0.0,
            "trust_value": 0.0,
            "annoyance_value": 0.0,
            "affection": -18.0,
            "trust_score": 0.0,
            "psychological_pressure": 0.0,
            "trauma_score": 0.0,
            "mood": "平静",
            "custom_label": "",
            "interaction_count": 0,
            "relationship_level": 2,
        }
        hub._merge_local_impression_snapshot(
            result,
            imp,
            "u",
            protected_metrics={"affection", "trust_score", "psychological_pressure", "trauma_score", "mood"},
        )
        self.assertEqual(result["affection"], -18.0)
        self.assertEqual(result["trust_score"], 0.0)
        self.assertEqual(result["mood"], "平静")

    def test_impression_hub_snapshot_prefers_tracker_trust_as_canonical(self):
        hub = ImpressionEvolutionHub("demo")
        imp = hub.get_or_create("u", "user")
        imp.truth.mutual_understanding_score = 0.9
        tracker = SimpleNamespace(
            get_user_state=lambda *_args, **_kwargs: SimpleNamespace(
                affection=0.0,
                trust_score=-14.0,
                psychological_pressure=0.0,
                trauma_score=0.0,
                mood="平静",
            )
        )
        with patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=tracker,
        ), patch(
            "src.person_info.person_info.get_unified_profile_hub",
            return_value=SimpleNamespace(get_top_relationships=lambda *_args, **_kwargs: []),
        ):
            result = asyncio.run(hub.assemble_relation_snapshot("u"))
        self.assertEqual(result["trust_value"], -14.0)
        self.assertEqual(result["trust_score"], -14.0)

    def test_impression_hub_snapshot_mirrors_local_trust_when_tracker_neutral(self):
        hub = ImpressionEvolutionHub("demo")
        imp = hub.get_or_create("u", "user")
        imp.truth.mutual_understanding_score = 0.6
        tracker = SimpleNamespace(
            get_user_state=lambda *_args, **_kwargs: SimpleNamespace(
                affection=0.0,
                trust_score=0.0,
                psychological_pressure=0.0,
                trauma_score=0.0,
                mood="平静",
            )
        )
        with patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=tracker,
        ), patch(
            "src.person_info.person_info.get_unified_profile_hub",
            return_value=SimpleNamespace(get_top_relationships=lambda *_args, **_kwargs: []),
        ):
            result = asyncio.run(hub.assemble_relation_snapshot("u"))
        self.assertEqual(result["trust_value"], 30.0)
        self.assertEqual(result["trust_score"], 30.0)

    def test_world_snapshot_emotion_state_uses_tracker_trust_only_as_fallback(self):
        snap = WorldSnapshot(target_user=TargetUserState(trust_value=0.0))
        tracker = SimpleNamespace(
            get_user_state=lambda *_args, **_kwargs: SimpleNamespace(
                affection=0.0,
                trust_score=-11.0,
                trauma_score=0.0,
                psychological_pressure=0.0,
                mood="平静",
            )
        )
        with patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=tracker,
        ):
            _collect_emotion_state_sync(snap, "demo-stream", "user-1")
        self.assertEqual(snap.target_user.trust_value, -11.0)
        self.assertEqual(snap.target_user.trust_score, -11.0)

        snap = WorldSnapshot(target_user=TargetUserState(trust_value=22.0))
        with patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=tracker,
        ):
            _collect_emotion_state_sync(snap, "demo-stream", "user-1")
        self.assertEqual(snap.target_user.trust_value, 22.0)
        self.assertEqual(snap.target_user.trust_score, -11.0)

    def test_mark_user_replied_resolve_ledger_uses_ack_result(self):
        latest_msg = SimpleNamespace(user_id="user-1", processed_plain_text="收到")
        for ack_value, expected_resolve in ((True, False), (False, True)):
            with self.subTest(ack_value=ack_value):
                ledger = FakeLedger(ack_value=ack_value)
                integration_hub = FakeIntegrationHub()
                chat = make_minimal_chat()
                chat._unanswered_bot_turns = 1
                chat._last_proactive_target_user_id = "user-1"
                with patch(
                    "src.chat.proactive.proactive_decider.get_proactive_decider",
                    return_value=FakeDecider(ledger),
                ), patch(
                    "src.chat.proactive.proactive_integration_hub.get_proactive_integration_hub",
                    return_value=integration_hub,
                ):
                    chat._mark_user_replied(latest_msg)
                self.assertEqual(chat._unanswered_bot_turns, 0)
                self.assertEqual(integration_hub.user_calls[-1], ("demo-stream", "user-1", "收到"))
                self.assertEqual(integration_hub.outcome_calls[-1]["resolve_ledger"], expected_resolve)

    def test_proactive_outcome_compensation_keeps_structured_consequence(self):
        ledger = FakeOutcomeLedger()
        hub = ProactiveIntegrationHub()
        hub._lazy_load_modules = lambda: None
        hub._decider = SimpleNamespace(ledger=ledger)
        hub._emotion_core = None
        consequence = hub.on_proactive_outcome(
            channel_id="demo-stream",
            has_reply=True,
            reply_sentiment=0.0,
            reply_relevance=0.1,
            resolve_ledger=True,
            target_user="user-9",
        )
        self.assertEqual(len(ledger.resolve_with_consequence_calls), 1)
        channel_id, recorded_consequence, target_user = ledger.resolve_with_consequence_calls[0]
        self.assertEqual(channel_id, "demo-stream")
        self.assertEqual(recorded_consequence.outcome_tag, "deflected")
        self.assertEqual(target_user, "user-9")
        self.assertFalse(ledger.resolve_pending_calls)
        self.assertEqual(consequence.outcome_tag, "deflected")

    def test_collect_emotion_state_uses_channel_lookup_for_memoir(self):
        hub = ProactiveIntegrationHub()
        fake_memoir = FakeMemoirCabinet()
        hub._emotion_core = SimpleNamespace(
            tick=lambda channel_id, silence_sec, unanswered: SimpleNamespace(
                boredom=0.1,
                loneliness=0.2,
                environmental_fatigue=0.3,
                social_desire=0.4,
                mood="calm",
                energy=0.9,
                curiosity=0.5,
                proactive_willingness=0.6,
                feeling_description="ok",
                silence_duration=silence_sec,
                unanswered_count=unanswered,
                monitoring=False,
            )
        )
        hub._quiet_monitor = SimpleNamespace(measure_silence_sec=lambda _channel_id: 12.0)
        hub._memoir_cabinet = fake_memoir
        result = hub._collect_emotion_state("channel-x")
        self.assertEqual(result["unanswered_count"], 3)
        self.assertEqual(fake_memoir.lookup_by_channel_calls, ["channel-x"])
        self.assertFalse(fake_memoir.lookup_sync_calls)

    def test_collect_session_data_uses_channel_lookup_for_memoir(self):
        hub = ProactiveIntegrationHub()
        fake_memoir = FakeMemoirCabinet()
        hub._memoir_cabinet = fake_memoir
        result = hub._collect_session_data("channel-y")
        self.assertEqual(result["exchange_tally"], 7)
        self.assertEqual(result["consecutive_timeouts"], 3)
        self.assertTrue(result["is_waiting"])
        self.assertEqual(fake_memoir.lookup_by_channel_calls, ["channel-y"])
        self.assertFalse(fake_memoir.lookup_sync_calls)

    def test_proactive_integration_hub_records_awareness_content(self):
        hub = ProactiveIntegrationHub()
        hub._lazy_load_modules = lambda: None
        hub._emotion_core = None
        hub._quiet_monitor = None
        hub._memoir_cabinet = None
        awareness = FakeSelfAwareness()
        hub._self_awareness = awareness

        hub.on_bot_message_sent("demo-stream", user_id="user-1", was_proactive=True, content="  主动打个招呼  ")
        hub.on_user_message_received("demo-stream", user_id="user-2", content="  我看到了  ")

        self.assertEqual(awareness.action_calls[-1]["content"], "主动打个招呼")
        self.assertEqual(awareness.action_calls[-1]["result"], "sent")
        self.assertEqual(awareness.event_calls[-1]["content"], "我看到了")

    def test_collect_self_awareness_returns_recent_content_summaries(self):
        hub = ProactiveIntegrationHub()
        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(msg_id="m1", content="  最近发过的话  ", sent_at=10.0),
        ]
        awareness.recent_actions = [
            SimpleNamespace(action_type="reply", target="user-1", content="  已回复  ", result="sent", timestamp=20.0),
        ]
        awareness.recent_events = [
            SimpleNamespace(event_type="user_message", user_id="user-1", content="  对方说你好  ", timestamp=30.0),
        ]
        awareness.stats = {"stream_id": "demo-stream", "total_messages": 1}
        hub._self_awareness = awareness

        result = hub._collect_self_awareness("demo-stream")

        self.assertEqual(result["recent_messages_count"], 1)
        self.assertEqual(result["recent_actions_count"], 1)
        self.assertEqual(result["recent_events_count"], 1)
        self.assertEqual(result["recent_messages"][0]["content"], "最近发过的话")
        self.assertEqual(result["recent_actions"][0]["content"], "已回复")
        self.assertEqual(result["recent_events"][0]["content"], "对方说你好")

    def test_on_bot_sent_separates_proactive_and_reactive_counters(self):
        ledger = FakeLedger()
        integration_hub = FakeIntegrationHub()
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "上一条回复", "ts": time.time(), "message_id": "bot-msg-0"}]
        with patch(
            "src.chat.proactive.proactive_decider.get_proactive_decider",
            return_value=FakeDecider(ledger),
        ), patch(
            "src.chat.proactive.proactive_integration_hub.get_proactive_integration_hub",
            return_value=integration_hub,
        ):
            chat._on_bot_sent(was_proactive=False)
            self.assertEqual(chat._bot_reply_counter, 1)
            self.assertEqual(len(chat._reactive_reply_timeline), 1)
            self.assertEqual(len(chat._proactive_reply_timeline), 0)
            self.assertEqual(chat._hourly_reply_count, 1)
            self.assertEqual(chat._hourly_proactive_reply_count, 0)
            chat._on_bot_sent(was_proactive=True)
            self.assertEqual(chat._bot_reply_counter, 2)
            self.assertEqual(len(chat._reactive_reply_timeline), 1)
            self.assertEqual(len(chat._proactive_reply_timeline), 1)
            self.assertEqual(chat._unanswered_bot_turns, 1)
            self.assertEqual(chat._hourly_reply_count, 2)
            self.assertEqual(chat._hourly_proactive_reply_count, 1)
            self.assertEqual(integration_hub.bot_calls[-1], ("demo-stream", "", True, "上一条回复"))
            self.assertEqual(ledger.register_calls[-1], ("demo-stream", "intent-1"))

    def test_build_synthetic_self_messages_prefers_self_awareness_records(self):
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "fallback reply", "ts": 1.0, "message_id": "fallback-msg"}]
        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(msg_id="bot-msg-1", content="第一条自我账本", sent_at=10.0),
            SimpleNamespace(msg_id="bot-msg-2", content="第二条自我账本", sent_at=20.0),
        ]

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ):
            result = chat._build_synthetic_self_messages()

        self.assertEqual(
            [item.processed_plain_text for item in result],
            ["第一条自我账本", "第二条自我账本"],
        )
        self.assertEqual(result[0].message_id, "bot-msg-1")
        self.assertEqual(result[1].message_id, "bot-msg-2")

    def test_recent_bot_texts_prefers_self_awareness_records(self):
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "fallback reply", "ts": 1.0, "message_id": "fallback-msg"}]
        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(msg_id="bot-msg-1", content="第一条自我账本", sent_at=10.0),
            SimpleNamespace(msg_id="bot-msg-2", content="第二条自我账本", sent_at=20.0),
        ]

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ):
            result = chat._recent_bot_texts()

        self.assertEqual(result, ["第一条自我账本", "第二条自我账本"])

    def test_build_self_reply_memory_uses_primary_bot_text_source(self):
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "fallback reply", "ts": 1.0, "message_id": "fallback-msg"}]
        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(msg_id="bot-msg-1", content="第一条自我账本", sent_at=10.0),
            SimpleNamespace(msg_id="bot-msg-2", content="第二条自我账本", sent_at=20.0),
        ]

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ):
            memory = chat._build_self_reply_memory()
            continuity = chat._build_self_continuity_context()

        self.assertIn("第一条自我账本", memory)
        self.assertIn("第二条自我账本", memory)
        self.assertNotIn("fallback reply", memory)
        self.assertIn("第一条自我账本", continuity)
        self.assertIn("第二条自我账本", continuity)

    def test_self_reply_recognizer_build_recent_message_snapshots_merges_awareness_and_chain(self):
        recognizer = SelfReplyRecognizer()
        recognizer._bot_messages.clear()
        recognizer._content_hash_map.clear()
        recognizer._reply_chains.clear()
        recognizer._message_to_chain.clear()
        recognizer._bot_user_ids.clear()
        recognizer.mark_bot_message(
            message_id="bot-msg-1",
            content="第一条自我账本",
            source=MessageSource.BOT_REPLY,
            response_to="user-msg-1",
            context={"stream_id": "demo-stream", "target_user_id": "user-1"},
        )

        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(
                msg_id="bot-msg-1",
                content="第一条自我账本",
                sent_at=10.0,
                context={"target_user_id": "user-1"},
            ),
        ]

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ):
            snapshots = recognizer.build_recent_message_snapshots("demo-stream", limit=3)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["message_id"], "bot-msg-1")
        self.assertEqual(snapshots[0]["text"], "第一条自我账本")
        self.assertEqual(snapshots[0]["source"], "bot_reply")
        self.assertEqual(snapshots[0]["response_to"], "user-msg-1")
        self.assertEqual(snapshots[0]["target_user_id"], "user-1")
        self.assertTrue(snapshots[0]["has_recognizer_record"])
        self.assertGreaterEqual(snapshots[0]["chain_depth"], 2)
        recognizer._bot_messages.clear()
        recognizer._content_hash_map.clear()
        recognizer._reply_chains.clear()
        recognizer._message_to_chain.clear()

    def test_collect_recent_bot_utterance_entries_prefers_recognizer_snapshot_contract(self):
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "fallback reply", "ts": 1.0, "message_id": "fallback-msg"}]
        recognizer = SimpleNamespace(
            build_recent_message_snapshots=lambda _channel_id, limit=3: [
                {
                    "text": "统一快照正文",
                    "ts": 12.0,
                    "message_id": "bot-msg-12",
                    "source": "bot_reply",
                    "response_to": "user-msg-1",
                    "chain_id": "chain-1",
                    "chain_depth": 2,
                    "target_user_id": "user-1",
                    "has_recognizer_record": True,
                }
            ][:limit]
        )

        with patch(
            "src.core.self_reply_recognizer.get_self_reply_recognizer",
            return_value=recognizer,
        ):
            entries = chat._collect_recent_bot_utterance_entries(limit=3)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "统一快照正文")
        self.assertEqual(entries[0]["message_id"], "bot-msg-12")
        self.assertEqual(entries[0]["chain_id"], "chain-1")

    def test_reply_coordinator_recent_bot_texts_prefers_self_awareness_records(self):
        coordinator = ReplyCoordinator()
        coordinator.remember_bot_utterance("demo-stream", "fallback reply")
        awareness = FakeSelfAwareness()
        awareness.recent_messages = [
            SimpleNamespace(msg_id="bot-msg-1", content="第一条自我账本", sent_at=10.0),
            SimpleNamespace(msg_id="bot-msg-2", content="第二条自我账本", sent_at=20.0),
        ]

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ):
            entries = coordinator.recent_bot_utterance_entries("demo-stream")
            texts = coordinator.recent_bot_texts("demo-stream")

        self.assertEqual([item["text"] for item in entries], ["第一条自我账本", "第二条自我账本"])
        self.assertEqual(texts, ["第一条自我账本", "第二条自我账本"])
        self.assertEqual(entries[0]["message_id"], "bot-msg-1")

    def test_reply_coordinator_bind_utterance_message_id_updates_transport_cache(self):
        coordinator = ReplyCoordinator()
        coordinator.remember_bot_utterance("demo-stream", "测试回复正文")

        self.assertTrue(coordinator.bind_utterance_message_id("demo-stream", "测试回复正文", "bot-msg-9"))
        entries = coordinator.recent_bot_utterance_entries("demo-stream")
        self.assertEqual(entries[-1]["message_id"], "bot-msg-9")

    def test_register_bot_message_record_writes_self_awareness_message(self):
        chat = make_minimal_chat()
        chat._recent_bot_utterances = [{"text": "测试回复正文", "ts": time.time()}]
        awareness = FakeSelfAwareness()

        with patch(
            "src.core.self_reply_recognizer.get_self_reply_recognizer",
            return_value=SimpleNamespace(
                register_bot_user=lambda *_args, **_kwargs: None,
                mark_bot_message=lambda **_kwargs: None,
            ),
        ), patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ), patch(
            "src.chat.heart_flow.reply_coordinator.acquire_reply_coordinator",
            return_value=SimpleNamespace(bind_utterance_message_id=lambda *_args, **_kwargs: True),
        ):
            chat._register_bot_message_record(
                reply_text="测试回复正文",
                loop_info={"message_id": "bot-msg-9"},
                target_message=SimpleNamespace(message_id="user-msg-1", user_id="user-1"),
                was_proactive=False,
            )

        self.assertEqual(awareness.message_calls[-1]["msg_id"], "bot-msg-9")
        self.assertEqual(awareness.message_calls[-1]["content"], "测试回复正文")
        self.assertEqual(awareness.message_calls[-1]["context"]["response_to"], "user-msg-1")
        self.assertEqual(chat._recent_bot_utterances[-1]["message_id"], "bot-msg-9")

    def test_collect_recent_bot_utterance_entries_falls_back_to_reply_coordinator_cache(self):
        chat = make_minimal_chat()
        coordinator = SimpleNamespace(
            recent_bot_utterance_entries=lambda _channel_id, limit=3: [
                {"text": "协调器缓存正文", "ts": 12.0, "message_id": "bot-msg-12"}
            ][:limit]
        )
        awareness = FakeSelfAwareness()

        with patch(
            "src.modules.recall.self_awareness.get_self_awareness",
            return_value=awareness,
        ), patch(
            "src.chat.heart_flow.reply_coordinator.acquire_reply_coordinator",
            return_value=coordinator,
        ):
            entries = chat._collect_recent_bot_utterance_entries(limit=3)

        self.assertEqual(entries, [{"text": "协调器缓存正文", "ts": 12.0, "message_id": "bot-msg-12"}])

    def test_self_restraint_proactive_source_uses_proactive_timeline(self):
        maker = FakeDecisionMaker()
        chat = make_minimal_chat()
        now = time.time()
        chat._proactive_reply_timeline = [now - 10, now - 20, now - 30, now - 40]
        chat._bot_reply_timeline = [now - 10, now - 20]
        with patch(
            "src.chat.heart_flow.heartflow_decision.acquire_decision_maker",
            return_value=maker,
        ), patch(
            "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
            return_value=SimpleNamespace(_ensure_channel=lambda _sid: None),
        ):
            proactive_result = asyncio.run(chat._run_self_restraint_check([], source="proactive"))
            reactive_result = asyncio.run(chat._run_self_restraint_check([], source="reactive"))
        self.assertEqual(proactive_result["recent_reply_burst"], 4)
        self.assertEqual(reactive_result["recent_reply_burst"], 2)

    def test_prepare_decision_messages_marks_sources_and_keeps_primary_target_on_incoming(self):
        chat = make_minimal_chat()
        now = time.time()
        chat._recent_bot_utterances = [{"text": "刚刚我说过的话", "ts": now}]
        incoming = SimpleNamespace(
            user_id="user-in",
            processed_plain_text="这条是当前消息",
            plain_text="这条是当前消息",
            content="这条是当前消息",
            timestamp=now - 5,
            message_id="msg-in",
        )
        context_item = SimpleNamespace(
            message_id="ctx-1",
            user_id="user-old",
            user_name="旧人",
            content="旧上下文消息",
            timestamp=now + 5,
        )

        fake_context_manager = SimpleNamespace(
            get_recent_context=lambda _channel_id, limit=10: [context_item],
            search_context=lambda _channel_id, _query, limit=5: [],
        )
        with patch(
            "src.modules.context_manager.get_context_manager",
            return_value=fake_context_manager,
        ):
            decision_messages = chat._prepare_decision_messages(
                incoming_batch=[incoming],
                filtered_messages=[incoming],
                pinged_msg=None,
            )

        sources = [chat._get_decision_message_source(msg) for msg in decision_messages]
        self.assertIn("incoming", sources)
        self.assertIn("historical_context", sources)
        self.assertIn("synthetic_self", sources)
        self.assertEqual(chat._get_latest_human_message(decision_messages).user_id, "user-in")
        self.assertEqual(len(chat._last_decision_incoming_messages), 1)
        self.assertEqual(len(chat._last_decision_context_messages), 1)
        self.assertEqual(len(chat._last_decision_self_messages), 1)

    def test_decision_context_packet_prefers_incoming_user_over_context_backfill(self):
        chat = make_minimal_chat()
        incoming = chat._tag_decision_message(
            SimpleNamespace(
                user_id="user-now",
                user_nickname="当前用户",
                processed_plain_text="当前轮消息",
                content="当前轮消息",
                timestamp=10.0,
            ),
            "incoming",
        )
        context_msg = chat._tag_decision_message(
            SimpleNamespace(
                user_id="user-old",
                user_nickname="旧上下文用户",
                processed_plain_text="旧上下文",
                content="旧上下文",
                timestamp=20.0,
            ),
            "historical_context",
        )
        bot_msg = chat._tag_decision_message(
            SimpleNamespace(
                user_id="bot",
                processed_plain_text="刚刚我说过的话",
                content="刚刚我说过的话",
                timestamp=30.0,
            ),
            "synthetic_self",
        )

        packet = chat._build_decision_context_packet([incoming, context_msg, bot_msg])
        self.assertEqual(packet.target_user_id, "user-now")
        self.assertEqual(packet.target_name, "当前用户")
        self.assertEqual(packet.latest_user_text, "当前轮消息")
        self.assertEqual(packet.latest_bot_text, "刚刚我说过的话")

    def test_repetition_pressure_ignores_historical_context_backfill(self):
        chat = make_minimal_chat()
        incoming = chat._tag_decision_message(
            SimpleNamespace(
                user_id="user-now",
                processed_plain_text="现在这句并不重复",
                content="现在这句并不重复",
                timestamp=10.0,
            ),
            "incoming",
        )
        historical = [
            chat._tag_decision_message(
                SimpleNamespace(
                    user_id=f"user-old-{idx}",
                    processed_plain_text="哈哈哈哈",
                    content="哈哈哈哈",
                    timestamp=20.0 + idx,
                ),
                "historical_context",
            )
            for idx in range(3)
        ]

        result = chat._analyze_repetition_pressure([incoming, *historical])
        self.assertFalse(result["detected"])
        self.assertEqual(result["exact_repeat_count"], 0)

    def test_run_peek_with_reflection_uses_voice_governor(self):
        chat = make_minimal_chat()
        chat._cached_scene_snapshot = SimpleNamespace(suitable_to_join=False, atmosphere=None)
        chat._cached_presence_state = SimpleNamespace(quiet_preference=1.0)

        async def _noop_observe(_incoming_batch):
            return None

        async def _should_not_run(*args, **kwargs):
            raise AssertionError("peek path should not bypass voice governor")

        chat._run_peek_observe_loop = _noop_observe
        chat._invoke_inner_voice = _should_not_run
        result = asyncio.run(
            chat._run_peek_with_reflection(
                [SimpleNamespace(user_id="user-1", processed_plain_text="看看这个", content="看看这个")]
            )
        )
        self.assertIsNone(result)
        self.assertEqual(chat._last_voice_ts, 0.0)

    def test_group_sense_analyze_uses_internal_history_window_not_only_current_batch(self):
        from src.modules.perception.group_sense import GroupSense

        sense = GroupSense()
        now = time.time()
        sense.analyze(
            "demo-stream",
            [
                {"sender_id": "user-1", "timestamp": now - 40, "text": "attention 修复"},
                {"sender_id": "user-2", "timestamp": now - 30, "text": "继续推进"},
                {"sender_id": "user-1", "timestamp": now - 20, "text": "先别回退"},
            ],
        )

        result = sense.analyze(
            "demo-stream",
            [
                {"sender_id": "user-3", "timestamp": now - 5, "text": "最后补一句"},
            ],
        )

        self.assertGreaterEqual(result.message_count_5min, 4)
        self.assertGreaterEqual(result.active_user_count, 3)

    def test_analyze_group_sense_merges_scene_contract_into_group_context_signal(self):
        from src.core.group_scene_state import AtmosphereType

        chat = make_minimal_chat()
        chat._cached_scene_snapshot = SimpleNamespace(
            suitable_to_join=False,
            join_unsuitable_reason="别人正在对线",
            atmosphere=AtmosphereType.CASUAL_CHAT,
            dominant_speaker="user-9",
        )
        chat._engine_orch = DummyOrch(
            {
                "scene_state": SimpleNamespace(
                    active_topics=lambda limit=3: ["刀盾修复", "attention", "夜聊"][:limit]
                )
            }
        )
        fake_result = SimpleNamespace(
            activity_level="活跃",
            message_count_5min=6,
            active_user_count=3,
            silence_duration_seconds=12.0,
            burst_detected=False,
            burst_density=0.0,
            controversy_detected=False,
            dominant_users=["user-9"],
            needs_topic=False,
            topic_hints=["attention", "补丁"],
            description="活跃中",
        )

        with patch(
            "src.modules.perception.group_sense.get_group_sense",
            return_value=SimpleNamespace(analyze=lambda **_kwargs: fake_result),
        ):
            result = asyncio.run(
                chat._analyze_group_sense(
                    [SimpleNamespace(user_id="user-1", timestamp=time.time(), processed_plain_text="来一条", content="来一条")]
                )
            )

        self.assertFalse(result["scene_suitable_to_join"])
        self.assertEqual(result["hard_block_reason"], "scene_constraint:unsuitable_to_join")
        self.assertEqual(result["joinability_source"], "group_scene_state")
        self.assertEqual(result["pressure_source"], "group_sense")
        self.assertEqual(result["topic_hints"], ["刀盾修复", "attention", "夜聊", "补丁"])

    def test_apply_scene_hard_constraints_uses_group_context_signal_contract(self):
        chat = make_minimal_chat()
        chat._cached_scene_snapshot = SimpleNamespace(suitable_to_join=True, join_unsuitable_reason="", atmosphere=None)

        block = chat._apply_scene_hard_constraints(
            group_context_signal={
                "scene_suitable_to_join": False,
                "scene_join_unsuitable_reason": "群里还在对话",
            }
        )

        self.assertEqual(block, "scene_constraint:unsuitable_to_join")

    def test_has_targeted_bot_message_ignores_historical_context_backfill(self):
        chat = make_minimal_chat()
        historical = chat._tag_decision_message(
            SimpleNamespace(
                user_id="user-old",
                processed_plain_text="@bot 旧上下文提及",
                content="@bot 旧上下文提及",
                is_at=True,
            ),
            "historical_context",
        )
        incoming = chat._tag_decision_message(
            SimpleNamespace(
                user_id="user-now",
                processed_plain_text="当前轮只是普通说话",
                content="当前轮只是普通说话",
                is_at=False,
                is_mentioned=False,
                mentioned_me=False,
            ),
            "incoming",
        )
        self.assertFalse(chat._has_targeted_bot_message([historical, incoming]))

    def test_force_wake_admin_ignores_historical_admin_context(self):
        chat = make_minimal_chat()
        historical_admin = chat._tag_decision_message(
            SimpleNamespace(user_id="admin-1", processed_plain_text="旧管理员消息", content="旧管理员消息"),
            "historical_context",
        )
        incoming_user = chat._tag_decision_message(
            SimpleNamespace(user_id="user-now", processed_plain_text="当前普通消息", content="当前普通消息"),
            "incoming",
        )
        with patch.object(global_config.chat, "admin_force_wake_qq_ids", ["admin-1"]):
            self.assertFalse(chat._is_force_wake_admin([historical_admin, incoming_user]))
            self.assertTrue(chat._is_force_wake_admin([incoming_user, chat._tag_decision_message(
                SimpleNamespace(user_id="admin-1", processed_plain_text="当前管理员消息", content="当前管理员消息"),
                "incoming",
            )]))

    def test_context_manager_recent_context_honors_requested_limit(self):
        from src.modules.context_manager import ContextManager, create_context_message

        manager = ContextManager()
        channel_id = "ctx-demo"
        now = time.time()
        for idx in range(8):
            manager.add_message(
                channel_id,
                create_context_message(
                    content=f"消息{idx}",
                    user_id=f"user-{idx}",
                    message_id=f"msg-{idx}",
                    timestamp=now + idx,
                ),
            )
        result = manager.get_recent_context(channel_id, limit=8)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0].message_id, "msg-0")
        self.assertEqual(result[-1].message_id, "msg-7")

    def test_flow_decision_summary_backfills_winner_layer(self):
        chat = make_minimal_chat()
        chat._reset_decision_trace()
        chat._last_legacy_gate = "block"
        chat._build_unified_flow_snapshot = lambda final_decision: SimpleNamespace(
            voice_action="hold",
            reply_desire=3,
            voice_should_reply=False,
            voice_mood="tired",
            voice_thinking="先看着",
            boredom=0.1,
            environmental_fatigue=0.2,
            loneliness=0.0,
            social_desire=0.3,
            watch_state="peek",
            phase="standby",
            is_resting=False,
            rest_remaining_sec=0.0,
            blocker="夜间",
        )
        chat._emit_flow_decision_summary("final_decision", "skip")
        self.assertEqual(chat._last_decision_trace["winner_layer"], "gateway_block")
        self.assertEqual(
            chat._last_flow_decision_summary["decision_trace"]["winner_layer"],
            "gateway_block",
        )

    def test_apply_post_reply_state_supports_dataclass_snapshot_copy(self):
        chat = make_minimal_chat()
        applied = {}
        chat._build_unified_flow_snapshot = lambda final_decision: UnifiedFlowSnapshot(
            channel_id="demo-stream",
            phase=FlowPhase.STANDBY.value,
            watch_state="peek",
            is_resting=False,
            blocker="旧阻断",
            final_decision=final_decision,
        )
        chat._apply_unified_flow_snapshot = lambda snapshot, **kwargs: applied.update(
            {"snapshot": snapshot, "kwargs": kwargs}
        )

        chat._apply_post_reply_state(did_reply=True, reason="回复完成")

        snapshot = applied["snapshot"]
        self.assertEqual(snapshot.phase, FlowPhase.ENGAGED.value)
        self.assertEqual(snapshot.watch_state, "engaged")
        self.assertEqual(snapshot.final_decision, "reply")
        self.assertEqual(snapshot.blocker, "")
        self.assertEqual(applied["kwargs"]["watch_reason"], "post_reply:engaged")

    def test_external_proactive_trace_is_reset_and_seeded(self):
        chat = make_minimal_chat()
        chat._last_decision_trace = {
            "semantic_route_summary": {"stale": True},
            "narration_should_reply": False,
            "narration_action_intent": "old",
            "narration_reply_strategy": "old",
            "legacy_gate": "block",
            "autonomy_guard_reason": "old guard",
            "voice_action": "hold",
            "model_should_reply": False,
            "final_decision": "skip",
            "winner_layer": "gateway_block",
        }
        chat._prime_external_proactive_trace("idle_proactive", desire_level=8)
        self.assertEqual(
            chat._last_decision_trace["semantic_route_summary"],
            {"source": "idle_proactive", "proactive": True},
        )
        self.assertTrue(chat._last_decision_trace["narration_should_reply"])
        self.assertEqual(chat._last_decision_trace["voice_action"], "proactive")
        self.assertTrue(chat._last_decision_trace["model_should_reply"])
        self.assertEqual(chat._last_decision_trace["winner_layer"], "algo_reply")
        self.assertEqual(chat._last_decision_trace["final_decision"], "reply")
        self.assertEqual(chat._last_desire_level, 8.0)

    def test_apply_night_cycle_modulation_clears_stale_soft_wake_state(self):
        chat = make_minimal_chat()
        chat._night_soft_wake_info = {"action": "soft_wake", "reason": "stale"}
        fake_d6 = SimpleNamespace(_get_night_mode=lambda _sid: None)
        with patch(
            "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
            return_value=fake_d6,
        ):
            result = chat._apply_night_cycle_modulation(time.time())
        self.assertIsNone(result)
        self.assertIsNone(chat._night_soft_wake_info)

    def test_normalize_night_action_falls_back_to_sleep_resist(self):
        self.assertEqual(
            EnhancedHeartFChatting._normalize_night_action("mystery_mode"),
            "sleep_resist",
        )
        self.assertEqual(
            EnhancedHeartFChatting._normalize_night_action("deep_sleep"),
            "sleep_resist",
        )
        self.assertEqual(
            EnhancedHeartFChatting._normalize_night_action(
                "deep_sleep", allow_internal=True
            ),
            "deep_sleep",
        )

    def test_prefetch_memory_hint_requires_real_relation_signal(self):
        chat = make_minimal_chat()
        chat.chat_stream = SimpleNamespace(stream_id="demo-stream")
        chat._last_relation_snapshot = {"social_value": 5.0}
        chat._get_latest_human_message = lambda messages: messages[-1]
        chat._extract_message_content = lambda msg: msg.processed_plain_text
        target_message = SimpleNamespace(
            user_id="user-1",
            user_nickname="Alice",
            processed_plain_text="你好",
        )

        with patch(
            "src.chat.utils.chat_message_builder.build_readable_messages",
            return_value="history",
        ), patch(
            "src.memory_system.memory_retrieval.build_memory_retrieval_prompt",
            new=AsyncMock(return_value=""),
        ) as memory_mock:
            result = asyncio.run(chat._prefetch_memory_hint([target_message]))

        self.assertEqual(result, "")
        self.assertEqual(memory_mock.await_args.kwargs["relation_context"], "")

    def test_prefetch_memory_hint_emits_relation_priority_for_real_high_signal(self):
        chat = make_minimal_chat()
        chat.chat_stream = SimpleNamespace(stream_id="demo-stream")
        chat._last_relation_snapshot = {"affection": 72.0, "trust_value": 81.0}
        chat._get_latest_human_message = lambda messages: messages[-1]
        chat._extract_message_content = lambda msg: msg.processed_plain_text
        target_message = SimpleNamespace(
            user_id="user-1",
            user_nickname="Alice",
            processed_plain_text="你好",
        )

        with patch(
            "src.chat.utils.chat_message_builder.build_readable_messages",
            return_value="history",
        ), patch(
            "src.memory_system.memory_retrieval.build_memory_retrieval_prompt",
            new=AsyncMock(return_value=""),
        ) as memory_mock:
            asyncio.run(chat._prefetch_memory_hint([target_message]))

        self.assertIn("[关系优先]", memory_mock.await_args.kwargs["relation_context"])

    def test_target_user_topic_requires_real_affection_signal(self):
        def evaluate_topic_bias(relation_snapshot):
            chat = make_minimal_chat()
            chat._engine_orch = DummyOrch(
                {
                    "scene_state": SimpleNamespace(
                        active_topic_slots=lambda limit=3: [
                            SimpleNamespace(ownership_type=lambda: "target_user_topic")
                        ]
                    )
                }
            )
            chat._last_relation_snapshot = relation_snapshot
            chat._cached_pattern_evidence = []
            chat._cached_self_references = []
            chat._cached_narration_plan = None
            chat._cached_multimodal_summary = None
            chat._cached_presence_state = None
            chat._cached_metabolism_state = None
            chat._cached_voice = None
            chat._cached_night_phase = None
            chat._cached_watch_level = None
            chat._cached_user_negative_emotion = 0.0
            chat._cached_ritual_behavior = None
            chat._safety_fusion_initialized = False
            chat.build_planner_injection_prompt = lambda *_args, **_kwargs: ""
            chat.register_cross_engine_outputs = lambda: None
            chat._get_dashboard_verdict = lambda: {}

            verdict = SimpleNamespace(
                activation_bar=0.7,
                breakdown={},
                fused_score=0.6,
                rationale="",
                should_proceed=True,
                intent_override=False,
            )
            fake_decider = SimpleNamespace(evaluate=lambda bundle: verdict)
            fake_d6 = SimpleNamespace(
                _ensure_channel=lambda _sid: SimpleNamespace(
                    chat_pool=50.0,
                    chat_ceiling=100.0,
                    thinking_value=50.0,
                    thinking_ceiling=100.0,
                    activity_level=50.0,
                    social_value=0.0,
                )
            )

            with patch(
                "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
                return_value=fake_d6,
            ), patch(
                "src.chat.proactive.proactive_decider.get_proactive_decider",
                return_value=fake_decider,
            ), patch(
                "src.core.multi_factor_decision_engine.get_multi_factor_engine",
                side_effect=RuntimeError("skip"),
            ), patch(
                "src.core.safety_boundary_fusion.get_safety_fusion_engine",
                side_effect=RuntimeError("skip"),
            ):
                return chat._evaluate_proactive_decision(0.0)

        no_signal = evaluate_topic_bias({})
        self.assertNotIn("target_user_low_affinity", no_signal.breakdown)
        self.assertNotIn("target_user_high_affinity", no_signal.breakdown)
        self.assertAlmostEqual(no_signal.activation_bar, 0.7)

        low_affection = evaluate_topic_bias({"affection": 10.0})
        self.assertIn("target_user_low_affinity", low_affection.breakdown)
        self.assertAlmostEqual(low_affection.activation_bar, 0.78)

        high_affection = evaluate_topic_bias({"affection": 72.0})
        self.assertIn("target_user_high_affinity", high_affection.breakdown)
        self.assertAlmostEqual(high_affection.activation_bar, 0.65)

    def test_emit_target_profile_keeps_negative_affection_in_output(self):
        chat = make_minimal_chat()
        chat._last_user_id = "user-1"
        chat._tick_world_snapshot = None
        chat._resolved_emo_state = None
        chat._inject_realtime_emotion = lambda payload: payload
        chat._resolve_relation_view = lambda *_args, **_kwargs: {
            "affection": -12.5,
            "trust_value": 0.0,
            "annoyance_value": 0.0,
            "trauma_score": 0.0,
            "psychological_pressure": 0.0,
            "custom_label": "",
            "social_value": 0.0,
        }
        fake_d6 = SimpleNamespace(
            _ensure_channel=lambda _sid: SimpleNamespace(social_value=0.0, annoyance_level=0.0),
        )
        tracker = FakeEmotionTracker()
        with patch(
            "src.modules.modcore.dynamic_persona.emotion_tracker.get_emotion_tracker",
            return_value=tracker,
        ), patch(
            "src.chat.heart_flow.heartFC_chat_enhanced.EnergyChainDimension.get_instance",
            return_value=fake_d6,
        ), patch("src.chat.heart_flow.heartFC_chat_enhanced.logger.info") as mock_info:
            asyncio.run(chat._emit_target_profile("user-1"))
        logged_text = " ".join(str(call.args[0]) for call in mock_info.call_args_list)
        self.assertIn("好感=-12.5", logged_text)

    def test_admin_force_safety_guard_reason_blocks_low_info_and_harassment(self):
        chat = make_minimal_chat()
        low_info_reason = chat._admin_force_safety_guard_reason(
            gateway_result={"reason": "allow"},
            relation_result={},
            repetition_signal={"detected": True, "low_info_cluster": True, "reason": "低信息重复施压"},
            harassment_signal={},
        )
        harassment_reason = chat._admin_force_safety_guard_reason(
            gateway_result={"reason": "allow"},
            relation_result={"behavior_signal": {"category": "harassing", "severity": 0.9}},
            repetition_signal={},
            harassment_signal={"detected": True, "reason": "持续骚扰输入"},
        )
        hostile_reason = chat._admin_force_safety_guard_reason(
            gateway_result={"reason": "allow"},
            relation_result={"behavior_signal": {"category": "hostile", "severity": 0.9}},
            repetition_signal={},
            harassment_signal={},
        )
        self.assertIn("低信息", low_info_reason)
        self.assertIn("骚扰", harassment_reason)
        self.assertIn("hostile", hostile_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
