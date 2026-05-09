import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.core.world_snapshot import WorldSnapshot
from src.common.data_models.heartflow_models import (
    BehaviorIntent,
    VoiceVerdict,
)
from src.config.core_config_engine import get_core_config

import src.chat.prompts.catalog  # noqa: F401 注册提示词
from src.chat.utils.prompt_builder import global_prompt_manager

logger = get_logger("独白系统")


def _inner_voice_view() -> Dict[str, Any]:
    try:
        return get_core_config().resolve_module_view("inner_voice").values
    except Exception:
        return {}


class ConversationScene(Enum):
    """对话场景枚举"""

    SOLO = "solo"
    ASSEMBLY = "assembly"


class SubjectMatter(Enum):
    """话题类型枚举"""

    UNKNOWN = "unknown"
    SALUTATION = "salutation"
    INQUIRY = "inquiry"
    DIRECTIVE = "directive"
    SENTIMENT = "sentiment"
    KNOWLEDGE = "knowledge"
    CHITCHAT = "chitchat"


def _load_time_bands() -> Dict[str, Tuple[int, int, int]]:
    bands_cfg = _inner_voice_view().get("time_bands", {}) or {}
    result = {}
    for name, cfg in bands_cfg.items():
        start = int(cfg.get("start", 0))
        end = int(cfg.get("end", 24))
        adj = int(cfg.get("adjustment", 0))
        result[name] = (start, end, adj)
    if not result:
        result = {
            "deep_night": (2, 6, -2),
            "late_night": (0, 2, -1),
            "evening": (22, 24, -1),
            "morning": (6, 9, 0),
            "daytime": (9, 18, 1),
            "afternoon": (18, 22, 0),
        }
    return result


TIME_BANDS = _load_time_bands()


@dataclass
class BoundaryContext:
    """护栏上下文 — 汇聚决策修正所需全部数据（含社交值）"""

    scene: ConversationScene = ConversationScene.ASSEMBLY
    mentioned_me: bool = False
    replying_to_me: bool = False
    raw_text: str = ""
    text_length: int = 0
    rapport_label: str = ""
    personal_impression: str = ""
    fondness: float = 0.0
    credence: float = 0.0
    mental_drain: float = 0.0
    endurance: float = 0.0
    irritation: float = 0.0
    wound_score: float = 0.0
    readiness: float = 0.0
    chat_reserve: float = 0.0
    brain_reserve: float = 0.0
    streak_count: int = 0
    clock_hour: int = 0
    subject: SubjectMatter = SubjectMatter.UNKNOWN
    assurance: float = 0.0
    involvement: float = 0.0
    social_value: float = 0.0
    positive_dim: float = 0.0
    negative_dim: float = 0.0
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    psychological_pressure: float = 0.0
    interaction_count: int = 0
    relationship_level: int = 2
    custom_label: str = ""
    trend_direction: str = "稳定"
    speaker_id: str = ""
    speaker_profile_summary: str = ""
    recalled_memory: str = ""
    execution_hint: str = ""
    is_blocked: bool = False
    is_admin: bool = False


class SelfDialogueEngine:
    """
    自我对话引擎：构建提示词 → 调用 LLM → 解析 JSON → 护栏修正 → 输出 VoiceVerdict
    每个频道一个实例，由全局缓存管理。
    支持跨轮记忆：保留近期想法、未完成话头、沉默理由。
    """

    _THOUGHT_WINDOW = 8
    _THREAD_CAPACITY = 5

    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._tag = f"[独白:{channel_id[:8]}]"
        # 跨轮持久状态
        self._recent_thoughts: List[str] = []
        self._unfinished_threads: List[str] = []
        self._per_user_attitudes: Dict[str, str] = {}
        self._silence_rationale: str = ""
        self._last_desire_level: int = 0
        self._last_speaker_id: str = ""
        self._last_reflection_ts: float = 0.0
        self._rounds_since_spoke: int = 0

    # ---- 主入口 ----

    async def generate_reflection(
        self,
        heart_state_label: str,
        raw_text: str,
        speaker_name: str,
        dialogue_history: List[Dict],
        assurance_score: float = 0.5,
        involvement_score: float = 0.5,
        mental_drain: float = 0.0,
        endurance: float = 100.0,
        irritation: float = 0.0,
        wound_score: float = 0.0,
        readiness: float = 1.0,
        speaker_id: str = "",
        world_snapshot: Optional["WorldSnapshot"] = None,
        extra_context: Optional[Dict[str, str]] = None,
        is_admin: bool = False,
    ) -> VoiceVerdict:
        """
        生成内心想法和回复欲望
        LLM 决定想回就回、不想回就不回，护栏仅做一致性修正
        如果提供 world_snapshot，直接从快照读取关系和资源数据
        """
        try:
            is_mentioned = self._detect_mention(raw_text)
            from src.core.world_snapshot import get_relation_number

            if world_snapshot is not None:
                rapport_data = world_snapshot.to_rapport_dict()
                res = world_snapshot.self_resources
                chat_val = res.chat_ratio() * 100.0
                thinking_val = res.thinking_ratio() * 100.0
                streak = res.consecutive_replies
            else:
                logger.debug(
                    f"{self._tag} 未收到 world_snapshot，退回兼容采集路径: "
                    "_retrieve_rapport_data + _retrieve_resource_levels"
                )
                rapport_data = self._retrieve_rapport_data(
                    speaker_name, speaker_id
                )
                chat_val, thinking_val, streak = (
                    self._retrieve_resource_levels()
                )
            subject = self._classify_subject(raw_text)
            is_assembly = bool(
                self.channel_id and not self.channel_id.startswith("private")
            )
            ctx = BoundaryContext(
                scene=(
                    ConversationScene.ASSEMBLY
                    if is_assembly
                    else ConversationScene.SOLO
                ),
                mentioned_me=is_mentioned,
                raw_text=raw_text,
                text_length=len(raw_text),
                rapport_label=str(rapport_data.get("legacy_relationship_label", "") or ""),
                personal_impression=str(
                    rapport_data.get("personal_impression", "")
                    or rapport_data.get("relationship", "")
                    or ""
                ),
                fondness=get_relation_number(
                    rapport_data,
                    "affection",
                    aliases=("favorability",),
                ),
                credence=get_relation_number(
                    rapport_data,
                    "trust_value",
                    aliases=("trust_score",),
                ),
                mental_drain=mental_drain,
                endurance=endurance,
                irritation=max(
                    irritation,
                    get_relation_number(
                        rapport_data,
                        "annoyance_value",
                        aliases=("annoyance",),
                    ),
                ),
                wound_score=wound_score,
                readiness=readiness,
                chat_reserve=chat_val,
                brain_reserve=thinking_val,
                streak_count=streak,
                clock_hour=time.localtime().tm_hour,
                subject=subject,
                assurance=assurance_score,
                involvement=involvement_score,
                social_value=get_relation_number(
                    rapport_data,
                    "social_value",
                    aliases=("favorability",),
                ),
                positive_dim=get_relation_number(rapport_data, "positive_dim"),
                negative_dim=get_relation_number(rapport_data, "negative_dim"),
                trust_value=get_relation_number(
                    rapport_data,
                    "trust_value",
                    aliases=("trust_score",),
                ),
                annoyance_value=get_relation_number(
                    rapport_data,
                    "annoyance_value",
                    aliases=("annoyance",),
                ),
                psychological_pressure=get_relation_number(
                    rapport_data,
                    "psychological_pressure",
                ),
                interaction_count=int(
                    get_relation_number(rapport_data, "interaction_count")
                ),
                relationship_level=int(
                    get_relation_number(rapport_data, "relationship_level", 2)
                ),
                custom_label=str(rapport_data.get("custom_label", "") or ""),
                trend_direction=str(
                    rapport_data.get("trend_direction", "稳定") or "稳定"
                ),
                speaker_id=speaker_id,
                speaker_profile_summary=str(
                    rapport_data.get("profile_summary", "") or ""
                ),
                recalled_memory=str(
                    rapport_data.get("memory_summary", "") or ""
                ),
                execution_hint=str(
                    rapport_data.get("execution_hint", "") or ""
                ),
                is_admin=is_admin,
            )
            # 合并来自 heartFC 的富上下文（画像、记忆、执行提示）
            if extra_context:
                _extra_profile = str(extra_context.get("profile_summary", "") or "").strip()
                _extra_memory = str(extra_context.get("memory_summary", "") or "").strip()
                _extra_exec = str(extra_context.get("execution_hint", "") or "").strip()
                if _extra_profile:
                    existing = ctx.speaker_profile_summary
                    ctx.speaker_profile_summary = (
                        f"{existing}，{_extra_profile}" if existing else _extra_profile
                    )
                if _extra_memory:
                    existing = ctx.recalled_memory
                    ctx.recalled_memory = (
                        f"{existing}\n{_extra_memory}" if existing else _extra_memory
                    )
                if _extra_exec:
                    existing = ctx.execution_hint
                    ctx.execution_hint = (
                        f"{existing}；{_extra_exec}" if existing else _extra_exec
                    )
                # 用实时数值覆盖旧快照中的滞后关系值
                _rt_ann = extra_context.get("rt_annoyance", "")
                if _rt_ann:
                    try:
                        _ann_f = float(_rt_ann)
                        if _ann_f > ctx.annoyance_value:
                            ctx.annoyance_value = _ann_f
                    except (ValueError, TypeError):
                        pass
                _rt_prs = extra_context.get("rt_pressure", "")
                if _rt_prs:
                    try:
                        _prs_f = float(_rt_prs)
                        if _prs_f > ctx.psychological_pressure:
                            ctx.psychological_pressure = _prs_f
                    except (ValueError, TypeError):
                        pass
                # 实时屏蔽状态
                _rt_blk = extra_context.get("rt_blocked", "")
                if _rt_blk and _rt_blk.lower() in ("true", "1"):
                    ctx.is_blocked = True
                # 实时烦躁也同步刷新 irritation（用于 _summarize_condition）
                if ctx.annoyance_value > ctx.irritation:
                    ctx.irritation = ctx.annoyance_value
            # 查询意图池最强未完成意图，供提示词和输出同时使用
            dominant_intent_data = None
            dominant_intent_desc = ""
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                pool = get_intention_pool()
                top_intents = pool.get_active_intentions(
                    self.channel_id, limit=1
                )
                if top_intents:
                    top = top_intents[0]
                    age_min = top.age_seconds() / 60.0
                    dominant_intent_data = {
                        "intent_id": top.intent_id,
                        "kind": top.kind.value,
                        "target_user": top.target_user,
                        "description": top.description[:120],
                        "urgency": round(float(top.effective_urgency()), 3),
                        "age_minutes": round(age_min, 1),
                        "fail_count": top.fail_count,
                    }
                    dominant_intent_desc = (
                        f"[{top.kind.value}] urgency={dominant_intent_data['urgency']:.2f} "
                        f"age={age_min:.0f}min "
                        f"desc={top.description[:60]}"
                    )
            except Exception as _e:
                logger.debug(f"{self._tag} 异常: {_e}")
            prompt_text = await self._assemble_focused_prompt(
                heart_state_label,
                raw_text,
                speaker_name,
                dialogue_history,
                ctx,
                pending_intent_desc=dominant_intent_desc,
            )
            llm_output = await self._invoke_llm(prompt_text)
            if llm_output == "__SAFETY_BLOCK__":
                return VoiceVerdict.make_error("safety_block")
            if not llm_output:
                return VoiceVerdict.make_error("llm_empty")
            verdict = self._extract_structured_reply(
                llm_output, raw_text, speaker_name
            )
            original_desire = verdict.reply_desire_level
            # 模型未显式输出 next_action 时，用模型原始 desire 推导行为意图
            # 必须在 _apply_boundaries 之前完成，以便算法层识别"模型已决策"
            if not verdict.next_action:
                # 高烦躁/已屏蔽时降级行为（避免公式盲目给 reply）
                _high_annoy = ctx.annoyance_value > 60 or ctx.is_blocked
                if _high_annoy:
                    if original_desire >= 8:
                        verdict.next_action = "wait"
                    else:
                        verdict.next_action = "observe"
                elif verdict.should_reply is True:
                    verdict.next_action = "reply"
                elif verdict.should_reply is False:
                    verdict.next_action = "observe"
                elif original_desire >= 7:
                    verdict.next_action = "reply"
                elif original_desire >= 4:
                    verdict.next_action = "wait"
                else:
                    verdict.next_action = "observe"
                logger.info(
                    f"{self._tag} [推导行为] next_action={verdict.next_action} "
                    f"(should_reply={verdict.should_reply}, desire={original_desire})"
                )
            verdict = self._apply_boundaries(verdict, ctx)
            verdict = self._stabilize_low_info_reflection(verdict, ctx)
            verdict = self._stabilize_single_speaker_reference(verdict, ctx)
            if verdict.reply_desire_level != original_desire:
                logger.info(
                    f"{self._tag} 欲望修正 {original_desire} -> {verdict.reply_desire_level}"
                )
            # 语义意图提取（优先使用 LLM 直接行为决策）
            verdict.intents = self._extract_semantic_intents(
                verdict.thinking,
                verdict.reply_desire_level,
                speaker_id,
                raw_text,
                next_action=verdict.next_action,
            )
            # 挂载最强未完成意图
            verdict.dominant_unfinished_intent = dominant_intent_data
            # 更新跨轮持久状态
            self._update_persistent_state(verdict, speaker_id)
            if verdict.thinking:
                _thinking_source = str(getattr(verdict, "thinking_source", "") or "unknown")
                logger.info(f"{self._tag} 心里想({_thinking_source}): {verdict.thinking[:80]}")
            primary = verdict.intents[0] if verdict.intents else None
            dom_tag = (
                dominant_intent_data["kind"]
                if dominant_intent_data
                else "none"
            )
            logger.info(
                f"{self._tag} 完成 | 想法='{verdict.thinking[:40] if verdict.thinking else '无'}' "
                f"| source={str(getattr(verdict, 'thinking_source', '') or 'unknown')} "
                f"| intents={len(verdict.intents)} "
                f"| primary={primary.intent_type if primary else 'none'}"
                f"| dominant_intent={dom_tag}"
                f"| silence_rounds={self._rounds_since_spoke}"
            )
            return verdict
        except Exception as exc:
            logger.error(f"{self._tag} 生成失败: {exc}")
            return VoiceVerdict.make_error(str(exc))

    # ---- 提示词构建 ----

    async def _assemble_focused_prompt(
        self,
        heart_label: str,
        raw_text: str,
        speaker_name: str,
        history: List[Dict],
        ctx: BoundaryContext,
        pending_intent_desc: str = "",
    ) -> str:
        """组装聚焦式提示词"""
        persona = self._retrieve_persona()
        lines = []
        for msg in history[-20:]:
            role = msg.get("role", "user")
            if role == "system":
                continue
            body = msg.get("content", "")[:300]
            if body:
                prefix = "我" if role == "assistant" else speaker_name
                lines.append(f"{prefix}: {body}")
        dialogue_block = "\n".join(lines) if lines else "(刚开始聊天)"
        rapport_desc = self._describe_rapport(speaker_name, ctx)
        condition_desc = self._summarize_condition(ctx)
        scene_label = (
            "群聊" if ctx.scene == ConversationScene.ASSEMBLY else "私聊"
        )
        profile_desc = (
            ctx.speaker_profile_summary or "(暂无稳定画像，按当前互动判断)"
        )
        memory_desc = ctx.recalled_memory or "(暂无直接相关记忆)"
        execution_desc = (
            ctx.execution_hint
            or "像人一样自然地想，可以联想、回忆、吐槽、疑惑。"
        )
        # 跨轮记忆上下文注入
        persistent_ctx = self._build_memory_context()
        if persistent_ctx:
            memory_desc = f"{memory_desc}\n{persistent_ctx}"
        # 未完成意图注入
        if pending_intent_desc:
            memory_desc = (
                f"{memory_desc}\n你心中最强的未完成念头: {pending_intent_desc}"
            )
        mention_note = ""
        if ctx.mentioned_me:
            mention_note = "\n对方直接叫了你的名字。你自行决定要不要回应，不要因为被叫名字就强迫自己回。"
        _is_new_context = (
            not lines and ctx.fondness < 10 and not ctx.recalled_memory
        )
        if _is_new_context:
            mention_note += (
                "\n注意：这是你们第一次交流，你对对方几乎不了解。"
                "不要用'哈''嗯''哦'这种单字敷衍——你是个有独立人格的人，"
                "就算不熟也应该有自己的判断和想法。"
                "想想这句话可能是什么意思，表达你真实的困惑或好奇。"
            )
        prompt = await global_prompt_manager.format_prompt(
            "inner_voice_focused",
            persona=persona,
            scene_label=scene_label,
            dialogue_block=dialogue_block,
            speaker_name=speaker_name,
            raw_text=raw_text,
            rapport_desc=rapport_desc,
            condition_desc=condition_desc,
            profile_desc=profile_desc,
            memory_desc=memory_desc,
            execution_desc=execution_desc,
            chat_reserve=f"{ctx.chat_reserve:.0f}",
            brain_reserve=f"{ctx.brain_reserve:.0f}",
            mention_note=mention_note,
        )
        return prompt

    # ---- 关系描述 ----

    def _describe_rapport(
        self, speaker_name: str, ctx: BoundaryContext
    ) -> str:
        """根据好感度和自由个人印象生成关系文字描述。"""
        aff = ctx.fondness
        impression = str(ctx.personal_impression or "").strip()
        impression_note = f"你对{speaker_name}的个人印象是：{impression}。" if impression else ""
        if aff < -30:
            return f"{impression_note}你对{speaker_name}明显反感(好感{aff:.0f})，本能就想躲开或敷衍。"
        if aff < -10:
            return f"{impression_note}你对{speaker_name}印象偏负面(好感{aff:.0f})，容易先烦一下。"
        if aff < 10:
            return f"{impression_note}你还没完全看清{speaker_name}(好感{aff:.0f})，先别过度解读。"
        if aff < 40:
            return f"{impression_note}你对{speaker_name}整体还能自然接话(好感{aff:.0f})。"
        return f"{impression_note}你对{speaker_name}挺有好感(好感{aff:.0f})，会想顺着多聊几句。"

    # ---- 状态描述 ----

    def _summarize_condition(self, ctx: BoundaryContext) -> str:
        """生成当前身心状态文字描述（含社交值）"""
        parts = []
        if ctx.is_blocked:
            parts.append("已经屏蔽了这人，根本不想搭理")
        if ctx.chat_reserve < 30:
            parts.append("精力快耗尽了")
        elif ctx.chat_reserve < 60:
            parts.append("有点累")
        if ctx.brain_reserve < 30:
            parts.append("思考值不足")
        if ctx.irritation > 50:
            parts.append("很烦躁")
        elif ctx.irritation > 30:
            parts.append("有点烦")
        if ctx.wound_score > 5:
            parts.append("心里有创伤很敏感")
        if ctx.psychological_pressure > 60:
            parts.append("心理压力很大")
        elif ctx.psychological_pressure > 35:
            parts.append("心理压力不小")
        if ctx.social_value < -30:
            parts.append("对这人印象很差")
        elif ctx.social_value < -10:
            parts.append("对这人不太喜欢")
        elif ctx.social_value > 40:
            parts.append("对这人挺有好感")
        elif ctx.social_value > 70:
            parts.append("很喜欢这人")
        if ctx.annoyance_value > 70:
            parts.append("这人烦死了，不想理")
        elif ctx.annoyance_value > 50:
            parts.append("这人让我很烦")
        elif ctx.annoyance_value > 30:
            parts.append("这人有点烦人")
        if ctx.trust_value < -30:
            parts.append("不信任这人")
        elif ctx.trust_value > 50:
            parts.append("信任这人")
        if ctx.personal_impression:
            parts.append(f"个人印象:{ctx.personal_impression}")
        elif ctx.custom_label:
            parts.append(f"旧关系备注:{ctx.custom_label}")
        if ctx.trend_direction == "下降":
            parts.append("关系在变差")
        elif ctx.trend_direction == "上升":
            parts.append("关系在变好")
        return "，".join(parts) if parts else "状态还行，脑子清醒"

    # ---- 护栏修正 ----

    def _apply_boundaries(
        self, verdict: VoiceVerdict, ctx: BoundaryContext
    ) -> VoiceVerdict:
        """多层护栏修正确保 thought 与 desire 逻辑一致"""
        inner_voice_cfg = _inner_voice_view()
        mention_min_desire = int(inner_voice_cfg.get("mention_min_desire", 6))
        private_min_desire = int(
            inner_voice_cfg.get("private_chat_min_desire", 5)
        )
        res_penalty_max = int(inner_voice_cfg.get("resource_penalty_max", 5))
        res_thresh = inner_voice_cfg.get("resource_thresholds", {}) or {}
        desire = verdict.reply_desire_level
        res_penalty = self._compute_resource_deficit(ctx, res_thresh)
        if res_penalty > 0 and desire > 1:
            old = desire
            desire = max(1, desire - min(res_penalty, res_penalty_max))
            if desire != old:
                logger.info(
                    f"{self._tag} [资源惩罚] chat={ctx.chat_reserve:.0f} "
                    f"think={ctx.brain_reserve:.0f} streak={ctx.streak_count} | {old}->{desire}"
                )
        social_penalty = self._compute_social_deficit(ctx)
        if social_penalty > 0 and desire > 1:
            old = desire
            desire = max(1, desire - social_penalty)
            if desire != old:
                logger.info(
                    f"{
                        self._tag} [社交值惩罚] social={
                        ctx.social_value:.0f} "
                    f"trust={
                        ctx.trust_value:.0f} annoyance={
                        ctx.annoyance_value:.0f} | {old}->{desire}"
                )
        # 当模型已给出行为决策(next_action)时，跳过被提及保底和私聊保底
        # 这些场景信息已在 prompt 中提供，模型自行判断是否回复
        _model_decided = bool(verdict.next_action)
        if not _model_decided:
            if ctx.mentioned_me and desire < mention_min_desire:
                _annoy_cap = ctx.annoyance_value
                if _annoy_cap >= 80:
                    _effective_floor = min(mention_min_desire - 2, desire + 1)
                    old = desire
                    desire = max(desire, _effective_floor)
                    if desire != old:
                        logger.info(
                            f"{self._tag} [被提及降级保底] 烦躁{_annoy_cap:.0f}≥80 {old}->{desire}"
                        )
                elif _annoy_cap >= 60:
                    _effective_floor = min(mention_min_desire - 1, desire + 1)
                    old = desire
                    desire = max(desire, _effective_floor)
                    if desire != old:
                        logger.info(
                            f"{self._tag} [被提及降级保底] 烦躁{_annoy_cap:.0f}≥60 {old}->{desire}"
                        )
                else:
                    old = desire
                    desire = mention_min_desire
                    logger.info(f"{self._tag} [被提及保底] {old}->{desire}")
            if ctx.scene == ConversationScene.SOLO and desire < private_min_desire:
                old = desire
                desire = private_min_desire
                logger.info(f"{self._tag} [私聊保底] {old}->{desire}")
        else:
            logger.debug(
                f"{self._tag} [跳过保底] 模型已决策 action={verdict.next_action}"
            )
        if (
            ctx.social_value <= 10
            and ctx.trust_value <= 10
            and ctx.annoyance_value <= 20
        ):
            thought_text = str(verdict.thinking or "")
            if thought_text:
                softened = re.sub(r"[。！？!?]+", "。", thought_text).strip()
                if len(softened) > 90:
                    softened = softened[:90].rstrip("，,；; ") + "。"
                verdict.thinking = softened
            if desire > 6:
                old = desire
                desire = 6
                logger.info(f"{self._tag} [低关系降躁] {old}->{desire}")
        if (
            not ctx.is_admin
            and (
                ctx.annoyance_value >= 35
                or ctx.mental_drain >= 55
                or ctx.readiness <= 0.35
            )
        ):
            hard_cap = 4 if not ctx.mentioned_me else 5
            if desire > hard_cap:
                old = desire
                desire = hard_cap
                logger.info(
                    f"{self._tag} [旧版式硬压制] annoyance={ctx.annoyance_value:.0f} "
                    f"drain={ctx.mental_drain:.0f} readiness={ctx.readiness:.2f} | {old}->{desire}"
                )
        if not ctx.is_admin and ctx.psychological_pressure >= 45:
            hard_cap = 3 if not ctx.mentioned_me else 5
            if desire > hard_cap:
                old = desire
                desire = hard_cap
                logger.info(
                    f"{self._tag} [心理压力硬压制] pressure={ctx.psychological_pressure:.0f} | {old}->{desire}"
                )
        if (
            not ctx.is_admin
            and ctx.social_value <= -20
            and ctx.trust_value <= 0
            and ctx.annoyance_value >= 25
        ):
            hard_cap = 3 if not ctx.mentioned_me else 5
            if desire > hard_cap:
                old = desire
                desire = hard_cap
                logger.info(
                    f"{
                        self._tag} [关系负向硬压制] social={
                        ctx.social_value:.0f} trust={
                        ctx.trust_value:.0f} "
                    f"annoyance={
                        ctx.annoyance_value:.0f} | {old}->{desire}"
                )
        normalized_raw = re.sub(r"\s+", "", str(ctx.raw_text or "")).strip()
        low_info_short = 0 < len(normalized_raw) <= 8
        repeated_chunk = False
        if normalized_raw:
            repeated_chunk = len(normalized_raw) >= 2 and len(
                set(normalized_raw)
            ) <= max(1, len(normalized_raw) // 3)
        repeated_phrase = False
        if normalized_raw:
            half = len(normalized_raw) // 2
            if half >= 2 and len(normalized_raw) % 2 == 0:
                repeated_phrase = (
                    normalized_raw[:half] == normalized_raw[half:]
                )
        if low_info_short and (repeated_chunk or repeated_phrase):
            thought_text = str(verdict.thinking or "").strip()
            if desire > 4 and not ctx.mentioned_me:
                old = desire
                desire = 4
                logger.info(f"{self._tag} [短句复读护栏] {old}->{desire}")
            if thought_text and len(thought_text) <= 6:
                verdict.thinking = f"{thought_text.rstrip('。！？!?')}，像在重复一句没说清楚的话。"
                verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+low_info_guard"
                if not verdict.current_mood or verdict.current_mood == "疑惑":
                    verdict.current_mood = "无聊"
        # 当 LLM 直接给出行为决策时，跳过基于文本分析的一致性修正
        _has_action = bool(verdict.next_action)
        if not _has_action:
            sentiment_val = self._gauge_sentiment(verdict.thinking)
            if sentiment_val > 4.0 and desire < 4:
                old = desire
                desire = min(10, desire + 2)
                logger.info(
                    f"{self._tag} [一致性] thought积极但desire低 {old}->{desire}"
                )
            elif sentiment_val < -4.0 and desire > 7:
                old = desire
                desire = max(1, desire - 2)
                logger.info(
                    f"{self._tag} [一致性] thought消极但desire高 {old}->{desire}"
                )
            rest_val = self._gauge_rest_desire(verdict.thinking)
            if rest_val > 4.0 and desire > 3:
                old = desire
                desire = min(desire, 2)
                logger.info(f"{self._tag} [一致性] 明确休息意图 {old}->{desire}")
        # 时段调节：仅在模型没有直接行为决策时才应用
        # prompt 已包含状态描述（精力/时间相关信息），模型能自主判断
        if not _model_decided:
            time_shift = self._time_period_shift(ctx.clock_hour)
            if time_shift != 0 and desire > 1:
                old = desire
                desire = max(1, min(10, desire + time_shift))
                if desire != old:
                    logger.info(
                        f"{self._tag} [时段调节] hour={ctx.clock_hour} adj={time_shift} | {old}->{desire}"
                    )
        desire = max(1, min(10, desire))
        verdict.reply_desire_level = desire
        # 护栏修正后同步 next_action：当 desire 被大幅压低时降级主动行为
        _action = verdict.next_action
        if _action in ("reply", "followup") and desire < 3:
            verdict.next_action = "observe" if desire <= 1 else "wait"
            logger.info(
                f"{self._tag} [护栏降级行为] {_action}→{verdict.next_action} (desire={desire})"
            )
        elif _action in ("rest", "disengage") and ctx.mentioned_me:
            # 被直接提及时，不允许直接放下手机/休息，至少等一下
            verdict.next_action = "wait"
            logger.info(
                f"{self._tag} [护栏提及覆盖] {_action}→wait (被提及，先等等看)"
            )
        # should_reply：模型已给出行为决策时从 next_action 推导，否则回退到欲望阈值
        if _model_decided:
            verdict.should_reply = verdict.next_action in ("reply", "followup")
        else:
            verdict.should_reply = desire >= 5
        return verdict

    def _stabilize_low_info_reflection(
        self, verdict: VoiceVerdict, ctx: BoundaryContext
    ) -> VoiceVerdict:
        """对低信息重复短句做模板化收束，避免独白无证据脑补题材。"""
        raw_text = str(ctx.raw_text or "").strip()
        normalized = re.sub(r"\s+", "", raw_text)
        if not normalized:
            return verdict

        is_short = len(normalized) <= 8
        repeated_phrase = False
        if len(normalized) >= 4 and len(normalized) % 2 == 0:
            half = len(normalized) // 2
            repeated_phrase = normalized[:half] == normalized[half:]

        has_repeated_segment = False
        if len(normalized) >= 4:
            for seg_len in range(2, len(normalized) // 2 + 1):
                seg = normalized[:seg_len]
                if len(normalized) % seg_len == 0 and all(
                    normalized[i: i + seg_len] == seg
                    for i in range(0, len(normalized), seg_len)
                ):
                    has_repeated_segment = True
                    break
                if (
                    len(normalized) >= seg_len * 3
                    and normalized.count(seg) >= 3
                ):
                    has_repeated_segment = True
                    break

        low_variety = len(set(normalized)) <= max(2, len(normalized) // 3)
        should_stabilize = is_short and (
            repeated_phrase or low_variety or has_repeated_segment
        )
        if not should_stabilize:
            return verdict

        current = str(verdict.thinking or "").strip()
        hostile_markers = ("发疯", "有病", "神经病", "脑残", "傻逼", "sb")
        if current and any(marker in current.lower() for marker in hostile_markers):
            verdict.thinking = "又在重复这句，先看懂再说。"
            verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+low_info_guard"
            if not verdict.current_mood or verdict.current_mood in ("疑惑", "困惑"):
                verdict.current_mood = "无聊"
            if verdict.reply_desire_level > 4 and not ctx.mentioned_me:
                verdict.reply_desire_level = max(1, verdict.reply_desire_level - 2)
                verdict.should_reply = verdict.reply_desire_level >= 5
            return verdict
        looks_like_guess = (
            len(current) > 6
            and verdict.reply_desire_level >= 5
            and not ctx.mentioned_me
        )
        if verdict.reply_desire_level > 4 and not ctx.mentioned_me:
            verdict.reply_desire_level = max(1, verdict.reply_desire_level - 2)
            verdict.should_reply = verdict.reply_desire_level >= 5

        if current and not looks_like_guess and len(current) <= 8:
            verdict.thinking = self._compress_thought(current)
            return verdict

        annoyance = float(ctx.annoyance_value or 0.0)
        if annoyance >= 35 or has_repeated_segment:
            verdict.thinking = "无不无聊啊，发这么多遍"
            verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+low_info_guard"
            if not verdict.current_mood or verdict.current_mood == "疑惑":
                verdict.current_mood = "心烦"
        elif annoyance >= 20:
            verdict.thinking = "还是这句，没说清楚，像在等我接前情。"
            verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+low_info_guard"
            if not verdict.current_mood or verdict.current_mood == "疑惑":
                verdict.current_mood = "无奈"
        else:
            verdict.thinking = "这句信息太少了，像在重复，我先看看再说。"
            verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+low_info_guard"
            if not verdict.current_mood or verdict.current_mood == "疑惑":
                verdict.current_mood = "无聊"
        verdict.thinking = self._compress_thought(verdict.thinking)
        return verdict

    def _stabilize_single_speaker_reference(
        self, verdict: VoiceVerdict, ctx: BoundaryContext
    ) -> VoiceVerdict:
        """独白聚焦当前说话人时，避免把用户和机器人误写成两个人。"""
        current = str(verdict.thinking or "").strip()
        if not current or not ctx.speaker_id:
            return verdict
        replacements = {
            "这俩人": "这人",
            "这两人": "这人",
            "他俩": "他",
            "他们俩": "他",
            "两个人": "这人",
        }
        normalized = current
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        if normalized != current:
            verdict.thinking = self._compress_thought(normalized)
            verdict.thinking_source = f"{verdict.thinking_source or 'unknown'}+single_speaker_guard"
        return verdict

    # ---- 情感分析 ----

    @staticmethod
    def _gauge_sentiment(thought: str) -> float:
        """基于语气强度而非词表估计 thought 情感方向"""
        if not thought:
            return 0.0
        text = thought.strip()
        length_factor = min(len(text) / 40.0, 1.5)
        exclamations = text.count("!") + text.count("！")
        questions = text.count("?") + text.count("？")
        pauses = text.count(".") + text.count("。") + text.count("…")
        line_breaks = text.count("\n")
        emphasis = min(4.0, exclamations * 0.9 + length_factor)
        restraint = min(4.0, pauses * 0.35 + line_breaks * 0.45)
        inquiry_pull = min(2.0, questions * 0.4)
        return max(-6.0, min(6.0, emphasis - restraint - inquiry_pull * 0.5))

    @staticmethod
    def _gauge_rest_desire(thought: str) -> float:
        """基于收束程度估计休息/停顿倾向"""
        if not thought:
            return 0.0
        text = thought.strip()
        if not text:
            return 0.0
        shortness = max(0.0, 1.0 - min(len(text) / 30.0, 1.0))
        pauses = text.count("…") + text.count("。") + text.count(".")
        soft_endings = sum(
            1
            for marker in ("算了", "先这样", "等等", "晚点")
            if marker in text
        )
        return min(6.0, shortness * 2.5 + pauses * 0.8 + soft_endings * 1.2)

    # ---- 惩罚计算 ----

    @staticmethod
    def _compute_resource_deficit(
        ctx: BoundaryContext, thresh: Dict[str, Any] = None
    ) -> int:
        """基于聊天精力、思考值、连续回复计算资源惩罚"""
        if thresh is None:
            thresh = {}
        chat_low = int(thresh.get("chat_reserve_low", 30))
        chat_mid = int(thresh.get("chat_reserve_mid", 50))
        brain_low = int(thresh.get("brain_reserve_low", 30))
        brain_mid = int(thresh.get("brain_reserve_mid", 50))
        streak_high = int(thresh.get("streak_high", 5))
        streak_mid = int(thresh.get("streak_mid", 3))
        penalty = 0
        if ctx.chat_reserve < chat_low:
            penalty += 2
        elif ctx.chat_reserve < chat_mid:
            penalty += 1
        if ctx.brain_reserve < brain_low:
            penalty += 2
        elif ctx.brain_reserve < brain_mid:
            penalty += 1
        if ctx.streak_count >= streak_high:
            penalty += 2
        elif ctx.streak_count >= streak_mid:
            penalty += 1
        return min(penalty, 5)

    @staticmethod
    def _time_period_shift(hour: int) -> int:
        """根据当前时段计算 desire 偏移"""
        if hour < 0:
            hour = time.localtime().tm_hour
        for _, (start, end, adj) in _load_time_bands().items():
            if start <= hour < end:
                return adj
        return 0

    @staticmethod
    def _compute_social_deficit(ctx: BoundaryContext) -> int:
        """基于社交值、信任值、厌烦值计算社交惩罚"""
        penalty = 0
        if ctx.social_value < -40:
            penalty += 3
        elif ctx.social_value < -20:
            penalty += 2
        elif ctx.social_value < -10:
            penalty += 1
        if ctx.trust_value < -40:
            penalty += 2
        elif ctx.trust_value < -20:
            penalty += 1
        if ctx.annoyance_value > 70:
            penalty += 3
        elif ctx.annoyance_value > 50:
            penalty += 2
        elif ctx.annoyance_value > 30:
            penalty += 1
        if ctx.relationship_level == 0:
            penalty += 2
        return min(penalty, 5)

    # ---- 辅助检测 ----

    def _detect_mention(self, raw_text: str) -> bool:
        """检测消息是否提及了自己"""
        try:
            from src.config.config import global_config

            bot_alias = global_config.bot.nickname or ""
            alt_names = global_config.bot.alias_names or []
            for name in [bot_alias] + alt_names:
                if name and name in raw_text:
                    return True
            if "@" in raw_text:
                return True
        except Exception as _exc:
            logger.warning(f"内心独白提及检测异常: {_exc}")
            return False
        return False

    def _classify_subject(self, text: str) -> SubjectMatter:
        """从结构信号推断话题类型，避免靠固定词表硬分类"""
        if not text:
            return SubjectMatter.UNKNOWN
        content = text.strip()
        if not content:
            return SubjectMatter.UNKNOWN
        lowered = content.lower()
        question_marks = content.count("?") + content.count("？")
        exclamations = content.count("!") + content.count("！")
        line_breaks = content.count("\n")
        length = len(content)
        punctuation_ratio = sum(
            1 for ch in content if ch in "!?！？。.,，；;:：…"
        ) / max(length, 1)
        has_link = "http://" in lowered or "https://" in lowered
        has_code_shape = (
            "```" in content
            or "=" in content
            or "(" in content
            and ")" in content
        )
        starts_with_at = content.startswith("@")

        if length <= 6 and exclamations == 0 and question_marks == 0:
            return SubjectMatter.SALUTATION
        if question_marks > 0:
            if has_link or has_code_shape or line_breaks > 0 or length > 60:
                return SubjectMatter.KNOWLEDGE
            return SubjectMatter.INQUIRY
        if starts_with_at and length < 20:
            return SubjectMatter.SALUTATION
        if line_breaks >= 2 or length > 90:
            return SubjectMatter.SENTIMENT
        if exclamations >= 2 or punctuation_ratio > 0.25:
            return SubjectMatter.SENTIMENT
        if has_link or has_code_shape:
            return SubjectMatter.KNOWLEDGE
        if length > 35:
            return SubjectMatter.CHITCHAT
        return SubjectMatter.UNKNOWN

    def _retrieve_resource_levels(self) -> Tuple[float, float, int]:
        """获取当前频道的聊天精力、思考值和连续回复数"""
        chat_val = 100.0
        thinking_val = 100.0
        streak = 0
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.channel_id)
            if _ch:
                chat_val = _ch.chat_ratio() * 100.0
                thinking_val = _ch.thinking_ratio() * 100.0
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.frequency_control import (
                acquire_frequency_manager,
            )

            fm = acquire_frequency_manager()
            streak = (
                fm.get_consecutive_replies(self.channel_id)
                if hasattr(fm, "get_consecutive_replies")
                else 0
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return chat_val, thinking_val, streak

    def _retrieve_persona(self) -> str:
        """获取人设描述片段"""
        try:
            from src.person_info.bot_identity import (
                IDENTITY_BOUNDARY_TEXT,
                get_bot_identity_manager,
            )

            identity = get_bot_identity_manager()
            persona_brief = identity.build_persona_brief(
                self.channel_id or None,
                include_name_prefix=True,
                include_style=True,
                include_interests=True,
                include_lore=True,
                fallback_text="自然说话，别端着，也别把自己说成系统或工具。",
            )
            if persona_brief:
                return (
                    f"[主人格摘要] {persona_brief}\n"
                    f"[身份边界] {IDENTITY_BOUNDARY_TEXT}"
                )
        except Exception as exc:
            logger.debug(f"构建人格提示失败: {exc}")
            from src.chat.utils.prompt_builder import global_prompt_registry

            fallback_tpl = global_prompt_registry._global_prompts.get(
                "inner_voice_persona_fallback"
            )
            return (
                str(fallback_tpl)
                if fallback_tpl
                else "你是一个拥有独立人格的灵魂"
            )

    def _retrieve_rapport_data(
        self, speaker_name: str, speaker_id: str
    ) -> Dict[str, Any]:
        """获取与说话者的关系数据"""
        result: Dict[str, Any] = {
            "relationship": "",
            "affection": 0.0,
            "trust": 0.0,
            "social_value": 0.0,
            "positive_dim": 0.0,
            "negative_dim": 0.0,
            "trust_value": 0.0,
            "annoyance_value": 0.0,
            "interaction_count": 0,
            "relationship_level": 2,
            "custom_label": "",
            "trend_direction": "稳定",
            "personal_impression": "",
            "impression_labels": [],
            "impression_source": "",
            "impression_updated_at": 0.0,
            "profile_summary": "",
            "memory_summary": "",
            "execution_hint": "先形成一句短的心里话，再决定要不要回；如果要回，后续回复必须顺着这句心里话。",
        }
        try:
            from src.core.world_snapshot import (
                build_relation_rapport_snapshot,
            )

            result.update(
                build_relation_rapport_snapshot(
                    channel_id=self.channel_id,
                    user_id=speaker_id,
                    user_name=speaker_name,
                )
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return result

    # ---- LLM 调用 ----

    async def _invoke_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM 生成原始回复文本"""
        def _is_safety_error(err_msg: str, err_type: str) -> bool:
            return (
                "EmptyContent" in err_type
                or "Safety violation" in err_msg
                or "HARASSMENT" in err_msg
            )

        async def _run_request(task_cfg, request_type: str) -> Optional[str]:
            from src.llm_models.utils_model import LLMRequest

            request = LLMRequest(task_cfg, request_type=request_type)
            try:
                response_text, _ = await asyncio.wait_for(
                    request.generate_response_async(prompt),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[内心独白] LLM请求超时(30s): {request_type}")
                response_text = None
            return response_text or None

        try:
            from src.config.config import model_config

            primary_exc: Optional[Exception] = None
            try:
                response_text = await _run_request(
                    model_config.model_task_config.focus_chat,
                    request_type="self_dialogue",
                )
                if response_text:
                    return response_text
            except Exception as exc:
                primary_exc = exc
                err_msg = str(exc)
                err_type = type(exc).__name__
                if _is_safety_error(err_msg, err_type):
                    logger.warning(f"{self._tag} 安全违规: {err_msg}")
                    return "__SAFETY_BLOCK__"
                logger.warning(f"{self._tag} 主模型调用失败，尝试utils兜底: {exc}")

            try:
                fallback_text = await _run_request(
                    model_config.model_task_config.utils,
                    request_type="self_dialogue.fallback",
                )
                if fallback_text:
                    logger.info(f"{self._tag} 独白已切换到utils兜底模型")
                    return fallback_text
            except Exception as fallback_exc:
                err_msg = str(fallback_exc)
                err_type = type(fallback_exc).__name__
                if _is_safety_error(err_msg, err_type):
                    logger.warning(f"{self._tag} 安全违规: {err_msg}")
                    return "__SAFETY_BLOCK__"
                logger.error(f"{self._tag} LLM兜底调用失败: {fallback_exc}")

            if primary_exc is not None:
                logger.error(f"{self._tag} LLM调用失败: {primary_exc}")
            return None
        except Exception as exc:
            logger.error(f"{self._tag} LLM调用失败: {exc}")
            return None

    # ---- 响应解析 ----

    def _extract_structured_reply(
        self,
        raw: str,
        raw_text: str,
        speaker_name: str,
    ) -> VoiceVerdict:
        """
        从 LLM 输出提取结构化 JSON
        依次尝试：标准JSON → 正则字段 → 数字+文本 → 语气结构兜底
        """
        text = raw.strip()
        # 去除 think 标签
        think_hit = re.search(r"</think>\s*(.+)", text, re.DOTALL)
        if think_hit:
            text = think_hit.group(1).strip()
        # 尝试提取 JSON 块（支持嵌套花括号）
        json_hit = re.search(r'\{.*?"thought".*?\}', text, re.DOTALL)
        if not json_hit:
            json_hit = re.search(r'\{.*?"(?:思考|想法|内心想法)".*?\}', text, re.DOTALL)
        if json_hit:
            candidate = json_hit.group(0)
            brace_depth = 0
            end_pos = 0
            for i, ch in enumerate(candidate):
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                candidate = candidate[:end_pos]
            try:
                data = json.loads(candidate)
                thought = str(
                    data.get("thought")
                    or data.get("thinking")
                    or data.get("思考")
                    or data.get("想法")
                    or data.get("内心想法")
                    or ""
                ).strip()
                desire_raw = (
                    data.get("desire")
                    or data.get("reply_desire")
                    or data.get("欲望")
                    or data.get("回复欲望")
                    or 5
                )
                desire_val = max(1, min(10, int(desire_raw)))
                mood = data.get("mood", data.get("情绪", ""))
                # 解析 LLM 直接给出的行为决策
                next_action_raw = str(data.get("next_action") or data.get("行动") or data.get("动作") or "").strip().lower()
                _valid_actions = {"reply", "followup", "observe", "wait", "lurk", "rest", "disengage"}
                next_action_val = next_action_raw if next_action_raw in _valid_actions else ""
                # 由 next_action 推导 should_reply
                should_reply_val = data.get("reply")
                if should_reply_val is None:
                    if next_action_val in ("reply", "followup"):
                        should_reply_val = True
                    elif next_action_val in ("observe", "rest", "disengage", "lurk"):
                        should_reply_val = False
                    elif desire_val >= 8:
                        should_reply_val = True
                    elif desire_val <= 2:
                        should_reply_val = False
                    else:
                        should_reply_val = bool(desire_val >= 5)
                comprehension_confidence = float(
                    data.get("comprehension_confidence", 0.5)
                )
                needs_upgrade = bool(data.get("needs_upgrade", False))
                if thought and len(thought) >= 4:
                    _compressed = self._compress_thought(thought)
                    if len(_compressed) >= 4:
                        return VoiceVerdict(
                            thinking=_compressed,
                            reply_desire_level=desire_val,
                            should_reply=should_reply_val,
                            next_action=next_action_val,
                            current_mood=mood,
                            comprehension_confidence=comprehension_confidence,
                            needs_upgrade=needs_upgrade,
                            thinking_source="llm_json",
                        )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # 正则提取字段
        num_hit = re.search(r'"?(?:desire|reply_desire|欲望|回复欲望)"?\s*[:=：]\s*(\d+)', text)
        thought_hit = re.search(
            r'"?(?:thought|thinking|思考|想法|内心想法)"?\s*[:=：]\s*["“]?(.+?)(?=["”]?\s*(?:[,，]|\n|\r|\s{2,})?\s*"?(?:desire|reply_desire|欲望|回复欲望|mood|情绪|next_action|行动)"?\s*[:=：]|$)',
            text,
            re.DOTALL,
        )
        # 尝试从非JSON文本中提取 next_action 关键词
        _text_action = ""
        _action_hit = re.search(
            r'"?(?:next_action|行动|动作)"?\s*[:=：]\s*"?(reply|followup|observe|wait|lurk|rest|disengage)"?',
            text,
            re.IGNORECASE,
        )
        if _action_hit:
            _text_action = _action_hit.group(1).lower()
        leading_num_hit = re.match(r"\s*(\d+)\s*[,，;；:：]?", text)
        field_desire_hit = num_hit or leading_num_hit
        if thought_hit and field_desire_hit:
            desire_val = max(1, min(10, int(field_desire_hit.group(1))))
            thought = thought_hit.group(1).strip()
            if thought and len(thought) >= 4:
                _compressed = self._compress_thought(thought)
                if len(_compressed) >= 4:
                    return VoiceVerdict(
                        thinking=_compressed,
                        reply_desire_level=desire_val,
                        next_action=_text_action,
                        thinking_source="llm_field",
                    )
        # 数字开头 + 余文
        plain_hit = re.match(r"(\d+)\s*(.*)", text, re.DOTALL)
        if plain_hit:
            desire_val = max(1, min(10, int(plain_hit.group(1))))
            remainder = plain_hit.group(2).strip()
            if remainder and len(remainder) >= 4:
                _compressed = self._compress_thought(remainder)
                if len(_compressed) >= 4:
                    return VoiceVerdict(
                        thinking=_compressed,
                        reply_desire_level=desire_val,
                        next_action=_text_action,
                        thinking_source="llm_plain",
                    )
            return VoiceVerdict(
                thinking=self._fabricate_reflection(
                    desire_val, raw_text, speaker_name
                ),
                reply_desire_level=desire_val,
                next_action=_text_action,
                thinking_source="local_fallback",
            )
        # 任意数字
        any_num = re.search(r"(\d+)", text)
        if any_num:
            desire_val = max(1, min(10, int(any_num.group(1))))
        else:
            sentiment_bias = self._gauge_sentiment(text)
            rest_bias = self._gauge_rest_desire(text)
            question_marks = text.count("?") + text.count("？")
            exclamations = text.count("!") + text.count("！")
            desire_val = 5
            if rest_bias > 3.5:
                desire_val = 2
            elif sentiment_bias > 2.5 or exclamations >= 2:
                desire_val = 7
            elif question_marks > 0:
                desire_val = 6
        if len(text) >= 4:
            _clean_text = text.strip()
            if _clean_text.startswith("{") or '"thought"' in _clean_text[:20]:
                _th_match = re.search(r'"thought"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', _clean_text)
                if _th_match:
                    _extracted = _th_match.group(1).strip()
                    if len(_extracted) >= 4:
                        _clean_text = _extracted
                else:
                    _bracket = re.search(r'"([^"]{4,})"', _clean_text)
                    if _bracket:
                        _clean_text = _bracket.group(1)
            _clean_text = re.sub(r'^[\{\[\']|[\}\]\']$', '', _clean_text).strip()
            _clean_text = re.sub(r'^["\']?(?:thought|mood|desire|next_action|action)\s*[:=]\s*["\']?', '', _clean_text).strip()
            _compressed = self._compress_thought(_clean_text)
            if len(_compressed) >= 4:
                return VoiceVerdict(
                    thinking=_compressed, reply_desire_level=desire_val,
                    next_action=_text_action,
                    thinking_source="llm_text",
                )
        return VoiceVerdict(
            thinking=self._fabricate_reflection(
                desire_val, raw_text, speaker_name
            ),
            reply_desire_level=desire_val,
            next_action=_text_action,
            thinking_source="local_fallback",
        )

    @staticmethod
    def _compress_thought(thought: str) -> str:
        """把模型输出压成一句短的脑内想法，避免长篇分析文。"""
        text = str(thought or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\.{2,}|…{2,}", "…", text)
        text = re.sub(r"([。！？!?；;…]){2,}", r"\1", text)
        first_cut = None
        for idx, ch in enumerate(text):
            if ch in "。！？!?；;":
                first_cut = idx
                break
        if first_cut is not None:
            text = text[: first_cut + 1]
        if len(text) > 36:
            text = text[:36].rstrip("，,；; ") + "。"
        return text

    @staticmethod
    def _fabricate_reflection(
        desire: int, raw_text: str, speaker_name: str
    ) -> str:
        """根据 desire 档位生成兜底 thought"""
        if desire >= 8:
            return "这句有点意思，我想接一下。"
        if desire >= 6:
            return "这个能接，我可以回。"
        if desire >= 4:
            return "先看看后面还有没有。"
        if desire >= 2:
            return "这句信息太少，先不急。"
        return "懒得接。"

    # ---- 等待时的内心独白 ----

    async def generate_idle_musing(
        self,
        speaker_name: str,
        progress: float,
    ) -> str:
        """生成等待期间的碎碎念"""
        waiting_cfg = _inner_voice_view().get("waiting_thoughts", {}) or {}
        if progress < 0.4:
            options = waiting_cfg.get(
                "short", ["对方可能在忙吧...", "再等等看", "不知道在做什么呢"]
            )
        elif progress < 0.7:
            options = waiting_cfg.get(
                "medium",
                ["等了一会了...", "是不是忘记回复了？", "嗯...还没消息"],
            )
        else:
            options = waiting_cfg.get(
                "long", ["等了挺久了", "要不要主动说点什么...", "快到时间了"]
            )
        return random.choice(options)

    # ---- 字段驱动意图构建（零关键词匹配）----
    def _extract_semantic_intents(
        self,
        thinking: str,
        desire: int,
        speaker_id: str,
        raw_text: str,
        next_action: str = "",
    ) -> List[BehaviorIntent]:
        """从 LLM 决策结果构建行为意图。

        优先使用 next_action（LLM 直接选择的行为），
        如果 next_action 为空则回退到 desire 数字推断。
        """
        intents: List[BehaviorIntent] = []
        thought_preview = (thinking or "")[:40] if thinking else ""
        # 优先走 LLM 直接行为决策
        if next_action:
            _priority_map = {
                "followup": max(1, desire - 1),
                "reply": max(1, desire - 2),
                "wait": 1,
                "observe": 0,
                "lurk": 0,
                "rest": 0,
                "disengage": 0,
            }
            _payload_map = {
                "followup": f"追问/展开({thought_preview})",
                "reply": f"回复({thought_preview})",
                "wait": f"先看看再说({thought_preview})",
                "observe": f"沉默观察({thought_preview})",
                "lurk": f"潜水扫一眼({thought_preview})",
                "rest": f"歇一会({thought_preview})",
                "disengage": f"放下手机({thought_preview})",
            }
            intents.append(
                BehaviorIntent(
                    intent_type=next_action,
                    target_id=speaker_id,
                    priority=_priority_map.get(next_action, 0),
                    payload=_payload_map.get(next_action, thought_preview),
                )
            )
            return intents
        # 回退：用 desire 数字推断行为
        if desire >= 7:
            intents.append(
                BehaviorIntent(
                    intent_type="followup",
                    target_id=speaker_id,
                    priority=max(1, desire - 2),
                    payload=f"高意愿主动({thought_preview})",
                )
            )
        elif desire >= 5:
            intents.append(
                BehaviorIntent(
                    intent_type="reply",
                    target_id=speaker_id,
                    priority=max(1, desire - 3),
                    payload=f"中高意愿回应({thought_preview})",
                )
            )
        elif desire >= 3:
            intents.append(
                BehaviorIntent(
                    intent_type="wait",
                    target_id=speaker_id,
                    priority=1,
                    payload=f"中低意愿观望({thought_preview})",
                )
            )
        else:
            intents.append(
                BehaviorIntent(
                    intent_type="observe",
                    target_id=speaker_id,
                    priority=0,
                    payload=f"低意愿沉默({thought_preview})",
                )
            )
        return intents

    # ---- 跨轮持久状态管理 ----
    def _update_persistent_state(
        self, verdict: VoiceVerdict, speaker_id: str
    ) -> None:
        """更新引擎跨轮记忆"""
        now = time.time()
        self._last_reflection_ts = now
        self._last_desire_level = verdict.reply_desire_level
        self._last_speaker_id = speaker_id
        # 保存近期想法
        if verdict.thinking:
            self._recent_thoughts.append(verdict.thinking[:200])
            if len(self._recent_thoughts) > self._THOUGHT_WINDOW:
                self._recent_thoughts = self._recent_thoughts[
                    -self._THOUGHT_WINDOW:
                ]
        # 追踪沉默轮次
        if verdict.reply_desire_level >= 5:
            self._rounds_since_spoke = 0
        else:
            self._rounds_since_spoke += 1
        # 记录沉默理由
        if verdict.reply_desire_level < 5 and verdict.thinking:
            self._silence_rationale = verdict.thinking[:200]
        else:
            self._silence_rationale = ""
        # 检测未完成话头 → 自动注入意图池实现跨轮回收
        thinking_lower = (verdict.thinking or "").lower()
        has_unfinished_hint = (
            verdict.reply_desire_level >= 3
            and verdict.reply_desire_level < 6
            and len(thinking_lower) > 4
        )
        if has_unfinished_hint:
            thread = f"{
                verdict.thinking[: 100]} (desire={
                verdict.reply_desire_level}) "
            self._unfinished_threads.append(thread)
            if len(self._unfinished_threads) > self._THREAD_CAPACITY:
                self._unfinished_threads = self._unfinished_threads[
                    -self._THREAD_CAPACITY:
                ]
            logger.debug(f"{self._tag} 记录未完成话头: {thread[:60]}")
            # 将未完成话头写入意图池，使其可被 planner 跨轮回收
            self._inject_thread_to_intention_pool(
                speaker_id=speaker_id,
                thread_desc=verdict.thinking[:200],
                desire=verdict.reply_desire_level,
            )
        # 更新对用户态度
        if speaker_id and verdict.thinking:
            attitude = self._infer_attitude(
                verdict.thinking, verdict.reply_desire_level,
                next_action=verdict.next_action,
            )
            if attitude:
                self._per_user_attitudes[speaker_id] = attitude

    def _infer_attitude(self, thinking: str, desire: int, next_action: str = "") -> str:
        """基于行为决策推断态度标签，优先使用 next_action"""
        if next_action:
            _action_attitude = {
                "followup": "积极",
                "reply": "好感",
                "wait": "中立",
                "observe": "中立",
                "lurk": "冷淡",
                "rest": "冷淡",
                "disengage": "冷淡",
            }
            return _action_attitude.get(next_action, "中立")
        if desire >= 7:
            return "积极"
        if desire >= 5:
            return "好感"
        if desire >= 3:
            return "中立"
        if desire <= 2:
            return "冷淡"
        return "中立"

    def _inject_thread_to_intention_pool(
        self,
        speaker_id: str,
        thread_desc: str,
        desire: int,
    ) -> None:
        """将未完成话头作为低优先意图注入意图池，使 planner 可在后续轮次回收。"""
        try:
            from src.chat.proactive.intention_pool import (
                get_intention_pool,
                IntentKind,
            )

            pool = get_intention_pool()
            # 根据话头中的关键词选择意图类型
            if desire >= 5:
                kind = IntentKind.FOLLOWUP_QUESTION
            else:
                kind = IntentKind.CONTINUE_TOPIC
            urgency = min(0.6, max(0.15, desire / 10.0 + 0.1))
            pool.submit(
                channel_id=self.channel_id,
                kind=kind,
                target_user=speaker_id,
                source="inner_voice_thread",
                description=thread_desc[:200],
                urgency=urgency,
                context_snippet=f"未完成话头(desire={desire})",
                expected_outcome="在后续轮次自然衔接话头",
            )
            logger.debug(
                f"{self._tag} 未完成话头注入意图池 kind={kind.value} "
                f"urgency={urgency:.2f} target={speaker_id[:8] if speaker_id else 'none'}"
            )
        except Exception as exc:
            logger.debug(f"{self._tag} 话头注入意图池失败: {exc}")

    # ---- 记忆上下文构建（供提示词使用） ----

    def _build_memory_context(self) -> str:
        """生成跨轮记忆上下文片段，注入提示词"""
        parts = []
        if self._recent_thoughts:
            recent = self._recent_thoughts[-3:]
            parts.append("近期内心想法: " + " | ".join(recent))
        if self._unfinished_threads:
            parts.append(
                "未说完的话头: " + " | ".join(self._unfinished_threads[-2:])
            )
        if self._silence_rationale and self._rounds_since_spoke >= 2:
            parts.append(f"上次沉默原因: {self._silence_rationale}")
        if self._rounds_since_spoke > 0:
            parts.append(f"已连续{self._rounds_since_spoke}轮没有说话")
        if (
            self._last_speaker_id
            and self._last_speaker_id in self._per_user_attitudes
        ):
            parts.append(
                f"对当前用户态度: {self._per_user_attitudes[self._last_speaker_id]}"
            )
        return "\n".join(parts) if parts else ""

    def export_observation_context(self) -> str:
        """导出最近观察记忆，供主动回复路径使用"""
        return self._build_memory_context()

    # ---- 自主思考入口（非消息触发） ----
    async def autonomous_reflection(
        self,
        channel_id: str,
        trigger_reason: str = "意图池驱动",
    ) -> VoiceVerdict:
        """非消息触发的自主思考
        由意图池或定时器驱动，无需新用户消息。
        产出可写回意图池的 VoiceVerdict。
        """
        try:
            persona = self._retrieve_persona()
            memory_ctx = self._build_memory_context()
            active_intentions = ""
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                pool = get_intention_pool()
                active = pool.get_active_intentions(channel_id, limit=3)
                if active:
                    active_intentions = "\n".join(
                        f"- {i.kind.value}: {i.description[:80]} (urgency={i.effective_urgency():.2f})"
                        for i in active
                    )
            except Exception as _e:
                logger.debug(f"{self._tag} 异常: {_e}")
            active_intentions_block = "(无活跃意图)"
            if active_intentions:
                active_intentions_block = (
                    "当前活跃意图:\n"
                    f"{active_intentions}"
                )
            prompt = f"""你是{persona}。
现在没有收到新消息，但你的内心有一些未完成的念头。
请基于以下上下文，产生一段简短的内心独白（1-2句），以及一个1-10的行动欲望值（desire）。
如果你觉得有话想说，desire给高；如果觉得没必要主动，desire给低。

{memory_ctx or '(暂无跨轮记忆)'}

{active_intentions_block}

触发原因: {trigger_reason}

请输出JSON: {{"thought": "你的内心想法", "desire": 数字1-10}}"""
            llm_output = await self._invoke_llm(prompt)
            if not llm_output or llm_output == "__SAFETY_BLOCK__":
                return VoiceVerdict.make_error("autonomous_failed")
            verdict = self._extract_structured_reply(llm_output, "", "")
            verdict.intents = self._extract_semantic_intents(
                verdict.thinking,
                verdict.reply_desire_level,
                self._last_speaker_id,
                "",
                next_action=verdict.next_action,
            )
            self._update_persistent_state(verdict, self._last_speaker_id)
            logger.info(
                f"{self._tag} 自主思考完成 | desire={verdict.reply_desire_level} "
                f"| trigger={trigger_reason[:30]}"
            )
            return verdict
        except Exception as exc:
            logger.error(f"{self._tag} 自主思考失败: {exc}")
            return VoiceVerdict.make_error(str(exc))


# ---- 全局实例缓存 ----
_engine_cache: Dict[str, SelfDialogueEngine] = {}


def get_self_dialogue_engine(channel_id: str) -> SelfDialogueEngine:
    """获取或创建频道对应的自我对话引擎"""
    if channel_id not in _engine_cache:
        _engine_cache[channel_id] = SelfDialogueEngine(channel_id)
    return _engine_cache[channel_id]


def teardown_engine(channel_id: str) -> None:
    """移除指定频道的自我对话引擎缓存"""
    _engine_cache.pop(channel_id, None)


def clear_all_engines() -> None:
    """清除全部自我对话引擎缓存（关机时调用）"""
    _engine_cache.clear()
