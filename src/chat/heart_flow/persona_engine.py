import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config

logger = get_logger("character_foundry")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


class BlendPhase(Enum):
    """过渡阶段"""

    IDLE = "idle"
    ENTERING = "entering"
    MERGING = "merging"
    DEPARTING = "departing"


@dataclass
class SlotRecord:
    """人格槽位记录"""

    slot_id: str = ""
    prompt_shard: str = ""
    source_tag: str = "dynamic"
    trait_labels: List[str] = field(default_factory=list)
    birth_ts: float = field(default_factory=time.time)
    last_touch_ts: float = field(default_factory=time.time)
    visit_count: int = 0

    def bump(self) -> None:
        self.last_touch_ts = time.time()
        self.visit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "prompt_shard": self.prompt_shard,
            "source_tag": self.source_tag,
            "trait_labels": list(self.trait_labels),
            "birth_ts": self.birth_ts,
            "last_touch_ts": self.last_touch_ts,
            "visit_count": self.visit_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlotRecord":
        return cls(
            slot_id=str(data.get("slot_id", "") or ""),
            prompt_shard=str(data.get("prompt_shard", "") or ""),
            source_tag=str(data.get("source_tag", "dynamic") or "dynamic"),
            trait_labels=_coerce_str_list(data.get("trait_labels", [])),
            birth_ts=_coerce_float(data.get("birth_ts"), time.time()),
            last_touch_ts=_coerce_float(
                data.get("last_touch_ts"), time.time()
            ),
            visit_count=_coerce_int(data.get("visit_count"), 0),
        )


@dataclass
class GradientSpec:
    """过渡规格"""

    phase: BlendPhase = BlendPhase.IDLE
    origin_slot_id: str = ""
    destination_slot_id: str = ""
    advancement: float = 0.0
    kickoff_ts: float = 0.0
    span_seconds: float = 6.0

    @property
    def in_motion(self) -> bool:
        return self.phase != BlendPhase.IDLE

    def origin_fraction(self) -> float:
        return max(0.0, 1.0 - self.advancement)

    def destination_fraction(self) -> float:
        return min(1.0, self.advancement)


@dataclass
class EffectivenessNote:
    """交互效果记忆条目"""

    note_id: str = ""
    slot_id: str = ""
    user_ref: str = ""
    scene_desc: str = ""
    effect_rating: float = 0.5
    retention_strength: float = 0.7
    inception_ts: float = field(default_factory=time.time)
    last_boost_ts: float = field(default_factory=time.time)
    boost_tally: int = 0
    _DECAY_RATE: float = 0.03

    def boost(self, amount: float = 0.1) -> None:
        self.retention_strength = min(1.0, self.retention_strength + amount)
        self.last_boost_ts = time.time()
        self.boost_tally += 1

    def wither(self) -> None:
        elapsed_hrs = (time.time() - self.last_boost_ts) / 3600.0
        self.retention_strength = max(
            0.0, self.retention_strength - self._DECAY_RATE * elapsed_hrs
        )

    @property
    def still_viable(self) -> bool:
        return self.retention_strength > 0.01

    def composite_score(self) -> float:
        return self.effect_rating * self.retention_strength


class CharacterFoundry:
    """
    角色铸造引擎

    管理有限容量的人格槽位池（LRU 逐出），
    支持平滑三阶段过渡（entering→merging→departing），
    极端创伤阈值时强制切入危机人格，
    交互记忆池通过 reinforce/decay 驱动偏好学习。
    """

    _solo: Optional["CharacterFoundry"] = None

    @classmethod
    def instance(cls) -> "CharacterFoundry":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo:
            cls._solo._save_snapshot()
            cls._solo._roster.clear()
            cls._solo._notes.clear()
        cls._solo = None

    def __init__(self):
        self._roster: Dict[str, SlotRecord] = {}
        self._active_slot_id: str = "baseline"
        self._gradient = GradientSpec()
        self._notes: List[EffectivenessNote] = []
        self._last_nonbaseline_ts: float = 0.0
        self._conf_cache: Dict[str, Any] = {}
        self._conf_ts: float = 0.0
        self._snapshot_dir: str = "data/huoli/persona_pool.json"
        self._refresh_conf()
        self._restore_snapshot()

    # ---- 配置 ----

    def _refresh_conf(self) -> None:
        now = time.time()
        if now - self._conf_ts < 60.0:
            return
        hub = get_core_config()
        ps = hub.personality_sliders_block()
        self._conf_cache = {
            "pool_capacity": int(ps.get("persona_pool_capacity", 10)),
            "default_blend_sec": float(
                ps.get("default_blend_duration_sec", 6.0)
            ),
            "auto_revert_sec": float(ps.get("persona_auto_revert_sec", 300.0)),
            "chaos_activate_threshold": float(
                ps.get("crisis_persona_chaos_threshold", 7.0)
            ),
            "crisis_base_duration": float(
                ps.get("crisis_base_duration_sec", 300.0)
            ),
            "crisis_extra_per_point": float(
                ps.get("crisis_extra_per_point_sec", 300.0)
            ),
            "severe_boundary": float(ps.get("extreme_severity_boundary", 9.0)),
            "memory_cap": int(ps.get("interaction_memory_cap", 200)),
            "effectiveness_floor": float(ps.get("effectiveness_floor", 0.2)),
            "high_effect_bar": float(ps.get("high_effect_threshold", 0.6)),
            "blend_enter_mark": float(ps.get("blend_enter_mark", 0.33)),
            "blend_merge_mark": float(ps.get("blend_merge_mark", 0.67)),
            "fragment_visible_threshold": float(
                ps.get("fragment_visible_threshold", 0.05)
            ),
        }
        self._conf_ts = now

    # ---- 槽位注册 & LRU ----

    def enroll_slot(
        self,
        slot_id: str,
        prompt_shard: str,
        source_tag: str = "dynamic",
        trait_labels: Optional[List[str]] = None,
    ) -> bool:
        """注册人格槽位，池满时淘汰最久未用的动态槽位"""
        if slot_id in self._roster:
            existing = self._roster[slot_id]
            existing.prompt_shard = prompt_shard
            existing.bump()
            return True
        cap = self._conf_cache.get("pool_capacity", 10)
        if len(self._roster) >= cap:
            if not self._expel_stale():
                logger.warning(f"[铸造] 池满且无法逐出 | 容量={cap}")
                return False
        self._roster[slot_id] = SlotRecord(
            slot_id=slot_id,
            prompt_shard=prompt_shard,
            source_tag=source_tag,
            trait_labels=trait_labels or [],
        )
        logger.info(
            f"[铸造] 注册槽位 {slot_id} | 来源={source_tag} | 当前={len(self._roster)}/{cap}"
        )
        return True

    def activate_slot(self, slot_id: str) -> bool:
        rec = self._roster.get(slot_id)
        if rec is None:
            return False
        rec.bump()
        self._active_slot_id = slot_id
        if slot_id != "baseline":
            self._last_nonbaseline_ts = time.time()
        return True

    def active_slot_id(self) -> str:
        return self._active_slot_id

    def active_prompt_shard(self) -> str:
        rec = self._roster.get(self._active_slot_id)
        return rec.prompt_shard if rec else ""

    def fetch_shard(self, slot_id: str) -> Optional[str]:
        rec = self._roster.get(slot_id)
        if rec:
            rec.bump()
            return rec.prompt_shard
        return None

    def fetch_record(self, slot_id: str) -> Optional[SlotRecord]:
        return self._roster.get(slot_id)

    def catalog_ids(self) -> List[str]:
        return list(self._roster.keys())

    def roster_count(self) -> int:
        return len(self._roster)

    def discard_slot(self, slot_id: str) -> bool:
        if slot_id == self._active_slot_id:
            return False
        return self._roster.pop(slot_id, None) is not None

    def _expel_stale(self) -> bool:
        """淘汰最久未访问的非固定槽位"""
        pinned_tags = frozenset({"manual", "pinned"})
        candidates = [
            r
            for r in self._roster.values()
            if r.source_tag not in pinned_tags
            and r.slot_id != self._active_slot_id
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda r: r.last_touch_ts)
        del self._roster[victim.slot_id]
        logger.debug(f"[铸造] 淘汰闲置槽位 {victim.slot_id}")
        return True

    # ---- 三阶段平滑过渡 ----

    def initiate_blend(
        self,
        from_id: str,
        to_id: str,
        duration: Optional[float] = None,
    ) -> None:
        """启动从 from_id 到 to_id 的平滑过渡"""
        if from_id == to_id:
            self._gradient = GradientSpec()
            return
        span = (
            duration
            if duration
            else self._conf_cache.get("default_blend_sec", 6.0)
        )
        self._gradient = GradientSpec(
            phase=BlendPhase.ENTERING,
            origin_slot_id=from_id,
            destination_slot_id=to_id,
            advancement=0.0,
            kickoff_ts=time.time(),
            span_seconds=max(0.5, span),
        )
        if to_id != "baseline":
            self._last_nonbaseline_ts = time.time()
        logger.info(f"[铸造] 开始过渡 {from_id} → {to_id} | 持续={span:.1f}s")

    def advance_blend(self) -> None:
        """推进过渡进度（每消息周期调用一次）"""
        if not self._gradient.in_motion:
            return
        elapsed = time.time() - self._gradient.kickoff_ts
        raw = elapsed / max(0.1, self._gradient.span_seconds)
        self._gradient.advancement = min(1.0, raw)
        enter_mark = self._conf_cache.get("blend_enter_mark", 0.33)
        merge_mark = self._conf_cache.get("blend_merge_mark", 0.67)
        if raw < enter_mark:
            self._gradient.phase = BlendPhase.ENTERING
        elif raw < merge_mark:
            self._gradient.phase = BlendPhase.MERGING
        else:
            self._gradient.phase = BlendPhase.DEPARTING
        if raw >= 1.0:
            self._gradient = GradientSpec()

    def needs_baseline_revert(self) -> bool:
        """检查是否超时需要回退基线人格"""
        if self._active_slot_id == "baseline":
            return False
        if self._last_nonbaseline_ts <= 0.0:
            return False
        limit = self._conf_cache.get("auto_revert_sec", 300.0)
        return (time.time() - self._last_nonbaseline_ts) > limit

    @property
    def blending_active(self) -> bool:
        return self._gradient.in_motion

    def weighted_shards(self) -> Tuple[str, float, str, float]:
        """返回 (origin_id, origin_w, dest_id, dest_w)"""
        if not self._gradient.in_motion:
            return ("", 0.0, "", 1.0)
        return (
            self._gradient.origin_slot_id,
            self._gradient.origin_fraction(),
            self._gradient.destination_slot_id,
            self._gradient.destination_fraction(),
        )

    def compose_blended_prompt(
        self, origin_text: str, destination_text: str
    ) -> str:
        """按权重拼接过渡期双人格片段"""
        if not self._gradient.in_motion:
            return destination_text
        ow = self._gradient.origin_fraction()
        dw = self._gradient.destination_fraction()
        threshold = self._conf_cache.get("fragment_visible_threshold", 0.05)
        segments = []
        if origin_text and ow > threshold:
            segments.append(f"[{ow:.0%}] {origin_text}")
        if destination_text and dw > threshold:
            segments.append(f"[{dw:.0%}] {destination_text}")
        return "\n".join(segments) if segments else destination_text

    # ---- 极端危机接管 ----

    def crisis_threshold_reached(self, chaos_level: float) -> bool:
        boundary = self._conf_cache.get("severe_boundary", 9.0)
        return chaos_level >= boundary

    def suggest_character_for_crisis(
        self, chaos_level: float, trauma_label: str = ""
    ) -> str:
        """根据混乱度推荐人格ID"""
        threshold = self._conf_cache.get("chaos_activate_threshold", 7.0)
        if chaos_level >= threshold:
            return "crisis_shield"
        crisis_labels = {
            "imminent_collapse",
            "total_breakdown",
            "flashback_active",
            "stress_outbreak",
        }
        if trauma_label.lower() in crisis_labels:
            return "crisis_shield"
        return "baseline"

    async def forge_crisis_character(
        self,
        chaos_level: float,
        trigger_desc: str = "",
    ) -> Dict[str, Any]:
        """为极端崩溃状态锻造危机人格"""
        clamped = min(10.0, max(0.0, chaos_level))
        context = trigger_desc[:200] if trigger_desc else "严重心理危机触发"
        boundary = self._conf_cache.get("severe_boundary", 9.0)
        base_dur = self._conf_cache.get("crisis_base_duration", 300.0)
        extra = self._conf_cache.get("crisis_extra_per_point", 300.0)
        protection_sec = base_dur + max(0.0, clamped - boundary) * extra
        intensity = "extreme" if clamped >= boundary + 0.5 else "severe"
        directive = (
            f"生成一个处于 {intensity} 级心理危机的角色配置。"
            f"创伤强度：{clamped}/10。"
            f"触发场景：{context}。"
            "角色应该表现出情感封锁或解离特征，回复应简短且碎片化。"
        )
        raw = await self._call_llm(directive)
        if raw:
            parsed = self._extract_json(raw)
            if parsed:
                parsed.setdefault("slot_id", f"crisis_{int(time.time())}")
                parsed["protection_duration_sec"] = protection_sec
                parsed["priority_tag"] = "critical"
                parsed["forged_by"] = "crisis_override"
                return parsed
        raise RuntimeError(f"危机人格锻造失败 | 混乱度={clamped}")

    # ---- 交互记忆 ----

    def log_interaction(
        self,
        slot_id: str,
        user_ref: str,
        scene_desc: str,
        was_positive: bool,
    ) -> None:
        """记录一次交互结果（强化或削弱记忆）"""
        existing = self._locate_note(slot_id, user_ref)
        if existing:
            if was_positive:
                existing.boost(0.1)
                existing.effect_rating = min(
                    1.0, existing.effect_rating + 0.05
                )
            else:
                existing.wither()
                existing.effect_rating = max(
                    0.0, existing.effect_rating - 0.08
                )
            return
        new_note = EffectivenessNote(
            note_id=uuid.uuid4().hex[:12],
            slot_id=slot_id,
            user_ref=user_ref,
            scene_desc=scene_desc[:300],
            effect_rating=0.6 if was_positive else 0.3,
            retention_strength=0.7,
        )
        self._notes.append(new_note)
        self._prune_notes()

    def decay_all_notes(self) -> None:
        """全局衰减 + 清除死亡条目"""
        for note in self._notes:
            note.wither()
        self._notes = [n for n in self._notes if n.still_viable]

    def best_slot_for_user(self, user_ref: str) -> Optional[str]:
        """返回该用户评价最高的人格 ID"""
        user_notes = [
            n for n in self._notes if n.user_ref == user_ref and n.still_viable
        ]
        if not user_notes:
            return None
        champion = max(user_notes, key=lambda n: n.composite_score())
        floor = self._conf_cache.get("effectiveness_floor", 0.2)
        if champion.composite_score() < floor:
            return None
        return champion.slot_id

    async def mint_character(
        self,
        context_text: str,
        style_hints: str = "",
        user_ref: str = "",
    ) -> Dict[str, Any]:
        """通过 LLM 铸造新人格摘要"""
        preference_hint = (
            self._compile_preference_hint(user_ref) if user_ref else ""
        )
        combined = "\n".join(filter(None, [style_hints, preference_hint]))
        prompt = (
            f"根据以下情境生成一个角色人格配置（JSON格式）：\n"
            f"情境：{context_text[:2000]}\n"
        )
        if combined:
            prompt += f"风格提示：{combined[:1000]}\n"
        prompt += (
            '输出 JSON：{"name": "人格名", "traits": ["特征1","特征2"], '
            '"tone": "语调描述", "prompt_fragment": "人格提示词片段"}'
        )
        raw = await self._call_llm(prompt)
        if raw:
            parsed = self._extract_json(raw)
            if parsed:
                return parsed
            return {"prompt_fragment": raw[:800], "parse_incomplete": True}
        return {"error": "LLM 无响应"}

    def _compile_preference_hint(self, user_ref: str) -> str:
        user_notes = sorted(
            [
                n
                for n in self._notes
                if n.user_ref == user_ref and n.still_viable
            ],
            key=lambda n: n.composite_score(),
            reverse=True,
        )[:5]
        if not user_notes:
            return ""
        bar = self._conf_cache.get("high_effect_bar", 0.6)
        lines = ["该用户历史偏好："]
        for n in user_notes:
            tag = "优" if n.effect_rating > bar else "中"
            lines.append(f"- [{tag}] {n.scene_desc}")
        return "\n".join(lines)

    def _locate_note(
        self, slot_id: str, user_ref: str
    ) -> Optional[EffectivenessNote]:
        for n in self._notes:
            if (
                n.slot_id == slot_id
                and n.user_ref == user_ref
                and n.still_viable
            ):
                return n
        return None

    def _prune_notes(self) -> None:
        cap = self._conf_cache.get("memory_cap", 200)
        if len(self._notes) <= cap:
            return
        self._notes.sort(key=lambda n: n.composite_score(), reverse=True)
        self._notes = self._notes[:cap]

    # ---- 持久化 ----

    def _save_snapshot(self) -> None:
        try:
            os.makedirs(
                os.path.dirname(self._snapshot_dir) or ".", exist_ok=True
            )
            payload = {
                "active_id": self._active_slot_id,
                "roster": [r.to_dict() for r in self._roster.values()],
                "notes": [
                    {
                        "note_id": n.note_id,
                        "slot_id": n.slot_id,
                        "user_ref": n.user_ref,
                        "scene_desc": n.scene_desc,
                        "effect_rating": n.effect_rating,
                        "retention_strength": n.retention_strength,
                        "inception_ts": n.inception_ts,
                        "last_boost_ts": n.last_boost_ts,
                        "boost_tally": n.boost_tally,
                    }
                    for n in self._notes
                    if n.still_viable
                ],
                "saved_at": time.time(),
            }
            with open(self._snapshot_dir, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except (OSError, TypeError) as exc:
            logger.debug(f"[铸造] 快照保存失败: {exc}")

    def _restore_snapshot(self) -> None:
        if not os.path.exists(self._snapshot_dir):
            return
        try:
            with open(self._snapshot_dir, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.debug("[铸造] 快照格式异常，跳过恢复")
                return
            for item in data.get("roster", []):
                if not isinstance(item, dict):
                    continue
                try:
                    record = SlotRecord.from_dict(item)
                except Exception as exc:
                    logger.debug(f"[铸造] 槽位记录恢复失败，已跳过: {exc}")
                    continue
                if record.slot_id:
                    self._roster[record.slot_id] = record
            saved_id = data.get("active_id", "baseline")
            if saved_id in self._roster:
                self._active_slot_id = saved_id
            for nd in data.get("notes", []):
                if not isinstance(nd, dict):
                    continue
                note = EffectivenessNote(
                    note_id=str(nd.get("note_id", "") or ""),
                    slot_id=str(nd.get("slot_id", "") or ""),
                    user_ref=str(nd.get("user_ref", "") or ""),
                    scene_desc=str(nd.get("scene_desc", "") or ""),
                    effect_rating=_coerce_float(
                        nd.get("effect_rating"), 0.5
                    ),
                    retention_strength=_coerce_float(
                        nd.get("retention_strength"), 0.5
                    ),
                    inception_ts=_coerce_float(
                        nd.get("inception_ts"), time.time()
                    ),
                    last_boost_ts=_coerce_float(
                        nd.get("last_boost_ts"), time.time()
                    ),
                    boost_tally=_coerce_int(nd.get("boost_tally"), 0),
                )
                if note.still_viable:
                    self._notes.append(note)
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            logger.debug(f"[铸造] 快照恢复失败: {exc}")

    # ---- LLM 调用 ----

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            req = LLMRequest(
                model_config.model_task_config.focus_chat,
                request_type="character_foundry",
            )
            text, _ = await asyncio.wait_for(
                req.generate_response_async(prompt),
                timeout=30.0,
            )
            return text
        except asyncio.TimeoutError:
            logger.error(f"[铸造] LLM 调用超时(30s)")
            return None

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict]:
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",\s*}", "}", match.group())
                fixed = re.sub(r",\s*]", "]", fixed)
                return json.loads(fixed)
            except Exception:
                return None

    # ---- 统计 ----

    def collect_metrics(self) -> Dict[str, Any]:
        return {
            "roster_size": len(self._roster),
            "active_slot": self._active_slot_id,
            "blending": self._gradient.in_motion,
            "memory_notes": len(self._notes),
            "viable_notes": sum(1 for n in self._notes if n.still_viable),
        }


def get_character_foundry() -> CharacterFoundry:
    """获取全局角色铸造引擎实例"""
    return CharacterFoundry.instance()
