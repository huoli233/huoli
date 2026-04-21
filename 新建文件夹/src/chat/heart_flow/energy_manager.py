import time
from typing import Any, Dict, List, Optional, Tuple, Set
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import EnergySnapshot

logger = get_logger("vitality_pool")

_social_affect_fuser = None
_shared_resources: Dict[str, Dict[str, float]] = {}
_group_members: Dict[str, Set[str]] = {}


class SharedResourceManager:
    """
    共享资源管理器

    管理群聊中三个共享指标：
    - chat_value: 聊天值（群聊共享）
    - social_value: 社交值（群聊共享）
    - activity_level: 活跃度（群聊共享）

    实现平均分配算法和独立扣除机制
    """

    def __init__(self):
        self._group_resources: Dict[str, Dict[str, float]] = {}
        self._group_ceiling: Dict[str, Dict[str, float]] = {
            "chat_value": 100.0,
            "social_value": 100.0,
            "activity_level": 100.0,
        }
        # 因素范围参数（不再使用固定扣除值）
        self._factor_bounds = {
            "chat_value": {
                "base_range": (0.15, 3.0),
                "frequency_range": (1.0, 1.8),
                "content_range": (0.8, 1.5),
                "final_clamp": (0.6, 5.0),
                "peek_ratio": 0.25,
            },
            "activity_level": {
                "base_range": (0.2, 2.5),
                "type_factors": {
                    "action": 1.0,
                    "think": 0.3,
                    "idle": 0.1,
                    "reply": 0.8,
                },
                "final_clamp": (0.1, 4.0),
            },
            "social_value": {
                "polarity_base": {
                    "negative": -0.8,
                    "positive": 0.5,
                    "annoyance": -0.4,
                    "neutral": 0.0,
                },
                "interaction_damping_cap": 0.4,
                "polarization_resistance": 0.3,
                "final_clamp": (-2.0, 1.5),
            },
        }

    def get_or_create_group(self, channel_id: str) -> Dict[str, float]:
        """获取或创建群组的共享资源池"""
        if channel_id not in self._group_resources:
            self._group_resources[channel_id] = {
                "chat_value": 100.0,
                "social_value": 0.0,
                "activity_level": 50.0,
                "member_count": 0,
                "last_update": time.time(),
            }
        return self._group_resources[channel_id]

    def register_member(self, channel_id: str, user_id: str) -> None:
        """注册群组成员"""
        global _group_members
        if channel_id not in _group_members:
            _group_members[channel_id] = set()
        _group_members[channel_id].add(user_id)
        resources = self.get_or_create_group(channel_id)
        resources["member_count"] = len(_group_members[channel_id])

    def get_shared_values(self, channel_id: str) -> Dict[str, float]:
        """获取群组共享值"""
        resources = self.get_or_create_group(channel_id)
        return {
            "chat_value": resources.get("chat_value", 100.0),
            "social_value": resources.get("social_value", 0.0),
            "activity_level": resources.get("activity_level", 50.0),
        }

    def apply_input_activity(
        self,
        channel_id: str,
        *,
        user_id: str = "",
        content_length: int = 0,
        is_repeat: bool = False,
    ) -> Dict[str, Any]:
        """用户输入到达时更新共享聊天值/活跃度/社交值。"""
        resources = self.get_or_create_group(channel_id)
        if user_id:
            self.register_member(channel_id, user_id)

        now = time.time()
        last_update = float(resources.get("last_update", now) or now)
        elapsed_minutes = max(0.0, (now - last_update) / 60.0)
        if elapsed_minutes > 0:
            self.recover_shared_values(channel_id, elapsed_minutes)
            resources = self.get_or_create_group(channel_id)

        content_scale = max(
            0.3,
            min(
                2.5,
                float(content_length or 0) / 24.0 if content_length else 0.6,
            ),
        )
        repeat_penalty = 0.75 if is_repeat else 0.0
        chat_boost = max(
            0.15, min(2.0, 0.3 + content_scale * 0.6 - repeat_penalty)
        )
        activity_boost = max(
            0.3,
            min(4.0, 0.5 + content_scale * 0.8 + (0.5 if is_repeat else 0.0)),
        )
        # 社交值增量：随内容丰富度动态缩放，重复则大幅扣减
        if is_repeat:
            social_delta = max(-0.6, -0.15 - content_scale * 0.1)
        else:
            social_delta = max(0.08, min(0.5, 0.1 + content_scale * 0.12))

        old_chat = float(resources.get("chat_value", 100.0) or 100.0)
        old_activity = float(resources.get("activity_level", 50.0) or 50.0)
        old_social = float(resources.get("social_value", 0.0) or 0.0)

        resources["chat_value"] = max(0.0, min(100.0, old_chat + chat_boost))
        resources["activity_level"] = max(
            0.0, min(100.0, old_activity + activity_boost)
        )
        resources["social_value"] = max(
            -100.0, min(100.0, old_social + social_delta)
        )
        resources["last_update"] = now

        return {
            "old_chat_value": old_chat,
            "new_chat_value": resources["chat_value"],
            "chat_delta": resources["chat_value"] - old_chat,
            "old_activity_level": old_activity,
            "new_activity_level": resources["activity_level"],
            "activity_delta": resources["activity_level"] - old_activity,
            "old_social_value": old_social,
            "new_social_value": resources["social_value"],
            "social_delta": resources["social_value"] - old_social,
            "is_repeat": is_repeat,
        }

    def distribute_evenly(
        self, channel_id: str, total_value: float, value_type: str
    ) -> float:
        """
        平均分配算法

        将总值平均分配给群组成员
        返回每个成员应得的值
        """
        global _group_members
        members = _group_members.get(channel_id, set())
        member_count = max(1, len(members))
        per_member = total_value / member_count
        return per_member

    def deduct_chat_value(
        self, channel_id: str, amount: float, reason: str = "reply"
    ) -> Dict[str, Any]:
        """聊天值动态扣除 - 因素链算法

        扣除流程：
        1. 活跃度映射基底：当前活跃度% × 系数 → [0.15, 3.0]
        2. 频率因子：距上次更新间隔越短 → 因子越高 [1.0, 1.8]
        3. 内容因子：消息量/复杂度映射 [0.8, 1.5]
        4. 最终 clamp [0.6, 5.0]
        窥屏场景取回复扣除的 peek_ratio 比例
        """
        resources = self.get_or_create_group(channel_id)
        bounds = self._factor_bounds["chat_value"]
        base_lo, base_hi = bounds["base_range"]
        freq_lo, freq_hi = bounds["frequency_range"]
        final_lo, final_hi = bounds["final_clamp"]

        activity_pct = resources.get("activity_level", 50.0) / 100.0
        base_cost = base_lo + (base_hi - base_lo) * activity_pct

        elapsed = time.time() - resources.get("last_update", time.time())
        if elapsed < 30:
            freq_factor = freq_hi
        elif elapsed < 120:
            freq_factor = freq_lo + (freq_hi - freq_lo) * max(
                0.0, 1.0 - elapsed / 120.0
            )
        else:
            freq_factor = freq_lo

        content_lo, content_hi = bounds["content_range"]
        if amount > 0 and reason not in ("reply", "peek"):
            content_factor = min(content_hi, content_lo + amount * 0.002)
        else:
            content_factor = 1.0

        raw_cost = base_cost * freq_factor * content_factor

        if reason == "peek":
            raw_cost *= bounds["peek_ratio"]

        deduct_amount = max(final_lo, min(final_hi, raw_cost))
        old_value = resources["chat_value"]
        new_value = max(0.0, old_value - deduct_amount)
        resources["chat_value"] = new_value
        resources["last_update"] = time.time()
        return {
            "old_value": old_value,
            "new_value": new_value,
            "deducted": deduct_amount,
            "reason": reason,
        }

    def deduct_social_value(
        self, channel_id: str, amount: float, reason: str = "neutral"
    ) -> Dict[str, Any]:
        """社交值因素链扣除 - 轻量级动态算法

        扣除流程：
        1. 极性基底：按 reason 查表得到方向性基底值
        2. 极化阻力：当前社交值越极端，变化效率越低
           coeff = 0.7 + 0.3 × (1 - |current| / 100)
        3. 交互频次阻尼：活跃成员越多，单次冲击越分散
           damping = 1.0 - min(member_count / 20, cap)
        4. 疲劳衰减：(可选) 如果有通道疲劳信息
        5. 最终 clamp [-2.0, +1.5]
        """
        resources = self.get_or_create_group(channel_id)
        bounds = self._factor_bounds["social_value"]
        final_lo, final_hi = bounds["final_clamp"]

        polarity_map = bounds["polarity_base"]
        base_delta = polarity_map.get(reason, 0.0)
        if reason == "neutral" and amount != 0:
            base_delta = max(-1.0, min(1.0, amount * 0.1))

        old_value = resources["social_value"]
        polarization_ratio = min(1.0, abs(old_value) / 100.0)
        balance_coeff = 0.7 + 0.3 * (1.0 - polarization_ratio)

        member_count = resources.get("member_count", 1)
        interaction_cap = bounds["interaction_damping_cap"]
        interaction_damping = 1.0 - min(
            max(0, member_count) / 20.0, interaction_cap
        )
        interaction_damping = max(0.3, interaction_damping)

        raw_delta = base_delta * balance_coeff * interaction_damping

        deduct_amount = max(final_lo, min(final_hi, raw_delta))
        new_value = max(-100.0, min(100.0, old_value + deduct_amount))
        resources["social_value"] = new_value
        resources["last_update"] = time.time()
        return {
            "old_value": old_value,
            "new_value": new_value,
            "deducted": (
                abs(deduct_amount) if deduct_amount < 0 else -deduct_amount
            ),
            "reason": reason,
        }

    def deduct_activity_level(
        self, channel_id: str, amount: float, reason: str = "action"
    ) -> Dict[str, Any]:
        """活跃度动态扣除 - 基于当前水位和动作类型的因素算法

        扣除流程：
        1. 水位映射基底：当前活跃度% × 0.025 → [0.2, 2.5]
        2. 动作类型因子：action=1.0, reply=0.8, think=0.3, idle=0.1
        3. 最终 clamp [0.1, 4.0]
        """
        resources = self.get_or_create_group(channel_id)
        bounds = self._factor_bounds["activity_level"]
        base_lo, base_hi = bounds["base_range"]
        final_lo, final_hi = bounds["final_clamp"]

        current_pct = resources.get("activity_level", 50.0) / 100.0
        base_cost = base_lo + (base_hi - base_lo) * current_pct

        type_factors = bounds["type_factors"]
        type_mult = type_factors.get(reason, 0.5)

        raw_cost = base_cost * type_mult
        deduct_amount = max(final_lo, min(final_hi, raw_cost))

        old_value = resources["activity_level"]
        new_value = max(0.0, min(100.0, old_value - deduct_amount))
        resources["activity_level"] = new_value
        resources["last_update"] = time.time()
        return {
            "old_value": old_value,
            "new_value": new_value,
            "deducted": deduct_amount,
            "reason": reason,
        }

    def recover_shared_values(
        self, channel_id: str, elapsed_minutes: float
    ) -> None:
        """
        共享值动态恢复机制

        低水位恢复慢（群组精力耗尽后需要冷却期），高水位恢复正常。
        活跃度向基线 50 衰减而非单调增长。
        """
        resources = self.get_or_create_group(channel_id)
        # 聊天值：动态恢复，低水位恢复慢
        cur_chat = float(resources.get("chat_value", 100.0) or 100.0)
        _chat_ratio = cur_chat / 100.0
        _chat_recovery_factor = max(0.25, _chat_ratio * 0.5 + 0.35)
        chat_recovery = elapsed_minutes * 1.5 * _chat_recovery_factor
        resources["chat_value"] = min(100.0, cur_chat + chat_recovery)
        # 活跃度：向基线 50 衰减（太高下降，太低上升），速率动态
        cur_activity = float(resources.get("activity_level", 50.0) or 50.0)
        _baseline = 50.0
        _activity_diff = cur_activity - _baseline
        if abs(_activity_diff) > 0.5:
            _decay_speed = max(0.3, min(2.0, abs(_activity_diff) / 40.0))
            _activity_shift = elapsed_minutes * _decay_speed
            if _activity_diff > 0:
                resources["activity_level"] = max(
                    _baseline, cur_activity - _activity_shift
                )
            else:
                resources["activity_level"] = min(
                    _baseline, cur_activity + _activity_shift
                )
        # 社交值：指数衰减趋向 0
        social_decay = elapsed_minutes * 0.015
        if resources["social_value"] > 0:
            resources["social_value"] = max(
                0.0, resources["social_value"] * (1 - social_decay)
            )
        elif resources["social_value"] < 0:
            resources["social_value"] = min(
                0.0, resources["social_value"] * (1 - social_decay)
            )
        resources["last_update"] = time.time()


_shared_resource_manager: Optional[SharedResourceManager] = None


def get_shared_resource_manager() -> SharedResourceManager:
    """获取共享资源管理器单例"""
    global _shared_resource_manager
    if _shared_resource_manager is None:
        _shared_resource_manager = SharedResourceManager()
    return _shared_resource_manager


def _get_affect_fuser():
    """延迟导入社交情感融合器"""
    global _social_affect_fuser
    if _social_affect_fuser is None:
        try:
            from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
from src.chat.heart_flow.social_value_dim import SocialValueDimension

            _social_affect_fuser = FondnessTrustDimension.get_instance()
        except Exception as e:
            logger.debug(f"初始化社交情感融合器失败: {e}")
    return _social_affect_fuser


class VitalityPoolManager:
    """
    双池能量管理器 - 完整融入社交值系统
    融合聊天精力值（chat_pool）与思考值（thinking_value），
    并完整集成社交值系统（social_value、trust_value、annoyance_value等）。
    所有数值参数来自 CoreSettingsHub，禁止硬编码。
    职责边界：管理能量状态和社交值状态，辅助回复决策。
    """

    _solo: Optional["VitalityPoolManager"] = None

    @classmethod
    def instance(cls) -> "VitalityPoolManager":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo is not None:
            cls._solo._channel_snapshots.clear()
        cls._solo = None

    def __init__(self):
        self._channel_snapshots: Dict[str, EnergySnapshot] = {}
        self._cfg_cache: Optional[Dict[str, Any]] = None
        self._cfg_ts: float = 0.0

    def _load_cfg(self) -> Dict[str, Any]:
        """从核心配置引擎读取双池能量参数，缓存60秒"""
        now = time.time()
        if self._cfg_cache is not None and (now - self._cfg_ts) < 60:
            return self._cfg_cache
        hub = get_core_config()
        block = hub.dual_pool_energy()
        self._cfg_cache = block
        self._cfg_ts = now
        return block

    def _chat_cfg(self) -> Dict[str, Any]:
        return self._load_cfg().get("chat_value", {})

    def _thinking_cfg(self) -> Dict[str, Any]:
        return self._load_cfg().get("thinking_power", {})

    def _decay_cfg(self) -> Dict[str, Any]:
        return self._load_cfg().get("decay", {})

    def _ensure_channel(self, channel_id: str) -> EnergySnapshot:
        """确保频道有快照，不存在则创建满值"""
        if channel_id not in self._channel_snapshots:
            cc = self._chat_cfg()
            bc = self._thinking_cfg()
            ceiling_chat = float(cc.get("ceiling", 100.0))
            ceiling_brain = float(bc.get("ceiling", 100.0))
            self._channel_snapshots[channel_id] = EnergySnapshot(
                chat_pool=ceiling_chat,
                chat_ceiling=ceiling_chat,
                thinking_value=ceiling_brain,
                thinking_ceiling=ceiling_brain,
                annoyance_level=0.0,
                captured_at=time.time(),
            )
        return self._channel_snapshots[channel_id]

    def _tick_recovery(self, snap: EnergySnapshot) -> None:
        """按经过时间恢复能量并衰减疲劳/烦恼"""
        now = time.time()
        if snap.captured_at <= 0:
            snap.captured_at = now
            return
        dc = self._decay_cfg()
        min_interval = float(dc.get("min_recovery_gap_seconds", 0))
        elapsed_min = (now - snap.captured_at) / 60.0
        if min_interval > 0 and (now - snap.captured_at) < min_interval:
            return
        cc = self._chat_cfg()
        bc = self._thinking_cfg()
        chat_recovery_rate = float(cc.get("recovery_rate_per_min", 5.0))
        brain_recovery_rate = float(bc.get("recovery_rate_per_sec", 0.005))
        annoyance_decay = float(dc.get("annoyance_decay_rate", 3.0))
        # 聊天值恢复：动态自适应，低水位恢复慢（精力耗尽后恢复需要休息）
        _chat_ratio = snap.chat_ratio()
        _chat_factor = max(0.2, _chat_ratio * 0.6 + 0.3)
        snap.chat_pool = min(
            snap.chat_ceiling,
            snap.chat_pool + elapsed_min * chat_recovery_rate * _chat_factor,
        )
        # 思考恢复：动态自适应，低水位恢复慢（深度疲劳），高水位恢复正常
        _think_ratio = snap.thinking_ratio()
        _recovery_factor = max(0.15, _think_ratio * 0.8 + 0.2)
        _brain_elapsed = now - snap.captured_at
        _brain_recovery = _brain_elapsed * brain_recovery_rate * _recovery_factor
        snap.thinking_value = min(
            snap.thinking_ceiling,
            snap.thinking_value + _brain_recovery,
        )
        if annoyance_decay > 0:
            snap.annoyance_level = max(
                0.0, snap.annoyance_level - elapsed_min * annoyance_decay
            )
        snap.captured_at = now

    def capture_snapshot(self, channel_id: str) -> EnergySnapshot:
        """获取频道的实时能量快照（已计算恢复）"""
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)
        shared_values = get_shared_resource_manager().get_shared_values(
            channel_id
        )
        snap.activity_level = float(
            shared_values.get("activity_level") or snap.activity_level
        )
        snap.social_value = float(shared_values.get("social_value") or 0.0)
        return snap

    async def capture_full_snapshot(
        self, channel_id: str, user_id: str = ""
    ) -> EnergySnapshot:
        """获取完整快照（能量 + 社交值）"""
        snap = self.capture_snapshot(channel_id)
        snap.user_id = user_id
        shared_values = get_shared_resource_manager().get_shared_values(
            channel_id
        )
        snap.activity_level = float(
            shared_values.get("activity_level") or snap.activity_level
        )
        if not user_id:
            return snap
        fuser = _get_affect_fuser()
        if fuser:
            try:
                affect = await fuser.get_snapshot(user_id, channel_id)
                snap.social_value = affect.social_score
                snap.trust_value = affect.trust_value
                snap.relationship_level = int(affect.phase)
                snap.custom_label = affect.phase_label
                snap.trend_direction = affect.trend
                record = await fuser.read_full_record(user_id, channel_id)
                if record:
                    snap.positive_dim = record.positive_dim
                    snap.negative_dim = record.negative_dim
                    snap.annoyance_value = record.annoyance_value
                    snap.interaction_count = record.interaction_count
            except Exception as e:
                logger.debug(f"获取社交情感状态失败: {e}")
        return snap

    def deplete_on_reply(
        self, channel_id: str, content_length: int = 0
    ) -> Dict[str, Any]:
        """回复消耗 - 动态自适应扣除

        聊天值消耗随水位递增：满水时每次回复 ~5%，低水时放缓
        思考值消耗随水位递增：满水时每次回复 ~3%，低水时放缓
        长消息额外消耗聊天+思考
        """
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)
        thinking_ratio = snap.thinking_ratio()
        brain_cost = max(0.5, min(3.5, thinking_ratio * 3.5))
        chat_ratio = snap.chat_ratio()
        # 聊天值动态消耗：满水时 ~5，低水时 ~1
        chat_cost = max(1.0, min(5.0, chat_ratio * 5.0))
        if content_length > 80:
            _extra = (content_length - 80) * 0.008
            chat_cost += min(2.0, _extra)
            brain_cost += min(1.5, _extra * 0.5)
        snap.thinking_value = max(0.0, snap.thinking_value - brain_cost)
        snap.chat_pool = max(0.0, snap.chat_pool - chat_cost)
        logger.debug(
            f"[{channel_id[:8]}] 回复消耗: chat-{chat_cost:.2f} brain-{brain_cost:.2f} "
            f"(ratio: chat={chat_ratio:.2f} thinking={thinking_ratio:.2f})"
        )
        return {
            "chat_pool": snap.chat_pool,
            "thinking_value": snap.thinking_value,
            "chat_cost": chat_cost,
            "thinking_cost": brain_cost,
        }

    def deplete_on_think(self, channel_id: str, complexity: float = 1.0) -> Dict[str, Any]:
        """思考消耗 - 动态自适应扣除

        消耗公式：
          基础消耗 = 当前水位比 × 2.5 → clamp [0.3, 2.5]
          最终消耗 = 基础消耗 × 复杂度系数 → clamp [0.2, 4.0]

        衰减曲线（复杂度=1.0）：
          10次思考后 ≈75%，20次后 ≈55%，30次后 ≈40%
        低水位时消耗自然放缓，模拟疲劳状态下思维迟钝。
        """
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)
        ratio = snap.thinking_ratio()
        # 基础消耗随水位动态调整：满水时消耗高，低水时消耗低
        base_cost = max(0.3, ratio * 2.5)
        # 复杂度系数（默认1.0，长文/多话题可传入更高值）
        final_cost = base_cost * max(0.5, min(2.0, complexity))
        final_cost = max(0.2, min(4.0, final_cost))
        snap.thinking_value = max(0.0, snap.thinking_value - final_cost)
        logger.debug(
            f"[{channel_id[:8]}] 思考消耗: -{final_cost:.2f} "
            f"(ratio={ratio:.2f} complexity={complexity:.1f}) "
            f"剩余={snap.thinking_value:.1f}"
        )
        return {
            "thinking_value": snap.thinking_value,
            "thinking_cost": final_cost,
        }

    def apply_manual_adjustment(
        self,
        channel_id: str,
        *,
        chat_delta: float = 0.0,
        thinking_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """统一应用聊天值/思考值调整，供上层主链做动态扣减时使用。"""
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)

        if chat_delta:
            snap.chat_pool = max(
                0.0, min(snap.chat_ceiling, snap.chat_pool + float(chat_delta))
            )
        if thinking_delta:
            snap.thinking_value = max(
                0.0,
                min(
                    snap.thinking_ceiling,
                    snap.thinking_value + float(thinking_delta),
                ),
            )

        return {
            "chat_pool": snap.chat_pool,
            "thinking_value": snap.thinking_value,
            "chat_delta": float(chat_delta),
            "thinking_delta": float(thinking_delta),
        }

    def deplete_on_glance(self, channel_id: str) -> Dict[str, Any]:
        """窥屏消耗 - 动态自适应

        满水时扫一眼消耗 ~1.5，低水位时放缓
        """
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)
        chat_ratio = snap.chat_ratio()
        chat_cost = max(0.3, min(1.5, chat_ratio * 1.5))
        snap.chat_pool = max(0.0, snap.chat_pool - chat_cost)
        return {"chat_pool": snap.chat_pool, "chat_cost": chat_cost}

    def replenish_on_input(
        self, channel_id: str, is_repeat: bool = False
    ) -> Dict[str, Any]:
        """用户消息后：少量回血 + 可能增烦恼"""
        snap = self._ensure_channel(channel_id)
        self._tick_recovery(snap)
        bonus = max(0.3, min(1.5, snap.chat_ratio() * 1.5))
        snap.chat_pool = min(snap.chat_ceiling, snap.chat_pool + bonus)
        if is_repeat:
            fc = self._load_cfg().get("decay", {})
            annoyance_add = float(fc.get("repeat_annoyance_add", 0))
            snap.annoyance_level = min(
                100.0, snap.annoyance_level + annoyance_add
            )
        return {"chat_pool": snap.chat_pool, "annoyance": snap.annoyance_level}

    def apply_rest_bonus(
        self, channel_id: str, rest_minutes: float = 0.0
    ) -> Dict[str, Any]:
        """休息恢复：大量回血并缓解疲劳"""
        snap = self._ensure_channel(channel_id)
        cc = self._chat_cfg()
        recovery_rate = float(cc.get("recovery_rate_per_min", 5.0))
        recovery = rest_minutes * recovery_rate * 2
        snap.chat_pool = min(snap.chat_ceiling, snap.chat_pool + recovery)
        snap.thinking_value = min(
            snap.thinking_ceiling, snap.thinking_value + recovery
        )
        snap.annoyance_level = max(
            0.0, snap.annoyance_level - rest_minutes * 3
        )
        snap.captured_at = time.time()
        logger.debug(f"[{channel_id[:8]}] 休息回血: +{recovery:.1f}")
        return {
            "chat_pool": snap.chat_pool,
            "thinking_value": snap.thinking_value,
        }

    def compute_eagerness(self, channel_id: str) -> Tuple[float, str]:
        """
        计算回复意愿度（0.05 ~ 1.0）和原因说明。
        综合思考值、聊天值、活跃度、社交值与烦躁度。
        """
        snap = self.capture_snapshot(channel_id)
        shared_values = get_shared_resource_manager().get_shared_values(
            channel_id
        )

        thinking_factor = snap.thinking_ratio()
        chat_value = float(shared_values.get("chat_value") or 0.0)
        activity_level = float(shared_values.get("activity_level") or 0.0)
        social_value = float(shared_values.get("social_value") or 0.0)

        chat_factor = max(
            0.0, min(1.0, chat_value / max(snap.chat_ceiling, 1.0))
        )
        activity_factor = max(0.0, min(1.0, activity_level / 100.0))
        social_factor = max(0.0, min(1.0, (social_value + 100.0) / 200.0))
        annoyance_cap = 0.5
        annoyance_penalty = min(snap.annoyance_level / 100.0, annoyance_cap)
        raw = (
            thinking_factor * 0.4
            + chat_factor * 0.25
            + activity_factor * 0.2
            + social_factor * 0.15
            - annoyance_penalty
        )
        eagerness = max(0.05, min(1.0, raw))
        reasons = []
        if chat_value < snap.chat_ceiling * 0.3:
            reasons.append(f"聊天值低({chat_value:.0f})")
        if activity_level < 30:
            reasons.append(f"活跃度低({activity_level:.0f})")
        if snap.thinking_value < snap.thinking_ceiling * 0.3:
            reasons.append(f"思考值低({snap.thinking_value:.0f})")
        if social_value < -20:
            reasons.append(f"社交值低({social_value:.0f})")
        if snap.annoyance_level > 30:
            reasons.append(f"烦躁({snap.annoyance_level:.0f})")
        reason = " | ".join(reasons) if reasons else "状态良好"
        return eagerness, reason

    async def compute_full_eagerness(
        self, channel_id: str, user_id: str = ""
    ) -> Tuple[float, str]:
        """
        计算完整回复意愿度 - 融合能量层 + 社交值层
        返回 (eagerness, reason)
        """
        snap = await self.capture_full_snapshot(channel_id, user_id)
        shared_values = get_shared_resource_manager().get_shared_values(
            channel_id
        )

        thinking_factor = snap.thinking_ratio()
        chat_value = float(shared_values.get("chat_value") or 0.0)
        activity_level = float(shared_values.get("activity_level") or 0.0)
        shared_social_value = float(shared_values.get("social_value") or 0.0)
        chat_factor = max(
            0.0, min(1.0, chat_value / max(snap.chat_ceiling, 1.0))
        )
        activity_factor = max(0.0, min(1.0, activity_level / 100.0))
        shared_social_factor = max(
            0.0, min(1.0, (shared_social_value + 100.0) / 200.0)
        )
        annoyance_cap = 0.5
        channel_annoyance_penalty = min(
            snap.annoyance_level / 100.0, annoyance_cap
        )
        social_factor = snap.social_ratio()
        trust_factor = snap.trust_ratio()
        user_annoyance_penalty = min(snap.annoyance_value / 100.0, 0.5)
        stage_factor = 0.8
        fuser = _get_affect_fuser()
        if fuser:
            try:
                stage_factor = fuser.get_phase_weight(user_id, channel_id)
            except Exception:
                logger.debug(f"获取阶段权重失败，使用默认值1.0")
        trend_factors = {
            "上升": 1.1,
            "稳定": 1.0,
            "下降": 0.85,
        }
        trend_factor = trend_factors.get(snap.trend_direction, 1.0)
        base_eagerness = (
            (
                thinking_factor * 0.3
                + chat_factor * 0.2
                + activity_factor * 0.15
                + shared_social_factor * 0.1
                + social_factor * 0.1
                + trust_factor * 0.05
            )
            * stage_factor
            * trend_factor
            - channel_annoyance_penalty
            - user_annoyance_penalty
        )
        eagerness = max(0.05, min(1.0, base_eagerness))
        reasons = []
        if chat_value < snap.chat_ceiling * 0.3:
            reasons.append(f"聊天值低({chat_value:.0f})")
        if activity_level < 30:
            reasons.append(f"活跃度低({activity_level:.0f})")
        if snap.thinking_value < snap.thinking_ceiling * 0.3:
            reasons.append(f"思考值低({snap.thinking_value:.0f})")
        if shared_social_value < -20:
            reasons.append(f"共享社交值低({shared_social_value:.0f})")
        if snap.social_value < -30:
            reasons.append(f"社交值低({snap.social_value:.0f})")
        if snap.annoyance_value > 50:
            reasons.append(f"用户烦恼({snap.annoyance_value:.0f})")
        if snap.relationship_level <= 1:
            label = snap.custom_label or self._get_level_description(
                snap.relationship_level
            )
            reasons.append(f"关系:{label}")
        if snap.trend_direction == "下降":
            reasons.append("关系下降中")
        reason = " | ".join(reasons) if reasons else "状态良好"
        return eagerness, reason

    def _get_level_description(self, level: int) -> str:
        """获取关系等级的默认描述"""
        descriptions = {
            0: "关系紧张",
            1: "不太熟悉",
            2: "有些熟悉",
            3: "比较熟悉",
            4: "关系亲密",
            5: "非常信任",
        }
        return descriptions.get(level, "有些熟悉")

    def suggest_brevity(self, channel_id: str) -> Tuple[str, str]:
        """
        建议回复长度档位。
        压力越大回复越短。
        """
        snap = self.capture_snapshot(channel_id)
        pressure = (
            snap.annoyance_level * 0.6
            - snap.chat_pool * 0.2
            - snap.thinking_value * 0.2
        )
        pressure = max(0, min(100, pressure))
        if pressure < 20:
            return "normal", "正常长度(8-20字)"
        elif pressure < 40:
            return "moderate", "适中长度(5-15字)"
        elif pressure < 60:
            return "brief", "简短(3-10字)"
        elif pressure < 80:
            return "minimal", "极简(2-6字)"
        return "ultra_minimal", "超短(1-3字)"

    def force_rest(self, channel_id: str, minutes: float = 0.0) -> None:
        """强制设置休息并立即应用恢复"""
        snap = self._ensure_channel(channel_id)
        if minutes > 0:
            self.apply_rest_bonus(channel_id, rest_minutes=minutes)
        snap.captured_at = time.time()
        logger.info(f"[{channel_id[:8]}] 强制休息 {minutes:.1f}分钟")

    def force_recover(self, channel_id: str, amount: float = 0.0) -> None:
        """强制注入能量"""
        snap = self._ensure_channel(channel_id)
        snap.chat_pool = min(snap.chat_ceiling, snap.chat_pool + amount)
        snap.thinking_value = min(
            snap.thinking_ceiling, snap.thinking_value + amount
        )

    def status_report(self, channel_id: str) -> Dict[str, Any]:
        """完整状态报告"""
        snap = self.capture_snapshot(channel_id)
        eagerness, reason = self.compute_eagerness(channel_id)
        brevity_level, brevity_desc = self.suggest_brevity(channel_id)
        return {
            "channel_id": channel_id,
            "chat_pool": round(snap.chat_pool, 1),
            "chat_ceiling": round(snap.chat_ceiling, 0),
            "thinking_value": round(snap.thinking_value, 1),
            "thinking_ceiling": round(snap.thinking_ceiling, 0),
            "chat_ratio": round(snap.chat_ratio(), 2),
            "thinking_ratio": round(snap.thinking_ratio(), 2),
            "annoyance": round(snap.annoyance_level, 1),
            "eagerness": round(eagerness, 2),
            "eagerness_reason": reason,
            "brevity_level": brevity_level,
            "brevity_desc": brevity_desc,
            "exhausted": snap.is_exhausted(),
            "social_value": round(snap.social_value, 1),
            "trust_value": round(snap.trust_value, 1),
            "annoyance_value": round(snap.annoyance_value, 1),
            "relationship_level": snap.relationship_level,
            "custom_label": snap.custom_label,
            "trend_direction": snap.trend_direction,
            "interaction_count": snap.interaction_count,
        }

    async def full_status_report(
        self, channel_id: str, user_id: str = ""
    ) -> Dict[str, Any]:
        """完整状态报告（包含社交值）"""
        snap = await self.capture_full_snapshot(channel_id, user_id)
        eagerness, reason = await self.compute_full_eagerness(
            channel_id, user_id
        )
        brevity_level, brevity_desc = self.suggest_brevity(channel_id)
        return {
            "channel_id": channel_id,
            "user_id": user_id,
            "chat_pool": round(snap.chat_pool, 1),
            "chat_ceiling": round(snap.chat_ceiling, 0),
            "thinking_value": round(snap.thinking_value, 1),
            "thinking_ceiling": round(snap.thinking_ceiling, 0),
            "chat_ratio": round(snap.chat_ratio(), 2),
            "thinking_ratio": round(snap.thinking_ratio(), 2),
            "annoyance": round(snap.annoyance_level, 1),
            "eagerness": round(eagerness, 2),
            "eagerness_reason": reason,
            "brevity_level": brevity_level,
            "brevity_desc": brevity_desc,
            "exhausted": snap.is_exhausted(),
            "social_value": round(snap.social_value, 1),
            "positive_dim": round(snap.positive_dim, 1),
            "negative_dim": round(snap.negative_dim, 1),
            "trust_value": round(snap.trust_value, 1),
            "annoyance_value": round(snap.annoyance_value, 1),
            "relationship_level": snap.relationship_level,
            "custom_label": snap.custom_label,
            "trend_direction": snap.trend_direction,
            "interaction_count": snap.interaction_count,
        }

    def bulk_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            cid: self.status_report(cid) for cid in self._channel_snapshots
        }

    def on_long_message_sent(self, channel_id: str, content: str = "") -> None:
        """长消息额外思考值消耗"""
        if len(content) <= 100:
            return
        snap = self._ensure_channel(channel_id)
        extra_cost = (len(content) - 100) * 0.01
        snap.thinking_value = max(0.0, snap.thinking_value - extra_cost)

    def compute_annoyance_penalty(self, channel_id: str) -> float:
        """
        综合烦恼惩罚系数 0~1。
        融合烦恼 + 能量缺口。
        """
        snap = self.capture_snapshot(channel_id)
        annoyance_norm = min(1.0, snap.annoyance_level / 100.0)
        energy_gap = 1.0 - min(snap.chat_ratio(), snap.thinking_ratio())
        a_penalty = 0.5 * annoyance_norm
        gap_penalty = energy_gap * 0.2
        total = a_penalty + gap_penalty
        return max(0.0, min(1.0, total))

    def persist_channel_energy(self, channel_id: str) -> None:
        """将指定频道的能量快照持久化到 persistent_state_db"""
        snap = self._channel_snapshots.get(channel_id)
        if snap is None:
            return
        payload = {
            "chat_pool": snap.chat_pool,
            "chat_ceiling": snap.chat_ceiling,
            "thinking_value": snap.thinking_value,
            "thinking_ceiling": snap.thinking_ceiling,
            "annoyance": snap.annoyance_level,
            "saved_at": time.time(),
        }
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            acquire_persistent_store().set_state("energy", channel_id, payload)
        except Exception as exc:
            logger.debug(f"[{channel_id[:8]}] 能量持久化失败: {exc}")

    def restore_channel_energy(self, channel_id: str) -> bool:
        """从 persistent_state_db 恢复频道能量快照，成功返回 True"""
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            data = acquire_persistent_store().get_state("energy", channel_id)
            if not data or not isinstance(data, dict):
                return False
            cc = self._chat_cfg()
            bc = self._thinking_cfg()
            ceiling_chat = float(cc.get("ceiling", 100.0))
            ceiling_brain = float(bc.get("ceiling", 100.0))
            snap = EnergySnapshot(
                chat_pool=min(
                    float(data.get("chat_pool", ceiling_chat)), ceiling_chat
                ),
                chat_ceiling=ceiling_chat,
                thinking_value=min(
                    float(data.get("thinking_value", ceiling_brain)),
                    ceiling_brain,
                ),
                thinking_ceiling=ceiling_brain,
                annoyance_level=float(data.get("annoyance", 0.0)),
                captured_at=time.time(),
            )
            self._channel_snapshots[channel_id] = snap
            logger.debug(f"[{channel_id[:8]}] 能量快照已恢复")
            return True
        except Exception as exc:
            logger.debug(f"[{channel_id[:8]}] 能量恢复失败: {exc}")
            return False

    def persist_all_channels(self) -> int:
        """批量持久化所有频道能量状态"""
        count = 0
        for cid in list(self._channel_snapshots.keys()):
            self.persist_channel_energy(cid)
            count += 1
        return count

    def collect_diagnostics(self) -> Dict[str, Any]:
        """运行时诊断信息，用于调试和健康检查"""
        active_channels = list(self._channel_snapshots.keys())
        exhausted = [
            cid
            for cid, snap in self._channel_snapshots.items()
            if snap.is_exhausted()
        ]
        high_annoyance = [
            cid
            for cid, snap in self._channel_snapshots.items()
            if snap.annoyance_level > 50.0
        ]
        config_info: Dict[str, Any] = {}
        try:
            config_info = {
                "chat_ceiling": float(self._chat_cfg().get("ceiling", 100.0)),
                "thinking_ceiling": float(
                    self._thinking_cfg().get("ceiling", 100.0)
                ),
                "decay_enabled": bool(self._load_cfg().get("enabled", True)),
            }
        except Exception:
            config_info = {"error": "配置未就绪"}
        return {
            "active_channel_count": len(active_channels),
            "exhausted_channels": exhausted,
            "high_annoyance_channels": high_annoyance,
            "config": config_info,
            "shared_resource_groups": len(
                get_shared_resource_manager()._group_resources
            ),
        }


def get_vitality_pool() -> VitalityPoolManager:
    """获取全局能量管理器"""
    return VitalityPoolManager.instance()


# ============================================================
# D6 维度接口：能量链条系统
# 包装 VitalityPoolManager + SharedResourceManager
# 使其符合 DimensionBase 统一协议
# ============================================================
import math
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import EnergyChainVote, EnergyStage


class EnergyChainDimension(DimensionBase):
    """
    D6 能量链条维度。
    包装 VitalityPoolManager（聊天值+思考值双池）和
    SharedResourceManager（群聊共享资源池），统一到维度接口。

    能量模型:
      - chat_pool: 聊天能量池（per-channel，与所有用户共享）
      - thinking_value: 思考能量池（per-channel）
      - 恢复采用动态阻尼式（低水位恢复慢）
      - 消耗采用链条百分比扣除（满水消耗高，低水自动放缓）
    """

    _singleton: Optional["EnergyChainDimension"] = None

    @classmethod
    def get_instance(cls) -> "EnergyChainDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._pool = VitalityPoolManager.instance()
        self._shared = get_shared_resource_manager()

    @property
    def dimension_name(self) -> str:
        return "energy_chain"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 3.0

    @property
    def persist_interval_sec(self) -> float:
        return 60.0

    def initialize(self):
        """从持久化存储恢复所有频道能量状态"""
        for channel_id in list(self._pool._channel_snapshots.keys()):
            self._pool.restore_channel_energy(channel_id)

    def tick(self, elapsed_sec: float) -> TickResult:
        """周期性驱动所有频道的能量恢复"""
        updated = False
        for channel_id in list(self._pool._channel_snapshots.keys()):
            snap = self._pool._channel_snapshots.get(channel_id)
            if snap is None:
                continue
            old_chat = snap.chat_pool
            old_think = snap.thinking_value
            self._pool._tick_recovery(snap)
            if (
                abs(snap.chat_pool - old_chat) > 0.01
                or abs(snap.thinking_value - old_think) > 0.01
            ):
                updated = True
        # 共享资源部分也恢复
        elapsed_min = elapsed_sec / 60.0
        for channel_id in list(self._shared._group_resources.keys()):
            self._shared.recover_shared_values(channel_id, elapsed_min)
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"频道数={len(self._pool._channel_snapshots)}"
            if updated
            else "",
        )

    def on_event(self, ctx: EventContext):
        """
        接收事件并更新能量。
        message_received: 用户消息少量回血
        reply_completed: 回复完成扣除聊天值+思考值
        """
        if not ctx.channel_id:
            return
        if ctx.event_type == "message_received":
            is_repeat = ctx.raw_extras.get("is_repeat", False)
            self._pool.replenish_on_input(ctx.channel_id, is_repeat=is_repeat)
            self._shared.apply_input_activity(
                ctx.channel_id,
                user_id=ctx.user_id,
                content_length=ctx.message_length,
                is_repeat=is_repeat,
            )
        elif ctx.event_type == "reply_completed":
            self._pool.deplete_on_reply(ctx.channel_id, ctx.reply_tokens)
            self._shared.deduct_chat_value(ctx.channel_id, ctx.reply_tokens, "reply")
            self._shared.deduct_activity_level(ctx.channel_id, ctx.reply_tokens, "reply")

    def vote(self, ctx: EventContext) -> EnergyChainVote:
        """
        根据当前频道能量状态生成投票。
        能量越低概率乘数越小，透支时建议强制拒绝。
        """
        if not ctx.channel_id:
            return EnergyChainVote()
        snap = self._pool.capture_snapshot(ctx.channel_id)
        chat_ratio = snap.chat_ratio()
        think_ratio = snap.thinking_ratio()
        ceiling = snap.chat_ceiling
        current = snap.chat_pool
        # 综合能量比（取较低的一方）
        combined_ratio = min(chat_ratio, think_ratio)
        # 确定能量阶段
        if combined_ratio > 0.8:
            stage = EnergyStage.FULL
        elif combined_ratio > 0.5:
            stage = EnergyStage.ADEQUATE
        elif combined_ratio > 0.2:
            stage = EnergyStage.LOW
        elif combined_ratio > 0.05:
            stage = EnergyStage.CRITICAL
        else:
            stage = EnergyStage.DEPLETED
        # 概率乘数：阻尼曲线
        if stage == EnergyStage.DEPLETED:
            prob = 0.05
        elif stage == EnergyStage.CRITICAL:
            prob = 0.3
        elif stage == EnergyStage.LOW:
            prob = 0.6 + combined_ratio * 0.5
        elif stage == EnergyStage.ADEQUATE:
            prob = 0.85 + combined_ratio * 0.15
        else:
            prob = 1.0
        # 是否强制拒绝（完全透支）
        force_refuse = snap.is_exhausted()
        # 预估消耗
        estimated_cost = max(0.01, min(0.15, chat_ratio * 0.05))
        # 检测夜间模式（简单用小时判断）
        import datetime
        current_hour = datetime.datetime.now().hour
        night_mode = current_hour < 6 or current_hour >= 23
        if night_mode:
            prob *= 0.7
        # 恢复阻尼系数
        recovery_damping = max(0.2, chat_ratio * 0.6 + 0.3)
        # token上限：能量低时限制回复长度
        tokens_cap = 0
        if stage == EnergyStage.CRITICAL:
            tokens_cap = 150
        elif stage == EnergyStage.LOW:
            tokens_cap = 350
        return EnergyChainVote(
            probability_factor=max(0.05, prob),
            force_refuse=force_refuse,
            attitude_tag="exhausted" if force_refuse else "",
            max_tokens_cap=tokens_cap,
            chat_energy_ratio=round(chat_ratio, 3),
            thinking_energy_ratio=round(think_ratio, 3),
            energy_stage=stage,
            estimated_cost=round(estimated_cost, 4),
            night_mode=night_mode,
            recovery_damping=round(recovery_damping, 3),
            chat_ceiling=ceiling,
            chat_current=round(current, 1),
            debug_reason=f"chat={chat_ratio:.2f} think={think_ratio:.2f} stage={stage.value}",
        )

    def serialize(self) -> dict:
        """导出所有频道的能量状态"""
        result = {}
        for cid, snap in self._pool._channel_snapshots.items():
            result[cid] = {
                "chat_pool": snap.chat_pool,
                "chat_ceiling": snap.chat_ceiling,
                "thinking_value": snap.thinking_value,
                "thinking_ceiling": snap.thinking_ceiling,
                "annoyance": snap.annoyance_level,
            }
        return result

    def deserialize(self, data: dict):
        """恢复所有频道的能量状态"""
        if not isinstance(data, dict):
            return
        for cid, vals in data.items():
            if not isinstance(vals, dict):
                continue
            cc = self._pool._chat_cfg()
            bc = self._pool._thinking_cfg()
            ceiling_chat = float(cc.get("ceiling", 100.0))
            ceiling_brain = float(bc.get("ceiling", 100.0))
            snap = EnergySnapshot(
                chat_pool=min(float(vals.get("chat_pool", ceiling_chat)), ceiling_chat),
                chat_ceiling=ceiling_chat,
                thinking_value=min(float(vals.get("thinking_value", ceiling_brain)), ceiling_brain),
                thinking_ceiling=ceiling_brain,
                annoyance_level=float(vals.get("annoyance", 0.0)),
                captured_at=time.time(),
            )
            self._pool._channel_snapshots[cid] = snap

    def calibrate(self, offline_seconds: float):
        """离线校准: 离线期间能量被动恢复"""
        offline_min = offline_seconds / 60.0
        recovery_cap = 0.8
        ratio = min(1.0, offline_min / 120.0)
        for snap in self._pool._channel_snapshots.values():
            recovery = offline_min * 2.0 * ratio * recovery_cap
            snap.chat_pool = min(snap.chat_ceiling, snap.chat_pool + recovery)
            snap.thinking_value = min(
                snap.thinking_ceiling, snap.thinking_value + recovery * 0.5
            )
            snap.annoyance_level = max(0.0, snap.annoyance_level - offline_min * 3.0)
            snap.captured_at = time.time()

    def get_state_summary(self) -> dict:
        summaries = {}
        for cid, snap in self._pool._channel_snapshots.items():
            summaries[cid[:8]] = {
                "chat_ratio": round(snap.chat_ratio(), 2),
                "think_ratio": round(snap.thinking_ratio(), 2),
                "annoyance": round(snap.annoyance_level, 1),
                "exhausted": snap.is_exhausted(),
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "channel_count": len(self._pool._channel_snapshots),
            "channels": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if channel_id and channel_id in self._pool._channel_snapshots:
            del self._pool._channel_snapshots[channel_id]
