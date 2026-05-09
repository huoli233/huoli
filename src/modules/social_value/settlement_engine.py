import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from src.common.logger import get_logger
from src.modules.social_value.social_calculator import SocialCalculator
from src.modules.social_value.runtime_config import social_value_module_view
from src.modules.social_value.social_storage import SocialStorage
from src.modules.social_value.models import (
    SocialValueRecord,
)

logger = get_logger("结算引擎")

# 分值硬边界
SCORE_FLOOR = -100.0
SCORE_CEILING = 100.0


@dataclass
class SettlementReport:
    """单次结算的完整诊断报告"""

    prior_score: float = 0.0
    score_after_decay: float = 0.0
    raw_delta: float = 0.0
    delta_after_saturation: float = 0.0
    delta_after_reversal: float = 0.0
    delta_after_damping: float = 0.0
    delta_after_cap: float = 0.0
    final_delta: float = 0.0
    settled_score: float = 0.0
    reversal_triggered: bool = False
    cap_triggered: bool = False
    hours_elapsed: float = 0.0
    is_first_encounter: bool = False


@dataclass
class DirectionLedger:
    """方向连续性记录簿
    用于追踪同向连续次数，判定反转奖励条件"""

    last_sign: int = 0
    consecutive_count: int = 0

    def feed(self, sign: int) -> int:
        """输入新方向符号，返回反转前的连续次数"""
        if sign == 0:
            return 0
        stale_count = 0
        if sign != self.last_sign and self.last_sign != 0:
            stale_count = self.consecutive_count
            self.consecutive_count = 1
            self.last_sign = sign
        else:
            self.consecutive_count += 1
            self.last_sign = sign
        return stale_count


class SettlementEngine:
    """量化结算引擎（Layer 1）

    职责：将行为信号转化为社交分值的量化变化，管控所有数值修正管线。
    整合了原 SocialValueCore 的衰减/阻尼/上限体系
    与原 BondSettler 的饱和曲线/反转奖励/大幅压制体系。

    修正管线（按顺序）:
      1. 时间衰减 → 将长期离线折算为分值自然回归
      2. 原始 delta 计算（由 SocialCalculator 驱动）
      3. 饱和衰减 → 越接近极值，变化越迟钝
      4. 反转奖励 → 连续同向后突然反转，给予放大
      5. 极化阻尼 → 极端分值区域额外阻尼
      6. 交互频次阻尼 → 高频交互者变化递减
      7. 合并上限 → 取 SVC 严重度上限与 BS 步长天花板的更严者
      8. 关系修正 → 新用户宽容 / 创伤敏感偏移
      9. 夹紧到 [-100, +100]
    """

    def __init__(
        self,
        calculator: SocialCalculator,
        storage: SocialStorage,
        config_engine: Optional[object] = None,
    ):
        self._calc = calculator
        self._store = storage
        # 方向连续性记录（按 user_id 索引）
        self._direction_books: Dict[str, DirectionLedger] = {}
        # 运行时配置缓存
        self._tuning = self._assemble_tuning()

    # ================================================================
    #  配置组装
    # ================================================================

    def _assemble_tuning(self) -> Dict[str, float]:
        """从配置画像合并为统一调优参数"""
        base = {
            # 衰减
            "decay_rate_per_hour": 0.015,
            # 饱和曲线陡度
            "saturation_steepness": 0.02,
            # 反转奖励
            "reversal_amplifier": 1.35,
            "reversal_required_streak": 3,
            # 极化阻尼权重
            "polarization_drag": 0.5,
            # 交互频次阻尼权重
            "frequency_drag": 0.15,
            # 大幅压制
            "large_step_threshold": 15.0,
            "large_step_compress_rate": 0.6,
            # 合并上限
            "severity_floor_cap": 0.5,
            "severity_ceil_cap": 2.0,
            "absolute_step_ceil": 20.0,
            # 新用户宽容
            "newcomer_interaction_bar": 10.0,
            "newcomer_softening": 0.6,
            # 创伤敏感
            "trauma_alarm_level": 5.0,
            "trauma_negative_amplify": 0.25,
            # 新用户初始值
            "first_encounter_seed": 5.0,
        }
        module_cfg = social_value_module_view("social_settlement")
        if isinstance(module_cfg, dict):
            for key in list(base.keys()):
                if key in module_cfg:
                    try:
                        base[key] = float(module_cfg[key])
                    except (ValueError, TypeError):
                        pass
        # 从人格滑块读取粘合引擎侧配置
        try:
            from src.config.core_config_engine import get_core_config

            ps = get_core_config().resolve_module_view("personality").values
            slider_mapping = {
                "saturation_steepness": "bond_saturation_steepness",
                "reversal_amplifier": "bond_reversal_multiplier",
                "reversal_required_streak": "bond_reversal_min_streak",
                "large_step_threshold": "bond_large_delta_threshold",
                "large_step_compress_rate": "bond_large_delta_dampen_rate",
                "absolute_step_ceil": "bond_step_ceiling",
            }
            for local_key, slider_key in slider_mapping.items():
                if slider_key in ps:
                    try:
                        base[local_key] = float(ps[slider_key])
                    except (ValueError, TypeError):
                        pass
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return base

    def _t(
        self, key: str, fallback: float, ctx: Optional[Dict[str, Any]] = None
    ) -> float:
        """读取调优值，优先从 ctx 覆盖"""
        if ctx and key in ctx:
            try:
                return float(ctx[key])
            except (ValueError, TypeError):
                pass
        return float(self._tuning.get(key, fallback))

    # ================================================================
    #  公开接口
    # ================================================================

    async def settle_event(
        self,
        user_id: str,
        channel_id: str,
        content: str,
        context: Dict[str, Any],
    ) -> SettlementReport:
        """从文本内容触发完整结算流水线

        外部调用此方法时传入原始聊天内容与上下文，
        引擎会自动完成行为分析 → 计算 → 修正 → 持久化。
        """
        behavior_signal = (
            context.get("behavior_signal")
            if isinstance(context, dict)
            else None
        )
        if isinstance(behavior_signal, dict) and behavior_signal.get(
            "behavior_type"
        ):
            behavior = {
                "behavior_type": str(
                    behavior_signal.get("behavior_type", "casual_chat")
                ),
                "severity": float(behavior_signal.get("severity", 0.5)),
                "intent": str(behavior_signal.get("intent", "chat")),
            }
        else:
            behavior = await self._infer_behavior(content, context)
        return await self.settle_from_behavior(
            user_id, channel_id, behavior, context
        )

    async def settle_from_behavior(
        self,
        user_id: str,
        channel_id: str,
        behavior: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> SettlementReport:
        """从已识别的行为结构体执行结算"""
        ctx = context or {}
        report = SettlementReport()
        # 读取当前记录
        record = await self._store.get(user_id, channel_id)
        if record is None:
            report.is_first_encounter = True
            seed = self._t("first_encounter_seed", 5.0, ctx)
            prior_score = seed
            interaction_seq = 0
            logger.info(f"[结算] 首次接触 uid={user_id[:8]} 种子分={seed:.1f}")
        else:
            prior_score = record.value
            interaction_seq = int(record.interaction_count)
        report.prior_score = prior_score
        # 步骤 1：时间衰减
        elapsed_h = 0.0
        if record and record.last_interaction > 0:
            elapsed_h = max(
                0.0, (time.time() - float(record.last_interaction)) / 3600.0
            )
        report.hours_elapsed = elapsed_h
        decayed_score = self._pipe_time_decay(prior_score, elapsed_h, ctx)
        report.score_after_decay = decayed_score
        # 步骤 2：计算原始 delta
        btype = (
            behavior.get("behavior_type", "neutral")
            if isinstance(behavior, dict)
            else getattr(behavior, "behavior_type", "neutral")
        )
        sev = (
            behavior.get("severity", 0.5)
            if isinstance(behavior, dict)
            else getattr(behavior, "severity", 0.5)
        )
        intent = (
            behavior.get("intent", "other")
            if isinstance(behavior, dict)
            else getattr(behavior, "intent", "other")
        )
        raw_delta = self._calc.calculate(
            behavior_type=btype,
            severity=sev,
            intent=intent,
            current_value=decayed_score,
            psychological_pressure=ctx.get("psychological_pressure", 0.0),
            training_resistance=ctx.get("training_resistance", 0.0),
            uid=user_id,
        )
        report.raw_delta = raw_delta
        # 步骤 3：饱和衰减
        sat_delta = self._pipe_saturation(raw_delta, decayed_score, ctx)
        report.delta_after_saturation = sat_delta
        # 步骤 4：反转奖励
        rev_delta, rev_hit = self._pipe_reversal(user_id, sat_delta, ctx)
        report.delta_after_reversal = rev_delta
        report.reversal_triggered = rev_hit
        # 步骤 5+6：极化阻尼 + 频次阻尼
        damped_delta = self._pipe_compound_damping(
            rev_delta, decayed_score, interaction_seq, ctx
        )
        report.delta_after_damping = damped_delta
        # 步骤 7：合并上限
        capped_delta, cap_hit = self._pipe_merged_cap(damped_delta, sev, ctx)
        report.delta_after_cap = capped_delta
        report.cap_triggered = cap_hit
        # 步骤 8：关系修正
        adjusted_delta = self._pipe_relation_shift(
            capped_delta, interaction_seq, ctx
        )
        report.final_delta = adjusted_delta
        # 步骤 9：夹紧 + 持久化
        settled = max(
            SCORE_FLOOR, min(SCORE_CEILING, decayed_score + adjusted_delta)
        )
        report.settled_score = settled
        await self._store.set(
            user_id=user_id, channel_id=channel_id, value=settled
        )
        return report

    async def decay_only(
        self, user_id: str, channel_id: str, elapsed_hours: float
    ) -> float:
        """仅执行时间衰减，不触发行为结算（供定时任务调用）"""
        record = await self._store.get(user_id, channel_id)
        if record is None:
            return 0.0
        new_val = self._pipe_time_decay(
            record.value, max(0.0, elapsed_hours), {}
        )
        await self._store.set(user_id, channel_id, new_val)
        return new_val

    async def read_score(self, user_id: str, channel_id: str) -> float:
        """只读取当前分值"""
        record = await self._store.get(user_id, channel_id)
        return record.value if record else 0.0

    async def read_full_record(
        self, user_id: str, channel_id: str
    ) -> Optional[SocialValueRecord]:
        """只读取完整记录"""
        return await self._store.get(user_id, channel_id)

    # ================================================================
    #  修正管线：各阶段独立函数
    # ================================================================

    def _pipe_time_decay(
        self, score: float, hours: float, ctx: Dict[str, Any]
    ) -> float:
        """时间衰减：分值向零点自然回归"""
        if abs(score) < 1e-6 or hours <= 0:
            return score
        rate = self._t("decay_rate_per_hour", 0.015, ctx)
        # 衰减因子限制在 [0, 0.95]，防止长时间离线一步归零
        factor = max(0.0, min(0.95, rate * hours))
        return score * (1.0 - factor)

    def _pipe_saturation(
        self, delta: float, current: float, ctx: Dict[str, Any]
    ) -> float:
        """饱和衰减：越接近极值，变化越迟钝
        使用 logistic 型抑制曲线：factor = 1 / (1 + k * r²)
        其中 r = |current| / cap"""
        if abs(delta) < 1e-8:
            return delta
        steep = self._t("saturation_steepness", 0.02, ctx)
        ratio = abs(current) / max(SCORE_CEILING, 1.0)
        suppression = 1.0 / (1.0 + steep * (ratio**2) * 100.0)
        return delta * suppression

    def _pipe_reversal(
        self, user_id: str, delta: float, ctx: Dict[str, Any]
    ) -> tuple:
        """反转奖励：连续同向达标后突然反转 → 放大变化"""
        if abs(delta) < 1e-8:
            return delta, False
        sign = 1 if delta > 0 else -1
        ledger = self._direction_books.get(user_id)
        if ledger is None:
            ledger = DirectionLedger()
            self._direction_books[user_id] = ledger
        stale_streak = ledger.feed(sign)
        required = int(self._t("reversal_required_streak", 3, ctx))
        if stale_streak >= required:
            amp = self._t("reversal_amplifier", 1.35, ctx)
            bonus = abs(delta) * (amp - 1.0)
            logger.debug(
                f"[结算] 反转奖励触发 uid={user_id[:8]} 旧连续={stale_streak} 加成={bonus:.3f}"
            )
            return delta + bonus * sign, True
        return delta, False

    def _pipe_compound_damping(
        self,
        delta: float,
        current: float,
        interaction_seq: int,
        ctx: Dict[str, Any],
    ) -> float:
        """复合阻尼：同时考虑极化程度与交互频次

        - 极化阻尼：极端区域变化阻力增大
        - 频次阻尼：长期用户变化递减
        合并公式避免两层各自独立扣减导致过度压制。
        """
        if abs(delta) < 1e-8:
            return delta
        pol_drag = self._t("polarization_drag", 0.5, ctx)
        freq_drag = self._t("frequency_drag", 0.15, ctx)
        pol_ratio = min(1.0, abs(current) / SCORE_CEILING)
        freq_ratio = min(1.0, max(0, interaction_seq) / 200.0)
        # 综合抑制因子，下限 0.25 防止极端场景趋近零
        combined = 1.0 - (pol_ratio * pol_drag + freq_ratio * freq_drag)
        combined = max(0.25, min(1.0, combined))
        return delta * combined

    def _pipe_merged_cap(
        self, delta: float, severity: float, ctx: Dict[str, Any]
    ) -> tuple:
        """合并上限：从两套配置取更严格的上限

        SVC 侧：severity_floor_cap ~ severity_ceil_cap 按严重度插值
        BS 侧：absolute_step_ceil 固定天花板
        最终取两者中更小的值。
        """
        floor_cap = self._t("severity_floor_cap", 0.5, ctx)
        ceil_cap = self._t("severity_ceil_cap", 2.0, ctx)
        abs_ceil = self._t("absolute_step_ceil", 20.0, ctx)
        # 大幅压制：超过阈值的部分额外压缩
        lg_thresh = self._t("large_step_threshold", 15.0, ctx)
        lg_compress = self._t("large_step_compress_rate", 0.6, ctx)
        working = delta
        if abs(working) > lg_thresh:
            excess = abs(working) - lg_thresh
            working = (lg_thresh + excess * lg_compress) * (
                1 if working > 0 else -1
            )
        # 严重度插值上限
        if ceil_cap < floor_cap:
            floor_cap, ceil_cap = ceil_cap, floor_cap
        s = min(1.0, max(0.0, float(severity)))
        sev_cap = floor_cap + (ceil_cap - floor_cap) * s
        # 取两者更严
        effective_cap = min(sev_cap, abs_ceil)
        capped = max(-effective_cap, min(effective_cap, working))
        was_capped = abs(capped) < abs(working) - 1e-6
        return capped, was_capped

    def _pipe_relation_shift(
        self, delta: float, interaction_seq: int, ctx: Dict[str, Any]
    ) -> float:
        """关系修正：新用户宽容 + 创伤敏感"""
        if abs(delta) < 1e-8:
            return delta
        result = delta
        # 新用户宽容：交互次数低于门槛时缩减变化幅度
        bar = self._t("newcomer_interaction_bar", 10.0, ctx)
        softening = self._t("newcomer_softening", 0.6, ctx)
        if interaction_seq < bar:
            result *= softening
        # 创伤敏感：高创伤分时放大负面变化
        trauma_alarm = self._t("trauma_alarm_level", 5.0, ctx)
        trauma_amp = self._t("trauma_negative_amplify", 0.25, ctx)
        trauma_score = max(0.0, float(ctx.get("trauma_score", 0.0)))
        if trauma_score >= trauma_alarm and result < 0:
            result *= 1.0 + trauma_amp
        return result

    # ================================================================
    #  行为推断（fallback）
    # ================================================================

    async def _infer_behavior(
        self, content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于关键词的行为推断（无外部模型信号时的兜底逻辑）"""
        text = content.lower() if content else ""
        btype = "casual_chat"
        sev = 0.5
        intent = "chat"
        # 正面关键词池
        warmth_words = [
            "谢谢",
            "感谢",
            "太棒了",
            "厉害",
            "牛",
            "赞",
            "喜欢",
            "爱",
            "可爱",
            "哈哈",
            "好",
            "不错",
            "可以",
            "真棒",
            "太好了",
            "开心",
            "高兴",
            "优秀",
            "完美",
            "厉害了",
            "牛逼",
            "棒极了",
            "太强了",
            "真厉害",
            "谢谢啦",
            "多谢",
            "辛苦了",
            "麻烦你了",
            "帮大忙了",
            "太感谢了",
        ]
        # 负面关键词池
        hostility_words = [
            "滚",
            "闭嘴",
            "烦人",
            "讨厌",
            "恶心",
            "垃圾",
            "废物",
            "傻",
            "笨",
            "蠢",
            "滚开",
            "走开",
            "别烦我",
            "烦死了",
            "真烦",
            "无聊",
            "没意思",
            "差劲",
            "太烂了",
            "什么破",
            "垃圾啊",
            "别说了",
            "别讲了",
            "闭嘴吧",
            "安静点",
            "吵死了",
        ]
        # 侮辱关键词池
        abuse_words = [
            "傻逼",
            "操你",
            "妈的",
            "草泥马",
            "去死",
            "滚蛋",
            "混蛋",
            "王八蛋",
            "畜生",
            "狗",
            "猪",
            "白痴",
            "弱智",
            "脑残",
        ]
        # 夸赞关键词池
        admiration_words = [
            "你真棒",
            "你太厉害了",
            "你是最棒的",
            "你真聪明",
            "你真厉害",
            "我崇拜你",
            "你太牛了",
            "你真优秀",
            "你真厉害啊",
        ]
        # 疑问关键词池
        inquiry_words = [
            "？",
            "?",
            "吗",
            "呢",
            "怎么",
            "什么",
            "为什么",
            "哪里",
            "谁",
            "如何",
            "怎样",
            "多少",
            "几",
            "是不是",
            "对不对",
            "好不好",
        ]
        # 问候关键词池
        salutation_words = [
            "你好",
            "早上好",
            "晚上好",
            "下午好",
            "嗨",
            "哈喽",
            "在吗",
            "在不在",
            "有人吗",
            "hello",
            "hi",
            "早啊",
            "晚安",
        ]
        warmth_hits = sum(1 for w in warmth_words if w in text)
        hostility_hits = sum(1 for w in hostility_words if w in text)
        abuse_hits = sum(1 for w in abuse_words if w in text)
        admiration_hits = sum(1 for w in admiration_words if w in text)
        inquiry_hits = sum(1 for w in inquiry_words if w in text)
        salutation_hits = sum(1 for w in salutation_words if w in text)
        if abuse_hits > 0:
            btype = "direct_insult"
            sev = min(1.0, 0.7 + abuse_hits * 0.1)
            intent = "insult"
        elif hostility_hits > warmth_hits + 2:
            btype = "hostile_remark"
            sev = min(0.9, 0.5 + hostility_hits * 0.05)
            intent = "hostile"
        elif admiration_hits > 0:
            btype = "genuine_praise"
            sev = min(1.0, 0.6 + admiration_hits * 0.15)
            intent = "compliment"
        elif warmth_hits > hostility_hits + 1:
            btype = "friendly_chat"
            sev = min(0.8, 0.4 + warmth_hits * 0.05)
            intent = "friendly"
        elif inquiry_hits > 0 and len(text) < 50:
            btype = "casual_inquiry"
            sev = 0.3
            intent = "inquiry"
        elif salutation_hits > 0:
            btype = "greeting"
            sev = 0.2
            intent = "greeting"
        elif inquiry_hits > 0:
            btype = "casual_chat"
            sev = 0.4
            intent = "chat"
        # 上下文微调
        interaction_flavor = context.get("interaction_type", "")
        if interaction_flavor == "bot_reply" and btype == "casual_chat":
            btype = "friendly_chat"
            sev = 0.3
            intent = "friendly"
        current_social = context.get("current_social", 0.0)
        if current_social < -30 and btype in ("casual_chat", "friendly_chat"):
            sev *= 0.7
        elif current_social > 30 and btype in ("casual_chat", "friendly_chat"):
            sev *= 1.2
        return {
            "behavior_type": btype,
            "severity": round(sev, 2),
            "intent": intent,
        }

    def collect_diagnostics(self) -> Dict[str, Any]:
        """收集运行时诊断信息"""
        return {
            "tracked_users": len(self._direction_books),
            "tuning_snapshot": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self._tuning.items()
            },
        }

    def get_behavior_category(self, behavior_type: str, intent: str) -> str:
        try:
            return self._calc.get_category_params(behavior_type, intent).category
        except Exception:
            return ""


_settlement_singleton = None


def get_settlement_engine() -> "SettlementEngine":
    """获取SettlementEngine单例（9步富管线结算引擎）"""
    global _settlement_singleton
    if _settlement_singleton is None:
        _calc = SocialCalculator()
        _store = SocialStorage()
        _settlement_singleton = SettlementEngine(_calc, _store)
    return _settlement_singleton
