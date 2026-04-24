import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.perception.runtime_config import perception_module_view

logger = get_logger("自感知")


@dataclass
class DecisionRecord:
    """决策记录"""

    timestamp: float = 0.0
    action: str = ""
    reason: str = ""
    confidence: float = 0.5
    context: str = ""


@dataclass
class SelfStatus:
    """自我状态"""

    current_time: str = ""
    time_of_day: str = ""
    session_duration_minutes: float = 0.0
    time_since_last_reply: str = "从未回复"
    decision_history: str = ""
    reply_count: int = 0
    active_streams: int = 0
    energy_level: str = "正常"
    mood_indicator: str = "平静"
    recent_decisions: List[DecisionRecord] = field(default_factory=list)
    interaction_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class SessionState:
    """会话状态"""

    start_time: float = 0.0
    last_reply_time: float = 0.0
    reply_count: int = 0
    decision_history: List[DecisionRecord] = field(default_factory=list)
    interaction_counts: Dict[str, int] = field(default_factory=dict)


class SelfSense:
    """自我感知 - 汇总机器人自身的状态信息"""

    def __init__(self, config_engine: Optional[Any] = None):
        del config_engine
        self._sessions: Dict[str, SessionState] = {}
        self._global_start_time = time.time()
        self._global_reply_count = 0
        self._max_history_size = 50
        self._load_config()

    def _load_config(self):
        """加载配置参数"""
        config = perception_module_view("perception_self_sense")
        self._max_history_size = int(config.get("max_history_size", 50))
        self._energy_decay_rate = float(
            config.get("energy_decay_rate", 0.01)
        )
        self._mood_sensitivity = float(
            config.get("mood_sensitivity", 0.5)
        )

    def get_status(self, stream_id: Optional[str] = None) -> SelfStatus:
        """
        获取自我状态

        Args:
            stream_id: 可选的流ID，用于获取特定会话的状态

        Returns:
            SelfStatus: 自我状态
        """
        status = SelfStatus()

        now = datetime.now()
        status.current_time = now.strftime("%H:%M")
        status.time_of_day = self._get_time_of_day(now.hour)

        session = self._sessions.get(stream_id) if stream_id else None
        if session:
            status.session_duration_minutes = (
                time.time() - session.start_time
            ) / 60.0
            if session.last_reply_time > 0:
                elapsed = time.time() - session.last_reply_time
                status.time_since_last_reply = self._format_elapsed(elapsed)
            else:
                status.time_since_last_reply = "从未回复"
            status.reply_count = session.reply_count
            status.recent_decisions = session.decision_history[-5:]
            status.interaction_stats = dict(session.interaction_counts)
            status.decision_history = self._build_decision_history(
                session.decision_history
            )
        else:
            status.session_duration_minutes = (
                time.time() - self._global_start_time
            ) / 60.0
            status.reply_count = self._global_reply_count
            status.time_since_last_reply = "从未回复"

        status.active_streams = len(self._sessions)

        status.energy_level = self._calc_energy_level(status)
        status.mood_indicator = self._calc_mood_indicator(status)

        return status

    def _get_time_of_day(self, hour: int) -> str:
        """获取时段描述"""
        if 0 <= hour < 6:
            return "凌晨"
        if 6 <= hour < 9:
            return "早上"
        if 9 <= hour < 12:
            return "上午"
        if 12 <= hour < 14:
            return "中午"
        if 14 <= hour < 18:
            return "下午"
        if 18 <= hour < 22:
            return "晚上"
        return "深夜"

    def _format_elapsed(self, seconds: float) -> str:
        """格式化时间间隔"""
        s = int(seconds)
        if s < 60:
            return f"{s}秒前"
        if s < 3600:
            return f"{s // 60}分钟前"
        if s < 86400:
            return f"{s // 3600}小时前"
        return f"{s // 86400}天前"

    def _calc_energy_level(self, status: SelfStatus) -> str:
        """计算能量等级"""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            base_energy = 0.4
        elif 6 <= hour < 12:
            base_energy = 0.9
        elif 12 <= hour < 14:
            base_energy = 0.7
        elif 14 <= hour < 18:
            base_energy = 0.8
        elif 18 <= hour < 22:
            base_energy = 0.6
        else:
            base_energy = 0.5

        reply_factor = max(0.5, 1.0 - status.reply_count * 0.01)
        energy = base_energy * reply_factor

        if energy >= 0.8:
            return "充沛"
        elif energy >= 0.6:
            return "正常"
        elif energy >= 0.4:
            return "一般"
        else:
            return "疲惫"

    def _calc_mood_indicator(self, status: SelfStatus) -> str:
        """计算心情指标"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            base_mood = "愉悦"
        elif 12 <= hour < 18:
            base_mood = "平静"
        elif 18 <= hour < 22:
            base_mood = "放松"
        else:
            base_mood = "安静"

        if status.reply_count > 50:
            return "忙碌"
        if status.reply_count > 20:
            return "充实"

        return base_mood

    def _build_decision_history(self, decisions: List[DecisionRecord]) -> str:
        """构建决策历史字符串"""
        if not decisions:
            return "暂无决策记录"

        recent = decisions[-5:]
        parts = []
        for d in recent:
            ts_str = datetime.fromtimestamp(d.timestamp).strftime("%H:%M")
            parts.append(f"[{ts_str}]{d.action}")
        return " → ".join(parts)

    def record_decision(
        self,
        stream_id: str,
        action: str,
        reason: str = "",
        confidence: float = 0.5,
        context: str = "",
    ):
        """
        记录决策

        Args:
            stream_id: 流ID
            action: 行动类型
            reason: 决策原因
            confidence: 置信度
            context: 上下文
        """
        if stream_id not in self._sessions:
            self._sessions[stream_id] = SessionState(start_time=time.time())

        session = self._sessions[stream_id]
        record = DecisionRecord(
            timestamp=time.time(),
            action=action,
            reason=reason,
            confidence=confidence,
            context=context,
        )
        session.decision_history.append(record)

        if len(session.decision_history) > self._max_history_size:
            session.decision_history = session.decision_history[
                -self._max_history_size:
            ]

        action_key = action.lower().replace(" ", "_")
        session.interaction_counts[action_key] = (
            session.interaction_counts.get(action_key, 0) + 1
        )

        logger.debug(
            f"记录决策: stream={stream_id}, action={action}, reason={reason}"
        )

    def record_reply(self, stream_id: str):
        """
        记录回复

        Args:
            stream_id: 流ID
        """
        if stream_id not in self._sessions:
            self._sessions[stream_id] = SessionState(start_time=time.time())

        session = self._sessions[stream_id]
        session.last_reply_time = time.time()
        session.reply_count += 1
        self._global_reply_count += 1

    def get_session_stats(self, stream_id: str) -> Optional[dict]:
        """
        获取会话统计

        Args:
            stream_id: 流ID

        Returns:
            会话统计字典，如果会话不存在则返回None
        """
        session = self._sessions.get(stream_id)
        if not session:
            return None

        now = time.time()
        return {
            "stream_id": stream_id,
            "duration_minutes": (now - session.start_time) / 60.0,
            "reply_count": session.reply_count,
            "decision_count": len(session.decision_history),
            "last_reply_time": session.last_reply_time,
            "time_since_last_reply": (
                now - session.last_reply_time
                if session.last_reply_time > 0
                else -1
            ),
            "interaction_counts": dict(session.interaction_counts),
        }

    def get_global_stats(self) -> dict:
        """获取全局统计"""
        now = time.time()
        total_replies = sum(s.reply_count for s in self._sessions.values())
        total_decisions = sum(
            len(s.decision_history) for s in self._sessions.values()
        )

        return {
            "uptime_minutes": (now - self._global_start_time) / 60.0,
            "active_streams": len(self._sessions),
            "total_replies": total_replies,
            "total_decisions": total_decisions,
            "global_reply_count": self._global_reply_count,
        }

    def clear_session(self, stream_id: str):
        """清除会话"""
        self._sessions.pop(stream_id, None)

    def clear_all_sessions(self):
        """清除所有会话"""
        self._sessions.clear()

    def get_time_context(self) -> dict:
        """获取时间上下文"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        return {
            "hour": hour,
            "minute": minute,
            "weekday": weekday,
            "weekday_name": [
                "周一",
                "周二",
                "周三",
                "周四",
                "周五",
                "周六",
                "周日",
            ][weekday],
            "time_of_day": self._get_time_of_day(hour),
            "is_weekend": weekday >= 5,
            "is_morning": 6 <= hour < 12,
            "is_afternoon": 12 <= hour < 18,
            "is_evening": 18 <= hour < 22,
            "is_night": hour >= 22 or hour < 6,
            "current_time_str": now.strftime("%H:%M"),
            "current_date_str": now.strftime("%Y-%m-%d"),
        }

    def should_be_quiet(self) -> bool:
        """判断是否应该安静（深夜时段）"""
        hour = datetime.now().hour
        return 0 <= hour < 6

    def get_activity_suggestion(self) -> str:
        """获取活动建议"""
        hour = datetime.now().hour
        if 6 <= hour < 9:
            return "早安时间，可以问候大家"
        elif 9 <= hour < 12:
            return "上午时段，适合活跃讨论"
        elif 12 <= hour < 14:
            return "午休时间，可以轻松聊天"
        elif 14 <= hour < 18:
            return "下午时段，适合持续互动"
        elif 18 <= hour < 22:
            return "晚间时段，可以放松交流"
        else:
            return "深夜时段，建议保持安静"


_self_sense_instance: Optional[SelfSense] = None


def get_self_sense(config_engine: Optional[Any] = None) -> SelfSense:
    """获取自我感知单例"""
    global _self_sense_instance
    if _self_sense_instance is None:
        _self_sense_instance = SelfSense(config_engine)
    return _self_sense_instance
