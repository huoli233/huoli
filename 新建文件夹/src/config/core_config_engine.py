import os
import copy
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("核心设置")
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_TEMPLATE_DIR = _PROJECT_ROOT / "template"
_CONFIG_DIR = _PROJECT_ROOT / "config"
_TOML_NAME = "core_config.toml"
_TEMPLATE_NAME = "core_config_template.toml"


class CoreSettingsHub:
    """核心聚合配置中枢，为心流、能量、主动回复等全部子系统提供统一配置"""

    _sole_ref: Optional["CoreSettingsHub"] = None
    _boot_lock = threading.Lock()
    # 已知的顶层段落名称，用于校验配置文件中是否存在拼写错误的段落
    _KNOWN_SECTIONS = frozenset(
        {
            "heartflow_timing",
            "dual_pool_energy",
            "penalty_caps",
            "emotion_stream",
            "proactive_schedule",
            "proactive_arbiter",
            "heartflow_decision",
            "message_processor",
            "frequency_control",
            "module_switches",
            "silence_detection",
            "trigger_thresholds",
            "inner_voice",
            "personality_sliders",
            "runtime_tuning",
        }
    )

    @classmethod
    def boot(
        cls,
        config_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
    ) -> "CoreSettingsHub":
        """启动配置中枢，程序启动时调用一次（线程安全）"""
        if cls._sole_ref is not None:
            return cls._sole_ref
        with cls._boot_lock:
            if cls._sole_ref is not None:
                return cls._sole_ref
            hub = cls.__new__(cls)
            hub._sections = {}
            hub._toml_path = (config_dir or _CONFIG_DIR) / _TOML_NAME
            hub._template_path = (
                template_dir or _TEMPLATE_DIR
            ) / _TEMPLATE_NAME
            hub._disk_mtime = 0.0
            hub._slider_cache = {}
            hub._slider_stale = True
            hub._do_init()
            cls._sole_ref = hub
            return hub

    @classmethod
    def hub(cls) -> "CoreSettingsHub":
        """获取已启动的配置中枢实例"""
        if cls._sole_ref is None:
            raise RuntimeError(
                "CoreSettingsHub 尚未启动，请先调用 CoreSettingsHub.boot()"
            )
        return cls._sole_ref

    @classmethod
    def teardown(cls) -> None:
        """销毁实例，仅测试用途"""
        cls._sole_ref = None

    def _do_init(self) -> None:
        """内部初始化流程"""
        if not self._toml_path.exists():
            self._stamp_from_template()
        self._ingest_disk()

    def _stamp_from_template(self) -> None:
        """从模板复制配置文件到运行目录"""
        if not self._template_path.exists():
            logger.error(f"配置模板缺失: {self._template_path}")
            raise FileNotFoundError(f"配置模板不存在: {self._template_path}")
        self._toml_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(self._template_path), str(self._toml_path))
        logger.info(f"已从模板部署配置: {self._toml_path}")

    def _ingest_disk(self) -> None:
        """从磁盘解析TOML并填充内部缓存"""
        filepath = str(self._toml_path)
        if not os.path.exists(filepath):
            logger.warning(f"配置文件不存在: {filepath}")
            return
        try:
            import tomlkit

            with open(filepath, "r", encoding="utf-8") as fh:
                doc = tomlkit.load(fh)
            self._sections = dict(doc)
            self._disk_mtime = os.path.getmtime(filepath)
            self._slider_stale = True
            # 校验段落名称：检测拼写错误或遗留的无效段落
            for section_name in self._sections:
                if (
                    isinstance(self._sections[section_name], dict)
                    and section_name not in self._KNOWN_SECTIONS
                ):
                    logger.warning(
                        f"配置文件存在未识别的段落 [{section_name}]，请确认是否拼写正确"
                    )
            logger.debug("核心配置已加载")
        except Exception as exc:
            logger.error(f"配置解析失败: {exc}")

    def hot_reload(self) -> bool:
        """检查文件是否被修改，变动则重新加载"""
        filepath = str(self._toml_path)
        if not os.path.exists(filepath):
            return False
        current = os.path.getmtime(filepath)
        if current <= self._disk_mtime:
            return False
        logger.info("检测到配置变更，执行热重载")
        self._ingest_disk()
        return True

    # ---- 通用读取 ----

    def fetch_block(self, block_name: str) -> Dict[str, Any]:
        """获取指定TOML段落的完整副本"""
        block = self._sections.get(block_name)
        if not isinstance(block, dict):
            return {}
        return copy.deepcopy(block)

    def fetch_item(
        self, block_name: str, key: str, fallback: Any = None
    ) -> Any:
        """获取指定段落下单个键值"""
        block = self._sections.get(block_name)
        if not isinstance(block, dict):
            return fallback
        val = block.get(key)
        if val is None:
            return fallback
        if isinstance(val, (dict, list)):
            return copy.deepcopy(val)
        return val

    def fetch_deep(self, *segments: str, fallback: Any = None) -> Any:
        """通过多级路径读取嵌套值"""
        cursor: Any = self._sections
        for seg in segments:
            if not isinstance(cursor, dict):
                return fallback
            cursor = cursor.get(seg)
            if cursor is None:
                return fallback
        if isinstance(cursor, (dict, list)):
            return copy.deepcopy(cursor)
        return cursor

    # ---- 段落快捷方法 ----

    def heartflow_timing(self) -> Dict[str, Any]:
        """心流状态机时序参数"""
        return self.fetch_block("heartflow_timing")

    def dual_pool_energy(self) -> Dict[str, Any]:
        """双池能量系统参数"""
        return self.fetch_block("dual_pool_energy")

    def proactive_schedule(self) -> Dict[str, Any]:
        """主动回复调度参数"""
        return self.fetch_block("proactive_schedule")

    def proactive_schedule_block(self) -> Dict[str, Any]:
        """兼容旧调用名。"""
        return self.proactive_schedule()

    def emotion_stream_block(self) -> Dict[str, Any]:
        """情绪流参数"""
        return self.fetch_block("emotion_stream")

    def penalty_caps_block(self) -> Dict[str, Any]:
        """惩罚上限参数"""
        return self.fetch_block("penalty_caps")

    def silence_detection_block(self) -> Dict[str, Any]:
        """沉默检测参数"""
        return self.fetch_block("silence_detection")

    def trigger_thresholds_block(self) -> Dict[str, Any]:
        """触发阈值参数"""
        return self.fetch_block("trigger_thresholds")

    def inner_voice_block(self) -> Dict[str, Any]:
        """内心独白参数"""
        return self.fetch_block("inner_voice")

    def personality_sliders_block(self) -> Dict[str, Any]:
        """人格滑块"""
        return self.fetch_block("personality_sliders")

    def module_switches_block(self) -> Dict[str, Any]:
        """模块开关"""
        return self.fetch_block("module_switches")

    def heartflow_decision_block(self) -> Dict[str, Any]:
        """心流决策参数"""
        return self.fetch_block("heartflow_decision")

    def message_processor_block(self) -> Dict[str, Any]:
        """消息处理管线参数"""
        return self.fetch_block("message_processor")

    def runtime_tuning_block(self) -> Dict[str, Any]:
        """运行时调优参数"""
        return self.fetch_block("runtime_tuning")

    def assemble_decision_config(self) -> Dict[str, Any]:
        """组装心流决策的完整配置包"""
        d = self.heartflow_decision_block()
        return {
            "calm_slot_ttl": float(d.get("calm_slot_ttl", 5.0)),
            "heated_slot_ttl": float(d.get("heated_slot_ttl", 2.0)),
            "pulse_span_seconds": float(d.get("pulse_span_seconds", 60.0)),
            "heated_pulse_limit": int(d.get("heated_pulse_limit", 5)),
            "slot_capacity_cap": int(d.get("slot_capacity_cap", 120)),
            "dimension_weights": d.get("dimension_weights", {}),
            "mood_factors": d.get("mood_factors", {}),
            "barrier_seeds": d.get("barrier_seeds", {}),
            "barrier_tuning": d.get("barrier_tuning", {}),
            "threat_weights": d.get("threat_weights", {}),
            "fusion": d.get("fusion", {}),
            "immediacy": d.get("immediacy", {}),
            "credibility": d.get("credibility", {}),
            "bond": d.get("bond", {}),
            "tactics": d.get("tactics", {}),
        }

    # ---- 聚合查询 ----

    def is_module_on(self, name: str) -> bool:
        """查询指定模块是否启用"""
        switches = self._sections.get("module_switches")
        if not isinstance(switches, dict):
            return True
        return bool(switches.get(name, True))

    def energy_ceilings(self) -> Tuple[float, float]:
        """获取聊天精力值和思考值的上限"""
        chat_ceil = float(
            self.fetch_deep(
                "dual_pool_energy", "chat_value", "ceiling", fallback=100.0
            )
        )
        thinking_ceil = float(
            self.fetch_deep(
                "dual_pool_energy", "thinking_power", "ceiling", fallback=100.0
            )
        )
        return (chat_ceil, thinking_ceil)

    def wait_timing_triple(self) -> Tuple[int, int, int]:
        """等待阶段三元组: (最大等待秒, 反思间隔, 最大反思轮数)"""
        t = self.heartflow_timing()
        return (
            int(t.get("max_idle_wait_seconds", 120)),
            int(t.get("reflect_interval_seconds", 30)),
            int(t.get("max_reflect_rounds", 3)),
        )

    def rest_stage_bundle(self) -> Dict[str, Any]:
        """休息阶段参数包"""
        t = self.heartflow_timing()
        return {
            "default_duration": int(t.get("rest_default_seconds", 300)),
            "peek_chance": float(t.get("peek_chance_ratio", 0.05)),
            "max_chase": int(t.get("max_chase_count", 2)),
        }

    def chat_pool_params(self) -> Dict[str, float]:
        """聊天值子段"""
        sub = self.fetch_deep("dual_pool_energy", "chat_value", fallback={})
        if not isinstance(sub, dict):
            sub = {}
        return {
            "ceiling": float(sub.get("ceiling", 100.0)),
            "recovery_rate_per_min": float(
                sub.get("recovery_rate_per_min", 5.0)
            ),
            "cost_per_reply": float(sub.get("cost_per_reply", 15.0)),
            "cost_per_peek": float(sub.get("cost_per_peek", 5.0)),
        }

    def thinking_pool_params(self) -> Dict[str, float]:
        """思考值子段"""
        sub = self.fetch_deep(
            "dual_pool_energy", "thinking_power", fallback={}
        )
        if not isinstance(sub, dict):
            sub = {}
        return {
            "ceiling": float(sub.get("ceiling", 100.0)),
            "cost_per_think": float(sub.get("cost_per_think", 3.0)),
            "cost_per_reply": float(sub.get("cost_per_reply", 8.0)),
            "recovery_rate_per_sec": float(
                sub.get("recovery_rate_per_sec", 0.005)
            ),
        }

    def decay_params(self) -> Dict[str, float]:
        """衰减子段"""
        sub = self.fetch_deep("dual_pool_energy", "decay", fallback={})
        if not isinstance(sub, dict):
            sub = {}
        return {
            "annoyance_decay_rate": float(
                sub.get("annoyance_decay_rate", 3.0)
            ),
            "min_recovery_gap_seconds": float(
                sub.get("min_recovery_gap_seconds", 6.0)
            ),
        }

    def penalty_caps(self) -> Dict[str, float]:
        """各项惩罚上限"""
        sec = self.penalty_caps_block()
        return {
            "trauma_cap": float(sec.get("trauma_cap", 0.6)),
            "annoyance_cap": float(sec.get("annoyance_cap", 0.5)),
            "frequency_bonus_cap": float(sec.get("frequency_bonus_cap", 0.3)),
        }

    def proactive_cooldown_sec(self) -> float:
        """主动回复最小冷却秒数"""
        return float(
            self.fetch_item("proactive_schedule", "min_cooldown_seconds", 60.0)
        )

    def think_concurrency_limit(self) -> int:
        """主动思考最大并发"""
        return int(
            self.fetch_item("proactive_schedule", "max_concurrent_thinks", 3)
        )

    def silence_idle_sec(self) -> float:
        """沉默检测触发秒数"""
        return float(
            self.fetch_item("silence_detection", "idle_trigger_seconds", 180.0)
        )

    def emotion_annoy_count(self) -> int:
        """触发烦恼的重复次数"""
        return int(
            self.fetch_item("emotion_stream", "annoyed_repeat_count", 3)
        )

    def voice_action_limit(self) -> int:
        """内心独白最多输出动作数"""
        return int(self.fetch_item("inner_voice", "max_output_actions", 5))

    def voice_desire_range(self) -> Tuple[int, int]:
        """回复欲望值范围"""
        lo = int(self.fetch_item("inner_voice", "desire_scale_min", 1))
        hi = int(self.fetch_item("inner_voice", "desire_scale_max", 10))
        return (lo, hi)

    def energy_overflow_ratio(self) -> float:
        """能量溢出触发比例"""
        return float(
            self.fetch_item(
                "trigger_thresholds", "energy_overflow_ratio", 0.85
            )
        )

    def social_warmth_floor(self) -> float:
        """社交温度下限"""
        return float(
            self.fetch_item("trigger_thresholds", "social_warmth_floor", 0.3)
        )

    def plan_trigger_gap(self) -> float:
        """规划触发最小间隔"""
        return float(
            self.fetch_item(
                "trigger_thresholds", "plan_trigger_interval", 600.0
            )
        )

    def curiosity_threshold(self) -> float:
        """好奇心爆发阈值"""
        return float(
            self.fetch_item(
                "trigger_thresholds", "curiosity_spike_threshold", 0.7
            )
        )

    # ---- 人格滑块 ----

    def slider(self, name: str, fallback: float = 0.5) -> float:
        """读取单个人格滑块，限制在0~1"""
        block = self._sections.get("personality_sliders")
        if not isinstance(block, dict):
            return fallback
        raw = block.get(name)
        if raw is None:
            return fallback
        return max(0.0, min(1.0, float(raw)))

    def compute_slider_factors(self) -> Dict[str, float]:
        """将人格滑块映射为各子系统的运行时乘数，结果缓存"""
        if not self._slider_stale and self._slider_cache:
            return copy.deepcopy(self._slider_cache)
        s = self.slider("sensitivity", 0.6)
        t = self.slider("tolerance", 0.4)
        r = self.slider("reactiveness", 0.5)
        rc = self.slider("recovery_speed", 0.5)
        w = self.slider("social_warmth", 0.6)
        c = self.slider("curiosity_drive", 0.6)
        e = self.slider("reply_eagerness", 0.5)
        result = {
            "emotion_volatility": 0.5 + s * 0.8,
            "negativity_buffer": 0.4 + t * 0.7,
            "action_speed": 0.6 + r * 0.6,
            "recovery_mod": 0.5 + rc * 0.8,
            "affinity_boost": 0.5 + w * 0.7,
            "curiosity_impulse": 0.4 + c * 0.8,
            "reply_inclination": 0.4 + e * 0.7,
        }
        self._slider_cache = result
        self._slider_stale = False
        return copy.deepcopy(result)

    def apply_slider(self, base: float, factor_key: str) -> float:
        """对基础值应用指定的人格因子乘数"""
        factors = self.compute_slider_factors()
        return base * factors.get(factor_key, 1.0)

    # ---- 跨段落组合 ----

    def assemble_wait_config(self) -> Dict[str, Any]:
        """组装等待阶段的完整配置包"""
        t = self.heartflow_timing()
        return {
            "max_wait_sec": int(t.get("max_idle_wait_seconds", 120)),
            "reflect_interval": int(t.get("reflect_interval_seconds", 30)),
            "max_reflect": int(t.get("max_reflect_rounds", 3)),
            "max_chase": int(t.get("max_chase_count", 2)),
        }

    def assemble_rest_config(self) -> Dict[str, Any]:
        """组装休息阶段的完整配置包"""
        t = self.heartflow_timing()
        return {
            "duration_sec": int(t.get("rest_default_seconds", 300)),
            "peek_chance": float(t.get("peek_chance_ratio", 0.05)),
            "cooldown_ms": int(t.get("transition_cooldown_ms", 500)),
        }

    def assemble_scheduler_config(self) -> Dict[str, Any]:
        """组装主动回复调度器的完整配置包"""
        p = self.proactive_schedule()
        return {
            "cooldown_sec": float(p.get("min_cooldown_seconds", 60.0)),
            "concurrency": int(p.get("max_concurrent_thinks", 3)),
            "timeout_sec": float(p.get("think_timeout_seconds", 30.0)),
            "max_retry": int(p.get("retry_max_count", 2)),
            "queue_size": int(p.get("queue_capacity", 20)),
            "mention_boost": float(p.get("priority_boost_on_mention", 2.0)),
        }

    def assemble_silence_config(self) -> Dict[str, Any]:
        """组装沉默监测的完整配置包"""
        s = self.silence_detection_block()
        return {
            "idle_sec": float(s.get("idle_trigger_seconds", 180.0)),
            "topic_cooldown": float(s.get("topic_launch_cooldown", 300.0)),
            "hourly_limit": int(s.get("frequency_limit_per_hour", 5)),
            "min_users": int(s.get("min_active_users", 0)),
        }

    def assemble_emotion_config(self) -> Dict[str, Any]:
        """组装情绪流的完整配置包"""
        e = self.emotion_stream_block()
        return {
            "window_sec": float(e.get("query_window_seconds", 300.0)),
            "max_queries": int(e.get("max_queries_per_window", 50)),
            "repeat_bar": int(e.get("repeat_detection_threshold", 3)),
            "refresh_sec": float(e.get("emotion_refresh_interval", 60.0)),
            "recent_window": float(e.get("recent_query_window_seconds", 60.0)),
            "annoy_count": int(e.get("annoyed_repeat_count", 3)),
            "social_scale": float(e.get("social_delta_scale", 0.02)),
        }

    def assemble_trigger_config(self) -> Dict[str, Any]:
        """组装触发系统的完整配置包"""
        t = self.trigger_thresholds_block()
        return {
            "overflow_ratio": float(t.get("energy_overflow_ratio", 0.85)),
            "warmth_floor": float(t.get("social_warmth_floor", 0.3)),
            "plan_gap_sec": float(t.get("plan_trigger_interval", 600.0)),
            "curiosity_bar": float(t.get("curiosity_spike_threshold", 0.7)),
        }

    def assemble_voice_config(self) -> Dict[str, Any]:
        """组装内心独白的完整配置包"""
        v = self.inner_voice_block()
        return {
            "action_cap": int(v.get("max_output_actions", 5)),
            "desire_lo": int(v.get("desire_scale_min", 1)),
            "desire_hi": int(v.get("desire_scale_max", 10)),
            "fallback": str(v.get("fallback_action", "observe")),
        }

    # ---- 工具 ----

    def frequency_control_block(self) -> Dict[str, Any]:
        """获取 [frequency_control] 段原始字典"""
        blk = self._sections.get("frequency_control")
        return dict(blk) if isinstance(blk, dict) else {}

    def assemble_frequency_config(self) -> Dict[str, Any]:
        """组装频率控制的完整配置包"""
        f = self.frequency_control_block()
        return {
            "baseline": float(f.get("baseline_multiplier", 1.0)),
            "floor": float(f.get("multiplier_floor", 0.1)),
            "ceiling": float(f.get("multiplier_ceiling", 5.0)),
            "decay_hl": float(f.get("decay_half_life_seconds", 300.0)),
            "long_msg_chars": int(f.get("adapt_long_msg_chars", 80)),
            "long_msg_delta": float(f.get("adapt_long_msg_delta", 0.05)),
            "short_msg_chars": int(f.get("adapt_short_msg_chars", 5)),
            "short_msg_delta": float(f.get("adapt_short_msg_delta", 0.03)),
            "fast_iv_sec": float(f.get("adapt_fast_interval_sec", 5.0)),
            "fast_iv_delta": float(f.get("adapt_fast_interval_delta", 0.08)),
            "slow_iv_sec": float(f.get("adapt_slow_interval_sec", 60.0)),
            "slow_iv_delta": float(f.get("adapt_slow_interval_delta", 0.05)),
            "silent_decay": float(f.get("silent_streak_decay", 0.9)),
            "tier_high": int(f.get("silent_streak_tier_high", 5)),
            "thr_high": int(f.get("silent_streak_threshold_high", 3)),
            "tier_mid": int(f.get("silent_streak_tier_mid", 3)),
            "thr_mid": int(f.get("silent_streak_threshold_mid", 2)),
            "thr_default": int(f.get("silent_streak_threshold_default", 1)),
            "engage_base_iv": float(f.get("engage_base_interval", 2.0)),
            "engage_min_iv": float(f.get("engage_min_interval", 1.0)),
            "engage_max_iv": float(f.get("engage_max_interval", 30.0)),
            "engage_stretch_n": int(f.get("engage_stretch_after_n", 5)),
            "engage_stretch_r": float(f.get("engage_stretch_ratio", 1.5)),
            "engage_shrink_skips": int(f.get("engage_shrink_after_skips", 3)),
            "engage_shrink_r": float(f.get("engage_shrink_ratio", 0.8)),
            "cooldown_sec": float(f.get("cooldown_duration", 60.0)),
            "stale_idle": float(f.get("stale_idle_seconds", 1800.0)),
            "win_max_req": int(f.get("window_max_requests", 10)),
            "win_sec": float(f.get("window_seconds", 60.0)),
            "burst_cap": int(f.get("burst_limit", 5)),
            "burst_cd": float(f.get("burst_cooldown_seconds", 10.0)),
        }

    def dump(self) -> Dict[str, Any]:
        """导出全部配置深拷贝"""
        return copy.deepcopy(self._sections)

    def block_names(self) -> list:
        """列出所有段落名"""
        return list(self._sections.keys())

    def summary(self) -> str:
        """配置概要"""
        n = len(self._sections)
        return f"CoreSettingsHub: {n}段, 路径={self._toml_path}"

    @staticmethod
    def _overlay_dicts(
        base: Dict[str, Any], patch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """递归合并两层字典，patch覆盖base"""
        merged = copy.deepcopy(base)
        for k, v in patch.items():
            if (
                k in merged
                and isinstance(merged[k], dict)
                and isinstance(v, dict)
            ):
                merged[k] = CoreSettingsHub._overlay_dicts(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged


def get_core_config() -> CoreSettingsHub:
    """全局快捷函数，获取核心配置中枢"""
    return CoreSettingsHub.hub()


def boot_core_config(
    config_dir: Optional[Path] = None, template_dir: Optional[Path] = None
) -> CoreSettingsHub:
    """全局快捷函数，启动核心配置中枢"""
    return CoreSettingsHub.boot(
        config_dir=config_dir, template_dir=template_dir
    )


def acquire_settings_hub() -> CoreSettingsHub:
    """兼容旧入口，获取或启动核心配置中枢。"""
    if CoreSettingsHub._sole_ref is not None:
        return CoreSettingsHub._sole_ref
    return CoreSettingsHub.boot()
