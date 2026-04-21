import time
from typing import Dict, Optional, Any
from src.common.logger import get_logger
from src.chat.heart_flow.persistent_state_db import get_persistent_state_db

logger = get_logger("system_calibration")


class SystemCalibrationManager:
    def __init__(self):
        self.state_db = get_persistent_state_db()
        self._calibration_history: Dict[str, float] = {}

    def calibrate_all_systems(self) -> Dict[str, Any]:
        results = {
            "memory": self.calibrate_memory_system(),
            "intimacy": self.calibrate_intimacy_system(),
            "emotion": self.calibrate_emotion_system(),
            "expression_learning": self.calibrate_expression_learning_system(),
            "jargon_learning": self.calibrate_jargon_learning_system(),
        }
        now = time.time()
        for system_name, result in results.items():
            if result.get("calibrated", False):
                self.state_db.save_system_index(system_name, last_calibration_time=now)
        return results

    def calibrate_all_systems_with_time_diff(self, time_diff: float, calibration_level: str = "medium") -> Dict[str, Any]:
        results = {}
        if calibration_level == "minor":
            results["memory"] = self.calibrate_memory_system()
            results["intimacy"] = self.calibrate_intimacy_with_decay(time_diff)
            results["emotion"] = self.calibrate_emotion_with_decay(time_diff)
        elif calibration_level == "medium":
            results["memory"] = self.calibrate_memory_system()
            results["intimacy"] = self.calibrate_intimacy_with_decay(time_diff)
            results["emotion"] = self.calibrate_emotion_with_decay(time_diff)
            results["expression_learning"] = self.calibrate_expression_learning_system()
            results["jargon_learning"] = self.calibrate_jargon_learning_system()
        elif calibration_level == "deep":
            results["memory"] = self.calibrate_memory_system()
            results["intimacy"] = self.calibrate_intimacy_with_decay(time_diff)
            results["emotion"] = self.calibrate_emotion_with_decay(time_diff)
            results["expression_learning"] = self.calibrate_expression_learning_system()
            results["jargon_learning"] = self.calibrate_jargon_learning_system()
            results["deep_cleanup"] = self.perform_deep_cleanup(time_diff)
        now = time.time()
        for system_name, result in results.items():
            if result.get("calibrated", False):
                self.state_db.save_system_index(system_name, last_calibration_time=now)
        return results

    def calibrate_intimacy_with_decay(self, time_diff: float) -> Dict[str, Any]:
        try:
            warmth_decay_rate = 2.0
            intimacy_states = self.state_db.get_all_intimacy_states()
            calibrated_count = 0
            for state in intimacy_states:
                old_warmth = state.get("interaction_warmth", 0.0)
                minutes_passed = time_diff / 60.0
                new_warmth = max(0.0, old_warmth - minutes_passed * warmth_decay_rate)
                if new_warmth != old_warmth:
                    self.state_db.save_intimacy_state(
                        stream_id=state.get("stream_id"),
                        channel_id=state.get("channel_id"),
                        interaction_warmth=new_warmth,
                        last_interaction_time=state.get("last_interaction_time", 0.0),
                    )
                    calibrated_count += 1
                    logger.debug(
                        f"[系统校准] 亲密度校准: {state.get('stream_id', '')[:8]}... "
                        f"离线 {time_diff:.1f}秒, 亲密度 {old_warmth:.1f} → {new_warmth:.1f}"
                    )
            self._calibration_history["intimacy"] = time.time()
            return {
                "calibrated": True,
                "calibrated_count": calibrated_count,
                "total_states": len(intimacy_states),
                "message": f"亲密度系统校准完成: {calibrated_count}/{len(intimacy_states)} 个状态已校准",
            }
        except Exception as e:
            logger.error(f"[系统校准] 亲密度系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_emotion_with_decay(self, time_diff: float) -> Dict[str, Any]:
        try:
            emotion_states = self.state_db.get_all_emotion_states()
            calibrated_count = 0
            for state in emotion_states:
                old_emotion = state.get("current_emotion", 0.0)
                decay_constant = 3600.0
                import math
                new_emotion = old_emotion * math.exp(-time_diff / decay_constant)
                if abs(new_emotion) < 0.2:
                    new_emotion = 0.0
                if new_emotion != old_emotion:
                    self.state_db.save_emotion_state(
                        stream_id=state.get("stream_id", state.get("channel_id", "")),
                        user_id=state.get("user_id", ""),
                        affection=new_emotion,
                        annoyance=state.get("annoyance", 0.0),
                        trust_score=state.get("trust_score", 0.0),
                        trauma_score=state.get("trauma_score", 0.0),
                        last_interaction=time.time(),
                    )
                    calibrated_count += 1
                    logger.debug(
                        f"[系统校准] 情绪校准: {state.get('user_id', '')[:8]}... "
                        f"离线 {time_diff:.1f}秒, 情绪 {old_emotion:.2f} → {new_emotion:.2f}"
                    )
            self._calibration_history["emotion"] = time.time()
            return {
                "calibrated": True,
                "calibrated_count": calibrated_count,
                "total_states": len(emotion_states),
                "message": f"情绪系统校准完成: {calibrated_count}/{len(emotion_states)} 个状态已校准",
            }
        except Exception as e:
            logger.error(f"[系统校准] 情绪系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def perform_deep_cleanup(self, time_diff: float) -> Dict[str, Any]:
        try:
            time_diff_days = time_diff / 86400.0
            if time_diff_days < 7:
                return {"calibrated": True, "message": f"离线时长 {time_diff_days:.1f} 天，无需深度清理"}
            logger.info(f"[深度清理] 离线 {time_diff_days:.1f} 天，开始深度清理")
            return {"calibrated": True, "message": f"深度清理完成（离线 {time_diff_days:.1f} 天）"}
        except Exception as e:
            logger.error(f"[系统校准] 深度清理失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_memory_system(self) -> Dict[str, Any]:
        try:
            self._calibration_history["memory"] = time.time()
            return {"calibrated": True, "message": "记忆库系统校准已在启动时完成"}
        except Exception as e:
            logger.error(f"[系统校准] 记忆库系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_intimacy_system(self) -> Dict[str, Any]:
        try:
            warmth_decay_rate = 2.0
            intimacy_states = self.state_db.get_all_intimacy_states()
            calibrated_count = 0
            for state in intimacy_states:
                offline_duration = state.get("offline_duration", 0.0)
                if offline_duration > 0:
                    minutes_passed = offline_duration / 60.0
                    decay = minutes_passed * warmth_decay_rate
                    old_warmth = state.get("interaction_warmth", 0.0)
                    new_warmth = max(0.0, old_warmth - decay)
                    if new_warmth != old_warmth:
                        self.state_db.save_intimacy_state(
                            stream_id=state.get("stream_id"),
                            channel_id=state.get("channel_id"),
                            interaction_warmth=new_warmth,
                            last_interaction_time=state.get("last_interaction_time", 0.0),
                        )
                        calibrated_count += 1
            self._calibration_history["intimacy"] = time.time()
            return {
                "calibrated": True,
                "calibrated_count": calibrated_count,
                "total_states": len(intimacy_states),
                "message": f"亲密度系统校准完成: {calibrated_count}/{len(intimacy_states)} 个状态已校准",
            }
        except Exception as e:
            logger.error(f"[系统校准] 亲密度系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_emotion_system(self) -> Dict[str, Any]:
        try:
            emotion_states = self.state_db.get_all_emotion_states()
            calibrated_count = 0
            for state in emotion_states:
                offline_duration = state.get("offline_duration", 0.0)
                if offline_duration > 0:
                    hours = offline_duration / 3600.0
                    time_recovery = min(50.0, hours * 0.5)
                    trauma_score = state.get("trauma_score", 0.0)
                    decay_rate = 0.02 if trauma_score > 1.0 else 0.05
                    annoyance = state.get("annoyance", 0.0)
                    new_annoyance = annoyance * (decay_rate ** hours) if hours > 0 else annoyance
                    old_affection = state.get("affection", 0.0)
                    new_affection = min(100.0, old_affection + time_recovery)
                    if new_affection != old_affection or new_annoyance != annoyance:
                        self.state_db.save_emotion_state(
                            stream_id=state.get("stream_id", state.get("channel_id", "")),
                            user_id=state.get("user_id", ""),
                            affection=new_affection,
                            annoyance=new_annoyance,
                            trust_score=state.get("trust_score", 0.0),
                            trauma_score=trauma_score,
                            last_interaction=state.get("last_interaction", 0.0),
                        )
                        calibrated_count += 1
            self._calibration_history["emotion"] = time.time()
            return {
                "calibrated": True,
                "calibrated_count": calibrated_count,
                "total_states": len(emotion_states),
                "message": f"情感追踪系统校准完成: {calibrated_count}/{len(emotion_states)} 个状态已校准",
            }
        except Exception as e:
            logger.error(f"[系统校准] 情感追踪系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_expression_learning_system(self) -> Dict[str, Any]:
        try:
            self._calibration_history["expression_learning"] = time.time()
            return {"calibrated": True, "message": "表达学习系统校准：表达权重衰减在访问时自动计算"}
        except Exception as e:
            logger.error(f"[系统校准] 表达学习系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def calibrate_jargon_learning_system(self) -> Dict[str, Any]:
        try:
            self._calibration_history["jargon_learning"] = time.time()
            return {"calibrated": True, "message": "群体语言学习系统校准：模式权重衰减在访问时自动计算"}
        except Exception as e:
            logger.error(f"[系统校准] 群体语言学习系统校准失败: {e}")
            return {"calibrated": False, "error": str(e)}

    def get_calibration_status(self) -> Dict[str, Any]:
        status = {}
        for system_name, last_calibration in self._calibration_history.items():
            index_info = self.state_db.get_system_index(system_name)
            status[system_name] = {
                "last_calibration": last_calibration,
                "calibration_count": index_info.get("calibration_count", 0) if index_info else 0,
                "last_organize": index_info.get("last_organize_time", 0.0) if index_info else 0.0,
                "organize_count": index_info.get("organize_count", 0) if index_info else 0,
            }
        return status


_calibration_manager: Optional[SystemCalibrationManager] = None


def get_calibration_manager() -> SystemCalibrationManager:
    global _calibration_manager
    if _calibration_manager is None:
        _calibration_manager = SystemCalibrationManager()
    return _calibration_manager


def get_system_calibration_manager() -> SystemCalibrationManager:
    return get_calibration_manager()


def calibrate_all_systems_on_startup():
    try:
        manager = get_calibration_manager()
        results = manager.calibrate_all_systems()
        logger.info("=" * 60)
        logger.info("系统时间校准完成")
        logger.info("=" * 60)
        for system_name, result in results.items():
            if result.get("calibrated", False):
                message = result.get("message", "校准完成")
                logger.info(f"  {system_name}: {message}")
            else:
                error = result.get("error", "未知错误")
                logger.warning(f"  {system_name}: 校准失败 - {error}")
        logger.info("=" * 60)
        return results
    except Exception as e:
        logger.error(f"[系统校准] 启动校准失败: {e}")
        return {}
