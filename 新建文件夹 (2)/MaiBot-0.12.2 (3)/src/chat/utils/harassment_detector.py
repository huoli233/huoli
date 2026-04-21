import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from src.common.logger import get_logger as get_styled_logger

logger = get_styled_logger("harassment_detector")

@dataclass
class HarassmentRecord:
    """骚扰记录"""
    user_id: str
    channel_id: str
    message_text: str
    message_hash: str
    timestamp: float
    intensity: float  # 骚扰强度 0-10


class HarassmentDetector:
    """重复骚扰检测器"""
    
    def __init__(self, 
                 time_window: int = 60,
                 max_records: int = 50,
                 min_similarity: float = 0.85,
                 harassment_threshold: int = 5,
                 intensity_threshold: float = 25.0
                 ):
        """
        Args:
            time_window: 检测时间窗口（秒）
            max_records: 每个用户最大记录数
            min_similarity: 消息相似度阈值
            harassment_threshold: 触发阈值（多少次相同消息）
            intensity_threshold: 累计强度阈值
        """
        self._time_window = time_window
        self._max_records = max_records
        self._min_similarity = min_similarity
        self._harassment_threshold = harassment_threshold
        self._intensity_threshold = intensity_threshold
        
        self._harassment_records: Dict[str, deque[HarassmentRecord]] = {}
        
        self._harassment_words = {
            "润": 1.0,
            "性骚扰": 6.0,
            "物化": 5.0,
            "冒犯": 3.0,
            "羞辱": 4.0,
            "支配感": 3.0,
            "低俗": 3.0,
            "亵渎": 4.0,
            "病态": 4.0,
            "骚扰": 5.0
        }
    
    def _get_key(self, user_id: str, channel_id: str) -> str:
        """生成用户-频道唯一键"""
        return f"{user_id}_{channel_id}"
    
    def _calculate_message_intensity(self, message_text: str) -> float:
        """计算消息的骚扰强度"""
        intensity = 0.0
        message_lower = message_text.lower()
        
        words = message_lower.split()
        unique_words = set(words)
        if len(unique_words) < len(words) / 2 and len(words) > 3:
            intensity += 1.5
        
        for word, score in self._harassment_words.items():
            if word in message_lower:
                intensity += score
        
        return min(intensity, 10.0)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简单的基于哈希的方法）"""
        if text1 == text2:
            return 1.0
        
        # 基于字符频率的相似度
        def get_char_freq(text):
            freq = {}
            for c in text:
                freq[c] = freq.get(c, 0) + 1
            return freq
        
        freq1 = get_char_freq(text1)
        freq2 = get_char_freq(text2)
        
        # 计算交集大小
        intersection = sum(min(freq1.get(c, 0), freq2.get(c, 0)) for c in set(freq1.keys()) & set(freq2.keys()))
        union = sum(max(freq1.get(c, 0), freq2.get(c, 0)) for c in set(freq1.keys()) | set(freq2.keys()))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_message_hash(self, message_text: str) -> str:
        """生成消息哈希"""
        return hashlib.md5(message_text.lower().encode()).hexdigest()
    
    def add_record(self, user_id: str, channel_id: str, message_text: str) -> None:
        """添加一条骚扰记录"""
        key = self._get_key(user_id, channel_id)
        
        if key not in self._harassment_records:
            self._harassment_records[key] = deque(maxlen=self._max_records)
        
        # 清理过期记录
        current_time = time.time()
        self._clean_expired_records(key, current_time)
        
        # 创建记录
        message_hash = self._generate_message_hash(message_text)
        intensity = self._calculate_message_intensity(message_text)
        record = HarassmentRecord(
            user_id=user_id,
            channel_id=channel_id,
            message_text=message_text,
            message_hash=message_hash,
            timestamp=current_time,
            intensity=intensity
        )
        
        self._harassment_records[key].append(record)
    
    def _clean_expired_records(self, key: str, current_time: float) -> None:
        """清理过期记录"""
        if key not in self._harassment_records:
            return
        
        # 过滤掉过期的记录
        self._harassment_records[key] = deque(
            [record for record in self._harassment_records[key] 
             if current_time - record.timestamp < self._time_window],
            maxlen=self._max_records
        )
    
    def detect_harassment(self, user_id: str, channel_id: str, message_text: str) -> Tuple[bool, Dict[str, any]]:
        """
        检测是否为重复骚扰
        
        Args:
            user_id: 用户ID
            channel_id: 频道ID
            message_text: 消息文本
        
        Returns:
            (是否为骚扰, 骚扰详情)
        """
        key = self._get_key(user_id, channel_id)
        current_time = time.time()
        
        # 清理过期记录
        self._clean_expired_records(key, current_time)
        
        if key not in self._harassment_records or len(self._harassment_records[key]) == 0:
            # 没有历史记录，添加当前记录并返回
            self.add_record(user_id, channel_id, message_text)
            return False, {}
        
        records = self._harassment_records[key]
        message_hash = self._generate_message_hash(message_text)
        current_intensity = self._calculate_message_intensity(message_text)
        
        # 计算相同消息的数量
        same_hash_count = 0
        similar_count = 0
        total_intensity = current_intensity
        
        for record in records:
            total_intensity += record.intensity
            
            if record.message_hash == message_hash:
                same_hash_count += 1
            elif self._calculate_similarity(record.message_text, message_text) >= self._min_similarity:
                similar_count += 1
        
        # 计算总相似消息数
        total_similar = same_hash_count + similar_count
        
        # 添加当前记录
        self.add_record(user_id, channel_id, message_text)
        
        # 检测是否触发骚扰条件
        is_harassment = False
        harassment_type = ""
        
        if total_similar >= self._harassment_threshold or total_intensity >= self._intensity_threshold:
            is_harassment = True
            
            if same_hash_count >= self._harassment_threshold:
                harassment_type = "exact_repeat"
            elif similar_count >= self._harassment_threshold:
                harassment_type = "similar_repeat"
            elif total_intensity >= self._intensity_threshold:
                harassment_type = "high_intensity"
        
        if is_harassment:
            logger.info(f"[骚扰检测] 用户{user_id[:8]}...在频道{channel_id}发送骚扰消息，类型:{harassment_type}，相同消息数:{same_hash_count}，相似消息数:{similar_count}，累计强度:{total_intensity:.1f}")
            
            return True, {
                "harassment_type": harassment_type,
                "same_hash_count": same_hash_count,
                "similar_count": similar_count,
                "total_similar": total_similar,
                "current_intensity": current_intensity,
                "total_intensity": total_intensity,
                "time_window": self._time_window,
                "threshold": self._harassment_threshold,
                "intensity_threshold": self._intensity_threshold,
                "message_text": message_text[:50] + "..." if len(message_text) > 50 else message_text
            }
        
        return False, {}
    
    def reset_user_records(self, user_id: str, channel_id: str) -> None:
        """重置用户的骚扰记录"""
        key = self._get_key(user_id, channel_id)
        if key in self._harassment_records:
            del self._harassment_records[key]
    
    def get_user_stats(self, user_id: str, channel_id: str) -> Dict[str, any]:
        """获取用户的骚扰统计信息"""
        key = self._get_key(user_id, channel_id)
        current_time = time.time()
        
        # 清理过期记录
        self._clean_expired_records(key, current_time)
        
        if key not in self._harassment_records:
            return {
                "record_count": 0,
                "total_intensity": 0.0,
                "recent_messages": []
            }
        
        records = self._harassment_records[key]
        total_intensity = sum(record.intensity for record in records)
        recent_messages = [{
            "text": record.message_text[:30] + "..." if len(record.message_text) > 30 else record.message_text,
            "timestamp": record.timestamp,
            "intensity": record.intensity
        } for record in sorted(records, key=lambda r: r.timestamp, reverse=True)[:5]]
        
        return {
            "record_count": len(records),
            "total_intensity": total_intensity,
            "recent_messages": recent_messages
        }


# 全局单例
def get_harassment_detector() -> HarassmentDetector:
    """获取全局的骚扰检测器单例"""
    if not hasattr(get_harassment_detector, "_instance"):
        get_harassment_detector._instance = HarassmentDetector()
    return get_harassment_detector._instance
