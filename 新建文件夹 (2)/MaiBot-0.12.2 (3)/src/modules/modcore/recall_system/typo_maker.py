import random
import time
from typing import Optional, List, Dict, Tuple
from src.common.logger import get_logger

logger = get_logger("recall")


class TypoMaker:
    """
    打错字生成器 - 学习型版本
    从记忆库学习流行的错字模式，自适应跟上潮流
    """
    # 基础键盘布局（作为后备）
    ADJACENT_KEYS = {
        'a': ['s', 'q', 'w', 'z'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 'r', 'd', 's'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'p', 'l', 'k'], 'p': ['o', 'l'],
        'q': ['w', 'a'], 'r': ['e', 't', 'f', 'd'], 's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'], 'u': ['y', 'i', 'j', 'h'], 'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'], 'x': ['z', 's', 'd', 'c'], 'y': ['t', 'u', 'h', 'g'], 'z': ['a', 's', 'x'],
    }
    ERROR_MODES = ['adjacent', 'swap', 'omit', 'duplicate', 'homophone', 'pinyin_mix', 'learned']

    def __init__(self):
        self._llm_bridge = None
        self._learned_patterns: Dict[str, List[str]] = {}  # 学习到的错字模式
        self._last_learn_time: float = 0.0
        self._learn_interval: float = 3600.0  # 每小时学习一次
        self._pattern_decay_time: float = 86400.0 * 7  # 7天后开始遗忘
        self._pattern_timestamps: Dict[str, float] = {}  # 记录模式的时间戳

    def _get_llm(self):
        if self._llm_bridge is None:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                self._llm_bridge = (LLMRequest, model_config)
            except Exception:
                pass
        return self._llm_bridge

    async def learn_from_memory(self, stream_id: str = "") -> int:
        """
        从记忆库学习流行的错字模式
        返回学习到的新模式数量
        """
        now = time.time()
        if now - self._last_learn_time < self._learn_interval:
            return 0
        
        try:
            from src.memory_system.memory_core import get_memory_manager
            mgr = get_memory_manager()
            
            # 获取最近的群聊记忆（最近7天）
            recent_time = now - 86400 * 7
            memories = mgr.search_memories(
                stream_id=stream_id,
                memory_type="group_chat",
                time_range=(recent_time, now),
                limit=200
            )
            
            if not memories:
                logger.debug("没有找到可学习的记忆")
                return 0
            
            # 提取文本内容
            texts = []
            for mem in memories:
                content = getattr(mem, 'content', '') or ''
                if content and len(content) > 2:
                    texts.append(content)
            
            if not texts:
                return 0
            
            # 使用小模型总结流行的错字模式
            llm = self._get_llm()
            if not llm:
                return 0
            
            LLMRequest, mc = llm
            sample_texts = texts[:50]  # 只取前50条
            combined = "\n".join(sample_texts[:30])  # 限制长度
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "typo_maker",
                "typo_learning.template",
                combined=combined
            )
            
            try:
                request = LLMRequest(mc.focus_chat, request_type="typo_learning")
                response_text, _ = await request.generate_response_async(prompt, max_tokens=500)
                
                if not response_text:
                    return 0
                
                # 解析JSON
                import json
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    return 0
                
                data = json.loads(json_match.group(0))
                patterns = data.get('patterns', [])
                
                learned_count = 0
                for p in patterns:
                    correct = p.get('correct', '')
                    typo = p.get('typo', '')
                    if correct and typo and correct != typo:
                        if correct not in self._learned_patterns:
                            self._learned_patterns[correct] = []
                        if typo not in self._learned_patterns[correct]:
                            self._learned_patterns[correct].append(typo)
                            self._pattern_timestamps[f"{correct}->{typo}"] = now
                            learned_count += 1
                            logger.info(f"学习到新错字模式: {correct} -> {typo}")
                
                self._last_learn_time = now
                self._forget_old_patterns(now)
                
                logger.info(f"从记忆库学习完成: 新增{learned_count}个模式, 总计{len(self._learned_patterns)}个")
                return learned_count
                
            except Exception as e:
                logger.debug(f"学习错字模式失败: {e}")
                return 0
                
        except Exception as e:
            logger.debug(f"从记忆库学习失败: {e}")
            return 0

    def _forget_old_patterns(self, now: float):
        """遗忘旧的模式"""
        to_remove = []
        for key, timestamp in self._pattern_timestamps.items():
            if now - timestamp > self._pattern_decay_time:
                to_remove.append(key)
        
        for key in to_remove:
            parts = key.split('->')
            if len(parts) == 2:
                correct, typo = parts
                if correct in self._learned_patterns:
                    if typo in self._learned_patterns[correct]:
                        self._learned_patterns[correct].remove(typo)
                    if not self._learned_patterns[correct]:
                        del self._learned_patterns[correct]
            del self._pattern_timestamps[key]
            logger.debug(f"遗忘旧模式: {key}")

    def _apply_learned_pattern(self, text: str) -> Optional[str]:
        """应用学习到的错字模式"""
        if not self._learned_patterns:
            return None
        
        # 随机选择一个学习到的模式
        available = [(k, v) for k, v in self._learned_patterns.items() if k in text]
        if not available:
            return None
        
        correct, typos = random.choice(available)
        typo = random.choice(typos)
        
        # 只替换一次
        return text.replace(correct, typo, 1)

    async def create_typo_with_llm(self, text: str, error_type: str = "random", stream_id: str = "") -> Tuple[str, str]:
        """
        使用LLM生成错字，优先使用学习到的模式
        """
        # 先尝试触发学习（如果到时间了）
        if stream_id:
            await self.learn_from_memory(stream_id)
        
        # 30%概率使用学习到的模式
        if self._learned_patterns and random.random() < 0.3:
            learned_result = self._apply_learned_pattern(text)
            if learned_result:
                logger.debug(f"使用学习模式: {text[:20]} -> {learned_result[:20]}")
                return learned_result, "learned"
        
        llm = self._get_llm()
        if llm is None:
            return self.create_typo_local(text), "local_fallback"
        
        error_instructions = {
            "typo": "故意打错1-2个字，比如把'你好'打成'你号'或'尼好'，像真人打字时手滑一样",
            "swap": "故意交换1-2个字的顺序，比如把'今天'打成'天今'，像打字太快手忙脚乱",
            "omit": "故意漏掉1-2个字，比如把'我今天很开心'打成'我今天开心'",
            "duplicate": "故意重复打1-2个字，比如把'好的'打成'好好的'或'好的的'",
            "pinyin": "故意用拼音同音错字，比如把'在'打成'再'，把'的'打成'得'",
            "random": "随机选择一种错误方式（打错字/交换/漏字/重复/同音错字），故意制造1-2处错误",
        }
        instruction = error_instructions.get(error_type, error_instructions["random"])
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.MODULE,
            "typo_maker",
            "typo_maker.template",
            instruction=instruction,
            text=text
        )
        try:
            LLMRequest, mc = llm
            request = LLMRequest(mc.focus_chat, request_type="typo_maker")
            response_text, _ = await request.generate_response_async(prompt, max_tokens=len(text) + 20)
            if response_text and len(response_text) > 0 and response_text != text:
                logger.debug(f"LLM生成错字: {text[:20]} -> {response_text[:20]}")
                return response_text.strip(), "llm"
        except Exception as e:
            logger.debug(f"LLM生成错字失败: {e}")
        return self.create_typo_local(text), "local_fallback"

    def create_typo_local(self, text: str, intensity: int = 1) -> str:
        """本地错字生成（后备方案）"""
        if not text or len(text) < 2:
            return text
        
        # 优先使用学习到的模式
        if self._learned_patterns and random.random() < 0.5:
            learned_result = self._apply_learned_pattern(text)
            if learned_result:
                return learned_result
        
        chars = list(text)
        positions = self._pick_positions(len(chars), intensity)
        for pos in positions:
            # 只使用基础的错误模式，不使用硬编码的同音字
            mode = random.choice(['adjacent', 'swap', 'omit', 'duplicate'])
            chars = self._apply_error(chars, pos, mode)
        result = ''.join(chars)
        if result == text:
            result = self._force_single_error(text)
        return result

    def create_typo(self, text: str, intensity: int = 1) -> str:
        if not text or len(text) < 2:
            return text
        chars = list(text)
        positions = self._pick_positions(len(chars), intensity)
        for pos in positions:
            mode = random.choice(self.ERROR_MODES)
            chars = self._apply_error(chars, pos, mode)
        result = ''.join(chars)
        if result == text:
            result = self._force_single_error(text)
        return result

    def _pick_positions(self, length: int, count: int) -> List[int]:
        max_errors = min(count, max(1, length // 4))
        available = list(range(length))
        return random.sample(available, min(max_errors, len(available)))

    def _apply_error(self, chars: List[str], pos: int, mode: str) -> List[str]:
        """应用错误模式（移除硬编码同音字）"""
        if pos >= len(chars):
            return chars
        char = chars[pos]
        if mode == 'adjacent':
            chars[pos] = self._adjacent_key_error(char)
        elif mode == 'swap' and pos + 1 < len(chars):
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif mode == 'omit':
            if random.random() < 0.5:
                chars[pos] = ''
        elif mode == 'duplicate':
            if random.random() < 0.4:
                chars.insert(pos, char)
        elif mode == 'learned':
            # 使用学习到的模式
            original = ''.join(chars)
            learned = self._apply_learned_pattern(original)
            if learned:
                return list(learned)
        return chars

    def _adjacent_key_error(self, char: str) -> str:
        """相邻键错误（仅英文）"""
        lower = char.lower()
        if lower in self.ADJACENT_KEYS:
            replacement = random.choice(self.ADJACENT_KEYS[lower])
            return replacement.upper() if char.isupper() else replacement
        return char

    def _force_single_error(self, text: str) -> str:
        """强制产生一个错误（移除硬编码同音字）"""
        # 优先使用学习到的模式
        if self._learned_patterns:
            learned = self._apply_learned_pattern(text)
            if learned:
                return learned
        
        chars = list(text)
        pos = random.randint(0, len(chars) - 1)
        char = chars[pos]
        
        # 只对英文和数字做简单替换
        if char.isalpha():
            new_char = self._adjacent_key_error(char)
            chars[pos] = new_char if new_char != char else ('x' if char.lower() != 'x' else 'z')
        elif char.isdigit():
            chars[pos] = str((int(char) + random.randint(1, 3)) % 10)
        else:
            # 中文字符：交换相邻字符
            if pos + 1 < len(chars):
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        return ''.join(chars)


_typo_maker: Optional[TypoMaker] = None


def get_typo_maker() -> TypoMaker:
    global _typo_maker
    if _typo_maker is None:
        _typo_maker = TypoMaker()
    return _typo_maker


def make_typo(text: str, intensity: int = 1) -> str:
    return get_typo_maker().create_typo(text, intensity)


async def make_typo_async(text: str, error_type: str = "random", stream_id: str = "") -> Tuple[str, str]:
    """异步生成错字，支持从记忆库学习"""
    return await get_typo_maker().create_typo_with_llm(text, error_type, stream_id)
