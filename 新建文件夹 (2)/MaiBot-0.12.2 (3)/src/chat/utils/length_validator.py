import re
from dataclasses import dataclass
from typing import Optional
from src.common.logger import get_logger

logger = get_logger("length_validator")


@dataclass
class LengthCheckResult:
    is_valid: bool
    final_text: str
    original_length: int
    final_length: int
    text_only_length: int


class UnifiedLengthChecker:
    def __init__(self):
        try:
            from src.config.config import global_config
            self.max_length = getattr(global_config, 'max_response_length', 200)
        except Exception:
            self.max_length = 200
        logger.info(f"[长度检查器] 初始化完成，max_length={self.max_length}")

    def check_and_adjust(
        self,
        text: str,
        stage: str = "unknown",
        allow_truncate: bool = True
    ) -> LengthCheckResult:
        original_length = len(text)
        text_only_length = self._count_text_only(text)
        logger.debug(
            f"[长度检查-{stage}] 文本长度统计：总字符={original_length}，纯文字={text_only_length}"
        )
        return LengthCheckResult(
            is_valid=True,
            final_text=text,
            original_length=original_length,
            final_length=original_length,
            text_only_length=text_only_length
        )

    def _count_text_only(self, text: str) -> int:
        punctuation_pattern = r'[，。！？、；：\u201c\u201d\u2018\u2019（）《》【】…—·,.!?;:\'\"(){}<>@#$%^&*\-_+=|/~`\s]'
        text_only = re.sub(punctuation_pattern, '', text)
        return len(text_only)


_length_checker_instance: Optional[UnifiedLengthChecker] = None


def get_length_checker() -> UnifiedLengthChecker:
    global _length_checker_instance
    if _length_checker_instance is None:
        _length_checker_instance = UnifiedLengthChecker()
    return _length_checker_instance
