import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import Counter
from src.common.logger import get_logger

logger = get_logger("sticker_review")


@dataclass
class StickerReviewResult:
    is_real_person: bool
    real_person_confidence: float
    is_compliant: bool
    compliance_confidence: float
    content_type: str
    usage_scenarios: List[str]
    should_accept: bool
    reason: str


class StickerContentReviewer:
    def __init__(self):
        self._review_cache: Dict[str, StickerReviewResult] = {}
        self._usage_feedback: Dict[str, List[Dict[str, Any]]] = {}

    async def review_sticker(self, image_path: str, content_hash: str) -> StickerReviewResult:
        if content_hash in self._review_cache:
            logger.debug(f"[审查缓存] 命中: {content_hash}")
            return self._review_cache[content_hash]
        try:
            analysis_result = await self._analyze_image(image_path)
            if not analysis_result:
                logger.warning(f"[图片分析] 失败: {image_path}")
                return self._create_reject_result("图片分析失败")
            review_prompt = self._build_review_prompt(analysis_result)
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="sticker_review")
            llm_response, _ = await request.generate_response_async(review_prompt, max_tokens=500)
            if not llm_response:
                logger.warning(f"[LLM审查] 无响应")
                return self._create_reject_result("LLM审查无响应")
            result = self._parse_review_response(llm_response)
            self._review_cache[content_hash] = result
            logger.info(
                f"[审查完成] 真人: {result.is_real_person}({result.real_person_confidence:.2f}), "
                f"合规: {result.is_compliant}({result.compliance_confidence:.2f}), "
                f"类型: {result.content_type}, 接受: {result.should_accept}"
            )
            return result
        except Exception as e:
            logger.error(f"[审查异常] {e}")
            return self._create_reject_result(f"审查异常: {e}")

    async def _analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            from src.chat.utils.utils_image import analyze_image_file
            result = await analyze_image_file(image_path)
            return result
        except Exception as e:
            logger.debug(f"[图片分析] 调用失败: {e}")
            return {"description": "无法分析", "objects": []}

    def _build_review_prompt(self, analysis_result: Dict[str, Any]) -> str:
        image_desc = analysis_result.get("description", "")
        detected_objects = analysis_result.get("objects", [])
        objects_str = ', '.join(detected_objects) if detected_objects else '无'
        
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.SYSTEM,
            "sticker_review",
            "review_sticker.template",
            image_desc=image_desc,
            detected_objects=objects_str
        )
        return prompt

    def _parse_review_response(self, llm_response: str) -> StickerReviewResult:
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("无法提取JSON")
            data = json.loads(json_str)
            return StickerReviewResult(
                is_real_person=data.get("is_real_person", False),
                real_person_confidence=float(data.get("real_person_confidence", 0.0)),
                is_compliant=data.get("is_compliant", True),
                compliance_confidence=float(data.get("compliance_confidence", 1.0)),
                content_type=data.get("content_type", "未知"),
                usage_scenarios=data.get("usage_scenarios", []),
                should_accept=data.get("should_accept", False),
                reason=data.get("reason", "")
            )
        except Exception as e:
            logger.error(f"[解析审查结果] 异常: {e}")
            return self._create_reject_result(f"解析失败: {e}")

    def _create_reject_result(self, reason: str) -> StickerReviewResult:
        return StickerReviewResult(
            is_real_person=False,
            real_person_confidence=0.0,
            is_compliant=False,
            compliance_confidence=0.0,
            content_type="未知",
            usage_scenarios=[],
            should_accept=False,
            reason=reason
        )

    async def record_usage_feedback(self, content_hash: str, scene_type: str, user_reaction: str):
        if content_hash not in self._usage_feedback:
            self._usage_feedback[content_hash] = []
        self._usage_feedback[content_hash].append({
            "scene_type": scene_type,
            "user_reaction": user_reaction,
            "timestamp": time.time()
        })
        logger.debug(f"[使用反馈] {content_hash}: {scene_type} -> {user_reaction}")

    async def get_dynamic_scenarios(self, content_hash: str) -> List[str]:
        if content_hash not in self._usage_feedback:
            return []
        feedback_list = self._usage_feedback[content_hash]
        positive_scenes = [fb["scene_type"] for fb in feedback_list if fb["user_reaction"] == "positive"]
        scene_counts = Counter(positive_scenes)
        return [scene for scene, _ in scene_counts.most_common(5)]


_reviewer_instance: Optional[StickerContentReviewer] = None


def get_sticker_reviewer() -> StickerContentReviewer:
    global _reviewer_instance
    if _reviewer_instance is None:
        _reviewer_instance = StickerContentReviewer()
    return _reviewer_instance
