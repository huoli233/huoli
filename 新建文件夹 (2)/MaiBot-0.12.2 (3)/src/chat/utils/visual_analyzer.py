import time
import hashlib
import base64
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from io import BytesIO
from src.common.logger import get_logger
from src.chat.utils.repeat_query_detector import get_repeat_query_detector

logger = get_logger("visual_analyzer")

try:
    import aiohttp
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

VISION_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "vision_cache"


@dataclass
class ImageInfo:
    url: str
    local_path: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    content_type: str = ""
    hash_value: str = ""


@dataclass
class AnalysisResult:
    description: str
    keywords: List[str] = field(default_factory=list)
    detected_objects: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    analyzed_at: float = field(default_factory=time.time)
    image_hash: str = ""
    is_rejected: bool = False
    rejection_reason: str = ""
    from_cache: bool = False
    is_self: bool = False
    self_match_score: float = 0.0
    self_matched_features: List[str] = field(default_factory=list)


class ImageCache:
    def __init__(self, cache_dir: Path, max_age_days: int = 7):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_age = max_age_days * 86400
        self._index: Dict[str, str] = {}

    def get_cache_path(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self._cache_dir / f"{url_hash}.jpg"

    def has_cached(self, url: str) -> bool:
        cache_path = self.get_cache_path(url)
        if not cache_path.exists():
            return False
        age = time.time() - cache_path.stat().st_mtime
        if age > self._max_age:
            try:
                cache_path.unlink()
            except Exception:
                pass
            return False
        return True

    def get_cached_path(self, url: str) -> Optional[str]:
        if self.has_cached(url):
            return str(self.get_cache_path(url))
        return None

    def save_to_cache(self, url: str, data: bytes) -> str:
        cache_path = self.get_cache_path(url)
        try:
            with open(cache_path, "wb") as f:
                f.write(data)
            return str(cache_path)
        except Exception as e:
            logger.error(f"[visual_analyzer] 缓存保存失败: {e}")
            return ""

    def cleanup_old(self):
        try:
            current = time.time()
            for file_path in self._cache_dir.glob("*.jpg"):
                age = current - file_path.stat().st_mtime
                if age > self._max_age:
                    file_path.unlink()
        except Exception as e:
            logger.debug(f"[visual_analyzer] 清理缓存失败: {e}")


class ImageDownloader:
    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._max_size = 10 * 1024 * 1024

    async def download(self, url: str) -> Tuple[bytes, str]:
        if not HAS_DEPS:
            return b"", ""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                    if resp.status != 200:
                        logger.warning(f"[visual_analyzer] 下载失败: HTTP {resp.status}")
                        return b"", ""
                    content_type = resp.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(f"[visual_analyzer] 非图片内容: {content_type}")
                    data = await resp.read()
                    if len(data) > self._max_size:
                        logger.warning(f"[visual_analyzer] 图片过大: {len(data)} bytes")
                        return b"", ""
                    return data, content_type
        except asyncio.TimeoutError:
            logger.warning(f"[visual_analyzer] 下载超时: {url}")
            return b"", ""
        except Exception as e:
            logger.error(f"[visual_analyzer] 下载异常: {e}")
            return b"", ""


class ImageProcessor:
    def __init__(self, max_size: Tuple[int, int] = (800, 800), quality: int = 75):
        self._max_size = max_size
        self._quality = quality

    def process(self, data: bytes) -> Tuple[bytes, ImageInfo]:
        if not HAS_DEPS:
            return data, ImageInfo(url="")
        try:
            img = Image.open(BytesIO(data))
            info = ImageInfo(url="", width=img.width, height=img.height, file_size=len(data))
            if img.width > self._max_size[0] or img.height > self._max_size[1]:
                ratio = min(self._max_size[0] / img.width, self._max_size[1] / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                info.width, info.height = new_size
            if img.mode != "RGB":
                img = img.convert("RGB")
            output = BytesIO()
            img.save(output, format="JPEG", quality=self._quality)
            processed = output.getvalue()
            info.file_size = len(processed)
            info.hash_value = hashlib.md5(processed).hexdigest()[:16]
            return processed, info
        except Exception as e:
            logger.error(f"[visual_analyzer] 处理图片失败: {e}")
            return data, ImageInfo(url="")

    def to_base64(self, data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")


class VisionApiCaller:
    def __init__(self):
        self._max_tokens: int = 800
        self._temperature: float = 0.7

    def set_params(self, max_tokens: int = 800, temperature: float = 0.7):
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def analyze(self, image_base64: str, prompt: str) -> str:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            model_name = model_config.image if hasattr(model_config, 'image') else model_config.focus_chat
            request = LLMRequest(model_name, request_type="vision_analysis")
            payload_kb = len(image_base64) / 1024
            logger.info(f"[VLM] 模型: {model_name} | 图片: {payload_kb:.1f}KB")
            response_text, _ = await request.generate_response_with_image(
                prompt=prompt,
                image_base64=image_base64,
                image_format="jpeg",
                max_tokens=self._max_tokens,
            )
            if response_text:
                logger.info(f"[VLM] 识别完成: {response_text[:50]}...")
                return response_text.strip()
            return ""
        except Exception as e:
            logger.error(f"[visual_analyzer] VLM图片分析失败: {e}")
            return ""

    async def analyze_with_custom_api(self, image_base64: str, prompt: str, api_url: str, api_key: str, model: str = "gpt-4o") -> str:
        if not HAS_DEPS:
            return ""
        try:
            url = f"{api_url.rstrip('/')}/chat/completions"
            payload_kb = len(image_base64) / 1024
            logger.info(f"[VLM] 模型: {model} | 图片: {payload_kb:.1f}KB")
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": self._max_tokens,
                }
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                logger.error(f"[visual_analyzer] API错误 (尝试 {attempt+1}/{max_retries}): {resp.status} - {error_text}")
                                if resp.status in [500, 502, 503, 504]:
                                    await asyncio.sleep(2)
                                    continue
                                return ""
                            result = await resp.json()
                            content = result["choices"][0]["message"]["content"]
                            logger.info(f"[VLM] 识别完成: {content[:50]}...")
                            return content
                    except aiohttp.ClientConnectorError as e:
                        logger.warning(f"[visual_analyzer] 连接失败 (尝试 {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(2)
                    except aiohttp.ServerDisconnectedError:
                        logger.warning(f"[visual_analyzer] 服务器断开连接 (尝试 {attempt+1}/{max_retries})，正在重试...")
                        await asyncio.sleep(2)
                    except Exception as e:
                        logger.error(f"[visual_analyzer] 未知错误 (尝试 {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(2)
                return ""
        except Exception as e:
            logger.error(f"[visual_analyzer] API调用彻底失败: {e}")
            return ""


ANALYSIS_CACHE_FILE = VISION_CACHE_DIR / "analysis_cache.json"
_VIS_CACHE_DB_KEY = "visual_analysis_cache"


class VisualAnalyzer:
    def __init__(self):
        self._cache = ImageCache(VISION_CACHE_DIR)
        self._downloader = ImageDownloader()
        self._processor = ImageProcessor()
        self._api_caller = VisionApiCaller()
        self._analysis_cache: Dict[str, AnalysisResult] = {}
        self._default_prompt = self._build_character_aware_prompt()
        self._avatar_prompt = self._build_avatar_prompt()
        self._bot_qq_id = self._load_bot_qq_id()
        self._last_cleanup = time.time()
        self._memory_decay_days = 7
        self._repeat_detector = get_repeat_query_detector()
        self._load_analysis_cache()

    def _load_bot_qq_id(self) -> str:
        try:
            from src.config.config import global_config
            return str(global_config.bot.qq_account or "")
        except Exception:
            return ""

    def _is_qq_avatar_url(self, url: str) -> bool:
        return "qlogo.cn" in url or "q.qlogo.cn" in url

    def _is_own_avatar(self, url: str) -> bool:
        if not self._bot_qq_id or not self._is_qq_avatar_url(url):
            return False
        return self._bot_qq_id in url

    def _build_avatar_prompt(self) -> str:
        try:
            from src.config.config import global_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            bot_name = global_config.bot.nickname
            visual_str = getattr(global_config.personality, 'character_visual', "")
            visual_hint = f"该角色的外观特征为：{visual_str}。" if visual_str else ""
            return get_prompt(
                PromptCategory.MODULE,
                "visual_analyzer",
                "avatar_prompt.template",
                bot_name=bot_name,
                visual_hint=visual_hint
            )
        except Exception:
            return "请描述这张头像中角色的外观特征，包括发色、瞳色、服装等，50-80字"

    def _check_self_in_description(self, description: str) -> Tuple[bool, float, List[str]]:
        try:
            from src.chat.utils.character_matcher import get_character_matcher
            matcher = get_character_matcher()
            result = matcher.match_visual(description)
            return result.is_match, result.score, result.matched_items
        except Exception:
            return False, 0.0, []

    def _build_character_aware_prompt(self) -> str:
        try:
            from src.config.config import global_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            bot_name = global_config.bot.nickname
            visual_str = getattr(global_config.personality, 'character_visual', '')
            visual_hint = f"你的外观特征：{visual_str}。" if visual_str else ""
            return get_prompt(
                PromptCategory.MODULE,
                "visual_analyzer",
                "character_aware_prompt.template",
                bot_name=bot_name,
                visual_hint=visual_hint
            )
        except Exception:
            return "请用中文详细描述这张图片的内容，包括人物特征、场景氛围和文字内容，50-100字。"

    def _get_cache_model(self):
        from src.common.database.database_model import ImageAnalysisCache
        return ImageAnalysisCache

    def _load_analysis_cache(self):
        try:
            IAC = self._get_cache_model()
            cutoff = time.time() - self._memory_decay_days * 86400
            rows = IAC.select().where(IAC.analyzed_at > cutoff)
            loaded = 0
            for row in rows:
                kw_list = row.keywords.split(",") if row.keywords else []
                obj_list = row.detected_objects.split(",") if row.detected_objects else []
                self._analysis_cache[row.content_hash] = AnalysisResult(
                    description=row.description,
                    keywords=kw_list,
                    detected_objects=obj_list,
                    confidence=row.confidence,
                    raw_response=row.raw_response,
                    analyzed_at=row.analyzed_at,
                    image_hash=row.image_hash,
                    is_rejected=row.is_rejected,
                    rejection_reason=row.rejection_reason,
                    from_cache=True
                )
                loaded += 1
            if loaded == 0:
                self._migrate_json_analysis_cache()
            elif loaded > 0:
                logger.info(f"[visual_analyzer] 已加载 {loaded} 条分析缓存")
        except Exception as e:
            logger.debug(f"[visual_analyzer] 加载分析缓存失败: {e}")

    def _migrate_json_analysis_cache(self):
        if not ANALYSIS_CACHE_FILE.exists():
            return
        try:
            with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            IAC = self._get_cache_model()
            count = 0
            for key, item in data.items():
                age_days = (time.time() - item.get('analyzed_at', 0)) / 86400
                if age_days >= self._memory_decay_days:
                    continue
                kw = item.get('keywords', [])
                obj = item.get('detected_objects', [])
                IAC.insert(
                    content_hash=key,
                    description=item.get('description', ''),
                    keywords=",".join(kw) if isinstance(kw, list) else str(kw),
                    detected_objects=",".join(obj) if isinstance(obj, list) else str(obj),
                    confidence=item.get('confidence', 0.0),
                    raw_response=item.get('raw_response', ''),
                    image_hash=item.get('image_hash', ''),
                    is_rejected=item.get('is_rejected', False),
                    rejection_reason=item.get('rejection_reason', ''),
                    analyzed_at=item.get('analyzed_at', time.time())
                ).on_conflict_ignore().execute()
                self._analysis_cache[key] = AnalysisResult(
                    description=item.get('description', ''),
                    keywords=kw, detected_objects=obj,
                    confidence=item.get('confidence', 0.0),
                    raw_response=item.get('raw_response', ''),
                    analyzed_at=item.get('analyzed_at', time.time()),
                    image_hash=item.get('image_hash', ''),
                    is_rejected=item.get('is_rejected', False),
                    rejection_reason=item.get('rejection_reason', ''),
                    from_cache=True
                )
                count += 1
            backup = ANALYSIS_CACHE_FILE.with_suffix('.json.bak')
            ANALYSIS_CACHE_FILE.rename(backup)
            logger.info(f"[visual_analyzer] {count}条图片缓存已从JSON迁移到结构化数据库")
        except Exception as e:
            logger.warning(f"[visual_analyzer] JSON迁移失败: {e}")

    def _save_analysis_cache(self):
        try:
            IAC = self._get_cache_model()
            for key, result in self._analysis_cache.items():
                if getattr(result, 'from_cache', False) and not getattr(result, '_dirty', False):
                    continue
                kw_str = ",".join(result.keywords) if isinstance(result.keywords, list) else str(result.keywords)
                obj_str = ",".join(result.detected_objects) if isinstance(result.detected_objects, list) else str(result.detected_objects)
                IAC.insert(
                    content_hash=key,
                    description=result.description,
                    keywords=kw_str,
                    detected_objects=obj_str,
                    confidence=result.confidence,
                    raw_response=result.raw_response,
                    image_hash=result.image_hash,
                    is_rejected=result.is_rejected,
                    rejection_reason=result.rejection_reason,
                    analyzed_at=result.analyzed_at
                ).on_conflict(
                    conflict_target=[IAC.content_hash],
                    update={
                        IAC.description: result.description,
                        IAC.keywords: kw_str,
                        IAC.detected_objects: obj_str,
                        IAC.confidence: result.confidence,
                        IAC.analyzed_at: result.analyzed_at,
                    }
                ).execute()
        except Exception as e:
            logger.debug(f"[visual_analyzer] 保存分析缓存失败: {e}")

    def set_api_params(self, max_tokens: int = 800, temperature: float = 0.7):
        self._api_caller.set_params(max_tokens, temperature)

    def set_default_prompt(self, prompt: str):
        self._default_prompt = prompt

    async def analyze_url(
        self, url: str, prompt: str = "",
        user_id: str = "", channel_id: str = "",
    ) -> AnalysisResult:
        cached_path = self._cache.get_cached_path(url)
        if cached_path:
            with open(cached_path, "rb") as f:
                image_data = f.read()
        else:
            image_data, _ = await self._downloader.download(url)
            if not image_data:
                return AnalysisResult(description="图片下载失败")
            processed, info = self._processor.process(image_data)
            self._cache.save_to_cache(url, processed)
            image_data = processed
        image_hash = hashlib.md5(image_data).hexdigest()
        is_own_avatar = self._is_own_avatar(url)
        is_avatar = self._is_qq_avatar_url(url)
        if is_own_avatar and not prompt:
            analysis_prompt = self._avatar_prompt
            logger.info(f"[图片分析] 检测到自己的QQ头像，使用头像专用分析")
        elif is_avatar and not prompt:
            analysis_prompt = self._default_prompt
        else:
            analysis_prompt = prompt or self._default_prompt
        if user_id and channel_id:
            repeat_count = self._repeat_detector.get_repeat_count(
                tool_type="image", content_hash=image_hash,
                user_id=user_id, channel_id=channel_id
            )
            if repeat_count > 0:
                should_reject, rejection_reason = await self._repeat_detector.should_reject_repeat_query(
                    tool_type="image", content_hash=image_hash,
                    user_id=user_id, channel_id=channel_id,
                    repeat_count=repeat_count + 1, tool_description="图片"
                )
                if should_reject:
                    self._repeat_detector.record_query(
                        tool_type="image", content_hash=image_hash,
                        user_id=user_id, channel_id=channel_id
                    )
                    logger.info(f"[图片拒绝] 用户{user_id[:8]}...重复提问被拒绝: {rejection_reason}")
                    return AnalysisResult(
                        description=rejection_reason, is_rejected=True,
                        rejection_reason=rejection_reason, image_hash=image_hash
                    )
            self._repeat_detector.record_query(
                tool_type="image", content_hash=image_hash,
                user_id=user_id, channel_id=channel_id
            )
        if image_hash in self._analysis_cache:
            cached = self._analysis_cache[image_hash]
            age_days = (time.time() - cached.analyzed_at) / 86400
            if age_days < self._memory_decay_days:
                logger.info(f"[图片去重] 检测到重复图片（哈希: {image_hash[:12]}...），使用缓存（{age_days:.1f}天前）")
                cached.analyzed_at = time.time()
                cached.from_cache = True
                return cached
            else:
                logger.debug(f"[图片去重] 记忆已过期（{age_days:.1f}天），重新分析")
                del self._analysis_cache[image_hash]
        image_base64 = self._processor.to_base64(image_data)
        logger.info(f"[图片分析] 开始分析新图片（哈希: {image_hash[:12]}...）")
        description = await self._api_caller.analyze(image_base64, analysis_prompt)
        if not description:
            return AnalysisResult(description="图片分析失败")
        is_self_detected, self_score, self_features = self._check_self_in_description(description)
        if is_own_avatar:
            is_self_detected = True
            self_score = max(self_score, 1.0)
        if is_self_detected:
            logger.info(f"[图片分析] 识别到自己！匹配度: {self_score:.2f}, 特征: {self_features}")
        result = AnalysisResult(
            description=description, raw_response=description, image_hash=image_hash,
            is_self=is_self_detected, self_match_score=self_score,
            self_matched_features=self_features,
        )
        self._analysis_cache[image_hash] = result
        self._save_analysis_cache()
        self._maybe_cleanup()
        return result

    async def analyze_base64(self, image_base64: str, prompt: str = "") -> AnalysisResult:
        try:
            image_data = base64.b64decode(image_base64)
            image_hash = hashlib.md5(image_data).hexdigest()
            analysis_prompt = prompt or self._default_prompt
            if image_hash in self._analysis_cache:
                cached = self._analysis_cache[image_hash]
                age_days = (time.time() - cached.analyzed_at) / 86400
                if age_days < self._memory_decay_days:
                    logger.info(f"[图片去重] 检测到重复图片（哈希: {image_hash[:12]}...），使用缓存（{age_days:.1f}天前）")
                    cached.analyzed_at = time.time()
                    cached.from_cache = True
                    return cached
                else:
                    del self._analysis_cache[image_hash]
            logger.info(f"[图片分析] 开始分析新图片（哈希: {image_hash[:12]}...）")
            description = await self._api_caller.analyze(image_base64, analysis_prompt)
            if not description:
                return AnalysisResult(description="图片分析失败")
            result = AnalysisResult(
                description=description, raw_response=description, image_hash=image_hash,
            )
            self._analysis_cache[image_hash] = result
            self._save_analysis_cache()
            self._maybe_cleanup()
            return result
        except Exception as e:
            logger.error(f"[visual_analyzer] analyze_base64失败: {e}")
            return AnalysisResult(description="图片分析失败")

    async def analyze_url_with_prompt(self, url: str, prompt: str) -> str:
        try:
            result = await self.analyze_url(url, prompt)
            return result.description if result else ""
        except Exception as e:
            logger.error(f"[visual_analyzer] analyze_url_with_prompt失败: {e}")
            return ""

    async def get_base64_content(self, url: str) -> str:
        cached_path = self._cache.get_cached_path(url)
        if cached_path:
            with open(cached_path, "rb") as f:
                image_data = f.read()
        else:
            image_data, _ = await self._downloader.download(url)
            if not image_data:
                logger.warning(f"[visual_analyzer] 下载失败，无法生成 Base64: {url}")
                return ""
            processed, _ = self._processor.process(image_data)
            self._cache.save_to_cache(url, processed)
            image_data = processed
        base64_str = self._processor.to_base64(image_data)
        return f"base64://{base64_str}"

    def _maybe_cleanup(self):
        if time.time() - self._last_cleanup > 3600:
            self._cache.cleanup_old()
            forget_threshold = time.time() - (self._memory_decay_days * 86400)
            before_count = len(self._analysis_cache)
            self._analysis_cache = {
                k: v for k, v in self._analysis_cache.items()
                if v.analyzed_at > forget_threshold
            }
            after_count = len(self._analysis_cache)
            if before_count != after_count:
                logger.debug(f"[图片去重] 海马体遗忘机制：清理了 {before_count - after_count} 条过期记忆")
            self._save_analysis_cache()
            self._last_cleanup = time.time()

    def get_stats(self) -> Dict:
        return {
            "analysis_cache_size": len(self._analysis_cache),
        }


_visual_analyzer: Optional[VisualAnalyzer] = None


def get_visual_analyzer() -> VisualAnalyzer:
    global _visual_analyzer
    if _visual_analyzer is None:
        _visual_analyzer = VisualAnalyzer()
    return _visual_analyzer


def init_visual_analyzer(config: Optional[Dict] = None) -> VisualAnalyzer:
    global _visual_analyzer
    _visual_analyzer = VisualAnalyzer()
    if config:
        vision_config = config.get("multimedia", config)
        max_tokens = vision_config.get("vision_max_tokens", 800)
        temperature = vision_config.get("vision_temperature", 0.7)
        _visual_analyzer.set_api_params(max_tokens, temperature)
        prompt = vision_config.get("visual_style", "")
        if prompt:
            enhanced = _inject_character_context(prompt)
            _visual_analyzer.set_default_prompt(enhanced)
    logger.info("[visual_analyzer] 初始化完成")
    return _visual_analyzer


def _inject_character_context(base_prompt: str) -> str:
    try:
        from src.config.config import global_config
        bot_name = global_config.bot.nickname
        alias_names = getattr(global_config.bot, 'alias_names', []) or []
        name_hint = f"'{bot_name}'"
        if alias_names:
            name_hint += "（也叫" + "、".join(alias_names[:3]) + "）"
        return f"{base_prompt}。如果出现名为{name_hint}的角色请明确指出。"
    except Exception:
        return base_prompt
