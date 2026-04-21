import os
import io
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, List, Annotated
from concurrent.futures import ThreadPoolExecutor
from fastapi import (
    APIRouter,
    HTTPException,
    Header,
    Query,
    UploadFile,
    File,
    Form,
    Cookie,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from src.common.logger import get_logger, PROJECT_ROOT
from src.common.database.database_model import Emoji
from src.webui.core.security import get_token_manager
from src.webui.core.auth import verify_auth_token_from_cookie_or_header

logger = get_logger("WebUI表情")
THUMBNAIL_CACHE_DIR = PROJECT_ROOT / "data" / "emoji_thumbnails"
THUMBNAIL_SIZE = (200, 200)
THUMBNAIL_QUALITY = 80
_thumbnail_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()
_thumbnail_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="thumbnail"
)
_generating_thumbnails: set[str] = set()
_generating_lock = threading.Lock()


def _get_thumbnail_lock(file_hash: str) -> threading.Lock:
    """获取指定文件哈希的锁"""
    with _locks_lock:
        if file_hash not in _thumbnail_locks:
            _thumbnail_locks[file_hash] = threading.Lock()
        return _thumbnail_locks[file_hash]


def _background_generate_thumbnail(source_path: str, file_hash: str) -> None:
    """后台生成缩略图"""
    try:
        _generate_thumbnail(source_path, file_hash)
    except Exception as e:
        logger.warning(f"后台生成缩略图失败 {file_hash}: {e}")
    finally:
        with _generating_lock:
            _generating_thumbnails.discard(file_hash)
        # 缩略图生成完毕（成功或失败），释放哈希锁防止泄漏
        with _locks_lock:
            _thumbnail_locks.pop(file_hash, None)


def _ensure_thumbnail_cache_dir() -> Path:
    """确保缩略图缓存目录存在"""
    THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return THUMBNAIL_CACHE_DIR


def _get_thumbnail_cache_path(file_hash: str) -> Path:
    """获取缩略图缓存路径"""
    return THUMBNAIL_CACHE_DIR / f"{file_hash}.webp"


def _generate_thumbnail(source_path: str, file_hash: str) -> Path:
    """生成缩略图并保存到缓存目录"""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL 未安装，无法生成缩略图")
        raise
    _ensure_thumbnail_cache_dir()
    cache_path = _get_thumbnail_cache_path(file_hash)
    lock = _get_thumbnail_lock(file_hash)
    with lock:
        if cache_path.exists():
            return cache_path
        try:
            with Image.open(source_path) as img:
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    img.seek(0)
                if img.mode in ("P", "PA"):
                    img = img.convert("RGBA")
                elif img.mode == "LA":
                    img = img.convert("RGBA")
                elif img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(
                    cache_path, "WEBP", quality=THUMBNAIL_QUALITY, method=6
                )
                logger.debug(f"生成缩略图: {file_hash} -> {cache_path}")
        except Exception as e:
            logger.warning(f"生成缩略图失败 {file_hash}: {e}")
            raise
    return cache_path


def cleanup_orphaned_thumbnails() -> tuple[int, int]:
    """清理孤立的缩略图缓存"""
    if not THUMBNAIL_CACHE_DIR.exists():
        return 0, 0
    valid_hashes = set()
    for emoji in Emoji.select(Emoji.emoji_hash):
        valid_hashes.add(emoji.emoji_hash)
    cleaned = 0
    kept = 0
    for cache_file in THUMBNAIL_CACHE_DIR.glob("*.webp"):
        file_hash = cache_file.stem
        if file_hash not in valid_hashes:
            try:
                cache_file.unlink()
                cleaned += 1
                logger.debug(f"清理孤立缩略图: {cache_file.name}")
            except Exception as e:
                logger.warning(f"清理缩略图失败 {cache_file.name}: {e}")
        else:
            kept += 1
    if cleaned > 0:
        logger.info(f"清理孤立缩略图: 删除 {cleaned} 个，保留 {kept} 个")
    return cleaned, kept


EmojiFile = Annotated[UploadFile, File(description="表情包图片文件")]
EmojiFiles = Annotated[
    List[UploadFile], File(description="多个表情包图片文件")
]
DescriptionForm = Annotated[str, Form(description="表情包描述")]
EmotionForm = Annotated[str, Form(description="情感标签")]
IsRegisteredForm = Annotated[bool, Form(description="是否直接注册")]
router = APIRouter(prefix="/emoji", tags=["Emoji"])


class EmojiResponse(BaseModel):
    """表情包响应"""

    id: int
    full_path: str
    format: str
    emoji_hash: str
    description: str
    query_count: int
    is_registered: bool
    is_banned: bool
    emotion: Optional[str]
    record_time: float
    register_time: Optional[float]
    usage_count: int
    last_used_time: Optional[float]


class EmojiListResponse(BaseModel):
    """表情包列表响应"""

    success: bool
    total: int
    page: int
    page_size: int
    data: List[EmojiResponse]


class EmojiUpdateRequest(BaseModel):
    """表情包更新请求"""

    description: Optional[str] = None
    is_registered: Optional[bool] = None
    is_banned: Optional[bool] = None
    emotion: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""

    emoji_ids: List[int]


def verify_auth_token(
    huoli_session: Optional[str] = None,
    authorization: Optional[str] = None,
) -> bool:
    """验证认证 Token"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


def emoji_to_response(emoji: Emoji) -> EmojiResponse:
    """将 Emoji 模型转换为响应对象"""
    return EmojiResponse(
        id=emoji.id,
        full_path=emoji.full_path,
        format=emoji.format,
        emoji_hash=emoji.emoji_hash,
        description=emoji.description,
        query_count=emoji.query_count,
        is_registered=emoji.is_registered,
        is_banned=emoji.is_banned,
        emotion=str(emoji.emotion) if emoji.emotion is not None else None,
        record_time=emoji.record_time,
        register_time=emoji.register_time,
        usage_count=emoji.usage_count,
        last_used_time=emoji.last_used_time,
    )


@router.get("/list", response_model=EmojiListResponse)
async def get_emoji_list(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索条件"),
    is_registered: Optional[bool] = Query(None, description="是否已注册筛选"),
    is_banned: Optional[bool] = Query(None, description="是否被禁用筛选"),
    format: Optional[str] = Query(None, description="格式筛选"),
    sort_by: Optional[str] = Query("usage_count", description="排序字段"),
    sort_order: Optional[str] = Query("desc", description="排序方向"),
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取表情包列表"""
    try:
        verify_auth_token(huoli_session, authorization)
        query = Emoji.select()
        if search:
            query = query.where(
                (Emoji.description.contains(search))
                | (Emoji.emoji_hash.contains(search))
            )
        if is_registered is not None:
            query = query.where(Emoji.is_registered == is_registered)
        if is_banned is not None:
            query = query.where(Emoji.is_banned == is_banned)
        if format:
            query = query.where(Emoji.format == format)
        sort_field_map = {
            "usage_count": Emoji.usage_count,
            "register_time": Emoji.register_time,
            "record_time": Emoji.record_time,
            "last_used_time": Emoji.last_used_time,
        }
        sort_field = sort_field_map.get(sort_by, Emoji.usage_count)
        if sort_order == "asc":
            query = query.order_by(sort_field.asc())
        else:
            query = query.order_by(sort_field.desc())
        total = query.count()
        offset = (page - 1) * page_size
        emojis = query.offset(offset).limit(page_size)
        data = [emoji_to_response(emoji) for emoji in emojis]
        return EmojiListResponse(
            success=True,
            total=total,
            page=page,
            page_size=page_size,
            data=data,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取表情包列表失败: {
                str(e)}",
        ) from e


@router.get("/{emoji_id}")
async def get_emoji_detail(
    emoji_id: int,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取表情包详细信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        return {"success": True, "data": emoji_to_response(emoji)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包详情失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取表情包详情失败: {
                str(e)}",
        ) from e


@router.patch("/{emoji_id}")
async def update_emoji(
    emoji_id: int,
    request: EmojiUpdateRequest,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """增量更新表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        update_data = request.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(
                status_code=400, detail="未提供任何需要更新的字段"
            )
        if (
            "is_registered" in update_data
            and update_data["is_registered"]
            and not emoji.is_registered
        ):
            update_data["register_time"] = time.time()
        for field, value in update_data.items():
            setattr(emoji, field, value)
        emoji.save()
        logger.info(
            f"表情包已更新: ID={emoji_id}, 字段: {list(update_data.keys())}"
        )
        return {
            "success": True,
            "message": f"成功更新 {
                len(update_data)} 个字段",
            "data": emoji_to_response(emoji),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"更新表情包失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新表情包失败: {
                str(e)}",
        ) from e


@router.delete("/{emoji_id}")
async def delete_emoji(
    emoji_id: int,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """删除表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        emoji_hash = emoji.emoji_hash
        emoji.delete_instance()
        logger.info(f"表情包已删除: ID={emoji_id}, hash={emoji_hash}")
        return {"success": True, "message": f"成功删除表情包: {emoji_hash}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"删除表情包失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"删除表情包失败: {
                str(e)}",
        ) from e


@router.get("/stats/summary")
async def get_emoji_stats(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取表情包统计数据"""
    try:
        verify_auth_token(huoli_session, authorization)
        total = Emoji.select().count()
        registered = Emoji.select().where(Emoji.is_registered).count()
        banned = Emoji.select().where(Emoji.is_banned).count()
        formats = {}
        for emoji in Emoji.select(Emoji.format):
            fmt = emoji.format
            formats[fmt] = formats.get(fmt, 0) + 1
        top_used = Emoji.select().order_by(Emoji.usage_count.desc()).limit(10)
        top_used_list = [
            {
                "id": emoji.id,
                "emoji_hash": emoji.emoji_hash,
                "description": emoji.description,
                "usage_count": emoji.usage_count,
            }
            for emoji in top_used
        ]
        return {
            "success": True,
            "data": {
                "total": total,
                "registered": registered,
                "banned": banned,
                "unregistered": total - registered,
                "formats": formats,
                "top_used": top_used_list,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取统计数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计数据失败: {
                str(e)}",
        ) from e


@router.post("/{emoji_id}/register")
async def register_emoji(
    emoji_id: int,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """注册表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        if emoji.is_registered:
            raise HTTPException(status_code=400, detail="该表情包已经注册")
        emoji.is_registered = True
        emoji.is_banned = False
        emoji.register_time = time.time()
        emoji.save()
        logger.info(f"表情包已注册: ID={emoji_id}")
        return {
            "success": True,
            "message": "表情包注册成功",
            "data": emoji_to_response(emoji),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"注册表情包失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"注册表情包失败: {
                str(e)}",
        ) from e


@router.post("/{emoji_id}/ban")
async def ban_emoji(
    emoji_id: int,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """禁用表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        emoji.is_banned = True
        emoji.is_registered = False
        emoji.save()
        logger.info(f"表情包已禁用: ID={emoji_id}")
        return {
            "success": True,
            "message": "表情包禁用成功",
            "data": emoji_to_response(emoji),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"禁用表情包失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"禁用表情包失败: {
                str(e)}",
        ) from e


@router.get("/{emoji_id}/thumbnail")
async def get_emoji_thumbnail(
    emoji_id: int,
    token: Optional[str] = Query(None, description="访问令牌"),
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
    original: bool = Query(False, description="是否返回原图"),
):
    """获取表情包缩略图"""
    try:
        token_manager = get_token_manager()
        is_valid = False
        if huoli_session and token_manager.verify_token(huoli_session):
            is_valid = True
        elif token and token_manager.verify_token(token):
            is_valid = True
        elif authorization and authorization.startswith("Bearer "):
            auth_token = authorization.replace("Bearer ", "")
            if token_manager.verify_token(auth_token):
                is_valid = True
        if not is_valid:
            raise HTTPException(status_code=401, detail="Token 无效或已过期")
        emoji = Emoji.get_or_none(Emoji.id == emoji_id)
        if not emoji:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {emoji_id} 的表情包"
            )
        if not os.path.exists(emoji.full_path):
            raise HTTPException(status_code=404, detail="表情包文件不存在")
        if original:
            mime_types = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "bmp": "image/bmp",
            }
            media_type = mime_types.get(
                emoji.format.lower(), "application/octet-stream"
            )
            return FileResponse(
                path=emoji.full_path,
                media_type=media_type,
                filename=f"{
                    emoji.emoji_hash}.{
                    emoji.format}",
            )
        cache_path = _get_thumbnail_cache_path(emoji.emoji_hash)
        if cache_path.exists():
            return FileResponse(
                path=str(cache_path),
                media_type="image/webp",
                filename=f"{
                    emoji.emoji_hash}_thumb.webp",
            )
        with _generating_lock:
            if emoji.emoji_hash not in _generating_thumbnails:
                _generating_thumbnails.add(emoji.emoji_hash)
                _thumbnail_executor.submit(
                    _background_generate_thumbnail,
                    emoji.full_path,
                    emoji.emoji_hash,
                )
        return JSONResponse(
            status_code=202,
            content={
                "status": "generating",
                "message": "缩略图正在生成中，请稍后重试",
                "emoji_id": emoji_id,
            },
            headers={"Retry-After": "1"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取表情包缩略图失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取表情包缩略图失败: {
                str(e)}",
        ) from e


@router.post("/batch/delete")
async def batch_delete_emojis(
    request: BatchDeleteRequest,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """批量删除表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        if not request.emoji_ids:
            raise HTTPException(
                status_code=400, detail="未提供要删除的表情包ID"
            )
        deleted_count = 0
        failed_count = 0
        failed_ids = []
        for emoji_id in request.emoji_ids:
            try:
                emoji = Emoji.get_or_none(Emoji.id == emoji_id)
                if emoji:
                    emoji.delete_instance()
                    deleted_count += 1
                    logger.info(f"批量删除表情包: {emoji_id}")
                else:
                    failed_count += 1
                    failed_ids.append(emoji_id)
            except Exception as e:
                logger.error(f"删除表情包 {emoji_id} 失败: {e}")
                failed_count += 1
                failed_ids.append(emoji_id)
        message = f"成功删除 {deleted_count} 个表情包"
        if failed_count > 0:
            message += f"，{failed_count} 个失败"
        return {
            "success": True,
            "message": message,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "failed_ids": failed_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"批量删除表情包失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"批量删除失败: {str(e)}"
        ) from e


EMOJI_REGISTERED_DIR = os.path.join("data", "emoji_registed")


@router.post("/upload")
async def upload_emoji(
    file: EmojiFile,
    description: DescriptionForm = "",
    emotion: EmotionForm = "",
    is_registered: IsRegisteredForm = True,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """上传并注册表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        if not file.content_type:
            raise HTTPException(status_code=400, detail="无法识别文件类型")
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {
                    file.content_type}，支持: {
                    ', '.join(allowed_types)}",
            )
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="文件内容为空")
        try:
            from PIL import Image

            with Image.open(io.BytesIO(file_content)) as img:
                img_format = img.format.lower() if img.format else "png"
                img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"无效的图片文件: {
                    str(e)}",
            ) from e
        with Image.open(io.BytesIO(file_content)) as img:
            img_format = img.format.lower() if img.format else "png"
        emoji_hash = hashlib.md5(file_content).hexdigest()
        existing_emoji = Emoji.get_or_none(Emoji.emoji_hash == emoji_hash)
        if existing_emoji:
            raise HTTPException(
                status_code=409,
                detail=f"已存在相同的表情包 (ID: {existing_emoji.id})",
            )
        os.makedirs(EMOJI_REGISTERED_DIR, exist_ok=True)
        timestamp = int(time.time())
        filename = f"emoji_{timestamp}_{emoji_hash[:8]}.{img_format}"
        full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
        counter = 1
        while os.path.exists(full_path):
            filename = (
                f"emoji_{timestamp}_{emoji_hash[:8]}_{counter}.{img_format}"
            )
            full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
            counter += 1
        with open(full_path, "wb") as f:
            f.write(file_content)
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"表情包文件已保存: {full_path}")
        emotion_str = (
            ",".join(e.strip() for e in emotion.split(",") if e.strip())
            if emotion
            else ""
        )
        current_time = time.time()
        emoji = Emoji.create(
            full_path=full_path,
            format=img_format,
            emoji_hash=emoji_hash,
            description=description,
            emotion=emotion_str,
            query_count=0,
            is_registered=is_registered,
            is_banned=False,
            record_time=current_time,
            register_time=current_time if is_registered else None,
            usage_count=0,
            last_used_time=None,
        )
        logger.info(f"表情包已上传并注册: ID={emoji.id}, hash={emoji_hash}")
        return {
            "success": True,
            "message": "表情包上传成功"
            + ("并已注册" if is_registered else ""),
            "data": emoji_to_response(emoji),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"上传表情包失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"上传失败: {str(e)}"
        ) from e


@router.post("/batch/upload")
async def batch_upload_emoji(
    files: EmojiFiles,
    emotion: EmotionForm = "",
    is_registered: IsRegisteredForm = True,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """批量上传表情包"""
    try:
        verify_auth_token(huoli_session, authorization)
        results = {
            "success": True,
            "total": len(files),
            "uploaded": 0,
            "failed": 0,
            "details": [],
        }
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        os.makedirs(EMOJI_REGISTERED_DIR, exist_ok=True)
        for file in files:
            try:
                if file.content_type not in allowed_types:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": f"不支持的文件类型: {file.content_type}",
                        }
                    )
                    continue
                file_content = await file.read()
                if not file_content:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": "文件内容为空",
                        }
                    )
                    continue
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(file_content)) as img:
                        img_format = (
                            img.format.lower() if img.format else "png"
                        )
                except Exception as e:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": f"无效的图片: {str(e)}",
                        }
                    )
                    continue
                emoji_hash = hashlib.md5(file_content).hexdigest()
                if Emoji.get_or_none(Emoji.emoji_hash == emoji_hash):
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "error": "已存在相同的表情包",
                        }
                    )
                    continue
                timestamp = int(time.time())
                filename = f"emoji_{timestamp}_{emoji_hash[:8]}.{img_format}"
                full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
                counter = 1
                while os.path.exists(full_path):
                    filename = f"emoji_{timestamp}_{emoji_hash[:8]}_{counter}.{img_format}"
                    full_path = os.path.join(EMOJI_REGISTERED_DIR, filename)
                    counter += 1
                with open(full_path, "wb") as f:
                    f.write(file_content)
                    f.flush()
                    os.fsync(f.fileno())
                emotion_str = (
                    ",".join(
                        e.strip() for e in emotion.split(",") if e.strip()
                    )
                    if emotion
                    else ""
                )
                current_time = time.time()
                emoji = Emoji.create(
                    full_path=full_path,
                    format=img_format,
                    emoji_hash=emoji_hash,
                    description="",
                    emotion=emotion_str,
                    query_count=0,
                    is_registered=is_registered,
                    is_banned=False,
                    record_time=current_time,
                    register_time=current_time if is_registered else None,
                    usage_count=0,
                    last_used_time=None,
                )
                results["uploaded"] += 1
                results["details"].append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "id": emoji.id,
                    }
                )
            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": str(e),
                    }
                )
        results["message"] = (
            f"成功上传 {
                results['uploaded']} 个，失败 {
                results['failed']} 个"
        )
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"批量上传表情包失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"批量上传失败: {str(e)}"
        ) from e


class ThumbnailCacheStatsResponse(BaseModel):
    """缩略图缓存统计响应"""

    success: bool
    cache_dir: str
    total_count: int
    total_size_mb: float
    emoji_count: int
    coverage_percent: float


class ThumbnailCleanupResponse(BaseModel):
    """缩略图清理响应"""

    success: bool
    message: str
    cleaned_count: int
    kept_count: int


@router.get(
    "/thumbnail-cache/stats", response_model=ThumbnailCacheStatsResponse
)
async def get_thumbnail_cache_stats(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取缩略图缓存统计信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        _ensure_thumbnail_cache_dir()
        cache_files = list(THUMBNAIL_CACHE_DIR.glob("*.webp"))
        total_count = len(cache_files)
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = round(total_size / (1024 * 1024), 2)
        emoji_count = Emoji.select().count()
        coverage_percent = round(
            (total_count / emoji_count * 100) if emoji_count > 0 else 0, 1
        )
        return ThumbnailCacheStatsResponse(
            success=True,
            cache_dir=str(THUMBNAIL_CACHE_DIR.absolute()),
            total_count=total_count,
            total_size_mb=total_size_mb,
            emoji_count=emoji_count,
            coverage_percent=coverage_percent,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取缩略图缓存统计失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"获取统计失败: {str(e)}"
        ) from e


@router.post(
    "/thumbnail-cache/cleanup", response_model=ThumbnailCleanupResponse
)
async def cleanup_thumbnail_cache(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """清理孤立的缩略图缓存"""
    try:
        verify_auth_token(huoli_session, authorization)
        cleaned, kept = cleanup_orphaned_thumbnails()
        return ThumbnailCleanupResponse(
            success=True,
            message=f"清理完成：删除 {cleaned} 个孤立缓存，保留 {kept} 个有效缓存",
            cleaned_count=cleaned,
            kept_count=kept,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"清理缩略图缓存失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"清理失败: {str(e)}"
        ) from e


@router.delete(
    "/thumbnail-cache/clear", response_model=ThumbnailCleanupResponse
)
async def clear_all_thumbnail_cache(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """清空所有缩略图缓存"""
    try:
        verify_auth_token(huoli_session, authorization)
        if not THUMBNAIL_CACHE_DIR.exists():
            return ThumbnailCleanupResponse(
                success=True,
                message="缓存目录不存在，无需清理",
                cleaned_count=0,
                kept_count=0,
            )
        cleaned = 0
        for cache_file in THUMBNAIL_CACHE_DIR.glob("*.webp"):
            try:
                cache_file.unlink()
                cleaned += 1
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {cache_file.name}: {e}")
        logger.info(f"已清空缩略图缓存: 删除 {cleaned} 个文件")
        return ThumbnailCleanupResponse(
            success=True,
            message=f"已清空所有缩略图缓存：删除 {cleaned} 个文件",
            cleaned_count=cleaned,
            kept_count=0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"清空缩略图缓存失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"清空失败: {str(e)}"
        ) from e
