from typing import Optional
from fastapi import Depends, Cookie, Header, Request
from .core.auth import get_current_token
from .core.security import get_token_manager
from .core.rate_limiter import check_auth_rate_limit


async def require_auth(
    request: Request,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    FastAPI 依赖：要求有效认证

    用于保护需要认证的路由，自动从 Cookie 或 Header 获取并验证 token

    Returns:
        验证通过的 token

    Raises:
        HTTPException 401: 认证失败
    """
    return get_current_token(request, huoli_session, authorization)


async def require_auth_with_rate_limit(
    request: Request,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
    _rate_limit: None = Depends(check_auth_rate_limit),
) -> str:
    """
    FastAPI 依赖：要求有效认证 + 频率限制

    组合了认证检查和频率限制，适用于敏感操作

    Returns:
        验证通过的 token

    Raises:
        HTTPException 401: 认证失败
        HTTPException 429: 请求过于频繁
    """
    return get_current_token(request, huoli_session, authorization)


def get_optional_token(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    FastAPI 依赖：可选获取 token（不验证）

    用于某些需要知道是否有 token 但不强制验证的场景

    Returns:
        token 字符串或 None
    """
    if huoli_session:
        return huoli_session
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None


async def verify_token_optional(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """
    FastAPI 依赖：可选验证 token

    返回 token 是否有效，不抛出异常

    Returns:
        True 如果 token 有效，否则 False
    """
    token = None
    if huoli_session:
        token = huoli_session
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")

    if not token:
        return False

    token_manager = get_token_manager()
    return token_manager.verify_token(token)
