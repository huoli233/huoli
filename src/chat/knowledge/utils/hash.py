import hashlib
from typing import Optional


def get_sha256(text: str) -> str:
    """计算文本的 SHA256 哈希值

    参数:
        text: 输入文本

    返回:
        64位十六进制哈希字符串
    """
    if not text:
        return ""
    sha256 = hashlib.sha256()
    sha256.update(text.encode("utf-8"))
    return sha256.hexdigest()


def get_md5(text: str) -> str:
    """计算文本的 MD5 哈希值

    参数:
        text: 输入文本

    返回:
        32位十六进制哈希字符串
    """
    if not text:
        return ""
    md5 = hashlib.md5()
    md5.update(text.encode("utf-8"))
    return md5.hexdigest()


def get_short_hash(text: str, length: int = 8) -> str:
    """获取短哈希值

    参数:
        text: 输入文本
        length: 哈希长度，默认8位

    返回:
        截断的哈希字符串
    """
    full_hash = get_sha256(text)
    return full_hash[:length] if full_hash else ""


def hash_file(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    """计算文件的哈希值

    参数:
        file_path: 文件路径
        algorithm: 哈希算法，支持 sha256, md5

    返回:
        哈希字符串，失败返回 None
    """
    try:
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        else:
            return None
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def verify_hash(
    text: str, expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """验证文本哈希值

    参数:
        text: 输入文本
        expected_hash: 期望的哈希值
        algorithm: 哈希算法

    返回:
        是否匹配
    """
    if algorithm == "sha256":
        actual_hash = get_sha256(text)
    elif algorithm == "md5":
        actual_hash = get_md5(text)
    else:
        return False
    return actual_hash == expected_hash.lower()
