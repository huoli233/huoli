import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
DATA_PATH = os.path.join(ROOT_PATH, "data")

INVALID_ENTITY = (
    "",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "我",
    "这",
    "那",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "为什么",
)

__all__ = ["CURRENT_DIR", "ROOT_PATH", "DATA_PATH", "INVALID_ENTITY"]
