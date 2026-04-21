import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(ROOT_PATH, "data")

INVALID_ENTITY = {
    "", " ", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "这个", "那个", "什么", "怎么", "哪里", "谁",
    "的", "了", "是", "在", "有", "不", "也", "都", "就",
    "吗", "呢", "啊", "吧", "哦", "嗯", "哈", "呵",
}
