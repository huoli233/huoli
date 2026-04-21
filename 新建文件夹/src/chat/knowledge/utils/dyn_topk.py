from typing import Any, List, Tuple


def dynamic_select_top_k(
    scores: List[Tuple[Any, float]],
    jump_factor: float = 0.5,
    variance_factor: float = 1.0,
) -> List[Tuple[Any, float, float]]:
    """动态 TopK 选择算法

    根据分数分布自动确定合适的返回数量，
    结合跳变点和方差信息进行动态阈值计算。

    参数:
        scores: 原始分数列表，每项为 (项目, 分数) 元组
        jump_factor: 跳变点权重因子，范围 [0, 1]
        variance_factor: 方差权重因子，控制方差对阈值的影响

    返回:
        过滤后的列表，每项为 (项目, 原始分数, 归一化分数) 元组
    """
    if not scores:
        return []
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    max_score = sorted_scores[0][1]
    min_score = sorted_scores[-1][1]
    if max_score == min_score:
        return [(s[0], s[1], 1.0) for s in sorted_scores]
    normalized = []
    for item, score in sorted_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((item, score, norm_score))
    jump_idx = 0
    max_jump = 0
    for i in range(1, len(normalized)):
        jump = abs(normalized[i][2] - normalized[i - 1][2])
        if jump > max_jump:
            max_jump = jump
            jump_idx = i
    jump_threshold = (
        normalized[jump_idx][2] if jump_idx < len(normalized) else 0
    )
    mean_score = sum(s[2] for s in normalized) / len(normalized)
    var_score = sum((s[2] - mean_score) ** 2 for s in normalized) / len(
        normalized
    )
    threshold = jump_factor * jump_threshold + (1 - jump_factor) * (
        mean_score + variance_factor * var_score
    )
    return [s for s in normalized if s[2] > threshold]


def adaptive_top_k(
    scores: List[Tuple[Any, float]],
    min_k: int = 1,
    max_k: int = 20,
    threshold: float = 0.3,
) -> List[Tuple[Any, float]]:
    """自适应 TopK 选择

    根据分数分布和阈值自动确定返回数量。

    参数:
        scores: 原始分数列表
        min_k: 最小返回数量
        max_k: 最大返回数量
        threshold: 最低分数阈值

    返回:
        过滤后的列表
    """
    if not scores:
        return []
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    max_score = sorted_scores[0][1]
    min_score = sorted_scores[-1][1]
    if max_score == min_score:
        return sorted_scores[:min_k]
    results = []
    for item, score in sorted_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        if norm_score >= threshold and len(results) < max_k:
            results.append((item, score))
        elif len(results) < min_k:
            results.append((item, score))
        else:
            break
    return results
