import asyncio
from src.common.logger import get_logger

logger = get_logger("dimension_bootstrap")


async def bootstrap_all_dimensions():
    """
    启动时注册所有11个维度到调度器。
    在 Heartflow.startup() 中调用一次。
    注册顺序不影响功能，但按依赖关系排列便于调试。
    """
    from src.chat.heart_flow.dimension_dispatcher import DimensionDispatcher
    dispatcher = await DimensionDispatcher.get_instance()
    registered = []
    # D1: 情绪三轴
    try:
        from src.chat.heart_flow.emotion_stream import EmotionAxisDimension
        d1 = EmotionAxisDimension.get_instance()
        dispatcher.register_dimension(d1)
        registered.append("D1_emotion_axis")
    except Exception as exc:
        logger.error(f"D1 情绪三轴注册失败: {exc}")
    # D2: 好感/信任双核
    try:
        from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
        d2 = FondnessTrustDimension.get_instance()
        dispatcher.register_dimension(d2)
        registered.append("D2_fondness_trust")
    except Exception as exc:
        logger.error(f"D2 好感信任注册失败: {exc}")
    # D2b: 讨厌度注册表
    try:
        from src.chat.heart_flow.dislike_registry import DislikeRegistryDimension
        d2b = DislikeRegistryDimension.get_instance()
        dispatcher.register_dimension(d2b)
        registered.append("D2b_dislike_registry")
    except Exception as exc:
        logger.error(f"D2b 讨厌度注册失败: {exc}")
    # D3: 频率控制
    try:
        from src.chat.heart_flow.frequency_control import FrequencyDimension
        d3 = FrequencyDimension.get_instance()
        dispatcher.register_dimension(d3)
        registered.append("D3_frequency_control")
    except Exception as exc:
        logger.error(f"D3 频率控制注册失败: {exc}")
    # D4: 用户状态检测
    try:
        from src.chat.heart_flow.user_state_detector import UserStateDimension
        d4 = UserStateDimension.get_instance()
        dispatcher.register_dimension(d4)
        registered.append("D4_user_state")
    except Exception as exc:
        logger.error(f"D4 用户状态注册失败: {exc}")
    # D5: 群体氛围
    try:
        from src.chat.heart_flow.group_atmosphere import GroupAtmosphereDimension
        d5 = GroupAtmosphereDimension.get_instance()
        dispatcher.register_dimension(d5)
        registered.append("D5_group_atmosphere")
    except Exception as exc:
        logger.error(f"D5 群体氛围注册失败: {exc}")
    # D6: 能量链条
    try:
        from src.chat.heart_flow.energy_manager import EnergyChainDimension
        d6 = EnergyChainDimension.get_instance()
        dispatcher.register_dimension(d6)
        registered.append("D6_energy_chain")
    except Exception as exc:
        logger.error(f"D6 能量链条注册失败: {exc}")
    # D7: 社交值
    try:
        from src.chat.heart_flow.social_value_dim import SocialValueDimension
        d7 = SocialValueDimension.get_instance()
        dispatcher.register_dimension(d7)
        registered.append("D7_social_value")
    except Exception as exc:
        logger.error(f"D7 社交值注册失败: {exc}")
    # D8: 心流等待状态
    try:
        from src.chat.heart_flow.heart_state_dim import HeartStateDimension
        d8 = HeartStateDimension.get_instance()
        dispatcher.register_dimension(d8)
        registered.append("D8_heart_state")
    except Exception as exc:
        logger.error(f"D8 心流等待注册失败: {exc}")
    # D10: 创伤核心
    try:
        from src.chat.heart_flow.trauma_fabric import TraumaDimension
        d10 = TraumaDimension.get_instance()
        dispatcher.register_dimension(d10)
        registered.append("D10_trauma")
    except Exception as exc:
        logger.error(f"D10 创伤核心注册失败: {exc}")
    # D11: 表面面具
    try:
        from src.chat.heart_flow.trauma_fabric import SurfaceMaskDimension
        d11 = SurfaceMaskDimension.get_instance()
        dispatcher.register_dimension(d11)
        registered.append("D11_surface_mask")
    except Exception as exc:
        logger.error(f"D11 表面面具注册失败: {exc}")
    logger.info(f"维度注册完成: {len(registered)}/11 → {', '.join(registered)}")
    return dispatcher
