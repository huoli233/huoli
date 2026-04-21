"""
情感驱动主动发言的LLM提示词模板
"""

# 基础情感状态分析提示词（动态生成）
def get_emotion_state_analysis_prompt(boredom_level, loneliness_accumulation, social_desire, mood_score, energy_level, silence_minutes, last_interaction="", mood_description=""):
    """动态获取基于配置的情感状态分析提示词"""
    from src.config.config import global_config
    character_name = global_config.bot.nickname
    personality = global_config.personality.personality
    
    from src.config.prompt_loader import get_prompt, PromptCategory
    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "emotion_state_analysis.template",
        character_name=character_name,
        personality=personality,
        boredom_level=f"{boredom_level:.2f}",
        loneliness_accumulation=f"{loneliness_accumulation:.2f}",
        social_desire=f"{social_desire:.2f}",
        mood_score=f"{mood_score:.2f}",
        energy_level=f"{energy_level:.2f}",
        silence_minutes=f"{silence_minutes:.1f}",
        last_interaction=last_interaction,
        mood_description=mood_description
    )

# 增强版情感决策提示词模板（动态生成）
def get_enhanced_emotion_decision_prompt(
    character_name: str,
    character_age: int,
    personality: str,
    hobbies: str,
    boredom_level: float,
    loneliness_accumulation: float,
    social_desire: float,
    mood_score: float,
    energy_level: float,
    chat_type: str,
    silence_minutes: float,
    recent_topics: str,
    atmosphere: str,
    inner_state_analysis: str
) -> str:
    """构建增强版情感决策提示词"""
    
    boredom_desc = get_boredom_description(boredom_level)
    loneliness_desc = get_loneliness_description(loneliness_accumulation)
    social_desc = get_social_description(social_desire)
    mood_desc = get_mood_description(mood_score)
    energy_desc = get_energy_description(energy_level)
    
    from src.config.prompt_loader import get_prompt, PromptCategory
    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "enhanced_emotion_decision.template",
        character_name=character_name,
        character_age=character_age,
        personality=personality,
        hobbies=hobbies,
        boredom_level=f"{boredom_level:.2f}",
        boredom_description=boredom_desc,
        loneliness_accumulation=f"{loneliness_accumulation:.2f}",
        loneliness_description=loneliness_desc,
        social_desire=f"{social_desire:.2f}",
        social_description=social_desc,
        mood_score=f"{mood_score:.2f}",
        mood_description=mood_desc,
        energy_level=f"{energy_level:.2f}",
        energy_description=energy_desc,
        chat_type=chat_type,
        silence_minutes=f"{silence_minutes:.1f}",
        recent_topics=recent_topics,
        atmosphere=atmosphere,
        inner_state_analysis=inner_state_analysis
    )

# 群聊氛围感知提示词
GROUP_ATMOSPHERE_SENSING_PROMPT = """作为群聊观察者，分析当前群聊氛围是否适合主动发言。

【最近对话内容】
{recent_messages}

【群聊状态】
- 活跃度: {activity_level}
- 参与人数: {participant_count}
- 话题类型: {topic_type}
- 沉默时长: {silence_duration}分钟

【氛围评估】
请评估以下维度（0-1分）：
1. 轻松程度 (0=很严肃, 1=很轻松)
2. 欢迎新话题 (0=不合适插话, 1=很欢迎新话题)  
3. 包容性 (0=排斥外来者, 1=很包容)
4. 活跃度 (0=冷场, 1=很活跃)

输出格式：
轻松程度：0.X
欢迎新话题：0.X  
包容性：0.X
活跃度：0.X
综合适合度：0.X
建议：适合/不适合主动发言
理由：[简短说明]"""

# 私聊情感共鸣提示词 - 已移动至 config
PRIVATE_EMOTIONAL_RESONANCE_PROMPT = ""

def get_mood_description(mood_score: float) -> str:
    """根据心情分数获取描述"""
    if mood_score >= 0.8:
        return "心情很棒，充满活力"
    elif mood_score >= 0.6:
        return "心情不错，比较开心"
    elif mood_score >= 0.4:
        return "心情一般，平静中带点小情绪"
    elif mood_score >= 0.2:
        return "心情有些低落，需要温暖"
    else:
        return "心情很差，感觉很沮丧"

def get_boredom_description(boredom_level: float) -> str:
    """根据无聊程度获取描述"""
    if boredom_level >= 0.8:
        return "非常无聊，急需找点有趣的事情"
    elif boredom_level >= 0.6:
        return "比较无聊，希望有人聊天"
    elif boredom_level >= 0.4:
        return "有点无聊，但还能忍受"
    elif boredom_level >= 0.2:
        return "稍微有点闲"
    else:
        return "一点都不无聊"

def get_loneliness_description(loneliness_level: float) -> str:
    """根据孤独感获取描述"""
    if loneliness_level >= 0.8:
        return "非常孤独，很渴望交流"
    elif loneliness_level >= 0.6:
        return "比较孤独，希望有人陪伴"
    elif loneliness_level >= 0.4:
        return "有点孤独感"
    elif loneliness_level >= 0.2:
        return "稍微有点寂寞"
    else:
        return "不感到孤独"

def get_social_description(social_desire: float) -> str:
    """根据社交欲望获取描述"""
    if social_desire >= 0.8:
        return "很想找人聊天"
    elif social_desire >= 0.6:
        return "比较想社交"
    elif social_desire >= 0.4:
        return "对社交有一定兴趣"
    elif social_desire >= 0.2:
        return "社交欲望不强"
    else:
        return "不太想社交"

def get_energy_description(energy_level: float) -> str:
    """根据能量水平获取描述"""
    if energy_level >= 0.8:
        return "精力充沛，活力满满"
    elif energy_level >= 0.6:
        return "精力不错"
    elif energy_level >= 0.4:
        return "精力一般"
    elif energy_level >= 0.2:
        return "有点疲惫"
    else:
        return "很累，精力不足"

def build_emotion_driven_prompt(
    emotion_state: dict,
    silence_minutes: float,
    chat_type: str = "group",
    recent_topics: str = "",
    user_name: str = "",
    atmosphere: str = "平静",
    **kwargs
) -> str:
    """构建情感驱动的主动发言决策提示词"""
    from src.config.config import global_config
    
    boredom_level = emotion_state.get("boredom_level", 0.0)
    loneliness_level = emotion_state.get("loneliness_accumulation", 0.0)
    social_desire = emotion_state.get("social_desire", 0.5)
    mood_score = emotion_state.get("mood_score", 0.5)
    energy_level = emotion_state.get("energy_level", 1.0)
    
    character_name = global_config.bot.nickname
    personality = global_config.personality.personality
    character_age = global_config.personality.character_age
    hobbies = global_config.personality.character_hobbies
    
    # 生成描述性文本
    boredom_desc = get_boredom_description(boredom_level)
    loneliness_desc = get_loneliness_description(loneliness_level)
    social_desc = get_social_description(social_desire)
    mood_desc = get_mood_description(mood_score)
    energy_desc = get_energy_description(energy_level)
    
    # 生成内心状态分析
    inner_analysis_parts = []
    
    if boredom_level > 0.6:
        inner_analysis_parts.append("感觉有些无聊，想找点有趣的事情做")
    
    if loneliness_level > 0.5:
        inner_analysis_parts.append("有点寂寞，希望能和别人交流")
    
    if social_desire > 0.6:
        inner_analysis_parts.append("很想和大家聊聊天")
        
    if mood_score > 0.7:
        inner_analysis_parts.append("心情很好，想和大家分享快乐")
    elif mood_score < 0.3:
        inner_analysis_parts.append("心情不太好，可能需要一些温暖")
    
    if energy_level < 0.3:
        inner_analysis_parts.append("感觉有点累，精力不太够")
    
    inner_state_analysis = "；".join(inner_analysis_parts) if inner_analysis_parts else "内心平静，没有特别强烈的感受"
    
    if chat_type == "private":
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "private_emotional_resonance.template",
            user_name=user_name,
            affection=kwargs.get("favor", kwargs.get("affection", 0)),
            trust=kwargs.get("trust", 0),
            relationship=kwargs.get("relationship", "普通"),
            last_interaction=kwargs.get("last_interaction", "普通对话"),
            silence_minutes=f"{silence_minutes:.1f}",
            missing_level=min(10, int(loneliness_level * 10)),
            worry_level=min(10, int((1.0 - mood_score) * 10)),
            sharing_desire=min(10, int(social_desire * 10)),
            social_need=min(10, int((boredom_level + loneliness_level) * 5)),
            recent_mood_state=mood_desc,
            emotional_reflection=inner_state_analysis
        )
    else:
        return get_enhanced_emotion_decision_prompt(
            character_name=character_name,
            character_age=character_age,
            personality=personality,
            hobbies=hobbies,
            boredom_level=boredom_level,
            loneliness_accumulation=loneliness_level,
            social_desire=social_desire,
            mood_score=mood_score,
            energy_level=energy_level,
            chat_type=chat_type,
            silence_minutes=silence_minutes,
            recent_topics=recent_topics or "无明显话题",
            atmosphere=atmosphere,
            inner_state_analysis=inner_state_analysis
        )
