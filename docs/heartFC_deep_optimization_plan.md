# heartFC_chat_enhanced.py 深度分析与优化方案

## 文档信息
- 文件: `src/chat/heart_flow/heartFC_chat_enhanced.py`
- 目标: 动态参数配置 + 状态监控 + 行为预测
- 原则: 不创建新模块，仅在现有文件增强

---

## 一、当前问题分析

### 1.1 固定值问题（需要改为动态百分比）

在 `heartFC_chat_enhanced.py` 中搜索到的所有固定阈值：

#### 摸鱼/休息相关固定值（第1460行附近）
```python
# 当前代码 - 固定值
elif loafing >= 0.78 and quiet_preference >= 0.55 and avoidance >= 0.40:
    # ...
elif loafing >= 0.58 and (quiet_preference >= 0.45 or avoidance >= 0.45):
    # ...
elif loafing >= 0.45 and max(boredom, loneliness, social_willingness) < 0.45:
    # ...
```

**问题**: 这些 0.78、0.58、0.45、0.55、0.40 全是硬编码

#### 无聊值相关固定值（第3576行附近）
```python
if _boredom > 55:      # 固定55
if _boredom > 40:      # 固定40
if _boredom < 35:      # 固定35
```

#### 厌烦值相关固定值（第10890行附近）
```python
if _boredom_val > 0.6:   # 固定0.6
if _boredom_val > 0.3:   # 固定0.3
```

#### 能量相关固定值（第3790行附近）
```python
if _loafing_sup > 0.40:  # 固定0.40
```

### 1.2 需要改为动态配置的所有固定值清单

| 位置 | 当前固定值 | 含义 | 建议改为 |
|------|-----------|------|----------|
| 第1463行 | `loafing >= 0.78` | 高摸鱼触发摸鱼模式 | `config.loafing.high_threshold` |
| 第1468行 | `loafing >= 0.58` | 中摸鱼触发窥屏 | `config.loafing.medium_threshold` |
| 第1492行 | `loafing >= 0.45` | 低摸鱼影响静默 | `config.loafing.low_threshold` |
| 第1463行 | `quiet_preference >= 0.55` | 安静偏好高 | `config.preference.quiet_high` |
| 第1468行 | `quiet_preference >= 0.45` | 安静偏好中 | `config.preference.quiet_medium` |
| 第1463行 | `avoidance >= 0.40` | 回避度高 | `config.avoidance.high` |
| 第1468行 | `avoidance >= 0.45` | 回避度中 | `config.avoidance.medium` |
| 第1492行 | `< 0.45` | 社交意愿低 | `config.social.low_willingness` |
| 第3576行 | `_boredom > 55` | 无聊触发窥屏 | `config.boredom.peek_trigger` |
| 第3614行 | `_boredom > 40` | 无聊触发浏览 | `config.boredom.scan_trigger` |
| 第3735行 | `_boredom < 35` | 无聊低时降级 | `config.boredom.downgrade_trigger` |
| 第10890行 | `_boredom_val > 0.6` | 高无聊降低回复 | `config.boredom.reply_suppress_high` |
| 第10895行 | `_boredom_val > 0.3` | 中无聊降低回复 | `config.boredom.reply_suppress_medium` |
| 第15253行 | `_boredom > 0.75` | 高无聊降激活条 | `config.boredom.activation_reduce` |
| 第18882行 | `_boredom_val > 0.5` | 触发无聊漂移 | `config.boredom.drift_trigger` |
| 第3790行 | `_loafing_sup > 0.40` | 摸鱼抑制社交 | `config.loafing.social_inhibit` |

---

## 二、动态参数配置系统设计方案

### 2.1 配置类设计（添加到现有文件顶部）

在 `heartFC_chat_enhanced.py` 的 import 区域之后，class 定义之前，添加：

```python
@dataclass
class DynamicThresholdConfig:
    """动态阈值配置 - 所有固定值集中管理"""
    
    # === 摸鱼系统阈值 ===
    class LoafingConfig:
        high_threshold: float = 0.78      # 高摸鱼触发摸鱼模式
        medium_threshold: float = 0.58     # 中摸鱼触发窥屏
        low_threshold: float = 0.45        # 低摸鱼影响静默
        social_inhibit: float = 0.40       # 摸鱼抑制社交意愿
    
    # === 偏好系统阈值 ===
    class PreferenceConfig:
        quiet_high: float = 0.55           # 安静偏好高
        quiet_medium: float = 0.45         # 安静偏好中
    
    # === 回避系统阈值 ===
    class AvoidanceConfig:
        high: float = 0.40                # 回避度高
        medium: float = 0.45              # 回避度中
    
    # === 社交系统阈值 ===
    class SocialConfig:
        low_willingness: float = 0.45     # 社交意愿低
    
    # === 无聊系统阈值 ===
    class BoredomConfig:
        peek_trigger: float = 55           # 无聊触发窥屏 (>这个值)
        scan_trigger: float = 40           # 无聊触发浏览
        downgrade_trigger: float = 35      # 无聊低时降级 (<这个值)
        reply_suppress_high: float = 0.6   # 高无聊降低回复
        reply_suppress_medium: float = 0.3 # 中无聊降低回复
        activation_reduce: float = 0.75    # 高无聊降激活条
        drift_trigger: float = 0.5        # 触发无聊漂移
    
    # === 主动发言系统阈值 ===
    class ProactiveConfig:
        willingness_threshold: float = 0.72  # 主动意愿阈值
        fatigue_limit: float = 0.70          # 疲劳限制
        satiation_limit: float = 0.60        # 饱腹限制
        withdrawal_limit: float = 0.50       # 放弃限制
        context_min_richness: float = 0.25   # 内容质量最低要求
    
    # === 内在时钟参数 ===
    class InternalClockConfig:
        circadian_morning_start: float = 9.0    # 上午开始
        circadian_afternoon_end: float = 17.0   # 下午结束
        circadian_night_start: float = 22.0     # 夜间开始
        fatigue_growth_rate: float = 0.015       # 疲劳增长速率
        satiation_growth_rate: float = 0.04      # 饱腹增长速率
        recovery_rate: float = 0.008             # 恢复速率
    
    # 实例化子配置
    loafing: LoafingConfig = field(default_factory=LoafingConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    social: SocialConfig = field(default_factory=SocialConfig)
    boredom: BoredomConfig = field(default_factory=BoredomConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    internal_clock: InternalClockConfig = field(default_factory=InternalClockConfig)
    
    @classmethod
    def from_toml(cls, config_path: str) -> "DynamicThresholdConfig":
        """从TOML文件加载配置"""
        try:
            import tomllib
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return cls(**data.get("thresholds", {}))
        except Exception:
            return cls()  # 加载失败使用默认值
    
    def to_dict(self) -> dict:
        """导出为字典（用于Web API）"""
        return {
            "loafing": asdict(self.loafing),
            "preference": asdict(self.preference),
            "avoidance": asdict(self.avoidance),
            "social": asdict(self.social),
            "boredom": asdict(self.boredom),
            "proactive": asdict(self.proactive),
            "internal_clock": asdict(self.internal_clock)
        }
```

### 2.2 配置文件格式（config/thresholds.toml）

创建 `config/thresholds.toml`：

```toml
[thresholds]

[thresholds.loafing]
high_threshold = 0.60
medium_threshold = 0.45
low_threshold = 0.35
social_inhibit = 0.35

[thresholds.preference]
quiet_high = 0.50
quiet_medium = 0.40

[thresholds.avoidance]
high = 0.35
medium = 0.40

[thresholds.social]
low_willingness = 0.40

[thresholds.boredom]
peek_trigger = 50
scan_trigger = 38
downgrade_trigger = 30
reply_suppress_high = 0.55
reply_suppress_medium = 0.28
activation_reduce = 0.70
drift_trigger = 0.48

[thresholds.proactive]
willingness_threshold = 0.70
fatigue_limit = 0.65
satiation_limit = 0.55
withdrawal_limit = 0.45
context_min_richness = 0.22

[thresholds.internal_clock]
circadian_morning_start = 9.0
circadian_afternoon_end = 17.0
circadian_night_start = 22.0
fatigue_growth_rate = 0.012
satiation_growth_rate = 0.035
recovery_rate = 0.007
```

---

## 三、状态监控API接口设计方案

### 3.1 在现有类中添加状态导出方法

在 `HeartFCChatEnhanced` 类中添加以下方法：

```python
def export_full_state(self) -> dict:
    """导出完整状态（用于Web监控）"""
    state = {
        # 基础信息
        "timestamp": time.time(),
        "channel_id": self.channel_id,
        
        # 核心情感状态
        "emotions": {
            "boredom": getattr(self, '_current_boredom', 0),
            "loneliness": getattr(self, '_current_loneliness', 0),
            "environmental_fatigue": getattr(self, '_current_env_fatigue', 0),
            "social_desire": getattr(self, '_current_social_desire', 0.5),
            "proactive_willingness": getattr(self, '_current_proactive_will', 0),
            "withdrawal_tendency": getattr(self, '_current_withdrawal', 0),
        },
        
        # 内在时钟
        "internal_clock": {
            "circadian_phase": getattr(self, '_circadian_phase', 12.0),
            "mental_fatigue": getattr(self, '_mental_fatigue', 0),
            "social_satiation": getattr(self, '_social_satiation', 0),
            "continuous_active_min": getattr(self, '_continuous_active_min', 0),
        },
        
        # Governor状态
        "governors": {
            "behavior": {
                "reply_mode": getattr(self, '_last_reply_mode', 'observe'),
                "interrupt_level": getattr(self, '_last_interrupt_level', 'ignore'),
                "allow_generation": getattr(self, '_last_allow_gen', False),
            },
            "rest": {
                "posture": getattr(self, '_last_posture', 'active'),
                "should_rest": getattr(self, '_should_rest', False),
                "should_loaf": getattr(self, '_should_loaf', False),
            },
            "model": {
                "tier": getattr(self, '_last_model_tier', 'small'),
                "rate_limited": getattr(self, '_rate_limited', False),
            }
        },
        
        # 组件可见性状态
        "component_visibility": self._calculate_component_visibility(),
        
        # 当前配置
        "config": self.config.to_dict() if hasattr(self, 'config') else {},
    }
    return state

def _calculate_component_visibility(self) -> dict:
    """计算组件显示/隐藏状态"""
    visibility = {}
    
    # 疲劳条 - 始终显示
    visibility["fatigue_bar"] = True
    
    # 摸鱼指示器 - 仅当摸鱼值>0.3时显示
    loafing = getattr(self, '_current_loafing', 0)
    visibility["loafing_indicator"] = loafing > 0.30
    
    # 无聊警告 - 仅当无聊>0.5时显示
    boredom = getattr(self, '_current_boredom', 0)
    visibility["boredom_warning"] = boredom > 0.50
    
    # 孤独指示器 - 仅当孤独>0.4时显示
    loneliness = getattr(self, '_current_loneliness', 0)
    visibility["loneliness_indicator"] = loneliness > 0.40
    
    # 创伤标记 - 仅当有创伤时显示
    trauma = getattr(self, '_trauma_score', 0)
    visibility["trauma_marker"] = trauma > 10
    
    # 预测面板 - 仅当主动意愿>0.4时显示
    willingness = getattr(self, '_current_proactive_will', 0)
    visibility["prediction_panel"] = willingness > 0.40
    
    # 调教状态 - 仅当调教阶段>0时显示
    training = getattr(self, '_training_stage', 0)
    visibility["training_status"] = training > 0
    
    return visibility
```

### 3.2 用户行为预测方法

在同一个类中添加：

```python
def predict_user_behavior(self, user_id: str, context: dict) -> dict:
    """
    预测用户下一步行为意图
    
    返回:
    - likely_actions: 可能的行为列表及概率
    - intent: 用户意图推断
    - confidence: 预测置信度
    - factors: 影响因素
    """
    factors = []
    actions = []
    
    # 获取用户状态
    user_state = self._get_user_state(user_id)
    if not user_state:
        return {"error": "用户状态不存在"}
    
    # 1. 基于好感度预测
    affection = user_state.affection
    if affection > 60:
        actions.append({"action": "主动互动", "probability": 0.7, "reason": "好感度高"})
        factors.append({"factor": "好感度", "value": affection, "impact": "positive"})
    elif affection < -20:
        actions.append({"action": "可能离开或沉默", "probability": 0.6, "reason": "好感度低"})
        factors.append({"factor": "好感度", "value": affection, "impact": "negative"})
    
    # 2. 基于最近交互频率预测
    recent_interactions = self._get_recent_interaction_count(user_id, minutes=10)
    if recent_interactions > 5:
        actions.append({"action": "继续活跃", "probability": 0.75, "reason": "近期活跃"})
        factors.append({"factor": "近期活跃度", "value": recent_interactions, "impact": "positive"})
    elif recent_interactions == 0 and context.get("silence_min", 0) > 15:
        actions.append({"action": "可能已离开", "probability": 0.5, "reason": "长时间无响应"})
        factors.append({"factor": "沉默时长", "value": context.get("silence_min", 0), "impact": "negative"})
    
    # 3. 基于消息内容预测
    last_msg = self._get_last_user_message(user_id)
    if last_msg:
        if "?" in last_msg:
            actions.insert(0, {"action": "等待回答", "probability": 0.85, "reason": "最后消息是问题"})
        elif any(word in last_msg for word in ["拜拜", "走了", "睡觉", "下线"]):
            actions.append({"action": "即将离开", "probability": 0.8, "reason": "告别关键词"})
        elif len(last_msg) < 5:
            actions.append({"action": "可能简短回应", "probability": 0.6, "reason": "短消息模式"})
    
    # 4. 基于时间模式预测
    hour = time.localtime().tm_hour
    if 23 <= hour or hour <= 6:
        actions.append({"action": "可能去睡觉", "probability": 0.65, "reason": "深夜时段"})
        factors.append({"factor": "时间段", "value": f"{hour}:00", "impact": "sleep_time"})
    
    # 计算置信度
    action_count = len(actions)
    if action_count == 0:
        actions = [{"action": "无法预测", "probability": 0.3, "reason": "数据不足"}]
        confidence = 0.2
    elif action_count == 1:
        confidence = actions[0]["probability"]
    else:
        confidence = max(a["probability"] for a in actions) * 0.8
    
    return {
        "user_id": user_id,
        "predicted_actions": sorted(actions, key=lambda x: x["probability"], reverse=True)[:3],
        "intent": self._infer_intent(actions),
        "confidence": confidence,
        "factors": factors,
        "prediction_time": time.time()
    }

def _infer_intent(self, actions: list) -> str:
    """从预测动作推断用户意图"""
    intents = []
    for action in actions:
        act = action["action"]
        if "互动" in act or "活跃" in act:
            intents.append("社交")
        elif "离开" in act or "走" in act or "睡" in act:
            intents.append("退出")
        elif "等待" in act:
            intents.append("期待回应")
        elif "无法" in act:
            intents.append("未知")
    
    if intents:
        return max(set(intents), key=intents.count)
    return "观察中"
```

---

## 四、Web端数据输出接口

### 4.1 HTTP API端点（添加到现有的FastAPI路由）

如果项目已有FastAPI应用，添加以下路由：

```python
@app.get("/api/state/{channel_id}")
async def get_robot_state(channel_id: str):
    """获取机器人完整状态"""
    heart_fc = get_heartFC_instance(channel_id)
    return heart_fc.export_full_state()

@app.get("/api/config/{channel_id}")
async def get_current_config(channel_id: str):
    """获取当前配置"""
    heart_fc = get_heartFC_instance(channel_id)
    return heart_fc.config.to_dict()

@app.put("/api/config/{channel_id}")
async def update_config(channel_id: str, config_data: dict):
    """更新配置（动态调整阈值）"""
    heart_fc = get_heartFC_instance(channel_id)
    # 更新配置逻辑...
    return {"status": "success", "new_config": heart_fc.config.to_dict()}

@app.get("/api/predict/user/{user_id}")
async def predict_user_action(user_id: str, channel_id: str):
    """预测用户行为"""
    heart_fc = get_heartFC_instance(channel_id)
    context = {"silence_min": heart_fc.get_silence_duration()}
    return heart_fc.predict_user_behavior(user_id, context)

@app.get("/api/components/visibility/{channel_id}")
async def get_component_visibility(channel_id: str):
    """获取组件显示/隐藏状态"""
    heart_fc = get_heartFC_instance(channel_id)
    return heart_fc._calculate_component_visibility()
```

### 4.2 WebSocket实时推送（可选增强）

如果需要实时推送，在现有WebSocket基础上添加：

```python
# 在心跳处理中添加状态推送
async def broadcast_state_update(self):
    """广播状态更新给所有连接的客户端"""
    state = self.export_full_state()
    await self.websocket_manager.broadcast({
        "type": "state_update",
        "data": state,
        "timestamp": time.time()
    })
```

---

## 五、组件动态显示/隐藏规则

### 5.1 组件列表及显示条件

| 组件名 | 显示条件 | 隐藏条件 | 说明 |
|--------|----------|----------|------|
| **fatigue_bar** | 始终显示 | 从不隐藏 | 疲劳进度条 |
| **loafing_indicator** | 摸鱼值 > 0.30 | 摸鱼值 ≤ 0.30 | 摸鱼状态指示器 |
| **boredom_warning** | 无聊值 > 0.50 | 无聊值 ≤ 0.50 | 无聊警告徽章 |
| **loneliness_indicator** | 孤独值 > 0.40 | 孤独值 ≤ 0.40 | 孤独感指示器 |
| **trauma_marker** | 创伤分数 > 10 | 创伤分数 ≤ 10 | 创伤标记 |
| **prediction_panel** | 主动意愿 > 0.40 | 主动意愿 ≤ 0.40 | 发言预测面板 |
| **training_status** | 调教阶段 > 0 | 调教阶段 = 0 | 调教状态面板 |
| **social_saturation** | 社交饱腹 > 0.30 | 社交饱腹 ≤ 0.30 | 社交饱腹指示器 |
| **internal_clock** | 始终显示 | 从不隐藏 | 内在时钟显示 |

### 5.2 前端判断逻辑示例

```javascript
// 根据后端返回的visibility对象控制组件显示
function renderComponent(componentName, visibility, children) {
    const isVisible = visibility[componentName];
    
    if (!isVisible) {
        return null; // 不渲染
    }
    
    // 渐入动画
    return (
        <div 
            className={`component ${componentName}`}
            style={{
                opacity: isVisible ? 1 : 0,
                transition: 'opacity 0.3s ease'
            }}
        >
            {children}
        </div>
    );
}
```

---

## 六、实施步骤清单

### 步骤1：添加配置类（30分钟）
- [ ] 在 `heartFC_chat_enhanced.py` 顶部添加 `DynamicThresholdConfig` 类
- [ ] 创建 `config/thresholds.toml` 配置文件
- [ ] 在 `__init__` 方法中加载配置

### 步骤2：替换固定值（1小时）
- [ ] 找到所有 `>= 0.78` 替换为 `self.config.loafing.high_threshold`
- [ ] 找到所有 `>= 0.58` 替换为 `self.config.loafing.medium_threshold`
- [ ] 以此类推替换所有固定值
- [ ] 测试确保功能正常

### 步骤3：添加状态导出方法（30分钟）
- [ ] 添加 `export_full_state()` 方法
- [ ] 添加 `_calculate_component_visibility()` 方法
- [ ] 确保返回正确的数据结构

### 步骤4：添加用户预测方法（45分钟）
- [ ] 添加 `predict_user_behavior()` 方法
- [ ] 实现 `_infer_intent()` 辅助方法
- [ ] 测试预测准确性

### 步骤5：添加API接口（30分钟）
- [ ] 添加HTTP GET端点获取状态
- [ ] 添加HTTP PUT端点更新配置
- [ ] 添加用户预测端点
- [ ] （可选）添加WebSocket推送

### 步骤6：前端集成（2小时）
- [ ] 创建状态展示页面
- [ ] 实现组件动态显示/隐藏
- [ ] 添加配置调整界面
- [ ] 添加预测结果展示

---

## 七、测试验证点

### 7.1 配置加载测试
- [ ] 默认配置正常工作
- [ ] TOML配置能正确覆盖默认值
- [ ] 配置修改后立即生效

### 7.2 动态阈值测试
- [ ] 修改 `loafing.high_threshold` 为 0.5 后，0.6的摸鱼值应触发摸鱼模式
- [ ] 修改 `boredom.peek_trigger` 为 30 后，35的无聊值应触发窥屏
- [ ] 所有阈值都能通过API动态调整

### 7.3 组件显隐测试
- [ ] 摸鱼值0.25时，loafing_indicator隐藏
- [ ] 摸鱼值0.35时，loafing_indicator显示
- [ ] 创伤为0时，trauma_marker隐藏
- [ ] 创伤为15时，trauma_marker显示

### 7.4 用户预测测试
- [ ] 好感度高的用户预测为"可能继续互动"
- [ ] 长时间未响应用户预测为"可能已离开"
- [ ] 发送问题的用户预测为"等待回答"
- [ ] 深夜时段用户预测为"可能去睡觉"

---

## 八、注意事项

1. **向后兼容**: 如果配置加载失败，必须使用默认值，不能报错
2. **性能考虑**: `export_full_state()` 不要过于频繁调用，建议缓存1秒
3. **线程安全**: 配置修改需要加锁，防止并发问题
4. **日志记录**: 配置变更应该记录日志，方便调试
5. **边界检查**: 所有百分比参数必须在 0.0 ~ 1.0 范围内

---

**文档结束**

*基于 heartFC_chat_enhanced.py 深度分析*
*版本: 1.0*
*日期: 2026-04-17*
