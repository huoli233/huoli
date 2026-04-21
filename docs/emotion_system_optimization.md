# 情感系统彻底深度优化方案

## 文档信息
- 版本: 1.0
- 日期: 2026-04-17
- 状态: 进行中

---

## 一、优化核心原则

### 1.1 真实累积原则
**现状问题**: 大多数状态值使用 `value = max(value * 0.6, target)` 的衰减公式  
**优化方向**: 改为真实累积，不会自动衰减，只有通过特定行为才能改变

```python
# 错误示例（当前）
state.boredom = max(state.boredom * 0.6, target_boredom)

# 正确示例（优化后）
state.boredom = min(3.0, state.boredom + growth_rate)
# 只有通过互动才能降低：state.boredom -= 0.15
```

### 1.2 情境感知原则
**现状问题**: 状态值计算只依赖时间，不考虑环境  
**优化方向**: 所有状态值都受情境影响（群里热闹/冷清、话题新鲜/重复等）

### 1.3 内在时间原则
**现状问题**: 使用系统时间，没有内在节律  
**优化方向**: 建立内在时钟（昼夜节律、疲劳累积、社交饱腹）

### 1.4 涌现而非触发原则
**现状问题**: 阈值触发（>=0.65就触发）  
**优化方向**: 渐变融合，综合多个因素自然涌现决策

### 1.5 转化而非线性原则
**现状问题**: 状态值线性增减  
**优化方向**: 极端状态会转化为其他状态（太无聊→放弃，太孤独→独立）

---

## 二、分层优化方案

### 第一层：核心状态枚举（已优化 ✅）

#### 2.1.1 VitalityLevel（精力层级）- 6档
**状态**: 已定义，无需优化  
**说明**: 枚举值本身不需要优化，但计算逻辑需要

#### 2.1.2 SocialPosture（社交姿态）- 6档
**状态**: 已定义，无需优化

#### 2.1.3 MentalState（心理状态）- 10档
**状态**: 已定义，无需优化

#### 2.1.4 AttentionMode（注意力模式）- 6档
**状态**: 已定义，无需优化

#### 2.1.5 GroupEngagement（群组参与度）- 5档
**状态**: 已定义，无需优化

---

### 第二层：D1-D11 维度投票系统（待优化 ⚠️）

#### D1: 情绪三轴投票（EmotionAxisVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| patience_snapshot | 快照值，无累积 | 改为真实耐心值，消耗后缓慢恢复 | 高 |
| annoyance_snapshot | 快照值，自动衰减 | 改为真实厌烦值，正面互动才能降低 | 高 |
| fatigue_snapshot | 快照值 | 改为真实疲劳值，休息才能恢复 | 高 |
| emotion_proactive_willingness | 简单计算 | 综合三轴状态自然涌现 | 中 |

**优化代码示例**:
```python
# 耐心值真实消耗
class EmotionAxisVote:
    patience: float = 100.0  # 真实耐心值，不是快照
    
    def consume_patience(self, amount: float):
        """消耗耐心，不会自动恢复"""
        self.patience = max(0.0, self.patience - amount)
        
    def recover_patience(self, positive_interaction: bool):
        """只有通过正面互动才能恢复"""
        if positive_interaction:
            self.patience = min(100.0, self.patience + 5.0)
```

#### D2: 好感/信任双核投票（FondnessTrustVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| fondness_value | 线性增减 | 真实累积，伤害后难以恢复 | 高 |
| trust_value | 线性增减 | 真实累积，背叛后大幅降低 | 高 |
| fondness_level | 枚举计算 | 基于真实值动态判断 | 中 |
| trust_level | 枚举计算 | 基于真实值动态判断 | 中 |

**优化方案**:
```python
# 好感度真实累积，不会自动衰减
class FondnessTrustVote:
    fondness: float = 0.0  # -100 ~ 100，真实累积
    trust: float = 0.0     # -100 ~ 100，真实累积
    
    def increase_fondness(self, amount: float, context: str):
        """增加好感，但边际效应递减"""
        if self.fondness > 50:
            amount *= 0.5  # 好感高时增加变慢
        self.fondness = min(100.0, self.fondness + amount)
        
    def decrease_fondness(self, amount: float, is_betrayal: bool = False):
        """降低好感，背叛时大幅降低且难以恢复"""
        if is_betrayal:
            amount *= 3.0  # 背叛惩罚3倍
            self.trust_break_count += 1
        self.fondness = max(-100.0, self.fondness - amount)
```

#### D3: 频率控制投票（FrequencyVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| adjust_factor | 简单调整 | 基于内在时钟自然调节 | 中 |
| dynamic_threshold | 固定阈值 | 动态学习个人节奏 | 中 |
| consecutive_skip_count | 计数器 | 改为真实"被忽视感"累积 | 中 |

#### D4: 用户意愿检测投票（UserStateVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| engagement | 5档枚举 | 连续值，更细腻 | 低 |
| want_to_chat_confidence | 简单计算 | 基于历史互动模式学习 | 中 |
| msg_frequency | 实时计算 | 建立长期互动节奏模型 | 中 |

#### D5: 群体氛围投票（GroupAtmosphereVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| atmosphere_tier | 5档枚举 | 连续值 + 历史趋势 | 中 |
| topic_relevance | 简单计算 | 基于语义理解的深度相关度 | 高 |

#### D6: 能量链条投票（EnergyChainVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| chat_energy_ratio | 比例计算 | 真实能量池，消耗后需恢复 | 高 |
| thinking_energy_ratio | 比例计算 | 真实能量池，思考消耗能量 | 高 |
| recovery_damping | 固定阻尼 | 基于疲劳度的动态恢复 | 中 |

**优化方案**:
```python
class EnergyChainVote:
    chat_energy: float = 100.0  # 真实能量值
    thinking_energy: float = 100.0
    
    def consume_chat_energy(self, amount: float, complexity: float):
        """消耗聊天能量，复杂对话消耗更多"""
        actual_cost = amount * (1.0 + complexity * 0.5)
        self.chat_energy = max(0.0, self.chat_energy - actual_cost)
        
    def recover_energy(self, rest_time_min: float):
        """休息恢复能量，但疲劳度高时恢复慢"""
        fatigue_penalty = 1.0 - self.fatigue_level * 0.5
        recovery = rest_time_min * 2.0 * fatigue_penalty
        self.chat_energy = min(100.0, self.chat_energy + recovery)
```

#### D7: 社交值投票（SocialValueVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| social_score | 线性计算 | 真实累积，长期互动建立 | 高 |
| positive_interactions | 计数器 | 加权计数，近期互动权重高 | 中 |
| negative_interactions | 计数器 | 负面互动长期影响 | 中 |

#### D8: 心流等待投票（HeartStateVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| wait_phase | 3档枚举 | 连续等待焦虑值 | 中 |
| heartbeat_proactive_willingness | 简单计算 | 基于等待时长的自然涌现 | 中 |

#### D9: 系统校准（CalibrationReport）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| emotion_decay_factor | 固定衰减 | 基于互动质量的动态衰减 | 中 |
| trauma_heal_factor | 固定恢复 | 创伤不会完全愈合，留下疤痕 | 高 |

#### D10: 创伤投票（TraumaVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| trauma_index | 0-10线性 | 真实累积，不会自动愈合 | 高 |
| cognitive_impairment | 固定值 | 基于创伤指数的动态影响 | 高 |
| flashback_risk | 简单计算 | 基于相似情境的触发概率 | 高 |
| inner_chaos | 独立计算 | 创伤导致的真实混乱 | 高 |

**优化方案**:
```python
class TraumaVote:
    trauma_index: float = 0.0  # 0-10，真实累积，不会自动衰减
    
    def add_trauma(self, severity: float, context: str):
        """增加创伤，严重事件留下深刻印象"""
        # 创伤不会自动消失
        self.trauma_index = min(10.0, self.trauma_index + severity)
        self.trauma_memories.append({
            "context": context,
            "severity": severity,
            "timestamp": time.time()
        })
        
    def heal_trauma(self, therapy_time: float, positive_experiences: int):
        """创伤只能通过长期正面体验缓慢愈合"""
        if positive_experiences > 10:  # 需要足够多正面体验
            heal_amount = therapy_time * 0.001  # 非常缓慢的愈合
            self.trauma_index = max(0.0, self.trauma_index - heal_amount)
        # 即使愈合到0，也会留下"创伤疤痕"
        if self.trauma_index < 1.0 and self.trauma_index > 0:
            self.has_trauma_scar = True
```

#### D11: 伪装投票（SurfaceMaskVote）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| mask_strength | 固定值 | 基于真实内外差异的动态伪装 | 高 |
| mask_fatigue | 简单累积 | 真实疲劳，长期伪装会崩溃 | 高 |
| detection_risk | 简单计算 | 基于伪装质量的真实风险 | 中 |

**优化方案**:
```python
class SurfaceMaskVote:
    inner_state: Dict[str, float]  # 真实内在状态
    outer_expression: Dict[str, float]  # 外在表现
    
    def calculate_mask_strength(self) -> float:
        """伪装强度 = 内外差异度"""
        difference = abs(self.inner_state["mood"] - self.outer_expression["mood"])
        return min(1.0, difference * 2.0)
        
    def accumulate_mask_fatigue(self, mask_duration_min: float):
        """伪装时间越长，疲劳越高，越容易崩溃"""
        base_fatigue = mask_duration_min * 0.02
        # 内外差异越大，伪装越累
        difficulty = self.calculate_mask_strength()
        self.mask_fatigue = min(100.0, self.mask_fatigue + base_fatigue * (1.0 + difficulty))
        
        if self.mask_fatigue > 80.0:
            self.behavior_mode = MaskBehaviorMode.CRACKING
        if self.mask_fatigue > 95.0:
            self.behavior_mode = MaskBehaviorMode.BROKEN
```

---

### 第三层：Energy Manager 能量管理（待优化 ⚠️）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| chat_pool | 自动恢复 | 真实消耗，需主动恢复 | 高 |
| thinking_value | 自动恢复 | 真实消耗，思考消耗能量 | 高 |
| annoyance_level | 自动衰减 | 真实累积，正面互动降低 | 高 |
| activity_level | 线性变化 | 基于真实活跃度的自然波动 | 中 |
| social_value | 简单计算 | 基于互动质量的长期建立 | 中 |

---

### 第四层：Emotion Driven Core（已优化 ✅）

**已完成的优化**:
- ✅ boredom: 真实累积，可达3.0
- ✅ loneliness: 真实累积，可达2.5
- ✅ environmental_fatigue: 真实累积，可达2.0
- ✅ social_desire: 增加多因素抑制
- ✅ proactive_willingness: 内在时钟主导
- ✅ internal_clock: 新增内在时间系统
- ✅ withdrawal_tendency: 极端状态转化

---

### 第五层：User Emotion State（待优化 ⚠️）

#### 5.1 基础关系值

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| affection | 线性增减 | 真实累积，边际效应 | 高 |
| trust_score | 线性增减 | 真实累积，背叛惩罚 | 高 |
| annoyance | 自动衰减 | 真实累积，难以消除 | 高 |
| trauma_score | 自动衰减 | 真实累积，不会愈合 | 高 |

#### 5.2 情绪维度（13个）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| joy | 简单设置 | 基于正面体验的真实累积 | 中 |
| anger | 简单设置 | 基于负面体验的真实累积 | 中 |
| sadness | 简单设置 | 真实累积，时间缓慢淡化 | 中 |
| fear | 简单设置 | 基于威胁体验的真实累积 | 中 |

#### 5.3 心理指标

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| psychological_pressure | 自动衰减 | 真实累积，需放松降低 | 高 |
| mental_fatigue | 自动衰减 | 真实累积，需休息恢复 | 高 |
| inner_chaos | 独立计算 | 多因素导致的真实混乱 | 中 |
| surface_mask | 简单设置 | 基于内外差异的动态伪装 | 中 |

#### 5.4 调教/训练状态

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| submission_level | 简单计算 | 基于长期互动的真实建立 | 中 |
| training_progress | 线性增加 | 非线性，有平台期 | 低 |
| training_resistance | 固定值 | 基于性格的动态抵抗 | 低 |

---

### 第六层：Heart Flow 心流状态（待优化 ⚠️）

| 字段名 | 当前问题 | 优化方案 | 优先级 |
|--------|----------|----------|--------|
| curiosity_level | 固定计算 | 基于内容质量的动态变化 | 高 |
| avoidance_tendency | 只和摸鱼相关 | 综合厌烦、创伤、疲劳 | 高 |
| quiet_preference | 固定值 | 基于环境和个人状态的动态 | 中 |
| fatigue_accumulated | 简单累加 | 真实疲劳，恢复缓慢 | 高 |

---

## 三、实施路线图

### 第一阶段：核心状态优化（1-2周）
- [ ] 优化 D1 情绪三轴投票
- [ ] 优化 D2 好感/信任双核
- [ ] 优化 D6 能量链条
- [ ] 优化 D10 创伤系统

### 第二阶段：用户关系优化（2-3周）
- [ ] 优化 UserEmotionState 基础关系值
- [ ] 优化情绪维度（13个）
- [ ] 优化心理指标

### 第三阶段：情境感知优化（2-3周）
- [ ] 优化 D5 群体氛围
- [ ] 优化 D11 伪装系统
- [ ] 优化 Heart Flow 心流状态

### 第四阶段：系统集成测试（1-2周）
- [ ] 集成所有优化
- [ ] 系统测试
- [ ] 调优参数

---

## 四、关键优化代码模板

### 4.1 真实累积模板
```python
def accumulate_state(self, current: float, growth: float, max_val: float, 
                      context_factor: float = 1.0) -> float:
    """通用真实累积函数"""
    adjusted_growth = growth * context_factor
    
    if current < 1.0:
        return min(max_val, current + adjusted_growth)
    else:
        # 超过1.0后边际效应
        return min(max_val * 1.5, current + adjusted_growth * 0.25)
```

### 4.2 情境感知模板
```python
def calculate_context_factor(self, context: Dict) -> float:
    """计算情境影响因子"""
    factor = 1.0
    
    if context.get("group_activity", 0.5) > 0.6:
        factor *= 1.5  # 群里热闹
    elif context.get("group_activity", 0.5) < 0.2:
        factor *= 0.3  # 群里冷清
        
    if context.get("topic_repetitiveness", 0.5) > 0.7:
        factor *= 1.8  # 话题重复
    elif context.get("topic_repetitiveness", 0.5) < 0.3:
        factor *= 0.5  # 话题新鲜
        
    return factor
```

### 4.3 状态转化模板
```python
def check_state_transformation(self, state: EmotionState):
    """检查极端状态转化"""
    # 太无聊 → 放弃
    if state.boredom > 1.5:
        withdrawal = min(1.0, (state.boredom - 1.5) * 0.8)
        state.withdrawal_tendency = max(state.withdrawal_tendency, withdrawal)
        
    # 太孤独 → 独立
    if state.loneliness > 1.3:
        independence = min(1.0, (state.loneliness - 1.3) * 0.6)
        state.social_desire *= (1 - independence * 0.5)
```

---

## 五、预期效果

### 5.1 AI行为变化
- **更自然**: 不会机械地按时间触发，而是基于内在状态自然涌现
- **更真实**: 有真实的疲劳、饱腹、厌烦，需要休息和恢复
- **更个性化**: 基于长期互动建立独特的关系模式
- **更有深度**: 创伤不会消失，伪装会疲劳，有真实的心理活动

### 5.2 用户体验提升
- **更沉浸**: AI的行为更符合人类心理预期
- **更有挑战**: 需要真正经营和AI的关系
- **更有趣**: AI有自己的"性格"和"节奏"

---

## 六、风险评估

### 6.1 技术风险
- **复杂度增加**: 系统更加复杂，调试难度增加
- **性能影响**: 更多计算可能影响性能
- **参数调优**: 大量参数需要精细调优

### 6.2 缓解措施
- 分阶段实施，逐步验证
- 增加详细的日志记录
- 建立参数调优工具

---

## 七、附录

### 7.1 已优化文件清单
- ✅ `emotion_driven_core.py` - 核心情感驱动

### 7.2 待优化文件清单
- ⚠️ `vote_types.py` - D1-D11维度投票
- ⚠️ `energy_manager.py` - 能量管理
- ⚠️ `emotion_tracker.py` - 用户情感追踪
- ⚠️ `heartFC_chat_enhanced.py` - 心流增强

### 7.3 关键参数参考
```python
# 真实累积参数
BOREDOM_GROWTH_RATE = 0.05  # 每分钟增长
BOREDOM_MAX = 3.0           # 最大值
LONELINESS_GROWTH_RATE = 0.08
LONELINESS_MAX = 2.5
FATIGUE_GROWTH_RATE = 0.015
FATIGUE_MAX = 1.0

# 内在时钟参数
CIRCADIAN_CYCLE = 24.0      # 昼夜周期（小时）
FATIGUE_RECOVERY_RATE = 0.008  # 疲劳恢复速率
SATIATION_RECOVERY_RATE = 0.02  # 饱腹恢复速率

# 阈值参数
PROACTIVE_THRESHOLD = 0.72  # 主动发言阈值
FATIGUE_INHIBITION = 0.7    # 疲劳抑制阈值
SATIATION_INHIBITION = 0.6  # 饱腹抑制阈值
```

---

**文档结束**

*最后更新: 2026-04-17*
