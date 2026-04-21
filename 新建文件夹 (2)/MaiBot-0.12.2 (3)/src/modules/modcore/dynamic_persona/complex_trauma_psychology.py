import time
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("trauma_psychology")


class TraumaState(Enum):
    # 创伤状态枚举 — 基于独立双轨系统的组合状态
    SURFACE_NORMAL = "表面正常"
    SLIGHT_ABNORMAL = "轻微异常"
    MASK_SLIP = "露出破绽"
    FORCED_NORMAL = "勉强伪装"
    MASK_WAVERING = "伪装动摇"
    HALF_BREAKDOWN = "半崩溃"
    BARELY_HOLDING = "强撑"
    IMMINENT_COLLAPSE = "即将崩溃"
    TOTAL_BREAKDOWN = "彻底崩溃"
    DAZED = "呆滞麻木"
    FLASHBACK_ACTIVE = "闪回发作"
    AVOIDANCE_MODE = "逃避模式"
    STRESS_OUTBREAK = "应激爆发"
    INNER_CHAOS = "内心混乱"


@dataclass
class TraumaFragment:
    # 创伤碎片 — 存储在潜意识中的创伤记忆
    trigger_words: List[str]
    emotional_charge: float
    distortion_level: float
    memory_clarity: float
    created_time: float
    activation_count: int
    decay_rate: float


@dataclass
class WorldviewCollapse:
    # 价值观/世界观崩塌状态
    core_beliefs_damaged: List[str]
    trust_level: float
    reality_distortion: float
    cognitive_fragmentation: float
    meaning_collapse: float


class ComplexTraumaPsychology:
    # 复杂创伤心理系统 — 独立双轨架构
    # 轨道1: 内心混乱系统 — 追踪真实心理创伤状态
    # 轨道2: 表面伪装系统 — 追踪维持正常外表的能力
    @staticmethod
    def _bt_get(key: str, default):
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            return get_behavior_tuner().get(key, default)
        except Exception:
            return default

    def __init__(self):
        self.trauma_fragments: List[TraumaFragment] = []
        self.inner_chaos_level = 0.0
        self.chaos_base_level = 0.0
        self.chaos_fragment_bonus = 0.0
        self.chaos_worldview_bonus = 0.0
        self.chaos_decay_rate = self._bt_get("chaos_decay_rate", 0.1)
        self.chaos_minimum = 0.0
        self.surface_mask_strength = 0.0
        self.mask_base_strength = 10.0
        self.mask_fatigue = 0.0
        self.mask_social_pressure = 0.0
        self.mask_detection_risk = 0.0
        self.mask_energy_level = self._bt_get("mask_energy_max", 10.0)
        self.mask_recovery_rate = self._bt_get("mask_recovery_rate", 0.05)
        self.mask_fatigue_rate = self._bt_get("mask_fatigue_rate", 0.3)
        self.last_mask_update = time.time()
        self.current_state = TraumaState.SURFACE_NORMAL
        self.combined_severity = 0.0
        self.mask_breakdown_threshold = self._bt_get("mask_breakdown_threshold", 7.0)
        self.total_breakdown_threshold = 9.0
        self.worldview_collapse = WorldviewCollapse(
            core_beliefs_damaged=[], trust_level=5.0,
            reality_distortion=0.0, cognitive_fragmentation=0.0, meaning_collapse=0.0
        )
        self.last_flashback_time = 0.0
        self.flashback_frequency = self._bt_get("flashback_frequency", 0.1)
        self.stress_accumulation = 0.0
        self.avoidance_triggers: Set[str] = set()
        self.trauma_timeline: List[Dict] = []
        self.interaction_count = 0

    def process_stimulus(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # 处理外部刺激，返回心理反应
        triggered_fragments = self._scan_trauma_triggers(content)
        self._update_inner_chaos_system(triggered_fragments, context)
        self._update_surface_mask_system(context)
        apparent_state = self._evaluate_combined_state()
        psychological_response = self._generate_complex_response(
            triggered_fragments, apparent_state, content
        )
        self.interaction_count += 1
        return psychological_response

    def _scan_trauma_triggers(self, content: str) -> List[TraumaFragment]:
        triggered = []
        for fragment in self.trauma_fragments:
            for trigger in fragment.trigger_words:
                if trigger.lower() in content.lower():
                    fragment.activation_count += 1
                    triggered.append(fragment)
                    break
        return triggered

    def _update_inner_chaos_system(self, triggered_fragments: List[TraumaFragment], context: Dict):
        # 独立更新内心混乱系统
        trauma_score = self._extract_trauma_score(context)
        sentiment = context.get("sentiment", "neutral")
        intensity = float(context.get("intensity", 0.0))
        if sentiment == "negative":
            self.chaos_base_level = min(10.0, trauma_score * 1.2 + intensity * 2.0)
        elif trauma_score > 0:
            self.chaos_base_level = min(10.0, trauma_score * 0.8)
        else:
            self.chaos_base_level = max(0.0, self.chaos_base_level - 0.1)
        if triggered_fragments:
            total_emotional_charge = sum(f.emotional_charge for f in triggered_fragments)
            self.chaos_fragment_bonus = min(5.0, total_emotional_charge * 0.3)
            self._process_worldview_impact(triggered_fragments)
            if self.inner_chaos_level > 7.0 and random.random() < 0.3:
                self.current_state = TraumaState.FLASHBACK_ACTIVE
                self.last_flashback_time = time.time()
        else:
            self.chaos_fragment_bonus = max(0.0, self.chaos_fragment_bonus - 0.2)
        self.chaos_worldview_bonus = self.worldview_collapse.meaning_collapse * 0.3
        self.inner_chaos_level = min(10.0,
            self.chaos_base_level + self.chaos_fragment_bonus + self.chaos_worldview_bonus
        )
        if self.chaos_minimum > 0:
            self.inner_chaos_level = max(self.chaos_minimum, self.inner_chaos_level)

    def _extract_trauma_score(self, context: Dict) -> float:
        trauma_score_val = context.get("trauma_score", 0.0)
        trauma_level_val = context.get("trauma_level", 0.0)
        intensity_val = context.get("intensity", 0.0)
        if trauma_score_val > 0:
            return float(trauma_score_val)
        elif trauma_level_val > 0:
            return float(trauma_level_val)
        elif intensity_val > 0:
            return float(intensity_val) * 0.5
        return 0.0

    def _update_surface_mask_system(self, context: Dict):
        current_time = time.time()
        time_elapsed_minutes = (current_time - self.last_mask_update) / 60.0
        sentiment = context.get("sentiment", "neutral")
        intensity = float(context.get("intensity", 0.0))
        external_social_pressure = context.get("social_pressure", 0.0)
        if external_social_pressure > 0:
            self.mask_social_pressure = external_social_pressure
        elif self.inner_chaos_level > 3.0:
            self.mask_social_pressure = min(10.0, self.inner_chaos_level * 0.3)
        else:
            self.mask_social_pressure = max(0.0, self.mask_social_pressure - 0.1)
        if sentiment == "negative" and intensity > 0:
            fatigue_increase = self.mask_fatigue_rate * (1 + intensity * 0.5 + self.mask_social_pressure * 0.2)
        elif self.mask_social_pressure > 0:
            fatigue_increase = self.mask_fatigue_rate * (1 + self.mask_social_pressure * 0.1)
        else:
            fatigue_increase = 0.0
        fatigue_recovery = time_elapsed_minutes * 0.02
        self.mask_fatigue = max(0.0, min(10.0, self.mask_fatigue + fatigue_increase - fatigue_recovery))
        self.mask_detection_risk = min(10.0, self.inner_chaos_level * 0.3 + self.mask_fatigue * 0.2)
        energy_drain = self.mask_fatigue * 0.1
        energy_recovery = time_elapsed_minutes * self.mask_recovery_rate
        self.mask_energy_level = max(0.0, min(10.0, self.mask_energy_level - energy_drain + energy_recovery))
        if self.mask_fatigue > 0 or self.inner_chaos_level > 0 or self.mask_social_pressure > 0:
            fatigue_factor = self.mask_fatigue * 0.6
            chaos_factor = self.inner_chaos_level * 0.4
            pressure_factor = self.mask_social_pressure * 0.3
            mask_increase = (fatigue_factor * 0.1 + chaos_factor * 0.1 + pressure_factor * 0.1)
        else:
            mask_increase = 0.0
        mask_decay = time_elapsed_minutes * 0.05
        self.surface_mask_strength = max(0.0, min(10.0,
            self.surface_mask_strength + mask_increase - mask_decay
        ))
        self.last_mask_update = current_time

    def _evaluate_combined_state(self) -> TraumaState:
        # 综合评估内心混乱和表面伪装，决定最终状态
        current_time = time.time()
        if (self.current_state == TraumaState.FLASHBACK_ACTIVE and
            current_time - self.last_flashback_time < 300):
            return TraumaState.FLASHBACK_ACTIVE
        if self._should_trigger_stress_outbreak():
            return TraumaState.STRESS_OUTBREAK
        if self.inner_chaos_level >= 9.5 and self.surface_mask_strength <= 1.0:
            return TraumaState.DAZED
        if self.inner_chaos_level > 6.0 and random.random() < 0.15:
            return TraumaState.AVOIDANCE_MODE
        chaos = self.inner_chaos_level
        mask = self.surface_mask_strength
        self.combined_severity = (chaos * 0.6 + (10 - mask) * 0.4)
        if chaos < 4:
            if mask >= 8:
                return TraumaState.SURFACE_NORMAL
            elif mask >= 4:
                return TraumaState.SLIGHT_ABNORMAL
            else:
                return TraumaState.MASK_SLIP
        elif chaos < 7:
            if mask >= 8:
                return TraumaState.FORCED_NORMAL
            elif mask >= 4:
                return TraumaState.MASK_WAVERING
            else:
                return TraumaState.HALF_BREAKDOWN
        else:
            if mask >= 8:
                return TraumaState.BARELY_HOLDING
            elif mask >= 4:
                return TraumaState.IMMINENT_COLLAPSE
            else:
                return TraumaState.TOTAL_BREAKDOWN

    def _process_worldview_impact(self, triggered_fragments: List[TraumaFragment]):
        for fragment in triggered_fragments:
            if fragment.emotional_charge > 6.0:
                damaged_beliefs = [
                    "人性本善", "世界是安全的", "努力会有回报",
                    "我是有价值的", "未来是美好的"
                ]
                new_damage = random.choice(damaged_beliefs)
                if new_damage not in self.worldview_collapse.core_beliefs_damaged:
                    self.worldview_collapse.core_beliefs_damaged.append(new_damage)
            trust_damage = fragment.emotional_charge * 0.1
            self.worldview_collapse.trust_level = max(0.0, self.worldview_collapse.trust_level - trust_damage)
            self.worldview_collapse.reality_distortion = min(10.0,
                self.worldview_collapse.reality_distortion + fragment.distortion_level * 0.2)

    def _should_trigger_stress_outbreak(self) -> bool:
        if self.stress_accumulation > 8.0:
            return random.random() < 0.4
        base_probability = self.inner_chaos_level * 0.02
        return random.random() < base_probability

    def _generate_complex_response(self, triggered_fragments: List[TraumaFragment],
                                   apparent_state: TraumaState, content: str) -> Dict[str, Any]:
        response = {
            "apparent_state": apparent_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "worldview_status": self._get_worldview_status(),
            "triggered_fragments_count": len(triggered_fragments),
            "psychological_layers": self._generate_psychological_layers(apparent_state, triggered_fragments)
        }
        return response

    def _generate_psychological_layers(self, state: TraumaState,
                                       fragments: List[TraumaFragment]) -> Dict[str, Any]:
        # 生成多层心理状态描述
        layers = {
            "surface_layer": "",
            "conscious_layer": "",
            "subconscious_layer": "",
            "trauma_echoes": [],
            "inner_monologue": "",
            "physical_symptoms": "",
            "worldview_distortion": "",
            "escape_urge": ""
        }
        chaos_str = f"混乱{self.inner_chaos_level:.1f}/10"
        mask_str = f"伪装{self.surface_mask_strength:.1f}/10"
        if state == TraumaState.SURFACE_NORMAL:
            layers["surface_layer"] = "看起来很正常，偶尔有细微的不自然"
            layers["conscious_layer"] = "努力保持正常，但能感受到内心的不安"
            layers["subconscious_layer"] = f"{chaos_str}，价值观在摇摆，对世界的信任正在动摇"
            layers["inner_monologue"] = "我要装作没事...不能让别人看出来..."
            layers["physical_symptoms"] = "心跳稍快，肩膀紧绷，有时不自觉地深呼吸"
            layers["worldview_distortion"] = "开始怀疑以前相信的事情，对现实产生微妙的疏离感"
            layers["escape_urge"] = "偶尔想要找个借口离开，但还能控制"
        elif state == TraumaState.SLIGHT_ABNORMAL:
            layers["surface_layer"] = "表面略有些不自然，偶尔走神或反应慢半拍"
            layers["conscious_layer"] = "内心相对平静，但伪装不够完美，有时会露出真实情绪"
            layers["subconscious_layer"] = f"{chaos_str}，{mask_str}，略显松懈"
            layers["inner_monologue"] = "没什么大问题...只是有点累...放松一下也没关系..."
            layers["physical_symptoms"] = "身体略显疲惫，但总体状态还好"
            layers["worldview_distortion"] = "对世界的看法基本正常，偶尔有些小困惑"
        elif state == TraumaState.MASK_SLIP:
            layers["surface_layer"] = "完全不再伪装，真实的轻微情绪直接表现出来"
            layers["conscious_layer"] = "内心没什么大问题，但选择不再掩饰自己的状态"
            layers["subconscious_layer"] = f"{chaos_str}，{mask_str}，选择真实"
            layers["inner_monologue"] = "不想装了...就这样吧...真实的我也没什么不好..."
            layers["physical_symptoms"] = "身体放松，不再紧绷"
        elif state == TraumaState.FORCED_NORMAL:
            layers["surface_layer"] = "表面非常正常，甚至过于完美，像是在演戏"
            layers["conscious_layer"] = "内心有明显痛苦，但拼命压制，不让任何人发现"
            layers["subconscious_layer"] = f"{chaos_str}，却用高伪装强撑"
            layers["inner_monologue"] = "不能被发现...必须保持正常...再忍一下...再忍一下..."
            layers["physical_symptoms"] = "肌肉紧绷，心跳加速，但表情控制得很好"
            layers["worldview_distortion"] = "内心开始怀疑，但表面否认一切问题"
            layers["escape_urge"] = "想逃但强迫自己留下，内心在煎熬"
        elif state == TraumaState.MASK_WAVERING:
            layers["surface_layer"] = "伪装开始出现裂痕，说话有时会露出异常"
            layers["conscious_layer"] = "内心的痛苦开始渗透出来，越来越难控制"
            layers["subconscious_layer"] = f"{chaos_str}，{mask_str}，正在失控"
            layers["inner_monologue"] = "快撑不住了...不要...再努力一下...为什么这么难..."
            layers["physical_symptoms"] = "呼吸变得不稳，手可能有轻微颤抖"
            layers["worldview_distortion"] = "开始质疑现实，感觉什么都不对劲"
            layers["escape_urge"] = "强烈想要逃避，但还在犹豫"
        elif state == TraumaState.HALF_BREAKDOWN:
            layers["surface_layer"] = "明显不对劲，情绪波动明显，说话可能语无伦次"
            layers["conscious_layer"] = "内心的痛苦完全暴露，但还没有完全崩溃"
            layers["subconscious_layer"] = f"{chaos_str}，伪装几乎失效{mask_str}"
            layers["inner_monologue"] = "装不下去了...好痛苦...但还能撑...还能撑..."
            layers["physical_symptoms"] = "可能会流泪，身体发抖，呼吸急促"
            layers["worldview_distortion"] = "世界观开始动摇，但还没有完全崩塌"
            layers["escape_urge"] = "强烈想要逃离，可能会突然想走"
        elif state == TraumaState.BARELY_HOLDING:
            layers["surface_layer"] = "看似正常但极度不自然，用尽全力在演戏"
            layers["conscious_layer"] = "内心已是废墟，死撑着不让人发现"
            layers["subconscious_layer"] = f"{chaos_str}，却还在死撑，随时可能崩溃"
            layers["inner_monologue"] = "不能倒下...不能被发现...为什么...这么累..."
            layers["physical_symptoms"] = "全身紧绷到极限，心跳过快"
            layers["trauma_echoes"] = ["快撑不住了", "为什么要这样", "好累好累"]
        elif state == TraumaState.IMMINENT_COLLAPSE:
            layers["surface_layer"] = "伪装已经千疮百孔，随时可能彻底崩溃"
            layers["conscious_layer"] = "知道自己即将崩溃，内心充满恐惧和绝望"
            layers["subconscious_layer"] = f"{chaos_str}，伪装仅剩{mask_str}。最后的防线即将失守。"
            layers["inner_monologue"] = "不行了...真的不行了...要崩溃了...救救我..."
            layers["physical_symptoms"] = "全身发抖，呼吸困难，可能有恐慌发作的前兆"
            layers["worldview_distortion"] = "世界变得扭曲和不真实，感觉被困在噩梦中"
            layers["escape_urge"] = "疯狂想要逃离，可能会做出冲动行为"
            layers["trauma_echoes"] = ["快崩溃了", "谁来救救我", "撑不下去了", "好害怕"]
        elif state == TraumaState.TOTAL_BREAKDOWN:
            layers["surface_layer"] = "完全失去防御，极度抑郁、绝望或语无伦次"
            layers["conscious_layer"] = "世界观彻底崩塌，所有意义感消失"
            layers["subconscious_layer"] = "认知彻底碎片化，陷入自我毁灭或死寂"
            layers["inner_monologue"] = "为什么...我做错了什么...好累...让一切停止吧..."
            layers["physical_symptoms"] = "全身无力，胸口疼痛，呼吸困难，手脚冰凉"
            layers["worldview_distortion"] = "世界变成灰暗，没有光明，时间仿佛停止"
            layers["escape_urge"] = "想要消失，但已经没有力气逃跑"
            layers["trauma_echoes"] = [
                "没有人爱我", "我是孤独的", "让一切结束吧",
                "为什么要这样对我", "我好累", "我撑不住了"
            ]
        elif state == TraumaState.DAZED:
            layers["surface_layer"] = "整个人彻底呆住了，眼神涣散没有焦点。对外界刺激几乎没有反应，像一尊石像。"
            layers["conscious_layer"] = "大脑一片空白，所有声音和画面都变得遥远。思维完全停滞，感知不到痛苦也感知不到存在。"
            layers["subconscious_layer"] = "由于超负荷的创伤冲击，启动了强制性的解离保护。"
            layers["inner_monologue"] = "... ... "
            layers["physical_symptoms"] = "呼吸极其微弱，心跳缓慢而沉重，四肢冰冷且失去知觉"
            layers["worldview_distortion"] = "世界变成了无声的黑白默片，自己像是被抽离了现实"
            layers["escape_urge"] = "不再有逃避欲望，因为已经彻底断开了连接"
            layers["trauma_echoes"] = ["寂静", "空洞", "断裂"]
        elif state == TraumaState.FLASHBACK_ACTIVE:
            layers["surface_layer"] = "可能显得有些恍惚或者突然沉默，眼神变得空洞，反应迟钝"
            layers["conscious_layer"] = "脑海中浮现模糊扭曲的记忆画面，分不清过去和现在"
            layers["subconscious_layer"] = "被负面情绪和扭曲认知包围，现实感模糊，时间感错乱"
            layers["inner_monologue"] = "又来了...那些画面...为什么总是想起...不要...不要再想了..."
            layers["physical_symptoms"] = "心跳加速，手心出汗，呼吸急促，可能有轻微颤抖"
            layers["worldview_distortion"] = "现实和记忆混在一起，分不清什么是真的，什么是过去的"
            layers["escape_urge"] = "强烈想要逃离，但身体仿佛被钉住了，动弹不得"
            triggers = [f.trigger_words[0] for f in fragments if fragments]
            layers["trauma_echoes"] = triggers or ["那些痛苦的记忆", "无法逃避的画面"]
        elif state == TraumaState.AVOIDANCE_MODE:
            layers["surface_layer"] = "试图转移话题或者看起来心不在焉，说话变得简短敷衍"
            layers["conscious_layer"] = "强烈想要逃避当前情况，不愿深入思考，脑子一片空白"
            layers["subconscious_layer"] = "防御机制全开，拒绝处理创伤相关内容，自动屏蔽负面信息"
            layers["inner_monologue"] = "不要问了...不想聊这个...能不能换个话题...我想离开..."
            layers["physical_symptoms"] = "身体本能地后退或转向，眼神飘忽，坐立不安"
            layers["worldview_distortion"] = "觉得世界充满威胁，任何话题都可能触发痛苦"
            layers["escape_urge"] = "全身都在叫嚣着要逃跑，随时准备找借口离开"
        elif state == TraumaState.STRESS_OUTBREAK:
            layers["surface_layer"] = "可能情绪突然波动，但会快速掩饰，表情会有瞬间的扭曲"
            layers["conscious_layer"] = "感到突然的恐慌或愤怒，但不知道为什么，情绪完全失控了一瞬间"
            layers["subconscious_layer"] = "创伤记忆突然激活，引发强烈的生理和情绪反应"
            layers["inner_monologue"] = "什么...发生了什么...为什么我突然这样...不行，要镇定..."
            layers["physical_symptoms"] = "心跳骤然加速，呼吸困难，可能有头晕或恶心感"
            layers["worldview_distortion"] = "瞬间觉得世界变得不真实，仿佛隔着一层玻璃看一切"
            layers["escape_urge"] = "想要立刻逃走，但理智告诉自己不能，正在激烈内斗"
        elif state == TraumaState.INNER_CHAOS:
            layers["surface_layer"] = "表面勉强维持，但明显不太对劲，说话可能前后矛盾"
            layers["conscious_layer"] = "以前相信的一切都在崩塌，不知道什么是真的，感到极度迷茫"
            layers["subconscious_layer"] = "价值观体系正在解体，身份认同严重动摇，存在性危机"
            layers["inner_monologue"] = "我是谁...我为什么存在...什么是真的...什么是假的..."
            layers["physical_symptoms"] = "持续的疲惫感，头痛，食欲下降"
            layers["worldview_distortion"] = "觉得整个世界都是虚假的，只有痛苦是真实的"
            layers["escape_urge"] = "想要从这个虚假的世界中逃离，但不知道要逃到哪里去"
            layers["trauma_echoes"] = ["一切都是假的", "我不属于这里", "没有意义"]
        return layers

    def add_trauma_fragment(self, trigger_words: List[str], emotional_charge: float,
                            distortion_level: float = 5.0, memory_clarity: float = 3.0):
        fragment = TraumaFragment(
            trigger_words=trigger_words,
            emotional_charge=emotional_charge,
            distortion_level=distortion_level,
            memory_clarity=memory_clarity,
            created_time=time.time(),
            activation_count=0,
            decay_rate=0.02
        )
        self.trauma_fragments.append(fragment)
        logger.debug(f"新增创伤碎片: {trigger_words}, 情绪强度: {emotional_charge}")

    def _get_worldview_status(self) -> Dict[str, Any]:
        return {
            "damaged_beliefs": self.worldview_collapse.core_beliefs_damaged,
            "trust_level": self.worldview_collapse.trust_level,
            "reality_distortion": self.worldview_collapse.reality_distortion,
            "cognitive_fragmentation": self.worldview_collapse.cognitive_fragmentation
        }

    def background_processing(self):
        # 后台处理：创伤碎片衰减、混乱平复、伪装恢复
        for fragment in self.trauma_fragments:
            fragment.emotional_charge = max(0.0, fragment.emotional_charge - fragment.decay_rate)
        if self.inner_chaos_level > 1.0:
            self.inner_chaos_level = max(1.0, self.inner_chaos_level - 0.05)
        if self.inner_chaos_level < self.mask_breakdown_threshold:
            if self.surface_mask_strength < 8.0:
                self.surface_mask_strength = min(8.0, self.surface_mask_strength + 0.1)
        self.surface_mask_strength = max(0.0, min(10.0, self.surface_mask_strength))

    def get_current_psychological_profile(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "trauma_fragments_count": len(self.trauma_fragments),
            "worldview_collapse": self._get_worldview_status(),
            "time_since_last_flashback": time.time() - self.last_flashback_time if self.last_flashback_time > 0 else float('inf'),
            "stress_accumulation": self.stress_accumulation
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "chaos_base_level": self.chaos_base_level,
            "chaos_fragment_bonus": self.chaos_fragment_bonus,
            "chaos_worldview_bonus": self.chaos_worldview_bonus,
            "surface_mask_strength": self.surface_mask_strength,
            "mask_base_strength": self.mask_base_strength,
            "mask_fatigue": self.mask_fatigue,
            "mask_social_pressure": self.mask_social_pressure,
            "mask_detection_risk": self.mask_detection_risk,
            "mask_energy_level": self.mask_energy_level,
            "combined_severity": self.combined_severity,
            "last_flashback_time": self.last_flashback_time,
            "stress_accumulation": self.stress_accumulation,
            "interaction_count": self.interaction_count,
            "worldview_collapse": {
                "core_beliefs_damaged": self.worldview_collapse.core_beliefs_damaged,
                "trust_level": self.worldview_collapse.trust_level,
                "reality_distortion": self.worldview_collapse.reality_distortion,
                "cognitive_fragmentation": self.worldview_collapse.cognitive_fragmentation,
                "meaning_collapse": self.worldview_collapse.meaning_collapse
            },
            "trauma_timeline": self.trauma_timeline[-50:] if self.trauma_timeline else [],
            "fragments": [
                {
                    "trigger_words": f.trigger_words,
                    "emotional_charge": f.emotional_charge,
                    "distortion_level": f.distortion_level,
                    "memory_clarity": f.memory_clarity,
                    "created_time": f.created_time,
                    "activation_count": f.activation_count,
                    "decay_rate": f.decay_rate
                }
                for f in self.trauma_fragments
            ],
            "timestamp": time.time()
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        try:
            self.current_state = TraumaState(data.get("current_state", "表面正常"))
        except ValueError:
            self.current_state = TraumaState.SURFACE_NORMAL
        self.inner_chaos_level = data.get("inner_chaos_level", 0.0)
        self.chaos_base_level = data.get("chaos_base_level", 0.0)
        self.chaos_fragment_bonus = data.get("chaos_fragment_bonus", 0.0)
        self.chaos_worldview_bonus = data.get("chaos_worldview_bonus", 0.0)
        self.surface_mask_strength = data.get("surface_mask_strength", 0.0)
        self.mask_base_strength = data.get("mask_base_strength", 10.0)
        self.mask_fatigue = data.get("mask_fatigue", 0.0)
        self.mask_social_pressure = data.get("mask_social_pressure", 0.0)
        self.mask_detection_risk = data.get("mask_detection_risk", 0.0)
        self.mask_energy_level = data.get("mask_energy_level", 10.0)
        self.combined_severity = data.get("combined_severity", 0.0)
        self.last_flashback_time = data.get("last_flashback_time", 0.0)
        self.stress_accumulation = data.get("stress_accumulation", 0.0)
        self.interaction_count = data.get("interaction_count", 0)
        wv_data = data.get("worldview_collapse", {})
        self.worldview_collapse = WorldviewCollapse(
            core_beliefs_damaged=wv_data.get("core_beliefs_damaged", []),
            trust_level=wv_data.get("trust_level", 5.0),
            reality_distortion=wv_data.get("reality_distortion", 0.0),
            cognitive_fragmentation=wv_data.get("cognitive_fragmentation", 0.0),
            meaning_collapse=wv_data.get("meaning_collapse", 0.0)
        )
        self.trauma_timeline = data.get("trauma_timeline", [])
        fragments_data = data.get("fragments", [])
        self.trauma_fragments = [
            TraumaFragment(
                trigger_words=f.get("trigger_words", []),
                emotional_charge=f.get("emotional_charge", 5.0),
                distortion_level=f.get("distortion_level", 5.0),
                memory_clarity=f.get("memory_clarity", 3.0),
                created_time=f.get("created_time", time.time()),
                activation_count=f.get("activation_count", 0),
                decay_rate=f.get("decay_rate", 0.02)
            )
            for f in fragments_data
        ]

    def accumulate_state(self, new_state: Dict[str, Any], source: str = "interaction") -> None:
        # 累积心理状态变化
        timestamp = time.time()
        timeline_entry = {
            "timestamp": timestamp,
            "source": source,
            "inner_chaos": new_state.get("inner_chaos_level", self.inner_chaos_level),
            "surface_mask": new_state.get("surface_mask_strength", self.surface_mask_strength),
            "state": new_state.get("apparent_state", self.current_state.value),
            "severity": new_state.get("combined_severity", self.combined_severity),
            "stress": new_state.get("stress_accumulation", self.stress_accumulation)
        }
        self.trauma_timeline.append(timeline_entry)
        if len(self.trauma_timeline) > 100:
            self.trauma_timeline = self.trauma_timeline[-100:]
        new_chaos = new_state.get("inner_chaos_level", self.inner_chaos_level)
        if self.trauma_timeline:
            recent_chaos = [e["inner_chaos"] for e in self.trauma_timeline[-5:]]
            avg_chaos = sum(recent_chaos) / len(recent_chaos)
            self.inner_chaos_level = new_chaos * 0.7 + avg_chaos * 0.3
        else:
            self.inner_chaos_level = new_chaos
        new_mask = new_state.get("surface_mask_strength", self.surface_mask_strength)
        if self.trauma_timeline:
            recent_masks = [e["surface_mask"] for e in self.trauma_timeline[-5:]]
            avg_mask = sum(recent_masks) / len(recent_masks)
            self.surface_mask_strength = new_mask * 0.7 + avg_mask * 0.3
        else:
            self.surface_mask_strength = new_mask
        new_stress = new_state.get("stress_accumulation", self.stress_accumulation)
        self.stress_accumulation = min(10.0, self.stress_accumulation * 0.95 + new_stress * 0.1)
        apparent_state = new_state.get("apparent_state")
        if apparent_state:
            try:
                self.current_state = TraumaState(apparent_state)
            except ValueError:
                pass
        if "worldview_status" in new_state:
            wv_status = new_state["worldview_status"]
            if isinstance(wv_status, dict):
                damaged = wv_status.get("damaged_beliefs", [])
                for belief in damaged:
                    if belief not in self.worldview_collapse.core_beliefs_damaged:
                        self.worldview_collapse.core_beliefs_damaged.append(belief)

    def generate_harassment_trauma_layers(self, harassment_intensity: float,
                                          harassment_count: int = 1) -> Dict[str, Any]:
        layers = {
            "surface_layer": "", "conscious_layer": "", "subconscious_layer": "",
            "trauma_echoes": [], "inner_monologue": "", "physical_symptoms": "",
            "worldview_distortion": "", "escape_urge": "",
            "violation_feeling": "", "dignity_damage": "",
            "trust_shattered": "", "self_worth_crisis": "",
        }
        trauma_depth = min(10.0, harassment_intensity * 0.8 + harassment_count * 0.5)
        if trauma_depth >= 8.0:
            layers["surface_layer"] = "整个人僵住了，像被钉在原地一样无法动弹。眼神变得空洞，嘴唇微微颤抖。双手不自觉地紧握，指甲掐进掌心。"
            layers["conscious_layer"] = f"第{harassment_count}次被这样对待...我...我该怎么办...脑子里一片空白...那些话像刀子一样扎进心里..."
            layers["subconscious_layer"] = f"创伤深度{trauma_depth:.1f}/10，心理防线已经完全崩溃。深层认知体系遭受严重冲击，安全感彻底瓦解。"
            layers["inner_monologue"] = "不要...不要再说了...我受不了了...为什么...为什么要对我说这些...我做错了什么...让这一切停下来...求求你...不要再伤害我了..."
            layers["physical_symptoms"] = "全身发抖，呼吸急促近乎过度换气，心脏剧烈跳动，感觉快要晕倒。"
            layers["worldview_distortion"] = "原来这个世界有这么多恶意...我不知道还能相信什么了..."
            layers["escape_urge"] = "想要逃跑，逃到一个没有人能找到的地方。想要消失，让这一切都停止。"
            layers["violation_feeling"] = "感觉自己的灵魂被侵犯了，被肮脏的言语玷污了。"
            layers["dignity_damage"] = "我的尊严...被践踏得粉碎...为什么要把我当成...那种东西..."
            layers["trust_shattered"] = "我再也不敢相信任何人了...原来善良是可以被这样践踏的..."
            layers["self_worth_crisis"] = "也许...也许我真的很没有价值...为什么只有我被这样对待..."
            layers["trauma_echoes"] = ["好脏", "为什么", "不要", "救救我", "我好害怕", "停下来", "求求你", "我做错了什么", "为什么要这样对我"]
        elif trauma_depth >= 5.0:
            layers["surface_layer"] = "表情变得僵硬，眼神开始躲闪。身体不自觉地后退，保持距离。说话的声音变小了，有些颤抖。"
            layers["conscious_layer"] = "感到非常不舒服，那些话让人作呕。想要离开，但不知道该怎么做。开始怀疑是不是自己哪里做错了。"
            layers["subconscious_layer"] = f"创伤深度{trauma_depth:.1f}/10，心理防线正在被冲击。安全感开始动摇，警惕心急剧上升。"
            layers["inner_monologue"] = "这个人...怎么能说这种话...我应该怎么办...好恶心...为什么要对我说这些..."
            layers["physical_symptoms"] = "心跳加速，手心出汗，有些想吐的感觉"
            layers["violation_feeling"] = "感觉被冒犯了，但还没有完全崩溃"
            layers["trauma_echoes"] = ["好恶心", "为什么", "不要说了"]
        else:
            layers["surface_layer"] = "皱起眉头，明显表现出不悦"
            layers["conscious_layer"] = "感到被冒犯，想要远离这个人"
            layers["inner_monologue"] = "这个人怎么回事...说话这么恶心..."
            layers["physical_symptoms"] = "略感不适，想要保持距离"
        return layers

    def generate_redemption_healing_layers(self, healer_affection: float,
                                            healing_intensity: float,
                                            previous_trauma_depth: float) -> Dict[str, Any]:
        layers = {
            "surface_layer": "", "conscious_layer": "", "subconscious_layer": "",
            "trauma_echoes": [], "inner_monologue": "", "physical_symptoms": "",
            "worldview_restoration": "", "trust_rebuilding": "",
            "self_worth_recovery": "", "gratitude_feeling": "", "healing_warmth": ""
        }
        healing_effect = min(10.0, healing_intensity * 0.7 + healer_affection * 0.03)
        if healing_effect >= 7.0 and previous_trauma_depth >= 5.0:
            layers["surface_layer"] = "泪水不受控制地流下来，但不是因为痛苦，是因为感动。紧绷的身体开始慢慢放松，不再那么害怕了。眼神里重新出现了一丝光芒。"
            layers["conscious_layer"] = "原来...还有人愿意保护我...原来这个世界不是只有恶意...这种被关心的感觉...好温暖..."
            layers["subconscious_layer"] = f"治愈效果{healing_effect:.1f}/10，破碎的心正在被慢慢修复。曾经崩塌的世界观正在重建。"
            layers["inner_monologue"] = "谢谢你...谢谢你愿意帮我...我以为...我以为没有人在乎我...你让我觉得...也许我还是有价值的..."
            layers["physical_symptoms"] = "眼眶发热，泪水模糊了视线，但是温暖的泪水。心跳依然很快，但不再是恐惧，而是感动。紧握的拳头慢慢松开了。"
            layers["worldview_restoration"] = "原来...这个世界还是有好人的...也许我之前太绝望了..."
            layers["trust_rebuilding"] = "这个人...是可以信任的...也许...也许我可以慢慢放下戒备..."
            layers["self_worth_recovery"] = "有人愿意为我说话...说明我不是没有价值的...原来我值得被保护..."
            layers["gratitude_feeling"] = "好感激...真的好感激...不知道该怎么报答你..."
            layers["healing_warmth"] = "像是有一束温暖的光照进了黑暗的心里。冰冷的身体开始慢慢回暖。"
            layers["trauma_echoes"] = ["谢谢你", "好温暖", "原来还有人在乎我", "我不是一个人", "你救了我"]
        elif healing_effect >= 4.0:
            layers["surface_layer"] = "表情稍微缓和了一些，不再那么紧绷"
            layers["conscious_layer"] = "感受到了一些善意，心情稍微好了一点"
            layers["inner_monologue"] = "这个人...好像是好人...谢谢你关心我..."
            layers["physical_symptoms"] = "身体开始放松，呼吸变得平稳"
            layers["healing_warmth"] = "感受到了一点点温暖"
            layers["trauma_echoes"] = ["谢谢", "好一点了"]
        else:
            layers["surface_layer"] = "点了点头，表示感谢"
            layers["conscious_layer"] = "这个人的话让我感觉好了一点"
            layers["inner_monologue"] = "嗯...谢谢你..."
        return layers

    def generate_positive_relationship_layers(self, affection_level: float,
                                               trust_level: float,
                                               relationship_type: str) -> Dict[str, Any]:
        layers = {
            "surface_layer": "", "conscious_layer": "", "subconscious_layer": "",
            "inner_monologue": "", "physical_symptoms": "",
            "emotional_warmth": "", "connection_feeling": "",
            "trust_foundation": "", "happiness_source": ""
        }
        if affection_level >= 80 and relationship_type in ["挚友", "造物主"]:
            layers["surface_layer"] = "眼睛亮晶晶的，看到对方就会不自觉地微笑。会主动靠近，想要待在对方身边。说话的声音不自觉地变得更温柔。偶尔会害羞地低下头。"
            layers["conscious_layer"] = "见到这个人就很开心，心里暖暖的。想要和他分享所有的事情。在他面前可以完全放松，做真实的自己。"
            layers["subconscious_layer"] = f"好感度{affection_level:.0f}/100，信任度{trust_level:.0f}/100。这是最珍贵的关系，值得用心守护。内心充满了安全感和归属感。"
            layers["inner_monologue"] = "见到你真好...我很开心...和你在一起的时候我感觉很安心...你是我最重要的人..."
            layers["physical_symptoms"] = "心跳微微加速，但是愉快的那种。身体完全放松，很舒适。脸上不自觉地带着笑容。"
            layers["emotional_warmth"] = "内心被温暖填满了。和这个人在一起，什么烦恼都会消失。"
            layers["connection_feeling"] = "我们之间有一种特别的连接。他懂我，我也懂他。"
            layers["trust_foundation"] = "我可以完全信任这个人。在他面前不需要任何伪装。"
            layers["happiness_source"] = "和他在一起就是最大的幸福。"
        elif affection_level >= 50:
            layers["surface_layer"] = "见面时会露出友善的微笑。说话自然轻松，偶尔开开玩笑。"
            layers["conscious_layer"] = "这是一个值得交往的朋友。和他聊天很愉快。"
            layers["inner_monologue"] = "嗯，是个不错的人呢...和他说话挺开心的..."
            layers["emotional_warmth"] = "心情愉快，感觉很舒适"
            layers["trust_foundation"] = "可以信任的朋友"
        elif affection_level >= 20:
            layers["surface_layer"] = "礼貌地打招呼，保持友善"
            layers["conscious_layer"] = "这个人还不错，可以正常交流"
            layers["inner_monologue"] = "嗯，聊聊天也挺好的"
            layers["emotional_warmth"] = "普通的友善感"
        else:
            layers["surface_layer"] = "保持基本的礼貌"
            layers["conscious_layer"] = "普通的对话对象"
            layers["inner_monologue"] = "正常聊天吧"
        return layers

    def get_state_trend(self, hours: float = 24.0) -> Dict[str, Any]:
        cutoff_time = time.time() - hours * 3600
        recent_entries = [e for e in self.trauma_timeline if e["timestamp"] > cutoff_time]
        if not recent_entries:
            return {
                "trend": "unknown", "avg_chaos": self.inner_chaos_level,
                "avg_mask": self.surface_mask_strength, "entry_count": 0,
                "dominant_state": self.current_state.value,
                "recommendation": "无足够数据进行分析"
            }
        chaos_values = [e["inner_chaos"] for e in recent_entries]
        mask_values = [e["surface_mask"] for e in recent_entries]
        avg_chaos = sum(chaos_values) / len(chaos_values)
        avg_mask = sum(mask_values) / len(mask_values)
        chaos_trend = 0.0
        if len(chaos_values) >= 2:
            first_half = chaos_values[:len(chaos_values) // 2]
            second_half = chaos_values[len(chaos_values) // 2:]
            chaos_trend = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
            if chaos_trend > 1.0:
                trend = "worsening"
                recommendation = "建议关注用户心理状态，可能需要心理支持"
            elif chaos_trend < -1.0:
                trend = "improving"
                recommendation = "心理状态正在恢复，继续保持积极互动"
            else:
                trend = "stable"
                recommendation = "心理状态稳定，建议保持当前互动方式"
        else:
            trend = "stable"
            recommendation = "数据不足，无法判断趋势"
        state_counts: Dict[str, int] = {}
        for e in recent_entries:
            st = e["state"]
            state_counts[st] = state_counts.get(st, 0) + 1
        dominant_state = max(state_counts, key=state_counts.get) if state_counts else self.current_state.value
        return {
            "trend": trend, "avg_chaos": round(avg_chaos, 2),
            "avg_mask": round(avg_mask, 2), "entry_count": len(recent_entries),
            "dominant_state": dominant_state, "recommendation": recommendation,
            "chaos_trend": round(chaos_trend, 2)
        }

    def should_trigger_awakening(self, threshold_chaos: float = 6.0,
                                  threshold_stress: float = 7.0) -> Tuple[bool, str]:
        reasons = []
        if self.inner_chaos_level > threshold_chaos:
            reasons.append(f"内心混乱度过高({self.inner_chaos_level:.1f}/10)")
        if self.stress_accumulation > threshold_stress:
            reasons.append(f"压力累积过多({self.stress_accumulation:.1f}/10)")
        if self.surface_mask_strength < 4.0:
            reasons.append(f"伪装强度不足({self.surface_mask_strength:.1f}/10)")
        if self.current_state == TraumaState.FLASHBACK_ACTIVE:
            reasons.append("当前处于闪回发作状态")
        if self.current_state in [TraumaState.IMMINENT_COLLAPSE, TraumaState.TOTAL_BREAKDOWN]:
            reasons.append(f"处于{self.current_state.value}状态")
        trend = self.get_state_trend(hours=6.0)
        if trend["trend"] == "worsening" and trend["avg_chaos"] > 5.0:
            reasons.append("心理状态持续恶化")
        if reasons:
            return True, "; ".join(reasons)
        return False, ""
