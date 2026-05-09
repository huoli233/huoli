import { useEffect, useMemo, useState } from "react";

import "./styles.css";

type OverviewChannel = {
  channel_id: string;
  chat_type?: string;
  chat_type_label?: string;
  class_name?: string;
  display_name?: string;
  idle_seconds?: number | null;
  is_internal_webui?: boolean;
  platform?: string;
  platform_label?: string;
  target_user_id?: string;
};

type ResidentCard = {
  label: string;
  icon: string;
  state: string;
  value: number;
  display_value: string;
  color: string;
};

type ActiveSignal = {
  key: string;
  label: string;
  family: string;
  family_label?: string;
  severity: string;
  value: number;
  display_value: string;
  trend: string;
  trend_label?: string;
  source_domain: string;
  source_domain_label?: string;
  icon: string;
  color: string;
};

type ParticipantImpact = {
  user_id: string;
  display_name: string;
  relationship_label: string;
  rapport_score: number;
  trust_score: number;
  irritation_load: number;
  trauma_load: number;
  pressure_load: number;
  chaos_load: number;
  mask_load: number;
  interaction_count: number;
  current_mood_hint: string;
  last_interaction_age_sec?: number;
  active_signals: string[];
  impact_rank: number;
  is_current_target?: boolean;
  in_current_scene?: boolean;
  recent_speaker?: boolean;
  recent_targeted_interaction?: boolean;
  source_scope?: string;
};

type SceneContext = {
  channel_id: string;
  scene_heat: string;
  scene_heat_label?: string;
  scene_heat_score: number;
  topic_focus: string[];
  active_user_count: number;
  session_phase: string;
  hot_count: number;
  warm_count: number;
};

type TimelineEvent = {
  at: number;
  label: string;
  detail: string;
  family: string;
};

type Presentation = {
  resident_overview: Record<string, ResidentCard>;
  active_signals: ActiveSignal[];
  resource_detail?: ResourceDetail;
  circadian_detail?: CircadianDetail;
  emotion_detail?: EmotionDetail;
  behavior_detail?: BehaviorDetail;
  timing_gate_detail?: TimingGateDetail;
  context_detail?: ContextDetail;
  memory_detail?: MemoryDetail;
  autonomy_detail?: AutonomyDetail;
  group_state_detail?: GroupStateDetail;
  current_user_detail?: CurrentUserDetail;
  attention_runtime_detail?: AttentionRuntimeDetail;
  safety_detail?: SafetyDetail;
  dynamic_trace?: DynamicTraceItem[];
  value_deltas?: Record<string, DynamicTraceItem>;
  initiative_state?: InitiativeState;
  scene_context: SceneContext;
  participant_impacts: ParticipantImpact[];
  timeline: TimelineEvent[];
  display_policy?: DisplayPolicy;
};

type Prediction = {
  speak_probability: number;
  probability_percent?: number;
  shared_verdict_id?: string;
  model_path?: string;
  should_reply?: boolean;
  complexity_label?: string;
  execution_stage?: string;
  execution_action?: string;
  runtime_sync_state?: string;
  runtime_sync_label?: string;
  content_source?: string;
  content_source_label?: string;
  eta_seconds: number;
  eta_label: string;
  generated_at?: number;
  deadline_at?: number;
  decision_label?: string;
  decision_reason?: string;
  content_direction: string;
  reply_length: string;
  tone: string;
  drive_score?: number;
  suppression_score?: number;
  gate_influence?: number;
  proactive_influence?: number;
  relationship_influence?: number;
  resource_influence?: number;
  scene_influence?: number;
  circadian_influence?: number;
  driving_factors: string[];
  suppressing_factors: string[];
};

type ResourceDetail = {
  energy_reserve_ratio: number;
  energy_phase: string;
  energy_phase_label?: string;
  chat_reserve: number;
  chat_capacity: number;
  chat_percent: number;
  thinking_reserve: number;
  thinking_capacity: number;
  thinking_percent: number;
  activity_index: number;
  irritation_load: number;
  social_field_score: number;
  last_recovery_source: string;
  last_irritation_relief_source: string;
};

type CircadianDetail = {
  phase: string;
  phase_label: string;
  time_band?: string;
  time_band_label?: string;
  time_band_description?: string;
  current_hour?: number;
  is_night: boolean;
  is_sleep_window?: boolean;
  is_night_social_window?: boolean;
  is_pressure_window?: boolean;
  mechanism_windows?: Record<string, {
    label?: string;
    range?: string;
    active?: boolean;
  }>;
  system_started_at?: number;
  last_evaluated_at?: number;
  sync_label?: string;
  sync_source?: string;
  is_sleeping: boolean;
  is_burnthrough: boolean;
  can_reply: boolean;
  drowsiness_value: number;
  sleep_debt: number;
  overnight_pressure: number;
  sleep_reserve: number;
  dawn_recovery_progress: number;
  peek_window_open: boolean;
  sleep_reply_used: number;
  sleep_reply_cap: number;
  remaining_sleep_replies: number;
  reply_quota_label: string;
  response_suppression: number;
  pressure_breakdown?: {
    total: number;
    chat_minutes: number;
    peek_minutes: number;
    think_intensity: number;
    interrupt_count: number;
    chat_component: number;
    peek_component: number;
    think_component: number;
    interrupt_component: number;
  };
  body_state_label: string;
  mood_hint: string;
  expression_style_label: string;
};

type EmotionDetail = {
  mood_bias: number;
  mood_label: string;
  curiosity_drive: number;
  social_desire: number;
  boredom_load: number;
  loneliness_load: number;
  environment_fatigue_load: number;
  initiative_drive: number;
  withdrawal_drive: number;
  energy_ratio: number;
  silence_seconds: number;
  unanswered_count: number;
  feeling_text: string;
};

type DecisionDetail = {
  reply_urgency: string;
  should_reply: boolean;
  decision_reason: string;
  detail_notes: string[];
  suggested_tone: string;
  decision_stage: string;
  confidence: number;
};

type ExecutionRuntimeDetail = {
  should_act: boolean;
  reply_sent: boolean;
  final_action: string;
  final_action_label?: string;
  execution_stage: string;
  execution_stage_label?: string;
  execution_reason: string;
  confidence: number;
  model_path: string;
  source: string;
  planner_action: string;
  blocker: string;
  blocking_factors: string[];
  driving_factors: string[];
};

type BehaviorGovernorDetail = {
  reply_mode: string;
  reply_mode_label: string;
  interrupt_level: string;
  interrupt_level_label: string;
  quote_policy: string;
  quote_policy_label: string;
  silence_policy: string;
  silence_policy_label: string;
  allow_generation: boolean;
  model_tier: string;
  model_tier_label: string;
  reason_codes: string[];
  reason_labels: string[];
};

type RestGovernorDetail = {
  posture: string;
  posture_label: string;
  interruption_policy: string;
  interruption_policy_label: string;
  should_rest: boolean;
  should_loaf: boolean;
  reason_codes: string[];
  reason_labels: string[];
};

type ModelGovernorDetail = {
  tier: string;
  tier_label: string;
  rate_limited: boolean;
  fallback_to_small: boolean;
  dynamic_cooldown_sec: number;
  dynamic_hourly_cap: number;
  reason_codes: string[];
  reason_labels: string[];
};

type TimingGateVerdict = {
  verdict_id: string;
  at: number;
  gate_result: string;
  gate_result_label?: string;
  stage: string;
  stage_label?: string;
  reason: string;
  source: string;
  final_action: string;
  final_action_label?: string;
  next_action: string;
  model_path: string;
  blocker: string;
  confidence: number;
};

type TimingGateDetail = {
  current: TimingGateVerdict;
  history: TimingGateVerdict[];
  history_count: number;
};

type BehaviorDetail = {
  watch_state: string;
  watch_state_label: string;
  flow_phase_label: string;
  reply_decision: DecisionDetail;
  execution_runtime: ExecutionRuntimeDetail;
  behavior_governor: BehaviorGovernorDetail;
  rest_governor: RestGovernorDetail;
  model_governor: ModelGovernorDetail;
};

type ContextDetail = {
  direct_target: boolean;
  quote_anchor: boolean;
  recent_human_activity: boolean;
  scene_suitable: boolean;
  watch_state: string;
  watch_state_label: string;
  attention_level: number;
  perception_engagement_pull: number;
  self_recent_messages_count: number;
  self_recent_actions_count: number;
  self_recent_events_count: number;
  memoir_phase: string;
  memoir_phase_label: string;
  memoir_consecutive_timeouts: number;
  current_target_user_id: string;
  reply_mode: string;
  reply_mode_label: string;
  interrupt_level: string;
  interrupt_level_label: string;
  silence_policy: string;
  silence_policy_label: string;
  behavior_reason_labels: string[];
  rest_reason_labels: string[];
};

type MemoryDetail = {
  ephemeral_short_term_count: number;
  planner_short_term_count: number;
  planner_long_term_count: number;
  long_term_total: number;
  long_term_by_category: Record<string, number>;
  memoir_phase: string;
  memoir_phase_label: string;
  memoir_exchange_tally: number;
  memoir_consecutive_timeouts: number;
  memoir_last_topic: string;
  memoir_last_mood: string;
  journal_count: number;
  governance_total: number;
  governance_utilization: number;
  governance_pending_evict: number;
  governance_last_op: string;
  health_score: number;
  health_grade: string;
  overload_gauge_level: string;
  overload_load_ratio: number;
  overload_amnesia_pressure: number;
  overload_emergency_needed: boolean;
  knowledge_entry_count: number;
  reactivation_summary: Record<string, number>;
};

type AutonomyDetail = {
  active_intentions: Array<{
    intent_id: string;
    kind: string;
    target_user: string;
    source: string;
    description: string;
    expected_outcome: string;
    effective_urgency: number;
    age_seconds: number;
    idle_seconds: number;
    attempt_count: number;
    fail_count: number;
    lifecycle: string;
  }>;
  intention_alive_count: number;
  intention_drive: number;
  max_intention_urgency: number;
  reward_score: number;
  pending_proactive_events: number;
  self_recent_messages_count: number;
  self_recent_actions_count: number;
  self_recent_events_count: number;
  background_event_type: string;
  background_event_label: string;
  background_event_priority: number;
  background_budget_remaining: number;
};

type GroupStateDetail = {
  scene_heat_label: string;
  scene_heat_score: number;
  messages_per_minute: number;
  active_user_count: number;
  participant_diversity: number;
  interaction_quality: number;
  complexity_level: number;
  social_density: number;
  suitable_to_join: boolean;
  join_unsuitable_reason: string;
  dominant_speaker: string;
  active_topic_count: number;
  topic_focus: string[];
  thread_count: number;
  session_phase: string;
  vexation: number;
  weariness: number;
  vitality: number;
  atmosphere_label: string;
  hot_count: number;
  warm_count: number;
};

type CurrentUserDetail = {
  user_id: string;
  display_name: string;
  relationship_label: string;
  rapport_score: number;
  trust_score: number;
  irritation_load: number;
  pressure_load: number;
  trauma_load: number;
  chaos_load: number;
  mask_load: number;
  interaction_count: number;
  current_mood_hint: string;
};

type AttentionRuntimeDetail = {
  state_label: string;
  visibility_threshold: number;
  process_ratio: number;
  peek_desire: number;
  interrupt_tolerance: number;
  silence_tolerance: number;
  time_since_last_look: number;
  consecutive_peeks_without_action: number;
  look_budget_state: number;
  process_budget_state: number;
  openness: number;
};

type SafetyDetail = {
  safety_level: string;
  safety_score: number;
  blocked: boolean;
  dominant_threat: string;
  bar_penalty: number;
  threat_evidence_summary: string;
};

type DynamicTraceItem = {
  key: string;
  label: string;
  current: number;
  previous: number;
  delta: number;
  trend_label: string;
  last_change_reason: string;
};

type InitiativeState = {
  boredom_load: number;
  loneliness_load: number;
  environment_fatigue_load: number;
  initiative_drive: number;
  withdrawal_drive: number;
  effect_label: string;
  chat_timing_label: string;
  effect_summary: string;
};

type DisplayPolicy = {
  resident: string[];
  active: string[];
  detail: string[];
  hidden: string[];
};

type MonitorPacket = {
  channel_id: string;
  snapshot_kind?: string;
  updated_at: number;
  server_time?: number;
  state_version?: number;
  stale_cache?: boolean;
  snapshot_source?: string;
  cache_reason?: string;
  domains: {
    circadian_rhythm?: {
      expression_style?: string;
      expression_style_label?: string;
    };
    [key: string]: unknown;
  };
  presentation: Presentation;
  prediction: Prediction;
};

type OverviewPayload = {
  snapshot_kind?: string;
  updated_at: number;
  server_time?: number;
  state_version?: number;
  stale_cache?: boolean;
  snapshot_source?: string;
  cache_reason?: string;
  active_count: number;
  hidden_internal_count?: number;
  channels: OverviewChannel[];
};

const severityRank: Record<string, number> = {
  critical: 5,
  high: 4,
  medium: 3,
  low: 2,
  info: 1,
};

const fallbackDisplayPolicy: DisplayPolicy = {
  resident: ["精力储备", "内在心情", "注意状态", "社交姿态", "安全护盾", "流转阶段", "场景热度", "发言预测"],
  active: ["无聊度/孤独感/疲劳感/退场倾向/主动意愿", "情绪低落/好奇心/社交欲显著变化", "浅睡/深睡/熬穿/清晨恢复/睡眠债", "烦躁/压力/创伤/混乱/伪装", "关系好感/信任显著偏高或偏低", "冷却窗口/等待时长/重新接入/会话升温"],
  detail: ["运行资源的聊天值和思考值", "时间线窗口、睡眠债、困意和熬夜压力", "情绪状态、主动倾向和当前感受", "行为机制和上下文感知", "会话整体状态", "当前对象状态", "安全护盾细节", "值变化原因", "发言预测的驱动和抑制因素"],
  hidden: ["内部阈值", "调试原因", "缓存字段", "旧命名残留", "纯计数器原值"],
};

function formatClock(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function percent(value: number | undefined | null): string {
  const safe = Math.max(0, Math.min(1, Number(value ?? 0)));
  return `${(safe * 100).toFixed(1)}%`;
}

function fixed(value: number | undefined | null, digits = 1): string {
  return Number(value ?? 0).toFixed(digits);
}

function humanSceneHeat(value: string | undefined): string {
  return {
    heated: "高热",
    lively: "活跃",
    normal: "正常",
    quiet: "偏安静",
    dead: "低活跃",
  }[value ?? ""] ?? (value || "-");
}

function humanExecutionStage(value: string | undefined): string {
  return {
    self_echo_gate: "自身消息回流跳过",
    short_batch_skip: "消息未达阈值",
    watch_gate_blackout: "黑屏观看阻断",
    peek_gate_observe: "窥屏后继续观察",
    night_gate_skip: "夜间节律阻断",
    autonomy_guard_skip: "自治守卫阻断",
    planner_cooldown_skip: "规划器冷却跳过",
    pipeline_early_exit: "输入早退出",
    dashboard_hard_block: "仪表盘硬阻断",
    pattern_route_skip: "群体模式硬路由",
    scene_constraint_skip: "会话场景硬约束",
    full_pipeline_entry: "完整管线入口",
    decision_runtime_skip: "初裁直接跳过",
    voice_action_rest: "内心要求休息",
    voice_action_disengage: "内心要求放下会话",
    voice_action_lurk: "内心要求潜水观察",
    legacy_block: "门控明确阻断",
    llm_autonomous_allow: "自主大模型放行",
    llm_autonomous_hold: "自主大模型保持观察",
    inner_voice_priority: "内心优先规则修正",
    post_voice_reluctant: "独白抗拒保持观察",
    post_gate_hold: "证据不足保持观察",
    post_gate_reply: "门控放行后转入回复",
    post_relation_block: "关系极端阻断",
    post_relation_guard: "关系情绪收缩",
    post_wait_intent_block: "等待意图压制",
    post_psychology_block: "心理门槛收缩",
    post_probability_block: "频率概率收缩",
    post_burst_block: "连续发言保护",
    post_negative_emotion_block: "负面情绪回避",
    pre_execution: "前置裁定阶段",
    reply_sent: "已成功发送",
    reply_aborted: "进入执行后中止",
    final_no_action: "最终未执行动作",
  }[value ?? ""] ?? (value || "-");
}

function channelOptionLabel(channel: OverviewChannel): string {
  const typeLabel = channel.chat_type_label ?? (channel.chat_type === "group" ? "群聊" : channel.chat_type === "private" ? "私聊" : "会话");
  const platformLabel = channel.platform_label ?? (channel.platform === "webui" ? "本地测试" : channel.platform || "本地");
  const name = channel.display_name || channel.channel_id.slice(0, 8);
  return `${typeLabel} · ${platformLabel} · ${name}`;
}

function conversationTypeLabel(channel: OverviewChannel | undefined): string {
  if (!channel) {
    return "等待会话";
  }
  return channel.chat_type_label ?? (channel.chat_type === "group" ? "群聊" : channel.chat_type === "private" ? "私聊" : "会话");
}

function conversationName(channel: OverviewChannel | undefined): string {
  if (!channel) {
    return "暂无活跃会话";
  }
  return channel.display_name || channel.target_user_id || channel.channel_id.slice(0, 12);
}

function conversationScopeLabel(channel: OverviewChannel | undefined): string {
  if (!channel) {
    return "会话";
  }
  return channel.chat_type === "group" ? "群聊" : channel.chat_type === "private" ? "私聊" : "会话";
}

function runtimeSyncLabel(prediction: Prediction | undefined): string {
  return prediction?.runtime_sync_label ?? "等待裁定同步";
}

function countText(value: number | undefined | null, unit: string): string {
  return `${Math.round(Number(value ?? 0))}${unit}`;
}

function hourText(value: number | undefined | null): string {
  const hour = Math.max(0, Math.min(23, Math.round(Number(value ?? new Date().getHours()))));
  return `${String(hour).padStart(2, "0")}:00`;
}

function mechanismWindowText(
  windows: CircadianDetail["mechanism_windows"] | undefined,
  key: string,
  fallback: string,
): string {
  const item = windows?.[key];
  const label = item?.label ?? fallback;
  const range = item?.range ?? "--:-- - --:--";
  return `${label} ${range}`;
}

function timestampText(value: number | undefined | null): string {
  const raw = Number(value ?? 0);
  if (raw <= 0) {
    return "等待同步";
  }
  return formatClock(raw);
}

function yesNo(value: boolean | undefined | null): string {
  return value ? "是" : "否";
}

function modelPathLabel(value: string | undefined): string {
  return {
    skip: "不调用模型",
    small: "小模型",
    large: "大模型",
    small_fallback: "小模型兜底",
  }[value ?? ""] ?? (value || "未裁定");
}

function expressionStyleLabel(value: string | undefined): string {
  return {
    normal: "正常",
    drowsy: "困倦短句",
    stubborn: "硬撑克制",
    irritable: "被扰烦躁",
    soft_night: "夜间放轻",
    arousal: "短时亢奋",
    burnthrough: "熬穿失衡",
  }[value ?? ""] ?? (value || "正常");
}

function circadianExpressionStyle(packet: MonitorPacket | null, detail: CircadianDetail | undefined): string {
  return expressionStyleLabel(
    detail?.expression_style_label ??
      packet?.domains?.circadian_rhythm?.expression_style_label ??
      packet?.domains?.circadian_rhythm?.expression_style,
  );
}

function connectionLabel(value: string): string {
  return {
    waiting: "等待会话",
    connecting: "监听中",
    live: "API实时同步",
    cached: "缓存快速恢复",
    reconnecting: "续连中",
    error: "同步异常",
  }[value] ?? "连接中";
}

function applyOverviewPayload(
  monitor: OverviewPayload | null | undefined,
  selectedChannel: string,
  setOverview: (value: OverviewPayload | null) => void,
  setSelectedChannel: (value: string) => void,
  setPacket: (value: MonitorPacket | null) => void,
): string {
  setOverview(monitor ?? null);
  const channels = monitor?.channels ?? [];
  if (channels.length === 0) {
    setSelectedChannel("");
    setPacket(null);
    return "";
  }
  const selectedStillExists = Boolean(selectedChannel) && channels.some((channel) => channel.channel_id === selectedChannel);
  const nextChannel = selectedStillExists ? selectedChannel : channels[0]?.channel_id ?? "";
  if (nextChannel !== selectedChannel) {
    setSelectedChannel(nextChannel);
  }
  return nextChannel;
}

function applyMonitorPacket(
  monitor: MonitorPacket | null | undefined,
  setPacket: (value: MonitorPacket | null) => void,
  setServerOffsetMs: (value: number) => void,
): void {
  if (!monitor) {
    setPacket(null);
    return;
  }
  const snapshotServerTime = Number(monitor.server_time ?? monitor.updated_at ?? 0);
  if (snapshotServerTime > 0) {
    setServerOffsetMs(snapshotServerTime * 1000 - Date.now());
  }
  setPacket(monitor);
}

function monitorUrl(path: string, version: number, timeoutSeconds = 25): string {
  const query = new URLSearchParams({
    after_version: String(Math.max(0, Math.floor(version || 0))),
    timeout: String(timeoutSeconds),
  });
  return `${path}?${query.toString()}`;
}

function metricTone(value: number): string {
  if (value >= 0.7) {
    return "high";
  }
  if (value >= 0.35) {
    return "medium";
  }
  return "low";
}

function etaText(prediction: Prediction | undefined, serverOffsetMs: number, nowMs: number): string {
  if (!prediction) {
    return "-";
  }
  const deadlineSeconds =
    Number(prediction.deadline_at ?? 0) ||
    Number(prediction.generated_at ?? 0) + Number(prediction.eta_seconds ?? 0);
  if (!deadlineSeconds) {
    return prediction.eta_label ?? "-";
  }
  const serverNow = (nowMs + serverOffsetMs) / 1000;
  const remaining = Math.max(0, deadlineSeconds - serverNow);
  if (remaining <= 30) {
    return "立即";
  }
  if (remaining < 3600) {
    return `约 ${Math.ceil(remaining / 60)} 分钟后`;
  }
  return `约 ${(remaining / 3600).toFixed(1)} 小时后`;
}

function elapsedSecondsText(startedAt: number | undefined | null, serverOffsetMs: number, nowMs: number): string {
  const started = Number(startedAt ?? 0);
  if (started <= 0) {
    return "等待同步";
  }
  const serverNow = (nowMs + serverOffsetMs) / 1000;
  return `${Math.max(0, Math.floor(serverNow - started))}秒`;
}

export function EmotionDashboard() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [selectedChannel, setSelectedChannel] = useState("");
  const [packet, setPacket] = useState<MonitorPacket | null>(null);
  const [loading, setLoading] = useState(false);
  const [connectionState, setConnectionState] = useState("waiting");
  const [serverOffsetMs, setServerOffsetMs] = useState(0);
  const [clockNowMs, setClockNowMs] = useState(() => Date.now());
  const [errorMessage, setErrorMessage] = useState("");

  const residentCards = useMemo(() => {
    const overviewMap = packet?.presentation?.resident_overview ?? {};
    return Object.entries(overviewMap);
  }, [packet]);

  const activeSignals = useMemo(() => {
    const items = [...(packet?.presentation?.active_signals ?? [])];
    items.sort((left, right) => {
      return (severityRank[right.severity] ?? 0) - (severityRank[left.severity] ?? 0);
    });
    return items;
  }, [packet]);

  const sceneContext = packet?.presentation?.scene_context;
  const participantImpacts = packet?.presentation?.participant_impacts ?? [];
  const secondaryParticipantImpacts = useMemo(() => {
    return participantImpacts.filter((item) => !item.is_current_target);
  }, [participantImpacts]);
  const timeline = packet?.presentation?.timeline ?? [];
  const prediction = packet?.prediction;
  const resourceDetail = packet?.presentation?.resource_detail;
  const circadianDetail = packet?.presentation?.circadian_detail;
  const emotionDetail = packet?.presentation?.emotion_detail;
  const behaviorDetail = packet?.presentation?.behavior_detail;
  const timingGateDetail = packet?.presentation?.timing_gate_detail;
  const contextDetail = packet?.presentation?.context_detail;
  const memoryDetail = packet?.presentation?.memory_detail;
  const autonomyDetail = packet?.presentation?.autonomy_detail;
  const groupStateDetail = packet?.presentation?.group_state_detail;
  const currentUserDetail = packet?.presentation?.current_user_detail;
  const attentionRuntimeDetail = packet?.presentation?.attention_runtime_detail;
  const safetyDetail = packet?.presentation?.safety_detail;
  const dynamicTrace = packet?.presentation?.dynamic_trace ?? [];
  const initiativeState = packet?.presentation?.initiative_state;
  const displayPolicy = packet?.presentation?.display_policy ?? fallbackDisplayPolicy;
  const selectedOverview = (overview?.channels ?? []).find(
    (channel) => channel.channel_id === selectedChannel,
  );
  const selectedScopeLabel = conversationScopeLabel(selectedOverview);
  const predictionPercent =
    prediction?.probability_percent ?? Number(((prediction?.speak_probability ?? 0) * 100).toFixed(1));
  const usingCachedSnapshot = Boolean(packet?.stale_cache || overview?.stale_cache);
  const displayConnectionState = selectedChannel ? (usingCachedSnapshot ? "cached" : connectionState) : "waiting";
  const liveEtaLabel = etaText(prediction, serverOffsetMs, clockNowMs);
  const liveSilenceSeconds = Math.max(
    0,
    Number(emotionDetail?.silence_seconds ?? 0) +
      Math.max(0, ((clockNowMs + serverOffsetMs) / 1000) - Number(packet?.server_time ?? packet?.updated_at ?? 0)),
  );

  useEffect(() => {
    let frame = 0;
    let last = 0;
    const tick = (timestamp: number) => {
      if (timestamp - last >= 100) {
        last = timestamp;
        setClockNowMs(Date.now());
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    let ignore = false;

    async function loadMonitorOverview() {
      setLoading(true);
      setErrorMessage("");
      try {
        const response = await fetch(monitorUrl("/api/heartflow/monitor/live", 0, 0.1), {
          credentials: "same-origin",
        });
        if (!response.ok) {
          throw new Error(`monitor ${response.status}`);
        }
        const data = await response.json();
        if (ignore) {
          return;
        }
        const monitor = data?.monitor as OverviewPayload | undefined;
        const initialChannel = applyOverviewPayload(monitor, "", setOverview, setSelectedChannel, setPacket);
        if (!initialChannel) {
          setConnectionState("waiting");
        }
      } catch (error) {
        if (!ignore) {
          setErrorMessage(`初始化状态页失败: ${String(error)}`);
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    void loadMonitorOverview();

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    let overviewVersion = Number(overview?.state_version ?? 0);

    async function watchOverview() {
      while (!cancelled) {
        try {
          const response = await fetch(monitorUrl("/api/heartflow/monitor/live", overviewVersion), {
            credentials: "same-origin",
            signal: controller.signal,
          });
          if (!response.ok) {
            throw new Error(`overview ${response.status}`);
          }
          const data = await response.json();
          if (cancelled) {
            return;
          }
          const monitor = data?.monitor as OverviewPayload | undefined;
          overviewVersion = Number(monitor?.state_version ?? overviewVersion);
          applyOverviewPayload(monitor, selectedChannel, setOverview, setSelectedChannel, setPacket);
          setConnectionState((current) => (current === "waiting" ? "waiting" : "live"));
          setErrorMessage("");
        } catch (error) {
          if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) {
            setConnectionState("reconnecting");
            setErrorMessage(`实时总览同步失败: ${String(error)}`);
            await new Promise((resolve) => window.setTimeout(resolve, 600));
          }
        }
      }
    }

    void watchOverview();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedChannel]);

  useEffect(() => {
    if (!selectedChannel) {
      setPacket(null);
      setConnectionState("waiting");
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    let packetVersion = Number(packet?.state_version ?? 0);

    async function watchChannel() {
      setConnectionState("connecting");
      while (!cancelled) {
        try {
          const response = await fetch(
            monitorUrl(`/api/heartflow/monitor/${encodeURIComponent(selectedChannel)}/live`, packetVersion),
            {
              credentials: "same-origin",
              signal: controller.signal,
            },
          );
          if (!response.ok) {
            throw new Error(`monitor ${response.status}`);
          }
          const data = await response.json();
          if (cancelled) {
            return;
          }
          const monitor = data?.monitor as MonitorPacket | undefined;
          packetVersion = Number(monitor?.state_version ?? packetVersion);
          applyMonitorPacket(monitor, setPacket, setServerOffsetMs);
          setConnectionState("live");
          setErrorMessage("");
        } catch (error) {
          if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) {
            setConnectionState("reconnecting");
            setErrorMessage(`实时状态同步失败: ${String(error)}`);
            await new Promise((resolve) => window.setTimeout(resolve, 600));
          }
        }
      }
    }

    void watchChannel();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedChannel]);

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">火力状态监控</p>
          <h1>机器人实时状态栏</h1>
          <p className="subtle">
            与 WebUI 会话同步显示。顶部显示常驻状态，卡片流只显示当前激活状态，关系、能量、会话感知和预测依据放在详情区。
          </p>
        </div>
        <div className="header-actions">
          <label className="channel-picker">
            <span>选择会话</span>
            <select
              value={selectedChannel}
              onChange={(event) => setSelectedChannel(event.target.value)}
            >
              {(overview?.channels ?? []).map((channel) => (
                <option key={channel.channel_id} value={channel.channel_id}>
                  {channelOptionLabel(channel)}
                </option>
              ))}
            </select>
          </label>
          <div className={`live-pill is-${displayConnectionState}`}>{connectionLabel(displayConnectionState)}</div>
        </div>
      </header>

      {loading && <div className="status-banner">正在同步状态数据…</div>}
      {errorMessage && <div className="status-banner is-error">{errorMessage}</div>}
      {!loading && !errorMessage && (overview?.active_count ?? 0) === 0 && (
        <div className="status-banner">当前没有活跃会话，状态页会等待新的 Heartflow 会话出现后接入。</div>
      )}

      <section className="command-grid">
        <article className="command-card">
          <span>当前会话</span>
          <strong>{conversationTypeLabel(selectedOverview)}</strong>
          <p>
            {selectedOverview
              ? `${selectedOverview.platform_label ?? selectedOverview.platform ?? "本地"} · ${conversationName(selectedOverview)}`
              : "暂无活跃会话"}
            {overview?.hidden_internal_count ? ` · 已隐藏 ${overview.hidden_internal_count} 个本地测试会话` : ""}
          </p>
        </article>
        <article className="command-card is-score">
          <span>下一步发言概率</span>
          <strong>{Number(predictionPercent).toFixed(1)}%</strong>
          <p>{prediction?.decision_label ?? "等待状态"} · {runtimeSyncLabel(prediction)} · {liveEtaLabel}</p>
        </article>
        <article className="command-card">
          <span>最后同步</span>
          <strong>{packet?.updated_at ? formatClock(packet.updated_at) : "-"}</strong>
          <p>{connectionLabel(displayConnectionState)} · {selectedScopeLabel}状态{usingCachedSnapshot ? "来自最近快照" : "实时更新"}</p>
        </article>
      </section>

      <section className="resident-grid">
        {residentCards.map(([key, card]) => (
          <article className="resident-card" key={key}>
            <div className="resident-topline">
              <span className="resident-icon">{card.icon}</span>
              <span className="resident-label">{card.label}</span>
              <span className="resident-value">{card.display_value}</span>
            </div>
            <div className="resident-state">{card.state}</div>
            <div className="resident-bar">
              <span className="resident-bar-fill" style={{ width: `${Math.max(6, Math.round(card.value * 100))}%`, background: card.color }} />
            </div>
          </article>
        ))}
      </section>

      <main className="dashboard-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>运行资源</h2>
            <span>{resourceDetail?.energy_phase_label ?? "未知"}</span>
          </div>
          <div className="metric-wall">
            <div className={`metric-tile tone-${metricTone(resourceDetail?.chat_percent ?? 0)}`}>
              <span>聊天值</span>
              <strong>{percent(resourceDetail?.chat_percent)}</strong>
              <p>{fixed(resourceDetail?.chat_reserve)} / {fixed(resourceDetail?.chat_capacity)}</p>
            </div>
            <div className={`metric-tile tone-${metricTone(resourceDetail?.thinking_percent ?? 0)}`}>
              <span>思考值</span>
              <strong>{percent(resourceDetail?.thinking_percent)}</strong>
              <p>{fixed(resourceDetail?.thinking_reserve)} / {fixed(resourceDetail?.thinking_capacity)}</p>
            </div>
            <div className="metric-tile">
              <span>社交场</span>
              <strong>{fixed(resourceDetail?.social_field_score)}</strong>
              <p>活跃度 {fixed(resourceDetail?.activity_index)}</p>
            </div>
            <div className="metric-tile tone-low">
              <span>资源烦躁</span>
              <strong>{fixed(resourceDetail?.irritation_load)}</strong>
              <p>{resourceDetail?.last_irritation_relief_source || "暂无缓解事件"}</p>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>时间线状态</h2>
            <span>{circadianDetail?.phase_label ?? "清醒"}</span>
          </div>
          <div className="metric-wall">
            <div className="metric-tile">
              <span>当前时段</span>
              <strong>{circadianDetail?.time_band_label ?? "等待同步"}</strong>
              <p>{hourText(circadianDetail?.current_hour)} · {circadianDetail?.time_band_description ?? "等待后端时间线"}</p>
            </div>
            <div className={`metric-tile tone-${metricTone((100 - (circadianDetail?.drowsiness_value ?? 0)) / 100)}`}>
              <span>困意值</span>
              <strong>{countText(circadianDetail?.drowsiness_value, "")}</strong>
              <p>{circadianDetail?.body_state_label ?? "正常"}</p>
            </div>
            <div className={`metric-tile tone-${metricTone(1 - (circadianDetail?.sleep_debt ?? 0))}`}>
              <span>睡眠债</span>
              <strong>{percent(circadianDetail?.sleep_debt)}</strong>
              <p>{circadianDetail?.mood_hint ?? "状态平稳"}</p>
            </div>
            <div className={`metric-tile tone-${metricTone((circadianDetail?.sleep_reserve ?? 100) / 100)}`}>
              <span>睡眠储备</span>
              <strong>{countText(circadianDetail?.sleep_reserve, "")}</strong>
              <p>清晨恢复 {percent(circadianDetail?.dawn_recovery_progress)}</p>
            </div>
            <div className="metric-tile">
              <span>夜间回复</span>
              <strong>{circadianDetail?.reply_quota_label ?? "不限"}</strong>
              <p>{circadianDetail?.can_reply ? "允许回复" : "建议继续休息"} · {circadianDetail?.peek_window_open ? "窥屏窗口打开" : "窥屏窗口关闭"}</p>
            </div>
            <div className="metric-tile">
              <span>熬夜压力</span>
              <strong>{countText(circadianDetail?.overnight_pressure, "")}</strong>
              <p>分项 {fixed(circadianDetail?.pressure_breakdown?.total, 2)} · {circadianDetail?.is_burnthrough ? "熬穿已激活" : circadianDetail?.is_sleeping ? "睡眠中" : circadianDetail?.is_pressure_window ? "压力窗口" : circadianDetail?.is_sleep_window ? "睡眠窗口" : "清醒时段"}</p>
            </div>
            <div className="metric-tile">
              <span>表达风格</span>
              <strong>{circadianExpressionStyle(packet, circadianDetail)}</strong>
              <p>回复抑制 {percent(circadianDetail?.response_suppression)}</p>
            </div>
            <div className="metric-tile">
              <span>状态同步</span>
              <strong>{circadianDetail?.sync_label ?? "运行时实时计算"}</strong>
              <p>上次 {timestampText(circadianDetail?.last_evaluated_at)} · 启动 {timestampText(circadianDetail?.system_started_at)}</p>
            </div>
          </div>
          <div className="detail-grid compact circadian-breakdown">
            <div className="detail-row">
              <span>压力分解</span>
              <strong>{fixed(circadianDetail?.pressure_breakdown?.total, 2)}</strong>
            </div>
            <div className="detail-row">
              <span>聊天累积</span>
              <strong>{fixed(circadianDetail?.pressure_breakdown?.chat_minutes, 1)}</strong>
            </div>
            <div className="detail-row">
              <span>窥屏累积</span>
              <strong>{fixed(circadianDetail?.pressure_breakdown?.peek_minutes, 1)}</strong>
            </div>
            <div className="detail-row">
              <span>思考负荷</span>
              <strong>{fixed(circadianDetail?.pressure_breakdown?.think_intensity, 2)}</strong>
            </div>
            <div className="detail-row">
              <span>打断次数</span>
              <strong>{circadianDetail?.pressure_breakdown?.interrupt_count ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>社交窗口</span>
              <strong>{mechanismWindowText(circadianDetail?.mechanism_windows, "night_social", "夜间社交")}</strong>
            </div>
            <div className="detail-row">
              <span>压力窗口</span>
              <strong>{mechanismWindowText(circadianDetail?.mechanism_windows, "overnight_pressure", "熬夜压力")}</strong>
            </div>
            <div className="detail-row">
              <span>睡眠窗口</span>
              <strong>{mechanismWindowText(circadianDetail?.mechanism_windows, "sleep_window", "睡眠窗口")}</strong>
            </div>
            <div className="detail-row">
              <span>反思窗口</span>
              <strong>{mechanismWindowText(circadianDetail?.mechanism_windows, "midnight_reflect", "凌晨反思")}</strong>
            </div>
            <div className="detail-row">
              <span>恢复窗口</span>
              <strong>{mechanismWindowText(circadianDetail?.mechanism_windows, "dawn_recover", "清晨恢复")}</strong>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>主动倾向</h2>
            <span>{initiativeState?.effect_label ?? "等待状态"}</span>
          </div>
          <div className="drive-radar">
            <div>
              <span>主动意愿</span>
              <strong>{percent(initiativeState?.initiative_drive)}</strong>
            </div>
            <div>
              <span>无聊度</span>
              <strong>{percent(initiativeState?.boredom_load)}</strong>
            </div>
            <div>
              <span>孤独感</span>
              <strong>{percent(initiativeState?.loneliness_load)}</strong>
            </div>
            <div>
              <span>疲劳感</span>
              <strong>{percent(initiativeState?.environment_fatigue_load)}</strong>
            </div>
            <div>
              <span>退场倾向</span>
              <strong>{percent(initiativeState?.withdrawal_drive)}</strong>
            </div>
          </div>
          <p className="panel-note">{initiativeState?.effect_summary ?? "主动倾向会进入发言预测，不再只是日志里的装饰值。"}</p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>情绪状态</h2>
            <span>{emotionDetail?.mood_label ?? "平静"}</span>
          </div>
          <div className="drive-radar">
            <div>
              <span>情绪基调</span>
              <strong>{percent(emotionDetail?.mood_bias)}</strong>
            </div>
            <div>
              <span>好奇心</span>
              <strong>{percent(emotionDetail?.curiosity_drive)}</strong>
            </div>
            <div>
              <span>社交欲</span>
              <strong>{percent(emotionDetail?.social_desire)}</strong>
            </div>
            <div>
              <span>无聊</span>
              <strong>{percent(emotionDetail?.boredom_load)}</strong>
            </div>
            <div>
              <span>孤独</span>
              <strong>{percent(emotionDetail?.loneliness_load)}</strong>
            </div>
            <div>
              <span>疲劳感</span>
              <strong>{percent(emotionDetail?.environment_fatigue_load)}</strong>
            </div>
            <div>
              <span>主动意愿</span>
              <strong>{percent(emotionDetail?.initiative_drive)}</strong>
            </div>
            <div>
              <span>退场倾向</span>
              <strong>{percent(emotionDetail?.withdrawal_drive)}</strong>
            </div>
          </div>
          <p className="panel-note">
            {emotionDetail?.feeling_text ?? "当前没有明显情绪波动"} · 已沉默 {countText(liveSilenceSeconds, "秒")} · 未回应 {emotionDetail?.unanswered_count ?? 0} 次
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>行为机制</h2>
            <span>{behaviorDetail?.flow_phase_label ?? "待命"}</span>
          </div>
          <div className="detail-grid compact">
            <div className="detail-row">
              <span>关注姿态</span>
              <strong>{behaviorDetail?.watch_state_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>行动裁定</span>
              <strong>{behaviorDetail?.reply_decision.reply_urgency ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>最终执行</span>
              <strong>{behaviorDetail?.execution_runtime.final_action_label || behaviorDetail?.execution_runtime.final_action || "-"}</strong>
            </div>
            <div className="detail-row">
              <span>执行阶段</span>
              <strong>{behaviorDetail?.execution_runtime.execution_stage_label || behaviorDetail?.execution_runtime.execution_stage || "-"}</strong>
            </div>
            <div className="detail-row">
              <span>行为模式</span>
              <strong>{behaviorDetail?.behavior_governor.reply_mode_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>打断等级</span>
              <strong>{behaviorDetail?.behavior_governor.interrupt_level_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>静默策略</span>
              <strong>{behaviorDetail?.behavior_governor.silence_policy_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>模型等级</span>
              <strong>{behaviorDetail?.behavior_governor.model_tier_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>实际模型路径</span>
              <strong>{modelPathLabel(behaviorDetail?.execution_runtime.model_path)}</strong>
            </div>
            <div className="detail-row">
              <span>休息姿态</span>
              <strong>{behaviorDetail?.rest_governor.posture_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>当前语气</span>
              <strong>{behaviorDetail?.reply_decision.suggested_tone ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>是否已发送</span>
              <strong>{yesNo(behaviorDetail?.execution_runtime.reply_sent)}</strong>
            </div>
          </div>
          <p className="panel-note">
            {behaviorDetail?.execution_runtime.execution_reason || behaviorDetail?.reply_decision.decision_reason || "暂无行为裁定理由"}
          </p>
          <div className="topic-wrap">
            {(behaviorDetail?.execution_runtime.blocking_factors ?? []).map((item) => (
              <span className="topic-chip" key={`exec-block-${item}`}>
                {item}
              </span>
            ))}
            {(behaviorDetail?.execution_runtime.driving_factors ?? []).map((item) => (
              <span className="topic-chip is-secondary" key={`exec-drive-${item}`}>
                {item}
              </span>
            ))}
            {(behaviorDetail?.behavior_governor.reason_labels ?? []).map((item) => (
              <span className="topic-chip is-secondary" key={`bg-${item}`}>
                {item}
              </span>
            ))}
            {(behaviorDetail?.rest_governor.reason_labels ?? []).map((item) => (
              <span className="topic-chip" key={`rg-${item}`}>
                {item}
              </span>
            ))}
          </div>
        </section>

        <section className="panel panel-timing-gate">
          <div className="panel-header">
            <h2>Timing Gate</h2>
            <span>{timingGateDetail?.current?.gate_result_label ?? "等待门控"}</span>
          </div>
          <div className="metric-wall">
            <div className="metric-tile">
              <span>当前裁定</span>
              <strong>{timingGateDetail?.current?.gate_result_label ?? "暂无"}</strong>
              <p>{timingGateDetail?.current?.reason ?? "尚未记录门控原因"}</p>
            </div>
            <div className="metric-tile">
              <span>门控阶段</span>
              <strong>{timingGateDetail?.current?.stage_label || humanExecutionStage(timingGateDetail?.current?.stage)}</strong>
              <p>来源 {timingGateDetail?.current?.source || "timing_gate"}</p>
            </div>
            <div className="metric-tile">
              <span>后续动作</span>
              <strong>{timingGateDetail?.current?.final_action_label || timingGateDetail?.current?.final_action || "-"}</strong>
              <p>next={timingGateDetail?.current?.next_action || "-"} · {modelPathLabel(timingGateDetail?.current?.model_path)}</p>
            </div>
            <div className="metric-tile">
              <span>历史记录</span>
              <strong>{timingGateDetail?.history_count ?? 0}</strong>
              <p>置信度 {percent(timingGateDetail?.current?.confidence)}</p>
            </div>
          </div>
          <div className="timing-gate-list">
            {(timingGateDetail?.history ?? []).length === 0 ? (
              <div className="empty-state">暂无门控历史，等待下一轮 Heartflow 裁定。</div>
            ) : (
              (timingGateDetail?.history ?? []).slice(-6).map((item, index) => (
                <article className="timing-gate-item" key={`${item.verdict_id || item.stage}-${index}`}>
                  <div>
                    <strong>{item.gate_result_label ?? item.gate_result}</strong>
                    <span>{item.stage_label || humanExecutionStage(item.stage)}</span>
                  </div>
                  <p>{item.reason || item.blocker || "暂无门控原因"}</p>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>上下文感知</h2>
            <span>{contextDetail?.watch_state_label ?? "瞥一眼"}</span>
          </div>
          <div className="detail-grid compact">
            <div className="detail-row">
              <span>直接目标</span>
              <strong>{yesNo(contextDetail?.direct_target)}</strong>
            </div>
            <div className="detail-row">
              <span>引用锚点</span>
              <strong>{yesNo(contextDetail?.quote_anchor)}</strong>
            </div>
            <div className="detail-row">
              <span>近期有人说话</span>
              <strong>{yesNo(contextDetail?.recent_human_activity)}</strong>
            </div>
            <div className="detail-row">
              <span>场景适合参与</span>
              <strong>{yesNo(contextDetail?.scene_suitable)}</strong>
            </div>
            <div className="detail-row">
              <span>关注等级</span>
              <strong>{contextDetail?.attention_level ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>感知拉力</span>
              <strong>{percent(contextDetail?.perception_engagement_pull)}</strong>
            </div>
            <div className="detail-row">
              <span>可见阈值</span>
              <strong>{percent(attentionRuntimeDetail?.visibility_threshold)}</strong>
            </div>
            <div className="detail-row">
              <span>处理比例</span>
              <strong>{percent(attentionRuntimeDetail?.process_ratio)}</strong>
            </div>
            <div className="detail-row">
              <span>窥屏欲望</span>
              <strong>{percent(attentionRuntimeDetail?.peek_desire)}</strong>
            </div>
            <div className="detail-row">
              <span>连续窥屏未行动</span>
              <strong>{attentionRuntimeDetail?.consecutive_peeks_without_action ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>回忆录阶段</span>
              <strong>{contextDetail?.memoir_phase_label ?? "开放会话"}</strong>
            </div>
            <div className="detail-row">
              <span>连续超时</span>
              <strong>{contextDetail?.memoir_consecutive_timeouts ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>行为模式</span>
              <strong>{contextDetail?.reply_mode_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>静默策略</span>
              <strong>{contextDetail?.silence_policy_label ?? "-"}</strong>
            </div>
          </div>
          <p className="panel-note">
            当前目标：{contextDetail?.current_target_user_id || "暂无"} · 自我近期消息 {contextDetail?.self_recent_messages_count ?? 0} 条 · 动作 {contextDetail?.self_recent_actions_count ?? 0} 条 · 事件 {contextDetail?.self_recent_events_count ?? 0} 条
          </p>
          <div className="topic-wrap">
            {(contextDetail?.behavior_reason_labels ?? []).map((item) => (
              <span className="topic-chip is-secondary" key={`ctx-bg-${item}`}>
                {item}
              </span>
            ))}
            {(contextDetail?.rest_reason_labels ?? []).map((item) => (
              <span className="topic-chip" key={`ctx-rg-${item}`}>
                {item}
              </span>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>记忆栈</h2>
            <span>{memoryDetail?.health_grade ?? "未知"}</span>
          </div>
          <div className="metric-wall">
            <div className={`metric-tile tone-${metricTone(1 - (memoryDetail?.overload_load_ratio ?? 0))}`}>
              <span>记忆负载</span>
              <strong>{percent(memoryDetail?.overload_load_ratio)}</strong>
              <p>{memoryDetail?.overload_emergency_needed ? "已触发紧急压力" : memoryDetail?.overload_gauge_level || "正常"}</p>
            </div>
            <div className="metric-tile">
              <span>短期记忆</span>
              <strong>{memoryDetail?.ephemeral_short_term_count ?? 0}</strong>
              <p>规划器短期 {memoryDetail?.planner_short_term_count ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>长期记忆</span>
              <strong>{memoryDetail?.long_term_total ?? 0}</strong>
              <p>规划器长期 {memoryDetail?.planner_long_term_count ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>知识条目</span>
              <strong>{memoryDetail?.knowledge_entry_count ?? 0}</strong>
              <p>健康分 {memoryDetail?.health_score ?? 100}</p>
            </div>
            <div className="metric-tile">
              <span>回忆录阶段</span>
              <strong>{memoryDetail?.memoir_phase_label ?? "开放会话"}</strong>
              <p>轮次 {memoryDetail?.memoir_exchange_tally ?? 0} · 超时 {memoryDetail?.memoir_consecutive_timeouts ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>治理状态</span>
              <strong>{memoryDetail?.governance_last_op ?? "无操作"}</strong>
              <p>待淘汰 {memoryDetail?.governance_pending_evict ?? 0} · 日志 {memoryDetail?.journal_count ?? 0}</p>
            </div>
          </div>
          <div className="topic-wrap">
            {Object.entries(memoryDetail?.long_term_by_category ?? {}).map(([key, value]) => (
              <span className="topic-chip is-secondary" key={`mem-${key}`}>
                {key}:{value}
              </span>
            ))}
            {Object.entries(memoryDetail?.reactivation_summary ?? {}).map(([key, value]) => (
              <span className="topic-chip" key={`react-${key}`}>
                {key}:{value}
              </span>
            ))}
          </div>
          <p className="panel-note">
            最近话题：{memoryDetail?.memoir_last_topic || "暂无"} · 最近心情：{memoryDetail?.memoir_last_mood || "暂无"}
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>主动意图</h2>
            <span>{autonomyDetail?.background_event_label || "无后台事件"}</span>
          </div>
          <div className="metric-wall">
            <div className={`metric-tile tone-${metricTone(autonomyDetail?.intention_drive ?? 0)}`}>
              <span>主动倾向总线</span>
              <strong>{percent(autonomyDetail?.intention_drive)}</strong>
              <p>活跃意图 {autonomyDetail?.intention_alive_count ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>最高意图紧迫度</span>
              <strong>{percent(autonomyDetail?.max_intention_urgency)}</strong>
              <p>待结算事件 {autonomyDetail?.pending_proactive_events ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>历史反馈分</span>
              <strong>{fixed(autonomyDetail?.reward_score, 2)}</strong>
              <p>近期动作 {autonomyDetail?.self_recent_actions_count ?? 0}</p>
            </div>
            <div className="metric-tile">
              <span>后台事件</span>
              <strong>{autonomyDetail?.background_event_label || "无"}</strong>
              <p>优先级 {fixed(autonomyDetail?.background_event_priority, 2)} · 剩余额度 {autonomyDetail?.background_budget_remaining ?? 0}</p>
            </div>
          </div>
          <div className="topic-wrap">
            {(autonomyDetail?.active_intentions ?? []).map((intent) => (
              <span className="topic-chip is-secondary" key={intent.intent_id}>
                {intent.kind}:{Math.round(intent.effective_urgency * 100)}%
              </span>
            ))}
          </div>
          <p className="panel-note">
            自我感知：最近消息 {autonomyDetail?.self_recent_messages_count ?? 0} 条，最近动作 {autonomyDetail?.self_recent_actions_count ?? 0} 条，最近事件 {autonomyDetail?.self_recent_events_count ?? 0} 条
          </p>
        </section>

        <section className="panel panel-signals">
          <div className="panel-header">
            <h2>当前激活状态</h2>
            <span>{activeSignals.length} 项</span>
          </div>
          <div className="signal-list">
            {activeSignals.length === 0 ? (
              <div className="empty-state">当前没有额外激活状态，系统处于相对平稳区间。</div>
            ) : (
              activeSignals.map((signal) => (
                <article className={`signal-card severity-${signal.severity}`} key={signal.key}>
                  <div className="signal-head">
                    <span className="signal-icon">{signal.icon}</span>
                    <div>
                      <h3>{signal.label}</h3>
                      <p>{signal.display_value}</p>
                    </div>
                  </div>
                  <div className="signal-meta">
                    <span>{signal.family_label ?? signal.family}</span>
                    <span>{signal.trend_label ?? signal.trend}</span>
                    <span>{signal.source_domain_label ?? signal.source_domain}</span>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>会话整体状态</h2>
            <span>{groupStateDetail?.scene_heat_label ?? sceneContext?.scene_heat_label ?? humanSceneHeat(sceneContext?.scene_heat)}</span>
          </div>
          <div className="detail-grid">
            <div className="detail-row">
              <span>场景热度</span>
              <strong>{groupStateDetail?.scene_heat_label ?? "-"} · {percent(groupStateDetail?.scene_heat_score)}</strong>
            </div>
            <div className="detail-row">
              <span>活跃人数</span>
              <strong>{groupStateDetail?.active_user_count ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>每分钟消息</span>
              <strong>{fixed(groupStateDetail?.messages_per_minute, 2)}</strong>
            </div>
            <div className="detail-row">
              <span>线程数</span>
              <strong>{groupStateDetail?.thread_count ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>主导发言者</span>
              <strong>{groupStateDetail?.dominant_speaker || "暂无"}</strong>
            </div>
            <div className="detail-row">
              <span>适合插话</span>
              <strong>{yesNo(groupStateDetail?.suitable_to_join)}</strong>
            </div>
            <div className="detail-row">
              <span>互动质量</span>
              <strong>{percent(groupStateDetail?.interaction_quality)}</strong>
            </div>
            <div className="detail-row">
              <span>会话烦躁 / 疲劳</span>
              <strong>{fixed(groupStateDetail?.vexation)} / {fixed(groupStateDetail?.weariness)}</strong>
            </div>
          </div>
          <p className="panel-note">
            氛围：{groupStateDetail?.atmosphere_label || "暂无"} · 不适合插话原因：{groupStateDetail?.join_unsuitable_reason || "无"}
          </p>
          <div className="topic-wrap">
            {(groupStateDetail?.topic_focus ?? []).length > 0 ? (
              groupStateDetail?.topic_focus.map((topic) => (
                <span className="topic-chip" key={topic}>
                  {topic}
                </span>
              ))
            ) : (
              <span className="empty-state inline">当前没有可提取的话题焦点。</span>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>当前对象状态</h2>
            <span>{currentUserDetail?.user_id ? currentUserDetail.display_name : "暂无"}</span>
          </div>
          {!currentUserDetail?.user_id ? (
            <div className="empty-state">当前没有明确焦点用户，用户影响区保持空状态。</div>
          ) : (
            <article className="participant-card">
              <header>
                <div>
                  <h3>{currentUserDetail.display_name}</h3>
                  <p>{currentUserDetail.relationship_label} · {currentUserDetail.current_mood_hint}</p>
                </div>
                <strong>{currentUserDetail.interaction_count}</strong>
              </header>
              <div className="detail-grid compact">
                <div className="detail-row">
                  <span>好感</span>
                  <strong>{currentUserDetail.rapport_score.toFixed(1)}</strong>
                </div>
                <div className="detail-row">
                  <span>信任</span>
                  <strong>{currentUserDetail.trust_score.toFixed(1)}</strong>
                </div>
                <div className="detail-row">
                  <span>烦躁</span>
                  <strong>{currentUserDetail.irritation_load.toFixed(1)}</strong>
                </div>
                <div className="detail-row">
                  <span>压力</span>
                  <strong>{currentUserDetail.pressure_load.toFixed(1)}</strong>
                </div>
                <div className="detail-row">
                  <span>创伤</span>
                  <strong>{currentUserDetail.trauma_load.toFixed(2)}</strong>
                </div>
                <div className="detail-row">
                  <span>混乱 / 伪装</span>
                  <strong>{currentUserDetail.chaos_load.toFixed(1)} / {currentUserDetail.mask_load.toFixed(1)}</strong>
                </div>
              </div>
            </article>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>多用户影响</h2>
            <span>{secondaryParticipantImpacts.length} 人</span>
          </div>
          {secondaryParticipantImpacts.length === 0 ? (
            <div className="empty-state">当前还没有其它高影响用户进入关系面板，系统暂时只锁定当前对象。</div>
          ) : (
            <div className="signal-list">
              {secondaryParticipantImpacts.map((impact) => (
                <article className="participant-card" key={impact.user_id}>
                  <header>
                    <div>
                      <h3>{impact.display_name}</h3>
                      <p>{impact.relationship_label} · {impact.current_mood_hint}</p>
                    </div>
                    <strong>{impact.interaction_count}</strong>
                  </header>
                  <div className="detail-grid compact">
                    <div className="detail-row">
                      <span>好感</span>
                      <strong>{impact.rapport_score.toFixed(1)}</strong>
                    </div>
                    <div className="detail-row">
                      <span>信任</span>
                      <strong>{impact.trust_score.toFixed(1)}</strong>
                    </div>
                    <div className="detail-row">
                      <span>烦躁</span>
                      <strong>{impact.irritation_load.toFixed(1)}</strong>
                    </div>
                    <div className="detail-row">
                      <span>压力</span>
                      <strong>{impact.pressure_load.toFixed(1)}</strong>
                    </div>
                    <div className="detail-row">
                      <span>创伤</span>
                      <strong>{impact.trauma_load.toFixed(2)}</strong>
                    </div>
                    <div className="detail-row">
                      <span>混乱 / 伪装</span>
                      <strong>{impact.chaos_load.toFixed(1)} / {impact.mask_load.toFixed(1)}</strong>
                    </div>
                  </div>
                  <div className="topic-wrap">
                    {(impact.active_signals ?? []).length > 0 ? (
                      impact.active_signals.map((label) => (
                        <span className="topic-chip is-secondary" key={`${impact.user_id}-${label}`}>
                          {label}
                        </span>
                      ))
                    ) : (
                      <span className="empty-state inline">当前没有额外激活信号。</span>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
          <p className="panel-note">
            这里只展示除当前对象之外、对本轮关系判断影响更大的用户，便于区分“单人状态”和“会话里其他人的影响”。
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>安全护盾</h2>
            <span>{safetyDetail?.safety_level ?? "安全"}</span>
          </div>
          <div className="metric-wall">
            <div className={`metric-tile tone-${metricTone(1 - (safetyDetail?.safety_score ?? 0))}`}>
              <span>安全风险</span>
              <strong>{percent(safetyDetail?.safety_score)}</strong>
              <p>{safetyDetail?.blocked ? "已阻断回复" : "未阻断"}</p>
            </div>
            <div className="metric-tile">
              <span>主导威胁</span>
              <strong>{safetyDetail?.dominant_threat || "无"}</strong>
              <p>护盾惩罚 {fixed(safetyDetail?.bar_penalty, 2)}</p>
            </div>
          </div>
          <p className="panel-note">{safetyDetail?.threat_evidence_summary ?? "暂无显著风险"}</p>
        </section>

        <section className="panel panel-policy">
          <div className="panel-header">
            <h2>值变化原因</h2>
            <span>{dynamicTrace.length} 项</span>
          </div>
          <div className="signal-list">
            {dynamicTrace.map((item) => (
              <article className="signal-card" key={item.key}>
                <div className="signal-head">
                  <span className="signal-icon">{item.delta > 0 ? "↗" : item.delta < 0 ? "↘" : "→"}</span>
                  <div>
                    <h3>{item.label}</h3>
                    <p>{item.trend_label} {item.delta >= 0 ? "+" : ""}{item.delta.toFixed(3)}</p>
                  </div>
                </div>
                <div className="signal-meta">
                  <span>当前 {item.current.toFixed(3)}</span>
                  <span>之前 {item.previous.toFixed(3)}</span>
                </div>
                <p className="panel-note">{item.last_change_reason}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>发言预测</h2>
            <span>{prediction?.decision_label ?? prediction?.eta_label ?? "-"}</span>
          </div>
          <div className="prediction-hero">
            <div className="prediction-score">
              <span>{predictionPercent}%</span>
              <small>开口概率</small>
            </div>
            <div>
              <p className="prediction-title">{prediction?.decision_label ?? "继续观察"}</p>
              <p className="prediction-subtitle">{prediction?.content_direction ?? "暂无预测方向"}</p>
              <p className="prediction-meta">
                语气：{prediction?.tone ?? "正常回应"} · 长度：{prediction?.reply_length ?? "-"} · ETA：{prediction?.eta_label ?? "-"} · 模型路径：{modelPathLabel(prediction?.model_path)}
              </p>
              <p className="prediction-meta">
                统一裁定：{prediction?.shared_verdict_id || "暂无"} · 是否回复：{yesNo(prediction?.should_reply)} · 执行阶段：{humanExecutionStage(prediction?.execution_stage)} · 复杂度：{prediction?.complexity_label ?? "普通"}
              </p>
              <p className="prediction-reason">{prediction?.decision_reason ?? "等待后端预测理由。"}</p>
            </div>
          </div>
          <div className="prediction-breakdown">
            <div>
              <span>开口驱动</span>
              <strong>{fixed(prediction?.drive_score, 2)}</strong>
            </div>
            <div>
              <span>抑制压力</span>
              <strong>{fixed(prediction?.suppression_score, 2)}</strong>
            </div>
            <div>
              <span>主动影响</span>
              <strong>{fixed(prediction?.proactive_influence, 2)}</strong>
            </div>
            <div>
              <span>关系影响</span>
              <strong>{fixed(prediction?.relationship_influence, 2)}</strong>
            </div>
            <div>
              <span>资源影响</span>
              <strong>{fixed(prediction?.resource_influence, 2)}</strong>
            </div>
            <div>
              <span>会话影响</span>
              <strong>{fixed(prediction?.scene_influence, 2)}</strong>
            </div>
            <div>
              <span>夜间影响</span>
              <strong>{fixed(prediction?.circadian_influence, 2)}</strong>
            </div>
          </div>
          <div className="prediction-columns">
            <div>
              <h3>驱动因素</h3>
              <ul>
                {(prediction?.driving_factors ?? []).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3>抑制因素</h3>
              <ul>
                {(prediction?.suppressing_factors ?? []).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="panel panel-policy">
          <div className="panel-header">
            <h2>显示规则</h2>
            <span>后端裁定</span>
          </div>
          <div className="policy-grid">
            <div>
              <h3>常驻</h3>
              {(displayPolicy?.resident ?? []).map((item) => <span key={item}>{item}</span>)}
            </div>
            <div>
              <h3>激活才显示</h3>
              {(displayPolicy?.active ?? []).map((item) => <span key={item}>{item}</span>)}
            </div>
            <div>
              <h3>详情区</h3>
              {(displayPolicy?.detail ?? []).map((item) => <span key={item}>{item}</span>)}
            </div>
            <div>
              <h3>不进入页面</h3>
              {(displayPolicy?.hidden ?? []).map((item) => <span key={item}>{item}</span>)}
            </div>
          </div>
        </section>

        <section className="panel panel-timeline">
          <div className="panel-header">
            <h2>最近状态时间线</h2>
            <span>{timeline.length} 条</span>
          </div>
          <div className="timeline-list">
            {timeline.map((event, index) => (
              <article className="timeline-item" key={`${event.label}-${index}`}>
                <div className="timeline-time">{formatClock(event.at)}</div>
                <div>
                  <h3>{event.label}</h3>
                  <p>{event.detail}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
