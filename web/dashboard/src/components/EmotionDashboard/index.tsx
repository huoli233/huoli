import { useEffect, useMemo, useRef, useState } from "react";

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
  eta_seconds: number;
  eta_label: string;
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
  is_night: boolean;
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
  updated_at: number;
  domains: Record<string, unknown>;
  presentation: Presentation;
  prediction: Prediction;
};

type OverviewPayload = {
  updated_at: number;
  active_count: number;
  hidden_internal_count?: number;
  channels: OverviewChannel[];
};

type ConfigScopeModule = {
  module: string;
  scope: "system" | "mixed" | "user" | string;
  semantic_domains: string[];
  user_editable_keys: string[];
  system_only_keys: string[];
  editable_key_count: number;
  system_key_count: number;
};

type ConfigScopeSnapshot = {
  updated_at: number;
  summary: {
    total: number;
    system: number;
    mixed: number;
    user: number;
    editable: number;
  };
  modules: ConfigScopeModule[];
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
  active: ["无聊/孤独/环境疲劳/撤离/主动意愿", "情绪低落/好奇心/社交欲显著变化", "浅睡/深睡/熬穿/黎明恢复/睡眠债", "烦躁/压力/创伤/混乱/伪装", "关系好感/信任显著偏高或偏低", "冷却窗口/等待时长/重新接入/群聊升温"],
  detail: ["资源账本的聊天值和思考值", "昼夜节律、睡眠债、困意和熬夜压力", "情绪账本、主动驱动和当前感受", "行为机制和上下文感知", "群聊整体状态", "当前对象状态", "安全护盾细节", "值变化原因", "发言预测的驱动和抑制因素"],
  hidden: ["内部阈值", "调试原因", "缓存字段", "旧命名残留", "纯计数器原值"],
};

function formatClock(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function wsUrl(channelId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const base = `${protocol}//${window.location.host}`;
  const query = channelId ? `?channel_id=${encodeURIComponent(channelId)}` : "";
  return `${base}/ws/state-monitor${query}`;
}

function percent(value: number | undefined | null): string {
  const safe = Math.max(0, Math.min(1, Number(value ?? 0)));
  return `${Math.round(safe * 100)}%`;
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
    scene_constraint_skip: "群场景硬约束",
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
  const typeLabel = channel.chat_type_label ?? (channel.chat_type === "group" ? "群聊" : "私聊");
  const platformLabel = channel.platform_label ?? (channel.platform === "webui" ? "本地测试" : channel.platform || "本地");
  const name = channel.display_name || channel.channel_id.slice(0, 8);
  return `${typeLabel} · ${platformLabel} · ${name}`;
}

function countText(value: number | undefined | null, unit: string): string {
  return `${Math.round(Number(value ?? 0))}${unit}`;
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

function connectionLabel(value: string): string {
  return {
    idle: "未连接",
    connecting: "连接中",
    live: "实时同步",
    reconnecting: "重连中",
    error: "连接异常",
  }[value] ?? value;
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

function scopeLabel(value: string): string {
  return {
    user: "用户可改",
    mixed: "部分可改",
    system: "系统级",
  }[value] ?? value;
}

export function EmotionDashboard() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [selectedChannel, setSelectedChannel] = useState("");
  const [packet, setPacket] = useState<MonitorPacket | null>(null);
  const [configScope, setConfigScope] = useState<ConfigScopeSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [connectionState, setConnectionState] = useState("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [configScopeError, setConfigScopeError] = useState("");
  const reconnectTimerRef = useRef<number | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

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
  const configScopeGroups = useMemo(() => {
    const modules = configScope?.modules ?? [];
    return {
      user: modules.filter((item) => item.scope === "user"),
      mixed: modules.filter((item) => item.scope === "mixed"),
      system: modules.filter((item) => item.scope === "system"),
    };
  }, [configScope]);
  const selectedOverview = (overview?.channels ?? []).find(
    (channel) => channel.channel_id === selectedChannel,
  );
  const predictionPercent =
    prediction?.probability_percent ?? Math.round((prediction?.speak_probability ?? 0) * 100);

  useEffect(() => {
    let ignore = false;

    async function bootstrap() {
      const [monitorResult, scopeResult] = await Promise.allSettled([
        fetch("/api/heartflow/monitor", {
          credentials: "same-origin",
        }).then((response) => response.json()),
        fetch("/api/heartflow/config-scope", {
          credentials: "same-origin",
        }).then((response) => response.json()),
      ]);

      if (ignore) {
        return;
      }

      if (monitorResult.status === "fulfilled") {
        const monitor = monitorResult.value?.monitor as OverviewPayload | undefined;
        setOverview(monitor ?? null);
        const initialChannel = monitor?.channels?.[0]?.channel_id ?? "";
        setSelectedChannel(initialChannel);
      } else {
        setErrorMessage(`初始化状态页失败: ${String(monitorResult.reason)}`);
      }

      if (scopeResult.status === "fulfilled") {
        setConfigScope(scopeResult.value?.config_scope ?? null);
        setConfigScopeError("");
      } else {
        setConfigScopeError(`读取配置分级失败: ${String(scopeResult.reason)}`);
      }
    }

    bootstrap();

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedChannel) {
      return;
    }

    let cancelled = false;

    async function loadMonitor() {
      setLoading(true);
      setErrorMessage("");
      try {
        const response = await fetch(`/api/heartflow/monitor/${encodeURIComponent(selectedChannel)}`, {
          credentials: "same-origin",
        });
        const data = await response.json();
        if (!cancelled) {
          setPacket(data?.monitor ?? null);
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(`拉取状态失败: ${String(error)}`);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadMonitor();

    return () => {
      cancelled = true;
    };
  }, [selectedChannel]);

  useEffect(() => {
    if (!selectedChannel) {
      return;
    }

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    let closedByCleanup = false;

    const connect = () => {
      setConnectionState("connecting");
      const socket = new WebSocket(wsUrl(selectedChannel));
      socketRef.current = socket;

      socket.onopen = () => {
        setConnectionState("live");
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { type?: string; data?: unknown };
          if (message.type === "state_snapshot" && message.data) {
            setPacket(message.data as MonitorPacket);
          }
        } catch (error) {
          setErrorMessage(`解析实时状态失败: ${String(error)}`);
        }
      };

      socket.onerror = () => {
        setConnectionState("error");
      };

      socket.onclose = () => {
        if (closedByCleanup) {
          return;
        }
        setConnectionState("reconnecting");
        reconnectTimerRef.current = window.setTimeout(() => {
          connect();
        }, 1800);
      };
    };

    connect();

    return () => {
      closedByCleanup = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [selectedChannel]);

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">火力状态监控</p>
          <h1>机器人实时状态栏</h1>
          <p className="subtle">
            免登录查看。顶部显示常驻状态，卡片流只显示当前激活状态，关系、能量、群聊感知和预测依据放在详情区。
          </p>
        </div>
        <div className="header-actions">
          <label className="channel-picker">
            <span>选择群聊 / 私聊</span>
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
          <div className={`live-pill is-${connectionState}`}>{connectionLabel(connectionState)}</div>
        </div>
      </header>

      {loading && <div className="status-banner">正在同步状态数据…</div>}
      {errorMessage && <div className="status-banner is-error">{errorMessage}</div>}

      <section className="command-grid">
        <article className="command-card">
          <span>当前群聊 / 私聊</span>
          <strong>{selectedOverview?.chat_type_label ?? "群聊 / 私聊"}</strong>
          <p>
            {selectedOverview
              ? `${selectedOverview.platform_label ?? selectedOverview.platform ?? "本地"} · ${
                  selectedOverview.display_name || selectedOverview.channel_id.slice(0, 12)
                }`
              : "暂无活跃会话"}
            {overview?.hidden_internal_count ? ` · 已隐藏 ${overview.hidden_internal_count} 个本地测试会话` : ""}
          </p>
        </article>
        <article className="command-card is-score">
          <span>是否该聊</span>
          <strong>{predictionPercent}%</strong>
          <p>{prediction?.decision_label ?? "等待状态"} · {modelPathLabel(prediction?.model_path)} · {prediction?.eta_label ?? "-"}</p>
        </article>
        <article className="command-card">
          <span>最后同步</span>
          <strong>{packet?.updated_at ? formatClock(packet.updated_at) : "-"}</strong>
          <p>{connectionLabel(connectionState)} · 状态页无需登录凭证</p>
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
            <h2>资源账本</h2>
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
            <h2>昼夜节律</h2>
            <span>{circadianDetail?.phase_label ?? "清醒"}</span>
          </div>
          <div className="metric-wall">
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
              <p>黎明恢复 {percent(circadianDetail?.dawn_recovery_progress)}</p>
            </div>
            <div className="metric-tile">
              <span>夜间回复</span>
              <strong>{circadianDetail?.reply_quota_label ?? "不限"}</strong>
              <p>{circadianDetail?.can_reply ? "允许回复" : "建议继续休息"} · {circadianDetail?.peek_window_open ? "窥屏窗口打开" : "窥屏窗口关闭"}</p>
            </div>
            <div className="metric-tile">
              <span>熬夜压力</span>
              <strong>{countText(circadianDetail?.overnight_pressure, "")}</strong>
              <p>{circadianDetail?.is_burnthrough ? "熬穿已激活" : circadianDetail?.is_sleeping ? "睡眠中" : circadianDetail?.is_night ? "夜间节律" : "清醒时段"}</p>
            </div>
            <div className="metric-tile">
              <span>表达风格</span>
              <strong>{circadianDetail?.expression_style_label ?? "正常"}</strong>
              <p>回复抑制 {percent(circadianDetail?.response_suppression)}</p>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>主动驱动</h2>
            <span>{initiativeState?.effect_label ?? "等待状态"}</span>
          </div>
          <div className="drive-radar">
            <div>
              <span>主动意愿</span>
              <strong>{percent(initiativeState?.initiative_drive)}</strong>
            </div>
            <div>
              <span>无聊负荷</span>
              <strong>{percent(initiativeState?.boredom_load)}</strong>
            </div>
            <div>
              <span>孤独负荷</span>
              <strong>{percent(initiativeState?.loneliness_load)}</strong>
            </div>
            <div>
              <span>环境疲劳</span>
              <strong>{percent(initiativeState?.environment_fatigue_load)}</strong>
            </div>
            <div>
              <span>撤离倾向</span>
              <strong>{percent(initiativeState?.withdrawal_drive)}</strong>
            </div>
          </div>
          <p className="panel-note">{initiativeState?.effect_summary ?? "主动驱动会进入发言预测，不再只是日志里的装饰值。"}</p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>情绪账本</h2>
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
              <span>疲劳</span>
              <strong>{percent(emotionDetail?.environment_fatigue_load)}</strong>
            </div>
            <div>
              <span>主动意愿</span>
              <strong>{percent(emotionDetail?.initiative_drive)}</strong>
            </div>
            <div>
              <span>撤离倾向</span>
              <strong>{percent(emotionDetail?.withdrawal_drive)}</strong>
            </div>
          </div>
          <p className="panel-note">
            {emotionDetail?.feeling_text ?? "当前没有明显情绪波动"} · 已沉默 {countText(emotionDetail?.silence_seconds, "秒")} · 未回应 {emotionDetail?.unanswered_count ?? 0} 次
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
            <h2>自主账本</h2>
            <span>{autonomyDetail?.background_event_label || "无后台事件"}</span>
          </div>
          <div className="metric-wall">
            <div className={`metric-tile tone-${metricTone(autonomyDetail?.intention_drive ?? 0)}`}>
              <span>主动驱动总线</span>
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
            <h2>群聊整体状态</h2>
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
              <span>群聊烦躁 / 疲劳</span>
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
            这里只展示除当前对象之外、对本轮关系判断影响更大的用户，便于区分“单人状态”和“群聊里其他人的影响”。
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
              <span>群聊影响</span>
              <strong>{fixed(prediction?.scene_influence, 2)}</strong>
            </div>
            <div>
              <span>昼夜影响</span>
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

        <section className="panel panel-config-scope">
          <div className="panel-header">
            <h2>配置分级</h2>
            <span>
              {configScope
                ? `${configScope.summary.editable} / ${configScope.summary.total} 可调 · ${formatClock(configScope.updated_at)}`
                : "等待配置分级"}
            </span>
          </div>
          {!configScope ? (
            <div className="empty-state">
              {configScopeError || "正在读取 system / mixed / user 分级..."}
            </div>
          ) : (
            <>
              <div className="scope-summary-grid">
                <div className="scope-summary-tile">
                  <span>总模块</span>
                  <strong>{configScope.summary.total}</strong>
                </div>
                <div className="scope-summary-tile">
                  <span>系统级</span>
                  <strong>{configScope.summary.system}</strong>
                </div>
                <div className="scope-summary-tile">
                  <span>部分可改</span>
                  <strong>{configScope.summary.mixed}</strong>
                </div>
                <div className="scope-summary-tile">
                  <span>用户可改</span>
                  <strong>{configScope.summary.user}</strong>
                </div>
              </div>

              <div className="scope-columns">
                {(["user", "mixed", "system"] as const).map((scope) => {
                  const modules = configScopeGroups[scope];
                  return (
                    <div className="scope-column" key={scope}>
                      <div className="scope-column-header">
                        <h3>{scopeLabel(scope)}</h3>
                        <span>{modules.length} 个模块</span>
                      </div>
                      {modules.length === 0 ? (
                        <div className="empty-state inline">当前没有这一类模块。</div>
                      ) : (
                        <div className="scope-module-list">
                          {modules.map((item) => (
                            <article className="scope-module-row" key={item.module}>
                              <div className="scope-module-head">
                                <strong>{item.module}</strong>
                                <span className={`scope-badge scope-badge-${scope}`}>{scopeLabel(scope)}</span>
                              </div>
                              <p className="scope-domain-text">
                                语义域：{item.semantic_domains.length > 0 ? item.semantic_domains.join(", ") : "无"}
                              </p>
                              {item.user_editable_keys.length > 0 ? (
                                <div className="scope-chip-row">
                                  {item.user_editable_keys.map((key) => (
                                    <span className="scope-chip" key={`${item.module}-${key}`}>
                                      {key}
                                    </span>
                                  ))}
                                </div>
                              ) : (
                                <p className="scope-note">用户不可直接修改，系统锁定 {item.system_key_count} 项。</p>
                              )}
                            </article>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}
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
