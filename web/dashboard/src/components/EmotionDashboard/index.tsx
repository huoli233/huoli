import { useEffect, useMemo, useRef, useState } from "react";

import "./styles.css";

type OverviewChannel = {
  channel_id: string;
  chat_type?: string;
  chat_type_label?: string;
  class_name?: string;
  idle_seconds?: number | null;
  platform?: string;
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
  interaction_count: number;
  active_signals: string[];
  impact_rank: number;
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
  initiative_state?: InitiativeState;
  scene_context: SceneContext;
  participant_impacts: ParticipantImpact[];
  timeline: TimelineEvent[];
  display_policy?: DisplayPolicy;
};

type Prediction = {
  speak_probability: number;
  probability_percent?: number;
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
  behavior_governor: BehaviorGovernorDetail;
  rest_governor: RestGovernorDetail;
  model_governor: ModelGovernorDetail;
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
  active: ["无聊/孤独/环境疲劳/撤离/主动意愿", "情绪低落/好奇心/社交欲显著变化", "浅睡/深睡/熬穿/黎明恢复/睡眠债", "烦躁/压力/创伤/混乱/伪装", "关系好感/信任显著偏高或偏低", "冷却窗口/等待时长/重新接入/群聊升温"],
  detail: ["资源账本的聊天值和思考值", "昼夜节律、睡眠债、困意和熬夜压力", "情绪账本、主动驱动和当前感受", "群聊感知、话题焦点、活跃人数", "目标用户关系、好感、信任、压力", "发言预测的驱动和抑制因素"],
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

function countText(value: number | undefined | null, unit: string): string {
  return `${Math.round(Number(value ?? 0))}${unit}`;
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

export function EmotionDashboard() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [selectedChannel, setSelectedChannel] = useState("");
  const [packet, setPacket] = useState<MonitorPacket | null>(null);
  const [loading, setLoading] = useState(false);
  const [connectionState, setConnectionState] = useState("idle");
  const [errorMessage, setErrorMessage] = useState("");
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
  const timeline = packet?.presentation?.timeline ?? [];
  const prediction = packet?.prediction;
  const resourceDetail = packet?.presentation?.resource_detail;
  const circadianDetail = packet?.presentation?.circadian_detail;
  const emotionDetail = packet?.presentation?.emotion_detail;
  const behaviorDetail = packet?.presentation?.behavior_detail;
  const initiativeState = packet?.presentation?.initiative_state;
  const displayPolicy = packet?.presentation?.display_policy ?? fallbackDisplayPolicy;
  const selectedOverview = (overview?.channels ?? []).find(
    (channel) => channel.channel_id === selectedChannel,
  );
  const predictionPercent =
    prediction?.probability_percent ?? Math.round((prediction?.speak_probability ?? 0) * 100);

  useEffect(() => {
    let ignore = false;

    async function bootstrap() {
      try {
        const monitorResponse = await fetch("/api/heartflow/monitor", {
          credentials: "same-origin",
        });
        const monitorData = await monitorResponse.json();
        if (ignore) {
          return;
        }
        const monitor = monitorData?.monitor as OverviewPayload | undefined;
        setOverview(monitor ?? null);
        const initialChannel = monitor?.channels?.[0]?.channel_id ?? "";
        setSelectedChannel(initialChannel);
      } catch (error) {
        if (!ignore) {
          setErrorMessage(`初始化状态页失败: ${String(error)}`);
        }
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
            <span>群聊 / 私聊</span>
            <select
              value={selectedChannel}
              onChange={(event) => setSelectedChannel(event.target.value)}
            >
              {(overview?.channels ?? []).map((channel) => (
                <option key={channel.channel_id} value={channel.channel_id}>
                  {channel.chat_type_label ?? (channel.chat_type === "group" ? "群聊" : "私聊")} · {channel.platform || "本地"} · {channel.channel_id.slice(0, 8)}
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
          <span>当前入口</span>
          <strong>{selectedOverview?.chat_type_label ?? "群聊 / 私聊"}</strong>
          <p>{selectedOverview?.platform || "本地"} · {selectedChannel ? selectedChannel.slice(0, 12) : "暂无活跃会话"}</p>
        </article>
        <article className="command-card is-score">
          <span>是否该聊</span>
          <strong>{predictionPercent}%</strong>
          <p>{prediction?.decision_label ?? "等待状态"} · {prediction?.eta_label ?? "-"}</p>
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
              <span>休息姿态</span>
              <strong>{behaviorDetail?.rest_governor.posture_label ?? "-"}</strong>
            </div>
            <div className="detail-row">
              <span>当前语气</span>
              <strong>{behaviorDetail?.reply_decision.suggested_tone ?? "-"}</strong>
            </div>
          </div>
          <p className="panel-note">
            {behaviorDetail?.reply_decision.decision_reason ?? "暂无行为裁定理由"}
          </p>
          <div className="topic-wrap">
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
            <h2>群聊情境</h2>
            <span>{sceneContext?.scene_heat_label ?? humanSceneHeat(sceneContext?.scene_heat)}</span>
          </div>
          <div className="detail-grid">
            <div className="detail-row">
              <span>场景热度</span>
              <strong>{sceneContext?.scene_heat_label ?? humanSceneHeat(sceneContext?.scene_heat)} · {percent(sceneContext?.scene_heat_score)}</strong>
            </div>
            <div className="detail-row">
              <span>活跃人数</span>
              <strong>{sceneContext?.active_user_count ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>会话阶段</span>
              <strong>{sceneContext?.session_phase || "-"}</strong>
            </div>
            <div className="detail-row">
              <span>热点用户</span>
              <strong>{sceneContext?.hot_count ?? 0}</strong>
            </div>
            <div className="detail-row">
              <span>温热用户</span>
              <strong>{sceneContext?.warm_count ?? 0}</strong>
            </div>
          </div>
          <div className="topic-wrap">
            {(sceneContext?.topic_focus ?? []).length > 0 ? (
              sceneContext?.topic_focus.map((topic) => (
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
            <h2>目标用户影响</h2>
            <span>{participantImpacts.length > 0 ? participantImpacts[0].display_name : "暂无"}</span>
          </div>
          {participantImpacts.length === 0 ? (
            <div className="empty-state">当前没有明确焦点用户，用户影响区保持空状态。</div>
          ) : (
            participantImpacts.map((user) => (
              <article className="participant-card" key={user.user_id}>
                <header>
                  <div>
                    <h3>{user.display_name}</h3>
                    <p>{user.relationship_label}</p>
                  </div>
                  <strong>{user.impact_rank.toFixed(2)}</strong>
                </header>
                <div className="detail-grid compact">
                  <div className="detail-row">
                    <span>好感</span>
                    <strong>{user.rapport_score.toFixed(1)}</strong>
                  </div>
                  <div className="detail-row">
                    <span>信任</span>
                    <strong>{user.trust_score.toFixed(1)}</strong>
                  </div>
                  <div className="detail-row">
                    <span>烦躁</span>
                    <strong>{user.irritation_load.toFixed(1)}</strong>
                  </div>
                  <div className="detail-row">
                    <span>创伤</span>
                    <strong>{user.trauma_load.toFixed(2)}</strong>
                  </div>
                  <div className="detail-row">
                    <span>压力</span>
                    <strong>{user.pressure_load.toFixed(1)}</strong>
                  </div>
                  <div className="detail-row">
                    <span>互动次数</span>
                    <strong>{user.interaction_count}</strong>
                  </div>
                </div>
                <div className="topic-wrap">
                  {user.active_signals.length > 0 ? (
                    user.active_signals.map((signal) => (
                      <span className="topic-chip is-secondary" key={signal}>
                        {signal}
                      </span>
                    ))
                  ) : (
                    <span className="empty-state inline">当前没有用户向激活信号。</span>
                  )}
                </div>
              </article>
            ))
          )}
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
              <p className="prediction-meta">语气：{prediction?.tone ?? "正常回应"} · 长度：{prediction?.reply_length ?? "-"} · ETA：{prediction?.eta_label ?? "-"}</p>
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
