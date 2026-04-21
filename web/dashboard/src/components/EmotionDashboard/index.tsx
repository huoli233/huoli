import { useEffect, useMemo, useRef, useState } from "react";

import "./styles.css";

type OverviewChannel = {
  channel_id: string;
  chat_type?: string;
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
  severity: string;
  value: number;
  display_value: string;
  trend: string;
  source_domain: string;
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
  active_signals: string[];
  impact_rank: number;
};

type SceneContext = {
  channel_id: string;
  scene_heat: string;
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
  scene_context: SceneContext;
  participant_impacts: ParticipantImpact[];
  timeline: TimelineEvent[];
};

type Prediction = {
  speak_probability: number;
  eta_seconds: number;
  eta_label: string;
  content_direction: string;
  reply_length: string;
  tone: string;
  driving_factors: string[];
  suppressing_factors: string[];
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

export function EmotionDashboard() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
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

  useEffect(() => {
    let ignore = false;

    async function bootstrap() {
      try {
        const authResponse = await fetch("/api/webui/auth/check", {
          credentials: "include",
        });
        const authData = await authResponse.json();
        if (ignore) {
          return;
        }
        const ok = Boolean(authData?.authenticated);
        setAuthenticated(ok);
        if (!ok) {
          setErrorMessage("当前状态页需要先通过 WebUI 登录，才能读取实时状态。");
          return;
        }

        const monitorResponse = await fetch("/api/heartflow/monitor", {
          credentials: "include",
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
          setAuthenticated(false);
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
    if (!authenticated || !selectedChannel) {
      return;
    }

    let cancelled = false;

    async function loadMonitor() {
      setLoading(true);
      setErrorMessage("");
      try {
        const response = await fetch(`/api/heartflow/monitor/${encodeURIComponent(selectedChannel)}`, {
          credentials: "include",
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
  }, [authenticated, selectedChannel]);

  useEffect(() => {
    if (!authenticated || !selectedChannel) {
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
  }, [authenticated, selectedChannel]);

  if (authenticated === null) {
    return <div className="dashboard-shell">状态台正在初始化…</div>;
  }

  if (!authenticated) {
    return (
      <div className="dashboard-shell">
        <div className="dashboard-auth-card">
          <p className="eyebrow">状态台不可用</p>
          <h1>当前还没有登录 WebUI</h1>
          <p>{errorMessage || "请先在主 WebUI 页面完成登录，状态台会自动复用同源认证 Cookie。"}</p>
          <a className="primary-link" href="/">
            返回 WebUI 登录
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Huoli Telemetry</p>
          <h1>机器人实时状态台</h1>
          <p className="subtle">
            只显示常驻状态与当前激活状态，未激活状态不会进入主页面。
          </p>
        </div>
        <div className="header-actions">
          <label className="channel-picker">
            <span>频道</span>
            <select
              value={selectedChannel}
              onChange={(event) => setSelectedChannel(event.target.value)}
            >
              {(overview?.channels ?? []).map((channel) => (
                <option key={channel.channel_id} value={channel.channel_id}>
                  {channel.channel_id} · {channel.chat_type === "group" ? "群聊" : "私聊"}
                </option>
              ))}
            </select>
          </label>
          <div className={`live-pill is-${connectionState}`}>{connectionState}</div>
        </div>
      </header>

      {loading && <div className="status-banner">正在同步状态数据…</div>}
      {errorMessage && <div className="status-banner is-error">{errorMessage}</div>}

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
                    <span>{signal.family}</span>
                    <span>{signal.trend}</span>
                    <span>{signal.source_domain}</span>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>群聊情境</h2>
            <span>{sceneContext?.scene_heat ?? "unknown"}</span>
          </div>
          <div className="detail-grid">
            <div className="detail-row">
              <span>场景热度</span>
              <strong>{sceneContext?.scene_heat ?? "-"}</strong>
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
            <span>{prediction?.eta_label ?? "-"}</span>
          </div>
          <div className="prediction-hero">
            <div className="prediction-circle">
              <span>{Math.round((prediction?.speak_probability ?? 0) * 100)}%</span>
            </div>
            <div>
              <p className="prediction-title">{prediction?.tone ?? "正常回应"}</p>
              <p className="prediction-subtitle">{prediction?.content_direction ?? "暂无预测方向"}</p>
              <p className="prediction-meta">建议长度：{prediction?.reply_length ?? "-"}</p>
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
