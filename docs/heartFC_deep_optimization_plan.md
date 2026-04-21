# heartFC 渐进拆分与第一阶段修复方案

## 结论

`src/chat/heart_flow/heartFC_chat_enhanced.py` 已经承担过多职责，不能继续把配置、监控、预测和 Web API 全部塞回主文件。第一阶段采用渐进拆分：先把关键阈值配置化，并新增只读状态导出能力；预测功能和主循环拆分放到后续阶段。

## 第一阶段已确定范围

- 新增 `src/chat/heart_flow/heartfc_thresholds.py`，统一读取 `[heartfc_thresholds]`，非法值按字段回退到历史硬编码默认值。
- 新增 `src/chat/heart_flow/heartfc_state_exporter.py`，只读聚合活跃 Heartflow 实例、缓存世界快照、D6 能量、情感驱动、休息态和 planner 状态。
- 新增 WebUI 只读接口：
  - `GET /api/heartflow/chats`
  - `GET /api/heartflow/state/{channel_id}`
  - `GET /api/heartflow/thresholds`
- 不新增 `config/thresholds.toml`，统一使用既有 `config/core_config.toml` 热重载体系。
- 不实现 `PUT` 配置接口，避免第一阶段引入写配置并发风险。
- 不实现用户行为预测，避免在主脑尚未瘦身前新增重复状态系统。

## 当前已配置化的阈值

```toml
[heartfc_thresholds]
rest_energy_ratio = 0.10
rest_thinking_ratio = 0.08
loafing_high = 0.60
loafing_medium = 0.45
loafing_idle = 0.45
quiet_high = 0.55
quiet_medium = 0.45
quiet_watch_low = 0.60
avoidance_high = 0.40
avoidance_medium = 0.45
social_low_willingness = 0.45
boredom_peek_trigger = 55.0
boredom_scan_trigger = 40.0
boredom_downgrade_trigger = 35.0
loafing_social_inhibit = 0.40
boredom_reply_suppress_high = 0.60
boredom_reply_suppress_medium = 0.30
boredom_activation_reduce = 0.75
boredom_drift_trigger = 0.50
```

这些默认值等价于迁移前的硬编码行为。后续调参应优先改配置，不直接改 `heartFC_chat_enhanced.py`。

## 后续拆分路径

1. 第二阶段抽 `heartfc_governors.py`：迁移 `BehaviorGovernorVerdict`、`RestGovernorVerdict`、`ModelGovernorVerdict` 和对应 evaluate/summarize 逻辑。
2. 第三阶段抽 `heartfc_watch.py`：迁移 watch transition、visibility decay、freshness drift。
3. 第四阶段抽 `heartfc_proactive.py`：迁移 proactive opportunity、decision、idle proactive、proactive reply。
4. 第五阶段评估 `heartfc_reply_flow.py`：只在前几阶段稳定后迁移 reply execution 与 settlement。

## 验证要求

- `uv run ruff check src --select F821`
- 新增模块与 WebUI router 可 import。
- 默认配置下，配置化前后的阈值行为保持一致。
- `/api/heartflow/state/{channel_id}` 只查询活跃实例，不自动创建 chat。
- `/api/heartflow/thresholds` 只返回有效配置，不写文件。
