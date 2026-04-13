# BUG-001 修复记录

## 基本信息
- Bug 编号: `BUG-001`
- 负责人: `01KP29EKTS9DTWDQPYSZ096CN7`
- 来源: `docs/bug_fix_list.md`
- 修复日期: `2026-04-13`

## 涉及文件
- `src/chat/brain_chat/brain_planner.py`
- `src/chat/brain_chat/brain_chat.py`

## 修复范围
- 拆分 `brain_planner` 的故障回退与业务性 `complete_talk`
- 引入显式降级动作 `planner_degraded`
- 在执行层为 `planner_degraded` 增加最小接线，避免被解释成自然收束

## 关键改动
1. `BrainPlanner` 新增 `DEGRADED_ACTION_TYPE = "planner_degraded"`。
2. 解析单动作时：
   - 缺省 `action` 不再回退为 `complete_talk`
   - 无效动作直接生成显式降级动作
   - 解析异常不再伪装成自然结束
3. 主规划流程中：
   - LLM 请求失败
   - 空响应
   - JSON 解析失败
   - 无可用动作
   统一改为 `_create_degraded_actions(...)`
4. `brain_chat.py` 新增 `planner_degraded` 执行分支：
   - 记录 `planner_degraded`
   - `action_done=False`
   - 短暂等待后继续下一轮

## 验证方式
- 搜索 `planner_degraded`，确认规划端与执行端均已接线
- 检查 `brain_planner.py` 中原先所有故障回退点，确认已不再落到 `complete_talk`
- 检查 `brain_chat.py` 中 `planner_degraded` 不会触发 `complete_talk` 的收束语义

## 与历史问题关系
- 与历史问题重复: 否
- 关联问题: `docs/bug_fix_list.md` 中 `BUG-001`
- 与既有修复记录关系: 当前为该问题的正式修复实现，早期仅有计划/骨架文档

## 影响主线
- 修复 P0: 避免模型故障被误判为对话自然结束
- 减少过早停话、错误沉默和错过应答时机
