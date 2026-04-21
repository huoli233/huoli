# huoli - AI 员工手册 (Dark Factory)

## 项目目标

完成任务 task_1775854176061_d3je22：扫描项目结构并理解架构

## 核心原则（三不）

- ❌ 人不写代码 — 所有代码由 AI 自动编写
- ❌ 人不审代码 — AI 自审 + Lint/Build 验证
- ❌ 人不测代码 — AI Build + 测试自动验证

## 强制工作流程

### 1. 领取任务
读取 `.dark-factory/task.json`，选择一个 `passes: false` 的任务。
选择标准：优先级高的先做，有依赖的等依赖完成。

### 2. 理解需求
仔细阅读任务描述。如果有 steps 字段，按步骤执行。
先用 read_file / glob / grep 了解相关代码现状。

### 3. 实现功能
遵循现有代码风格和架构模式。
不要过度工程化——KISS 原则。
每完成一个逻辑单元就验证一次。

### 4. 测试验证（强制！）
- 运行 lint 检查（无错误才能继续）
- 运行 build（构建成功才能继续）
- UI 相关修改必须在浏览器中验证

### 5. 记录进度
写入 `.dark-factory/progress.txt`：
```
## [ISO日期] - Task: [任务描述]

### What was done:
- [具体做了什么]

### Testing:
- [怎么测的，结果如何]

### Notes:
- [给后续 Agent 的信息]
```

### 6. 更新任务状态
修改 task.json：将当前任务的 `passes` 改为 `true`，填写 `completedAt`。

### 7. Git 提交
```bash
git add .
git commit -m "[任务简述] - completed"
```
规则补充：
- 每完成一轮代码改动，并完成该轮对应的验证后，立即执行一次提交
- 不要把多个改动轮次堆积到同一个提交里
- 后续继续修改时，继续在上一个提交基础上开发，并在该轮结束后再次提交
- 分支节奏固定为：
  - 最新改动先进入 `test`（测试版）
  - `develop`（开发版）故意落后 `test` 5 到 10 个改动提交；只有当 `test` 已经累计领先 `develop` 至少 5 个提交，并且这些提交已经完成验证时，才允许把这一批次推进到 `develop`
  - `main`（稳定版）不参与日常开发提交，也不按固定节奏自动更新；只有在用户明确确认“当前版本已经足够稳定、可以沉淀为稳定版”之后，才允许把经过验证的版本推进到 `main`
- 如果是新的功能/重构/机制修复，默认先提交到 `test`
- `develop` 的作用是作为阶段性开发基线和回退参考，不追求和 `test` 同步
- `main` 禁止直接承接日常开发提交；稳定版更新必须由用户明确触发

## 阻塞处理

如果无法完成任务：
- ❌ 不要 git commit
- ❌ 不要把 passes 设为 true
- ✅ 在 progress.txt 记录阻塞原因
- ✅ 输出 BLOCKED: [原因]

## 关键规则

1. One task per session — 专注一个任务
2. Test before mark complete — 全部验证通过才算完成
3. Browser test for UI — UI 改动必须浏览器验证
4. Document everything — progress.txt 是给后续 Agent 看的
5. Commit after every completed change round — 每完成一轮改动并验证通过，就立即提交
6. Branch flow: `test` carries latest validated changes, `develop` lags 5-10 commits, `main` updates only after explicit user approval
7. Never remove tasks — 只改 passes: false → true
8. Stop if blocked — 阻塞就停，不要假装完成
