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
git commit
```
默认使用仓库内模板文件 `.gitmessage-cn.txt` 生成中文提交说明。
规则补充：
- 每完成一轮代码改动，并完成该轮对应的验证后，立即执行一次提交
- 不要把多个改动轮次堆积到同一个提交里
- 后续继续修改时，继续在上一个提交基础上开发，并在该轮结束后再次提交
- 整个过程中的进度说明、提交说明、同步说明、删除说明、回滚说明，默认全部使用中文
- 分支节奏固定为：
  - 最新改动先进入 `test`（测试版）
  - `develop`（开发版）故意落后 `test` 5 到 10 个改动提交，用作开发回退基线；即使达到 5 个提交阈值，也不自动同步，必须先向用户汇报当前 ahead 数量，再由用户明确决定是否同步
  - `main`（稳定版）不参与日常开发提交，也不按固定节奏自动更新；只有在用户明确确认“当前版本已经足够稳定、可以沉淀为稳定版”之后，才允许把经过验证的版本推进到 `main`
- 如果是新的功能/重构/机制修复，默认先提交到 `test`
- `develop` 的作用是作为阶段性开发基线和回退参考，不追求和 `test` 同步
- `main` 禁止直接承接日常开发提交；稳定版更新必须由用户明确触发
- 任何同步动作开始前，必须先用中文明确说明：
  - 当前所在分支
  - 目标同步分支
  - `test` 相对 `develop`/`main` 的 ahead 数量
  - 本次同步会带过去哪些提交范围
- 所有新提交必须使用中文标题 + 中文正文，禁止只写一句简短英文
- 提交标题格式固定为：`类型：模块 - 本轮核心改动`
- 提交正文固定包含 6 段：
  - `本轮目标：`
  - `具体改动：`
  - `修复问题：`
  - `影响范围：`
  - `验证结果：`
  - `回滚说明：`
- 每次回复用户时，也必须提供中文结构化说明：
  - `本次改动`
  - `本次修复/优化点`
  - `影响范围`
  - `验证`
  - `提交`
  - `分支状态`
  - `回滚参考`
- 每次都要明确说明这次提交进入了哪个分支；如果只进了 `test`，要直接写明 `develop/main 尚未同步`
- 如果没有 push，必须明确写明 `仅本地提交，远程未更新`
- 只要涉及“删除模块 / 删除文件 / 删除接口 / 删除页面 / 删除状态字段 / 删除算法入口”，必须额外执行删除分支规则：
  - 先保留一条 `已存在分支`：命名格式固定为 `preserve/<主题>`，保存删除前的完整可运行版本，专门用于快速回滚
  - 再维护一条 `已删除分支`：命名格式固定为 `delete/<主题>`，承载实际删除后的版本，并在这条线上继续验证
  - 没有 `已存在分支` 时，不允许直接做结构性删除
  - 删除完成后，必须用中文说明：
    - 删了什么
    - 删除影响到哪些模块
    - `已存在分支` 是哪条
    - `已删除分支` 是哪条
    - 如果出问题要回滚到哪里
- 删除类改动默认不同步到 `develop` / `main`，除非用户明确确认同步

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
7. Chinese structured commit + rollback notes are mandatory for every new change round
8. Never remove tasks — 只改 passes: false → true
9. Stop if blocked — 阻塞就停，不要假装完成
