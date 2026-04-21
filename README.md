# huoli

huoli 是对 MaiBot 能力体系进行原创重写的项目。

## 当前阶段

- 已完成基础骨架
- 已完成配置系统与模型配置加载
- 已完成日志系统与数据库初始化
- 已完成 LLM 客户端抽象层
- 已完成最小可运行主程序入口

## 目录结构

- `src/config`: 配置加载与配置模型
- `src/common`: 日志、数据库、服务与生命周期
- `src/llm_models`: 模型客户端与请求抽象
- `src/chat`: 聊天相关模块（待逐步重写）
- `src/memory_system`: 记忆系统（待逐步重写）
- `src/plugin_system`: 插件系统（待逐步重写）
- `src/webui`: WebUI 后端（待逐步重写）

## 启动

1. 复制 `template/template.env` 为 `.env`
2. 填写 `config/model_config.toml` 的 API Key
3. 安装依赖后运行 `bot.py`
