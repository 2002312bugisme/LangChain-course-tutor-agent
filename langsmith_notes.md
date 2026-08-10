# LangSmith 功能地图（截至 2026-08，依据 docs.smith.langchain.com 官方文档源码查证）

> 写作约定：功能名 = 官方当前名称；状态 = GA（正式）/ Beta（试用）；每条关键能力均为文档可复现的行为描述。
> 主要来源：https://docs.smith.langchain.com/langsmith/（observability、manage-datasets、manage-prompts、use-studio、llm-gateway 等页面）

## 一、可观测性体系（原 "Monitoring" 已并入此处）

### 1. Tracing（追踪）— GA，核心功能
- 完整记录每次 LLM 调用链路（Trace）：每层的 Prompt 原文、模型输出、Token 用量、每节点耗时
- 接入方式：Python 侧设 `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`，或用 langsmith SDK 手动打点
- 通过 `run_name` / `tags` / `metadata` 给单次运行打标记，便于检索过滤

### 2. Debug（单条调试）
- 点进单条 Trace 逐层查看：输入/输出/耗时/Token，定位慢节点或出错节点

### 3. Dashboards（看板）
- 宏观视图：Token 消耗趋势、QPS、错误率、平均延迟（Latency）

### 4. Cost Tracking（成本追踪）
- 按模型/项目聚合的 Token 费用统计与预估

### 5. Alerts（告警）
- 阈值告警：错误率、延迟、成本超限时触发通知（可接 webhook/邮件）

## 二、测试与评估体系

### 6. Datasets（数据集）— GA
- 管理测试数据集：用户真实输入、边界用例（edge cases）
- 支持数据集版本与拆分（train/test）

### 7. Experiments（实验）
- 在同一数据集上跑新旧版本（改 Prompt / 换模型）的对比实验
- 输出逐条并排对比结果，量化差异

### 8. Evaluators（评估器）
- 三类：规则型（关键词/正则）、LLM-as-a-judge（模型打分，如相关性/幻觉检测）、Code evaluator（自定义 Python 打分函数）
- 可挂到实验上（离线）或生产 Trace 上（在线评估）

### 9. Annotation Queues（标注队列）
- 人工反馈通道：打分、纠正回答、贴标签
- 产出高质量标注数据，可回流为测试集或用于微调

## 三、提示词与调试工具

### 10. Prompts（提示词管理）— GA
- 提示词与代码解耦，云端管理；每个 Prompt 有 commit 历史（hash）
- 支持 Commit Tag、按 `name:production` / `name:<commit_hash>` 拉取
- 支持 Staging / Production 环境 promote / rollback
- 注意："Prompt Hub" 现在专指 LangChain Hub 的公共社区提示词库

### 11. Playground（演练场）— GA
- 网页端免代码试玩：选模型、调参数（temperature 等）、测 Prompt
- 一键把调好的 Prompt 存回 Prompts 仓库

### 12. Chat（对话界面）
- 网页端多轮对话调试入口，与 Playground 互补

### 13. Context Hub（上下文中心）
- 存**指令 + 工具的版本化包**（skills，如"客服技能包"），不是普通文本模板
- 可整体 promote 到 staging/production 环境，供多个项目复用

### 14. Studio（工作台）— 面向 LangGraph
- 可视化状态机流转调试
- 支持 **Interrupt**：在指定节点前/后暂停 → 人工修改 state → **Continue** 恢复（human-in-the-loop）
- 支持编辑节点输出后 **Fork** 出新 run 继续执行

## 四、部署与运行

### 15. Deployments（部署）
- 将 LangChain / LangGraph 应用部署为线上 API 服务（依托 LangGraph Cloud / Serverless）
- 内置高并发、队列、状态持久化

### 16. Sandboxes（沙盒）— GA
- 轻量级隔离运行环境：不污染生产，安全试运行/执行自动化脚本

### 17. Engine（引擎）
- LangSmith 运行时引擎（新功能，管理应用运行时的执行环境）

### 18. Assistants（助手）
- 管理可复用的助手配置（系统提示 + 工具组合的封装）

### 19. Automations（自动化）
- 自动化工作流：如自动触发评估、自动告警动作

## 五、模型网关

### 20. LLM Gateway（模型网关）— Beta
- 统一模型访问入口：集中管理各厂商 API Key（Provider Secrets）、额度（Credits）
- 提供路由、缓存、统一观测能力；端点：gateway.smith.langchain.com

## 学习优先级建议（实用排序）
1. **Tracing + Debug**：先学会看自己的调用链路（每个项目必用）
2. **Playground + Prompts**：快速调 Prompt 并版本化
3. 应用变复杂（RAG / 多 Agent）后：**Datasets + Experiments + Evaluators** 做量化回归
4. 上线后：**Dashboards + Alerts + Cost Tracking** 盯生产
5. 需要对外服务时再看 **Deployments / LLM Gateway**

## 已知不确定项
- "Models（模型注册表）"：官方文档未找到独立页面，可能在 UI 存在但无文档——未确认，勿写入正式文档
