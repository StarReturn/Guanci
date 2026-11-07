# Guanci · 数字人（数字亲人）后端

面向陪伴场景的 AI 数字人后端：用户上传亲人照片与声音，生成可实时语音对话的数字人。支持照片活化、形象构建、实时语音通话（ASR→LLM→TTS 流式）、文字聊天、方言语音、音色库、长期记忆与分享。

> 本仓库为**开源版**：包含数字人核心链路与基础设施，不包含商业化模块（见下）。克隆后**不能直接运行**，缺失项与获取方式见[「不能直接跑」清单](#克隆后为什么不能直接跑)。

---

## 架构总览

```
App（Android）
   │ REST / WebSocket
   ▼
client API（本仓库，FastAPI）──► PostgreSQL(pgvector) / Redis / MinIO / RabbitMQ(可选)
   │                                    ▲
   │ HTTP                               │ 回源拉取照片/回传素材包
   ▼                                    │
推理服务（不在本仓库）───────────────────┘
   ├─ photo-activation：照片活化（LivePortrait 逐帧推理，GPU 服务）
   └─ dhlive-builder：DH_live 素材包构建（打包/加密/校验/参考渲染质检）
云端 API：DashScope（qwen-plus LLM / CosyVoice TTS / Paraformer ASR）
```

**渲染引擎为 [DH_live](https://github.com/zixianwang21/DH_live)**（轻量 talking-head 渲染，WASM 实现）。数字人形象以"素材包"形式分发：本仓库负责调度构建与校验，渲染运行在端侧（Android WebView + WASM）。

### 核心链路

- **创建流水线**：照片预检（内容安全门 + `preprocess_token`）→ 授权链（consents）→ 照片活化 → DH_live 素材包构建（SHA-256 指纹完整性校验）→ 参考渲染质检 → 激活发布
- **实时对话**（`/api/v2/realtime/stream`，WebSocket）：流式 ASR → LLM → 流式 TTS 整轮单会话合成，支持自然打断（barge-in）、句级字幕时间戳、情绪指令
- **方言语音**：席位级 TTS 口音指令（`preset_voices.tts_instruction`）+ LLM 方言书写指令（`llm_dialect_prompt`）双层架构；音色库支持真人参考克隆
- **长期记忆**：RQ 异步提取/会话摘要（幂等）→ pgvector 检索 → 分层注入（人格→关系画像→记忆→会话摘要→滚动历史）

## 目录结构

```
app/
  api/v1/routes/   # REST：数字人 CRUD/构建/聊天/分享/音色库/记忆/对话…
  api/v2/routes/   # WebSocket 实时通话、素材包下载
  services/        # 领域服务：digital_human/dhlive_*/memory_*/tts·asr·llm 引擎客户端…
  core/            # 配置/数据库/JWT/指标
  middleware/      # 鉴权/限流
  worker/          # RQ 异步任务（构建/删除）
  schemas/ domain/
migrations/        # PostgreSQL DDL（含 pgvector）
tests/unit/        # 无依赖契约测试
```

## 技术栈

Python 3.11 · FastAPI · WebSocket · SQLAlchemy · PostgreSQL(pgvector) · Redis(RQ) · MinIO · Docker

## 克隆后为什么不能直接跑

| # | 缺失项 | 说明与获取方式 |
|---|---|---|
| 1 | **DH_live 渲染运行时（WASM）** | 渲染引擎为开源项目 [DH_live](https://github.com/zixianwang21/DH_live)，但其 **WASM 运行时与素材包构建工具链不随本仓库分发**（随端侧 SDK 与构建服务交付，需按 DH_live 项目说明自行构建或联系仓库作者获取授权版本） |
| 2 | **照片活化服务** | LivePortrait GPU 推理服务（独立容器 + 模型权重），需按 [LivePortrait](https://github.com/KwaiVGI/LivePortrait) 自行部署，本仓库通过 `photo_activation_service.py` 的 HTTP 地址对接 |
| 3 | **云端 API Key** | DashScope（LLM/TTS/ASR）密钥全部走环境变量，见 `app/core/config.py`，需自行开通[阿里云百炼](https://bailian.console.aliyun.com/) |
| 4 | **基础设施** | PostgreSQL 16 + pgvector、Redis、MinIO 需自行部署（`migrations/` 提供全部 DDL；无 docker-compose，避免误导为"开箱即用"） |
| 5 | **管理端与运营面** | 管理后台（音色库运营/供应商配置等）不在本仓库 |

开源版还做了如下裁剪（接口桩化，`app/services/entitlement_stub.py`）：**会员配额体系**（reserve/check/confirm 一律放行）、**支付**（微信/支付宝适配层）、**多账号 API 限流矩阵**（单账号直连）。生产部署请自行实现配额与风控。

## 快速开始（自建环境）

```bash
pip install -r requirements.txt        # Python 3.11+
# 1) 准备 PostgreSQL(pgvector)/Redis/MinIO，执行 migrations/*.sql
# 2) 配置环境变量（关键项见下表）
export APP_ROLE=client
export DASHSCOPE_API_KEY=sk-xxx
export STORAGE_PUBLIC_ENDPOINT_URL=http://<minio-host>:9000   # 活化/构建服务回源用
# 3) 自建活化与构建服务后，配置其地址
uvicorn app.main:app --host 0.0.0.0 --port 8091
```

### 关键环境变量（节选）

| 变量 | 用途 |
|---|---|
| `DASHSCOPE_API_KEY` | LLM/TTS/ASR 云端调用 |
| `JWT_SECRET` | 登录态签名（必填，否则启动报错） |
| `DATABASE_URL` / `REDIS_URL` / MinIO 三件套 | 基础设施连接 |
| `TTS_PROVIDER=aliyun_cosyvoice` | 默认 TTS 引擎 |
| 活化/构建服务地址 | 对接自建推理服务 |

## License

代码以源码形式提供，供学习与研究。DH_live、LivePortrait、Common Voice 等第三方组件/数据遵循各自许可。
