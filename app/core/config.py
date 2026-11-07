from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DigitalHuman Mobile Keyframe API"
    app_env: str = "local"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    public_base_url: str = "http://localhost:8090"

    database_url: str = "postgresql+psycopg://digitalhuman:digitalhuman@postgres:5432/digitalhuman"
    redis_url: str = "redis://redis:6379/0"

    local_storage_root: Path = Path("./storage")

    storage_provider: str = "s3"
    storage_endpoint_url: str = "http://minio:9000"
    # Public HTTPS endpoint used only when signing URLs for external services
    # such as Alibaba Cloud voice enrollment. Keep the internal endpoint above
    # for container-to-container traffic.
    storage_public_endpoint_url: str | None = None
    storage_bucket: str = "digitalhuman"
    storage_access_key: str = "digitalhuman"
    storage_secret_key: str = "digitalhuman"
    storage_region: str = "us-east-1"
    storage_use_ssl: bool = False
    storage_presigned_url_expires_seconds: int = 3600
    # 用户头像等资料类对象的签名有效期（App 每次启动拉 /me 会刷新签名）
    storage_presigned_url_expires_seconds: int = 3600

    keyframe_package_prefix: str = "avatars/keyframes"
    # ---- mobile_keyframe_v3 canvas / inference geometry ----
    # Legacy single-value ``keyframe_resolution`` is removed; production uses
    # explicit inference and canvas dimensions. See
    # docs/拼帧数字人实时对话与3比4画布完整调整方案.md §8.1.
    keyframe_package_format_version: int = 3
    keyframe_inference_width: int = 768
    keyframe_inference_height: int = 768
    keyframe_canvas_profile: str = "portrait_3_4"
    keyframe_canvas_width: int = 768
    keyframe_canvas_height: int = 1024
    keyframe_mouth_levels: int = 7
    keyframe_motion_lead_ms: int = 200
    keyframe_crossfade_ms: int = 40
    keyframe_quality_profile: str = "mvp_high"
    # ---- Vision model paths (read-only mount from /data) ----
    vision_model_root: str = "/models/vision"
    mediapipe_face_landmarker_path: str = "/models/vision/mediapipe/face_landmarker.task"
    rembg_model_name: str = "u2net"
    rembg_model_path: str = "/models/vision/rembg/u2net.onnx"
    keyframe_frame_format: str = "webp"  # "webp" (~60% smaller) or "png"
    keyframe_mouth_frame_count: int = 30
    # Observability
    log_format: str = "json"   # "json" (structlog) or "plain"
    log_level: str = "INFO"
    metrics_enabled: bool = True

    # Auth: split-key design so a leaked client key can't trigger builds,
    # and a leaked admin key can't be embedded in a mobile app.
    # Both are comma-separated to support zero-downtime rotation:
    #   ADMIN_API_KEY="new-key,old-key"
    admin_api_key: str = ""
    client_api_key: str = ""
    provider_secret_encryption_key: str = ""

    # Optional one-time bootstrap. If both values are set and no account with
    # this username exists, one super administrator is created at startup.
    bootstrap_super_admin_username: str = ""
    bootstrap_super_admin_password: str = ""

    # Role of THIS process — admin or client. Set via APP_ROLE env var.
    # Controls which router the entrypoint mounts.
    app_role: str = "all"  # "admin" | "client" | "all" (legacy compat)

    # Rate limits (slowapi syntax). Only enforced on client app.
    rate_limit_tts: str = "2/second"
    rate_limit_phoneme: str = "5/second"
    rate_limit_realtime: str = "1/second"
    rate_limit_package_read: str = "10/second"
    rate_limit_storage_uri: str = "memory://"  # or redis://redis:6379/1

    # ---- User system (P3) ----
    # JWT signing secret. CHANGE IN PRODUCTION (openssl rand -hex 32).
    # If empty in production, app refuses to start.
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_sec: int = 3600        # 1 hour
    jwt_refresh_token_ttl_sec: int = 2592000    # 30 days
    jwt_issuer: str = "guanci"

    # SMS verification
    sms_provider: str = "mock"   # "mock" (dev) | "aliyun" | "tencent"
    sms_aliyun_access_key: str = ""
    sms_aliyun_access_secret: str = ""
    sms_aliyun_sign_name: str = ""
    sms_aliyun_template_code: str = ""
    sms_code_ttl_sec: int = 300              # 5 min
    sms_resend_cooldown_sec: int = 60        # min gap between sends
    sms_ip_daily_limit: int = 10             # max codes per IP per day
    sms_phone_daily_limit: int = 5           # max codes per phone per day

    # WeChat OAuth (https://open.weixin.qq.com/)
    wechat_app_id: str = ""
    wechat_app_secret: str = ""
    # If your mobile app is a "Website App" use "web"; for "Mobile App" use "app"
    wechat_platform: str = "web"

    # Default per-user quota (used when seeding UserQuotaRecord on signup)
    default_quota_tts_daily: int = 100
    default_quota_build_daily: int = 5
    default_quota_download_daily: int = 20

    # Audit log retention (days). Older entries purged by a future cron.
    audit_retention_days: int = 90
    digital_human_delete_retention_days: int = 30

    tts_provider: str = "aliyun_cosyvoice"  # aliyun_cosyvoice | cosyvoice | fish_speech | indextts
    # TTS 朗读语速（CosyVoice speech_rate，合法域 0.5–2.0，默认 1.0 不传参）。
    # "说话慢"对策：只提速朗读，音调(pitch_rate)与 16kHz 采样率不动；
    # A/B 试听建议 1.0/1.1/1.2。注意 v3/v4/v3.5 系模型参数面与 v1/v2 有
    # 差异，启用前须以线上实际模型实测（重整方案 D1.3.2）
    tts_speech_rate: float = 1.0
    # 沙盒模拟支付（收银台未接入期间的测试通路）：POST /payments/orders/{no}/simulate-pay
    # 接入真实微信/支付宝收银台后置 false 关闭
    membership_sandbox_pay: bool = True
    # RabbitMQ（支付任务底座：durable+ack，杜绝 Redis 队列易失问题）
    mq_url: str = ""  # 开源版不含支付任务队列（RabbitMQ），如需请自行配置
    fish_speech_endpoint: str = "http://fish-speech:8080"
    fish_speech_tts_path: str = "/v1/tts"
    fish_speech_health_path: str = "/v1/health"
    fish_speech_timeout_seconds: int = 30
    fish_speech_api_key: str | None = None

    # ---- CosyVoice 3.0 (zero-shot streaming TTS) ----
    # Used when tts_provider == "cosyvoice".
    # Server: https://github.com/FunAudioLLM/CosyVoice (runtime/python/fastapi)
    cosyvoice_endpoint: str = "http://cosyvoice:9886"
    cosyvoice_health_path: str = "/health"
    cosyvoice_tts_stream_path: str = "/v1/tts/stream"   # SSE 流式
    cosyvoice_tts_batch_path: str = "/v1/tts"           # 非流式（build 阶段用）
    cosyvoice_timeout_seconds: int = 30
    cosyvoice_stream_chunk_size: int = 4096              # 流式 PCM 块大小（bytes）
    cosyvoice_stream_sample_rate: int = 24000

    # ---- Alibaba Cloud Model Studio CosyVoice v3.5 ----
    # Used when tts_provider == "aliyun_cosyvoice". Secrets must only be
    # provided through the runtime environment; never commit them to files.
    dashscope_api_key: str = ""
    aliyun_cosyvoice_workspace_id: str = ""
    aliyun_cosyvoice_model: str = "cosyvoice-v3.5-flash"
    aliyun_cosyvoice_default_voice_id: str = ""
    aliyun_cosyvoice_stream_sample_rate: int = 22050
    aliyun_cosyvoice_max_prompt_audio_length: float = 20.0
    aliyun_cosyvoice_enable_preprocess: bool = False
    # 44.1k 档声码器输出电平偏高（实测默认 volume=50 削波 20~140 样本/句，
    # 听感为电流声/滋滋爆点）。40 实测零削波（peak≈0.84），batch/流式共用。
    aliyun_cosyvoice_volume: int = 40

    # ---- Qwen-Audio-TTS / Qwen3-TTS-VC（百炼，独立于 CosyVoice 的第二 TTS 引擎）----
    # Used when tts_provider == "qwen_tts"（或 runtime_for("qwen_tts") 留档）。
    # 仅 qwen-audio-3.1-tts-flash/plus 支持复刻；qwen3-tts-flash 无克隆（2026-10-31 实测）。
    # 限流独立桶 180 RPM（CosyVoice 3 RPS 互不占用）；API Key 复用 DASHSCOPE_API_KEY。
    qwen_tts_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    qwen_tts_model: str = "qwen-audio-3.1-tts-flash"
    aliyun_cosyvoice_timeout_seconds: int = 45

    # ---- IndexTTS2 (zero-shot, batch) ----
    # 镜像：neosun/indextts2（all-in-one，含权重）
    # API：/upload_speaker + /tts_cached（无原生流式，用 chunk 模拟）
    indextts_endpoint: str = "http://indextts:8002"
    indextts_health_path: str = "/health"
    indextts_timeout_seconds: int = 60

    # ---- 构建期引擎（唯一视觉路线：dhlive，2026-08-25 旧图片路线移除）----
    # dhlive-builder（DH_live 素材构建容器）
    # 单机部署: http://dhlive-builder:9897（Docker 内网）
    # 分离部署: http://<推理机IP>:9897（业务/推理分离架构）
    dhlive_builder_endpoint: str = "http://dhlive-builder:9897"
    dhlive_builder_timeout_seconds: int = 600
    # 照片活化容器（快捷入口：照片 → 微动视频，LivePortrait 系 GPU 服务）
    dhlive_photo_activation_endpoint: str = "http://photo-activation:9898"
    dhlive_photo_activation_timeout_seconds: int = 300

    # ---- 内容安全（NSFW，用户上传照片唯一入口的内容门）----
    # enforce：检测并拒绝（检测器不可用时 fail-closed 拒绝上传）
    # disabled：明确关闭（联调/受控环境），不作为默认
    nsfw_guard_mode: str = "enforce"
    # NudeNet *_EXPOSED 标签判定阈值；COVERED（泳装等）不拦
    nsfw_explicit_threshold: float = 0.45

    # ---- ASR provider ----
    # "aliyun_asr"（生产默认，在线 DashScope Paraformer Realtime，复用 DASHSCOPE_API_KEY）
    #   | "funasr"（自托管流式 Docker，legacy profile，受 0.1.13 服务端 bug 影响）
    #   | "whisper"（旧批量，仅 debug，需 API 容器装 openai-whisper + GPU）
    asr_provider: str = "aliyun_asr"
    # Aliyun online ASR (DashScope Paraformer Realtime).
    aliyun_asr_model: str = "paraformer-realtime-v2"
    aliyun_asr_sample_rate: int = 16000
    aliyun_asr_timeout_seconds: int = 30
    # FunASR (legacy, self-hosted Docker). Only used under profiles=[legacy].
    funasr_endpoint: str = "ws://funasr:10095"
    funasr_mode: str = "2pass"                           # "online" | "offline" | "2pass"
    funasr_chunk_size: str = "5,10,5"                    # chunk_size str（FunASR spec）
    funasr_chunk_interval: int = 10
    funasr_timeout_seconds: int = 30

    # ---- Phoneme alignment provider ----
    # "whisperx_service"（生产默认，独立 CPU 服务，真实 CTC forced alignment）
    #   | "whisperx"（旧 in-process，仅 debug，镜像需装 whisperx extra）
    # simple/energy 已在 v3 删除，禁止作为生产降级。
    phoneme_provider: str = "whisperx_service"
    phoneme_aligner_endpoint: str = "http://phoneme-aligner:9890"
    phoneme_aligner_timeout_seconds: int = 10
    phoneme_aligner_required_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    phoneme_aligner_required_revision: str = "99ccb27"
    # Deterministic asset generation (LivePortrait builder container).
    liveportrait_endpoint: str = "http://liveportrait-builder:9895"
    liveportrait_timeout_seconds: int = 120
    # Legacy in-process WhisperX (debug only).
    whisperx_model_name: str = "tiny"
    whisperx_device: str = "cpu"
    whisperx_compute_type: str = "int8"

    # ---- GPU concurrency control ----
    # 单张 16GB GPU 的 MVP 安全水位。CosyVoice 容器内部也会串行推理。
    gpu_runtime_limit: int = 1
    cpu_fallback_threshold: int = 4
    # Build 期间 runtime 降到该值（让出显存给 EchoMimic）。
    gpu_runtime_limit_during_build: int = 1
    # 同时存在的 build 任务上限（EchoMimic 显存独占，通常为 1）。
    gpu_build_limit: int = 1

    # ---- Async build worker (P0 修复, 2026-07) ----
    # RQ 队列名，与 worker 进程一致
    worker_queue_name: str = "keyframe_build"
    # 单次 build 任务在 RQ 中的超时（秒）。超时后 RQ 把 job 标 failed，触发 worker 重启
    worker_job_timeout_seconds: int = 1800
    # running 超过该秒数视为 stale（worker 崩溃 / OOM），启动时重新入队
    worker_recover_stale_after_seconds: int = 600
    # v3 strict quality: ALL visemes are required, no critical/non-critical split.
    build_quality_max_attempts_per_asset: int = 3  # first attempt + 2 retries

    # ---- Cross-session long-term memory (relationship_key) ----
    # Independent Fernet key for sensitive memory ciphertext. MUST NOT be
    # reused for JWT, provider secrets, or object storage. Empty in local
    # dev; production loader enforces non-empty + differs from other keys.
    memory_encryption_key: str = ""
    memory_encryption_key_version: int = 1
    # Embedding HTTP service (BAAI/bge-small-zh-v1.5, dim=512, CPU).
    memory_embedding_endpoint: str = "http://memory-embedding:9887"
    memory_embedding_timeout_seconds: int = 10
    memory_embedding_required_model: str = "BAAI/bge-small-zh-v1.5"
    memory_embedding_required_revision: str = "7999e1d3359715c523056ef9478215996d62a620"
    memory_embedding_required_dim: int = 512
    # Hybrid retrieval thresholds. Below the floor a candidate is dropped,
    # regardless of importance or top-K demand.
    memory_dense_threshold: float = 0.30
    memory_lexical_threshold: float = 0.35
    memory_dense_weight: float = 0.45
    memory_lexical_weight: float = 0.15
    memory_importance_weight: float = 0.15
    memory_confidence_weight: float = 0.10
    memory_recency_weight: float = 0.10
    memory_user_confirmed_weight: float = 0.05
    # Session summary trigger.
    memory_session_summary_message_threshold: int = 20
    # Worker.
    memory_worker_lease_seconds: int = 120
    memory_worker_heartbeat_seconds: int = 30
    memory_worker_stale_recover_seconds: int = 300
    memory_worker_max_attempts: int = 5
    memory_worker_batch_size: int = 5
    memory_worker_poll_seconds: float = 1.5
    memory_worker_instance_id: str = ""  # auto-filled to hostname at runtime


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def validate_production_security(*, role: str) -> None:
    """Fail startup when a production process has unsafe security settings."""
    if settings.app_env.strip().lower() not in {"production", "prod"}:
        return
    if role not in {"admin", "client", "memory_worker", "worker", "migrate"}:
        raise RuntimeError(
            "APP_ROLE must be admin, client, worker, memory_worker, or migrate in production"
        )
    if not settings.jwt_secret.strip():
        raise RuntimeError("JWT_SECRET is required in production")
    if not settings.provider_secret_encryption_key.strip():
        raise RuntimeError("PROVIDER_SECRET_ENCRYPTION_KEY is required in production")
    # Memory encryption key is independent and must not match the provider key.
    from app.core.memory_crypto import ensure_production_memory_key_is_independent
    ensure_production_memory_key_is_independent()
