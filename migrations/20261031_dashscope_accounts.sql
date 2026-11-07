-- 百炼账号矩阵：多 API Key 限流桶，429 秒切下一账号（前端无感知）
-- 限流按主账号维度（实测 3 请求/秒/账号），N 个账号 = N 个独立桶。
CREATE TABLE IF NOT EXISTS dashscope_accounts (
    account_id  VARCHAR(64) PRIMARY KEY,
    name        VARCHAR(64) NOT NULL,
    api_key     TEXT NOT NULL,
    priority    INT NOT NULL DEFAULT 0,          -- 数值小者优先
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    note        VARCHAR(256),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 本地音色 <-> 各账号 voice_id 映射（复刻音色账号隔离，切换前提）
-- local_voice_key: "preset:<display_name>" | "dh:<digital_human_id>"
CREATE TABLE IF NOT EXISTS voice_account_map (
    account_id      VARCHAR(64) NOT NULL,
    local_voice_key VARCHAR(128) NOT NULL,
    remote_voice_id VARCHAR(256) NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, local_voice_key)
);
CREATE INDEX IF NOT EXISTS ix_vam_voice ON voice_account_map (local_voice_key);
