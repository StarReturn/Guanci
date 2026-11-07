-- 数字人分享层（2026-10-12 第二批立项）
-- 规则（甲方已定）：默认可分享 / 分享码+二维码 / 被分享者仅对话权 / 免费 /
--                    配额"谁用谁付"（消耗使用者自己的配额——现有 entitlement
--                    按 state.user_id 计量，天然满足，无需改动计费层）
CREATE TABLE IF NOT EXISTS dh_shares (
    share_id        UUID PRIMARY KEY,
    digital_human_id VARCHAR(64) NOT NULL,     -- 被分享的数字人
    owner_user_id   VARCHAR(64) NOT NULL,      -- 创建者
    grantee_user_id VARCHAR(64) DEFAULT NULL,  -- 被分享者（NULL=未兑换，码仍有效）
    share_code      VARCHAR(12) NOT NULL,      -- 兑换码（6 位字母数字）
    status          VARCHAR(16) NOT NULL DEFAULT 'active',  -- active | redeemed | revoked
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    redeemed_at     TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ,
    UNIQUE (share_code)
);
CREATE INDEX IF NOT EXISTS idx_dh_shares_dh ON dh_shares (digital_human_id);
CREATE INDEX IF NOT EXISTS idx_dh_shares_grantee ON dh_shares (grantee_user_id);
