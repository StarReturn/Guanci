-- 预设音色库（免费音源云端克隆注册后的 voice_id 台账）
-- 2026-10-12 建库定稿 / 2026-10-14 修订：24 种（陕西话 6 席为最大方言份额），
-- 音源 AISHELL-3（Apache 2.0），accent 列优先标记陕西/西北口音说话人
CREATE TABLE IF NOT EXISTS preset_voices (
    voice_id       VARCHAR(128) PRIMARY KEY,   -- cosyvoice-v3.5-flash-preset_xxx（平台账号注册）
    display_name   VARCHAR(64)  NOT NULL,      -- "温柔女声·青年"
    gender         VARCHAR(8)   NOT NULL,      -- female | male
    age_band       VARCHAR(16)  NOT NULL,      -- young | middle | senior
    style          VARCHAR(32),                -- 温柔/知性/活泼/沉稳/温暖/清朗/方言
    accent         VARCHAR(32),                -- 普通话/四川/上海/广东/陕西（可空=普通话）
    sample_url     TEXT,                       -- 试听音频（固定句合成，存 MinIO 预签名）
    source_note    VARCHAR(128),               -- 溯源：AISHELL-3 speaker ID + 许可证
    is_active      BOOLEAN DEFAULT TRUE,
    sort_order     INT DEFAULT 0,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- 示例（建库脚本批量插入后可删）
INSERT INTO preset_voices (voice_id, display_name, gender, age_band, style, accent, sort_order)
VALUES
    ('PLACEHOLDER', '温柔女声·青年', 'female', 'young', '温柔', '普通话', 1)
ON CONFLICT (voice_id) DO NOTHING;
