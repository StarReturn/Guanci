-- =============================================================================
-- 迁移：2026-10 线上运维变更沉淀（此前为手工 SQL/docker 操作，无迁移记录）
-- 来源：2026-09-25 ~ 10-08 生产排障与功能上线
-- 幂等：可重复执行
-- =============================================================================

-- 1) App 版本发布表（版本检查 GET /api/v1/app/version 的数据源）
CREATE TABLE IF NOT EXISTS app_release (
    id int PRIMARY KEY,
    version_code int NOT NULL,
    version_name text NOT NULL,
    download_url text,
    notes text,
    force_update boolean DEFAULT false,
    released_at timestamptz DEFAULT now()
);
INSERT INTO app_release (id, version_code, version_name)
VALUES (1, 5, '1.3.1')
ON CONFLICT (id) DO NOTHING;
-- 种子值需与最新发布保持一致：发新版时除更新本种子外，正式环境执行
-- UPDATE app_release SET version_code=<N>, version_name='<X>', download_url='<URL>';

-- 2) 每音色语速列（克隆音色继承参考音频慢速，按音色校正；NULL 走全局兜底 1.5）
ALTER TABLE digital_humans ADD COLUMN IF NOT EXISTS voice_speech_rate REAL;

-- 3) LLM 角色绑定约束更新：代码新增 memory_summary 角色，约束定义落后
ALTER TABLE llm_provider_bindings DROP CONSTRAINT IF EXISTS ck_llm_provider_binding_role;
ALTER TABLE llm_provider_bindings ADD CONSTRAINT ck_llm_provider_binding_role
    CHECK (role::text = ANY (ARRAY[
        'chat'::character varying,
        'memory_extract'::character varying,
        'memory_rerank'::character varying,
        'memory_summary'::character varying
    ]));

-- 4) 记忆角色绑定（memory_extract/memory_summary → qwen-plus）
--    缺失会导致全部记忆提取任务死亡（memory_extract_not_configured）
INSERT INTO llm_provider_bindings
    (binding_id, role, provider_id, enabled, config_overrides, probe_status, created_at, updated_at)
VALUES
    ('binding_memory_extract_qp', 'memory_extract',
     (SELECT provider_id FROM llm_providers WHERE model = 'qwen-plus' LIMIT 1),
     true, '{}', 'ok', now(), now()),
    ('binding_memory_summary_qp', 'memory_summary',
     (SELECT provider_id FROM llm_providers WHERE model = 'qwen-plus' LIMIT 1),
     true, '{}', 'ok', now(), now())
ON CONFLICT (role) DO NOTHING;
