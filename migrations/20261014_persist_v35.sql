-- =============================================================================
-- v3.5 持久化迁移（幂等，可重复执行）
-- 确保 tts_providers 的 model 和 default_voice_id 在任何环境重建后正确
-- =============================================================================

-- 1) 模型锁定 v3.5
UPDATE tts_providers
SET model = 'cosyvoice-v3.5-flash', updated_at = now()
WHERE type = 'aliyun_cosyvoice'
  AND model != 'cosyvoice-v3.5-flash';

-- 2) 默认音色：切换到 v3.5 生成的预设音色（温柔女声·青年）
--    此前指向 v2 克隆音色（ cosyvoice-v2-dhlive-dc61...），与 v3.5 模型不完全兼容
UPDATE tts_providers
SET default_voice_id = (
    SELECT voice_id FROM preset_voices
    WHERE display_name = '温柔女声·青年' AND is_active = true
    ORDER BY sort_order LIMIT 1
), updated_at = now()
WHERE type = 'aliyun_cosyvoice'
  AND (default_voice_id IS NULL
       OR default_voice_id LIKE 'cosyvoice-v2-%'
       OR default_voice_id = '');

-- 3) 存量数字人语速：v3.5 基线 3.5 字/s，1.2 档 = 4.2 字/s（正常）
UPDATE digital_humans
SET voice_speech_rate = 1.2
WHERE voice_storage_key IS NOT NULL
  AND deleted_at IS NULL
  AND (voice_speech_rate IS NULL OR voice_speech_rate != 1.2);

-- 4) 记忆角色绑定确认（幂等：不存在则插入）
INSERT INTO llm_provider_bindings
    (binding_id, role, provider_id, enabled, config_overrides, probe_status, created_at, updated_at)
SELECT
    'binding_memory_extract_qp', 'memory_extract',
    (SELECT provider_id FROM llm_providers WHERE model = 'qwen-plus' LIMIT 1),
    true, '{}', 'ok', now(), now()
WHERE NOT EXISTS (SELECT 1 FROM llm_provider_bindings WHERE role = 'memory_extract');

INSERT INTO llm_provider_bindings
    (binding_id, role, provider_id, enabled, config_overrides, probe_status, created_at, updated_at)
SELECT
    'binding_memory_summary_qp', 'memory_summary',
    (SELECT provider_id FROM llm_providers WHERE model = 'qwen-plus' LIMIT 1),
    true, '{}', 'ok', now(), now()
WHERE NOT EXISTS (SELECT 1 FROM llm_provider_bindings WHERE role = 'memory_summary');
