-- 席位级 TTS 指令：方言/老年口音保真的唯一正解（克隆只保音色不保口音）
-- 对话与试听共用（synthesize_batch / 流式会话注入，席位指令优先于情绪指令）
ALTER TABLE preset_voices ADD COLUMN IF NOT EXISTS tts_instruction VARCHAR(512);

UPDATE preset_voices SET tts_instruction = '请用陕西话，中年女性的亲切语气说话，用额、咋、美得很这些方言词' WHERE display_name = '陕味大姐姐·中年';
UPDATE preset_voices SET tts_instruction = '请用陕西话口音，知性中年女性的语气说话，带上方言词汇如额、谝、包' WHERE display_name = '陕味知心人·中年';
UPDATE preset_voices SET tts_instruction = '请用陕西话，慈祥的老年女性声音慢慢说，用方言词汇额、咋、美得很' WHERE display_name = '陕味奶奶·老年';
UPDATE preset_voices SET tts_instruction = '请用陕西话，厚实的中年男性声音说话，用额、咋、美得很' WHERE display_name = '陕味大哥·中年';
UPDATE preset_voices SET tts_instruction = '请用陕西方言，沉稳的中年男人声音说话，用方言词汇额、美得很' WHERE display_name = '陕味老哥·中年';
UPDATE preset_voices SET tts_instruction = '请用陕西话，和蔼苍老的老年男性声音慢慢说，带方言词汇额、咋、美得很' WHERE display_name = '陕味爷爷·老年';
UPDATE preset_voices SET tts_instruction = '请用四川话，热情的中年女性声音说话，用要得、巴适、啥子这些方言词' WHERE display_name = '川味热情·中年';
UPDATE preset_voices SET tts_instruction = '请用上海话，细腻温柔的中年女性声音说话，用侬、阿拉、交关' WHERE display_name = '沪味细腻·中年';
UPDATE preset_voices SET tts_instruction = '请用广东话，沉稳的中年男性声音说话' WHERE display_name = '粤味沉稳·中年';
UPDATE preset_voices SET tts_instruction = '请用东北话，幽默爽朗的中年男性声音说话，用哎呀妈呀、瞅、咋整' WHERE display_name = '东北幽默·中年';
UPDATE preset_voices SET tts_instruction = '请用河南话，亲切热络的中年女性声音说话，多用中、恁、得劲、俺这些方言词' WHERE display_name = '豫味大姐姐·中年';
UPDATE preset_voices SET tts_instruction = '请用河南话，实在厚道的中年男人声音说话，用中、恁、得劲这些方言词' WHERE display_name = '豫味老哥·中年';
UPDATE preset_voices SET tts_instruction = '请用苍老缓慢、略带沙哑颤动的老年男性声音慈祥地说，语速明显放慢' WHERE display_name = '慈祥爷爷·老年';
UPDATE preset_voices SET tts_instruction = '请用苍老低沉、气息略弱的老年男性声音平静缓慢地说' WHERE display_name = '沉稳爷爷·老年';
UPDATE preset_voices SET tts_instruction = '请用慈祥温和的老年女性声音慢慢说' WHERE display_name = '慈祥奶奶·老年';
