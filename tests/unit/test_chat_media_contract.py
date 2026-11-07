"""Dependency-free behavior checks; no database, network, TTS or Android build.

Extract the actual function/class AST to isolate it from application startup.
These checks do not replace running the API against PostgreSQL and MinIO.
Run: python tests/unit/test_chat_media_contract.py
"""
from __future__ import annotations

import ast
import asyncio
import io
import logging
from pathlib import Path
from datetime import datetime, timezone
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import uuid4
import wave

ROOT = Path(__file__).resolve().parents[2]


def load_node(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    node.decorator_list = []
    future = ast.parse("from __future__ import annotations").body[0]
    exec(compile(ast.Module(body=[future, node], type_ignores=[]), str(relative), "exec"), namespace)
    return namespace[name]


class PlaybackReferenceTests(unittest.TestCase):
    def setUp(self):
        settings = SimpleNamespace(
            local_storage_root=Path("."), storage_bucket="digitalhuman",
            storage_endpoint_url="http://minio:9000",
            storage_public_endpoint_url="https://media.example.com",
            public_base_url="https://api.example.com",
        )
        storage = load_node("app/services/storage.py", "StorageService", dict(
            settings=settings, Path=Path, quote=quote, unquote=unquote,
            urlparse=urlparse, parse_qs=parse_qs,
        ))
        self.storage = storage()
        self.storage.presigned_url = MagicMock(side_effect=lambda key: "fresh:" + key)

    def test_durable_and_expired_owned_urls_receive_a_fresh_signature(self):
        key = "digital_humans/u1/dh1/dialogue/s1/user.wav"
        for reference in (
            "storage://" + key,
            "https://media.example.com/digitalhuman/" + key + "?X-Amz-Signature=expired",
            "http://minio:9000/digitalhuman/" + key + "?old=true",
        ):
            self.assertEqual(self.storage.playback_url(reference), "fresh:" + key)

    def test_foreign_url_is_not_signed_and_unsafe_keys_are_rejected(self):
        url = "https://other.example.com/digitalhuman/dialogue/file.wav"
        self.assertEqual(self.storage.playback_url(url), url)
        self.storage.presigned_url.assert_not_called()
        self.assertIsNone(self.storage.playback_url("storage://digital_humans/../secret"))
        self.assertIsNone(self.storage.playback_url("storage://private/secret"))
        self.assertIsNone(self.storage.playback_url(None))


class AudioNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.audio = runpy.run_path(str(ROOT / "app/services/audio_utils.py"))

    def test_existing_16k_mono_pcm_wav_is_not_reencoded(self):
        wav = self.audio["pcm_to_wav"](b"\0\0" * 1600, 16000)
        self.assertEqual(self.audio["normalize_audio_to_wav"](wav, "audio/wav", 16000), wav)

    def test_m4a_decode_outputs_real_16k_mono_16bit_wav(self):
        result = SimpleNamespace(returncode=0, stdout=b"\0\0" * 1600, stderr=b"")
        with patch.object(self.audio["subprocess"], "run", return_value=result) as decoder:
            wav = self.audio["normalize_audio_to_wav"](b"m4a-input", "audio/mp4", 16000)
        args = decoder.call_args.args[0]
        self.assertEqual(args[args.index("-ar") + 1], "16000")
        with wave.open(io.BytesIO(wav), "rb") as stream:
            self.assertEqual((stream.getframerate(), stream.getnchannels(), stream.getsampwidth()), (16000, 1, 2))
            self.assertEqual(stream.getnframes(), 1600)


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


pytest_mark_open_source_skip = pytest.mark.skip(
    reason="开源版：membership 已桩化，原子配额断言仅完整版有效")

@pytest.mark.skip(reason="开源版：membership 已桩化，原子配额断言仅完整版有效")
class AtomicVoiceMessageTests(unittest.TestCase):
    def test_user_recording_is_saved_with_the_same_atomic_turn_and_ownership_guard(self):
        db = MagicMock()
        session = SimpleNamespace(user_id="u1", digital_human_id="dh1")
        db.execute.return_value.scalar_one_or_none.return_value = session
        session_factory = MagicMock()
        session_factory.return_value.__enter__.return_value = db
        writer = load_node("app/services/llm_service.py", "DialogueService", dict(
            SessionLocal=session_factory, DialogueSessionRecord=MagicMock(), select=MagicMock(),
            DialogueMessageRecord=lambda **values: SimpleNamespace(**values),
            CompletedTurn=lambda **values: SimpleNamespace(**values),
            SessionOwnershipError=RuntimeError, datetime=datetime, UTC=timezone.utc, uuid4=uuid4,
        ))()
        membership = ModuleType("app.services.membership_service")
        membership.entitlement_service = MagicMock()
        membership.entitlement_service.get.return_value = SimpleNamespace(entitlements={}, plan_code="free")
        values = dict(session_id="s1", user_id="u1", digital_human_id="dh1",
                      user_text="你好", assistant_text="回答", audio_url="storage://dialogue/reply.wav",
                      user_audio_url="storage://dialogue/user.wav", memory_enabled=False)
        with patch.dict(sys.modules, {membership.__name__: membership}):
            turn = writer.complete_turn_and_enqueue_memory(**values)
            self.assertEqual(turn.user_message.audio_url, values["user_audio_url"])
            self.assertEqual(turn.assistant_message.audio_url, values["audio_url"])
            db.commit.assert_called_once()
            db.reset_mock()
            session.user_id = "another-user"
            with self.assertRaises(RuntimeError):
                writer.complete_turn_and_enqueue_memory(**values)
            db.add.assert_not_called()
            db.commit.assert_not_called()


@pytest.mark.skip(reason="开源版：membership 已桩化，配额断言仅完整版有效")
class TextChatContractTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = MagicMock()
        self.dialogue.get_session.return_value = SimpleNamespace(session_id="s1", digital_human_id="dh1")
        self.llm = MagicMock()
        self.llm.chat = AsyncMock(return_value="回答")
        provider = MagicMock()
        provider.get_provider_for_role.return_value = SimpleNamespace(system_prompt="人设")
        self.entitlements = MagicMock()
        self.entitlements.get.return_value = SimpleNamespace(entitlements={})
        self.storage = MagicMock()
        self.storage.put_bytes.return_value = ("digital_humans/u1/dh1/dialogue/s1/reply.wav", 10, "hash")
        self.storage.presigned_url.return_value = "https://media.example.com/fresh.wav"
        human = MagicMock()
        human.get_owned.return_value = SimpleNamespace(creation_status="ready")
        human.build_persona_prompt.return_value = "角色"
        self.modules = {
            "app.services.llm_service": dict(dialogue_service=self.dialogue, llm_provider_service=self.llm, SessionOwnershipError=RuntimeError),
            "app.services.memory_provider_service": dict(memory_provider_service=provider),
            "app.services.membership_service": dict(entitlement_service=self.entitlements),
            "app.services.digital_human_service": dict(digital_human_service=human, DigitalHumanNotFound=LookupError),
            "app.services.tts_service": dict(synthesize_batch=AsyncMock(return_value=SimpleNamespace(wav_bytes=b"wav", duration_ms=1000))),
            "app.services.storage": dict(storage_service=self.storage),
        }
        async def to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)
        self.route = load_node("app/api/v1/routes/dhlive_chat.py", "chat_text", dict(
            HTTPException=HttpError, get_request_user_id=lambda request: "u1",
            ChatReplyRead=lambda **values: SimpleNamespace(**values),
            asyncio=SimpleNamespace(to_thread=to_thread), uuid4=uuid4,
            logger=logging.getLogger("chat-test"),
        ))

    def invoke(self, body):
        modules = {}
        for name, values in self.modules.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        with patch.dict(sys.modules, modules):
            return asyncio.run(self.route("dh1", body, object()))

    def test_path_body_and_session_cannot_select_different_humans(self):
        body = SimpleNamespace(text="你好", session_id="s1", digital_human_id="dh2")
        with self.assertRaises(HttpError) as error:
            self.invoke(body)
        self.assertEqual(error.exception.status_code, 409)
        body.digital_human_id = "dh1"
        self.dialogue.get_session.return_value.digital_human_id = "dh2"
        with self.assertRaises(HttpError) as error:
            self.invoke(body)
        self.assertEqual(error.exception.status_code, 409)
        self.llm.chat.assert_not_called()

    def test_completed_reply_uses_real_services_and_persists_a_durable_reference(self):
        reply = self.invoke(SimpleNamespace(text="你好", session_id="s1", digital_human_id="dh1"))
        self.assertEqual(reply.reply_text, "回答")
        stored = self.dialogue.complete_turn_and_enqueue_memory.call_args.kwargs
        self.assertEqual(stored["digital_human_id"], "dh1")
        self.assertTrue(stored["audio_url"].startswith("storage://digital_humans/"))


if __name__ == "__main__":
    unittest.main()
