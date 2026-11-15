"""
Tests for the FastAPI REST + WebSocket server.

Uses httpx.AsyncClient with TestClient transport for fast in-process testing.
"""

from __future__ import annotations

import asyncio
import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ai_ear.api.middleware import RequestBodyLimitMiddleware
from ai_ear.api.server import create_app
from ai_ear.core.config import Settings


# Use minimal settings for speed (no real ML models)
def _test_settings(**overrides) -> Settings:
    values = {
        "speech_enabled": False,
        "whisper_model": "tiny",
        "emotion_enabled": False,
        "music_enabled": False,
        "environment_enabled": True,
        "audio_sample_rate": 16_000,
        "audio_chunk_duration_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def app():
    return create_app(_test_settings())


def _run_request_body_limit(path, method, headers, messages, max_body_bytes):
    """Exercise the ASGI middleware with explicit (possibly chunked) receive events."""
    app_calls = []
    sent = []

    async def downstream(scope, receive, send):
        app_calls.append(True)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": bytes(body)})

    pending = iter(messages)

    async def receive():
        return next(pending, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=max_body_bytes)
    asyncio.run(middleware(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, response_body, app_calls


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0

    def test_health_timestamp_recent(self, client):
        before = time.time()
        resp = client.get("/health")
        after = time.time()
        ts = resp.json()["timestamp"]
        assert before <= ts <= after


class TestInfoEndpoint:
    def test_info_returns_config(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["whisper_model"] == "tiny"
        assert data["sample_rate"] == 16_000

    def test_info_version_present(self, client):
        resp = client.get("/info")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

class TestMemoryEndpoints:
    def test_context_empty(self, client):
        resp = client.get("/memory/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == ""
        assert not data["music_detected"]

    def test_transcript_empty(self, client):
        resp = client.get("/memory/transcript")
        assert resp.status_code == 200
        assert resp.json()["transcript"] == ""

    def test_events_empty(self, client):
        resp = client.get("/memory/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["events"] == []

    def test_events_invalid_type_returns_400(self, client):
        resp = client.get("/memory/events?event_type=not_a_real_type")
        assert resp.status_code == 400

    def test_context_window_param(self, client):
        resp = client.get("/memory/context?window_s=30")
        assert resp.status_code == 200
        assert resp.json()["window_s"] == 30.0


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_initial_state(self, client):
        resp = client.get("/pipeline/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert "chunks_processed" in stats
        assert stats["chunks_processed"] == 0.0


# ---------------------------------------------------------------------------
# Audio file analysis (requires soundfile)
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    """Generate a minimal in-memory WAV file."""
    try:
        import soundfile as sf
        samples = np.zeros(int(sr * duration_s), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        return b""


class TestAnalyseEndpoint:
    def test_analyse_silence_wav(self, client):
        wav = _make_wav_bytes(1.0, 16_000)
        if not wav:
            pytest.skip("soundfile not available")
        resp = client.post(
            "/analyse",
            files={"file": ("silence.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "chunk_id" in data
        assert "environment" in data

    def test_analyse_invalid_file_returns_422(self, client):
        resp = client.post(
            "/analyse",
            files={"file": ("garbage.wav", b"not audio at all", "audio/wav")},
        )
        assert resp.status_code == 422



class TestRequestBodyLimitMiddleware:
    def test_declared_oversized_content_length_is_rejected_before_parser(self):
        status, body, app_calls = _run_request_body_limit(
            "/analyse", "POST", [(b"content-length", b"9")], [], 8
        )
        assert status == 413
        assert json.loads(body)["detail"].endswith("8 bytes")
        assert app_calls == []

    def test_chunked_body_is_counted_and_rejected_when_limit_is_crossed(self):
        messages = [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]
        status, body, app_calls = _run_request_body_limit(
            "/analyse", "POST", [], messages, 8
        )
        assert status == 413
        assert json.loads(body)["detail"].endswith("8 bytes")
        assert app_calls == [True]

    def test_streamed_body_at_exact_limit_is_forwarded_unchanged(self):
        messages = [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ]
        status, body, app_calls = _run_request_body_limit(
            "/analyse", "POST", [], messages, 8
        )
        assert status == 200
        assert body == b"12345678"
        assert app_calls == [True]

    def test_non_analyse_routes_are_not_subject_to_upload_limit(self):
        messages = [{"type": "http.request", "body": b"123456789", "more_body": False}]
        status, body, app_calls = _run_request_body_limit(
            "/health", "GET", [(b"content-length", b"9")], messages, 8
        )
        assert status == 200
        assert body == b"123456789"
        assert app_calls == [True]

    def test_invalid_content_length_is_rejected(self):
        status, body, app_calls = _run_request_body_limit(
            "/analyse", "POST", [(b"content-length", b"not-a-number")], [], 8
        )
        assert status == 400
        assert json.loads(body)["detail"] == "Invalid Content-Length header"
        assert app_calls == []

    @pytest.mark.parametrize(
        "duplicate_values",
        [(b"8", b"9"), (b"8", b"8")],
        ids=("conflicting-values", "identical-values"),
    )
    def test_duplicate_content_length_headers_are_rejected_before_parser(self, duplicate_values):
        headers = [
            (b"content-length", duplicate_values[0]),
            (b"content-length", duplicate_values[1]),
        ]
        status, body, app_calls = _run_request_body_limit(
            "/analyse", "POST", headers, [], 8
        )
        assert status == 400
        assert json.loads(body)["detail"] == "Invalid Content-Length header"
        assert app_calls == []


class TestBoundedAudioIngress:
    def test_analyse_rejects_upload_over_configured_limit(self):
        settings = _test_settings(api_max_upload_bytes=16)
        with TestClient(create_app(settings)) as limited_client:
            response = limited_client.post(
                "/analyse",
                files={"file": ("oversized.wav", b"x" * 17, "audio/wav")},
            )
        assert response.status_code == 413

    def test_analyse_accepts_upload_at_configured_limit(self):
        wav = _make_wav_bytes(1.0, 16_000)
        if not wav:
            pytest.skip("soundfile not available")
        settings = _test_settings(api_max_upload_bytes=len(wav))
        with TestClient(create_app(settings)) as limited_client:
            response = limited_client.post(
                "/analyse",
                files={"file": ("silence.wav", wav, "audio/wav")},
            )
        assert response.status_code == 200

    def test_stream_rejects_oversized_binary_frame(self):
        settings = _test_settings(api_max_ws_frame_bytes=4)
        with TestClient(create_app(settings)) as limited_client:
            with limited_client.websocket_connect("/stream") as websocket:
                websocket.send_bytes(b"\x00" * 8)
                with pytest.raises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
        assert exc.value.code == 1009

    def test_stream_rejects_partial_float32_sample(self):
        settings = _test_settings(api_max_ws_frame_bytes=16)
        with TestClient(create_app(settings)) as limited_client:
            with limited_client.websocket_connect("/stream") as websocket:
                websocket.send_bytes(b"\x00" * 3)
                with pytest.raises(WebSocketDisconnect) as exc:
                    websocket.receive_bytes()
        assert exc.value.code == 1003

    def test_cli_passes_configured_websocket_frame_limit(self, monkeypatch):
        from ai_ear import __main__ as entrypoint

        settings = _test_settings(api_max_ws_frame_bytes=32)
        captured = {}
        monkeypatch.setattr(entrypoint, "Settings", lambda: settings)
        monkeypatch.setattr(entrypoint, "create_app", lambda current: object())
        monkeypatch.setattr(
            entrypoint.uvicorn,
            "run",
            lambda app, **kwargs: captured.update(kwargs),
        )

        assert entrypoint.main(["serve"]) == 0
        assert captured["ws_max_size"] == 32
