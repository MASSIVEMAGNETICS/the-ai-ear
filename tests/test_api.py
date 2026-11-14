"""
Tests for the FastAPI REST + WebSocket server.

Uses httpx.AsyncClient with TestClient transport for fast in-process testing.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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
