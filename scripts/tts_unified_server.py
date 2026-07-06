#!/usr/bin/env python3
"""Unified TTS Server v2 — three tracks, one process, lazy-load on demand.

Runs inside gold-team container (GPU 1 = RTX 3090).

Three tracks (shared GPU, loaded on-demand):
  - gpt_sovits:  中文轨道 (角色/IP克隆, ~4 GB VRAM) — via api_v2 subprocess
  - chatterbox:  英文轨道 (Chatterbox-Turbo, ~2 GB VRAM)
  - cosyvoice:   双语轨道 (CosyVoice-300M, ~2.5 GB VRAM)

Language routing:
  - Pure CJK               → gpt_sovits
  - Pure Latin             → chatterbox
  - Mixed CJK + Latin      → cosyvoice
  - Explicit backend param → override

Lifecycle:
  - Server starts with ZERO models loaded (no VRAM used)
  - First TTS request for a track → load model (~3-8s cold start)
  - Subsequent requests → warm inference (~0.5-2s)
  - No requests for IDLE_TIMEOUT → unload track → VRAM freed

API:
  POST /tts          — Synchronous TTS
  POST /tts/batch    — Batch TTS
  GET  /health       — Health + per-track status
  GET  /tracks       — Track status detail
  POST /tracks/load  — Pre-warm a track
  POST /tracks/unload — Force unload a track
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# ── Numba: use cache dir instead of disabling JIT ────────────────────
# DISABLE_JIT breaks GPT-SoVITS cnhubert. Use writable cache instead.
import numba
numba.config.CACHE_DIR = "/tmp/numba_cache"

import numpy as np
import torch

# ── Paths ───────────────────────────────────────────────────────────────
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", os.path.expanduser("~/CosyVoice"))
GPTSOVITS_ROOT = os.environ.get("GPTSOVITS_ROOT", os.path.expanduser("~/GPT-SoVITS"))
CHATTERBOX_ROOT = os.environ.get("CHATTERBOX_ROOT", os.path.expanduser("~/chatterbox"))
OUTPUT_DIR = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

# ── Language detection ──────────────────────────────────────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and not has_latin:
        return "zh"
    if has_latin and not has_cjk:
        return "en"
    return "auto"


# ═════════════════════════════════════════════════════════════════════════
# Track wrappers
# ═════════════════════════════════════════════════════════════════════════

class BaseTrack:
    name: str = "base"
    language: str = "auto"
    vram_mb: int = 0

    def __init__(self, idle_timeout: float = 300.0):
        self._model = None
        self._loaded = False
        self._loading = False
        self._last_used: float = 0.0
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def last_used(self) -> float:
        return self._last_used

    async def ensure_loaded(self) -> None:
        async with self._lock:
            if self._loaded:
                self._last_used = time.monotonic()
                return
            if self._loading:
                while self._loading:
                    await asyncio.sleep(0.5)
                return
            self._loading = True
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)
                self._loaded = True
                self._last_used = time.monotonic()
                logging.info("Track '%s' loaded", self.name)
            except Exception as e:
                logging.error("Track '%s' failed: %s", self.name, e)
                raise
            finally:
                self._loading = False

    def _load_model(self) -> None:
        raise NotImplementedError

    async def synthesize(self, text, voice="default", speed=1.0,
                         reference_audio="", output_path="") -> dict:
        raise NotImplementedError

    async def unload(self) -> None:
        async with self._lock:
            if not self._loaded:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._unload_model)

    def _unload_model(self) -> None:
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Track '%s' unloaded", self.name)

    def is_idle(self) -> bool:
        return self._loaded and (time.monotonic() - self._last_used > self._idle_timeout)

    def status(self) -> dict:
        return {
            "name": self.name, "language": self.language,
            "loaded": self._loaded, "loading": self._loading,
            "vram_mb": self.vram_mb,
            "idle_seconds": time.monotonic() - self._last_used if self._loaded else None,
        }


class CosyVoiceTrack(BaseTrack):
    """Track 3: CosyVoice-300M — Chinese + English + mixed."""
    name = "cosyvoice"
    language = "auto"
    vram_mb = 2500

    def _load_model(self) -> None:
        sys.path.insert(0, COSYVOICE_ROOT)
        sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        model_dir = os.path.join(COSYVOICE_ROOT, "pretrained_models", "CosyVoice-300M")
        self._model = AutoModel(model_dir=model_dir)
        logging.info("CosyVoice-300M loaded")

    async def synthesize(self, text, voice="default", speed=1.0,
                         reference_audio="", output_path="") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            import soundfile as sf
            spks = self._model.list_available_spks()

            if reference_audio and os.path.isfile(reference_audio):
                result = self._model.inference_zero_shot(
                    text, "", reference_audio, stream=False, speed=speed)
            elif spks:
                spk_id = voice if voice in spks else spks[0]
                result = self._model.inference_sft(
                    text, spk_id, stream=False, speed=speed)
            else:
                dummy_sr = 22050
                t = np.arange(0, 3.0, 1.0 / dummy_sr)
                dummy_wav = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
                dummy_path = os.path.join(OUTPUT_DIR, "_cosyvoice_ref.wav")
                sf.write(dummy_path, dummy_wav, dummy_sr)
                result = self._model.inference_cross_lingual(
                    text, dummy_path, stream=False, speed=speed)

            all_audio = []
            sr = 22050
            for model_output in result:
                tts_speech = model_output["tts_speech"]
                if hasattr(tts_speech, "cpu"):
                    tts_speech = tts_speech.cpu().numpy()
                if tts_speech.ndim == 3:
                    tts_speech = tts_speech.squeeze(0)
                if tts_speech.ndim == 1:
                    tts_speech = tts_speech.unsqueeze(0)
                all_audio.append(tts_speech)
            if not all_audio:
                return {"error": "No audio generated"}
            audio_np = (np.concatenate(all_audio, axis=1).squeeze(0)
                        if len(all_audio) > 1 else all_audio[0].squeeze(0))

            if not output_path:
                output_path = os.path.join(OUTPUT_DIR, f"tts_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_np, sr)
            return {
                "output_path": output_path,
                "duration_sec": round(len(audio_np) / sr, 2),
                "sample_rate": sr, "backend": "cosyvoice",
            }

        return await loop.run_in_executor(None, _synth)


class GPTSoVITSTrack(BaseTrack):
    """Track 1: GPT-SoVITS — Chinese voice cloning via api_v2 subprocess."""
    name = "gpt_sovits"
    language = "zh"
    vram_mb = 4000
    _api_port = 9988
    _process = None

    def _load_model(self) -> None:
        script = os.path.join(GPTSOVITS_ROOT, "api_v2.py")
        config = os.path.join(GPTSOVITS_ROOT, "GPT_SoVITS", "configs", "tts_infer.yaml")
        self._process = subprocess.Popen(
            ["python3", script, "-a", "127.0.0.1", "-p", str(self._api_port), "-c", config],
            cwd=GPTSOVITS_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env={**os.environ, "NUMBA_CACHE_DIR": "/tmp/numba_cache", "NUMBA_DISABLE_JIT": "0"},
        )
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._api_port}/", timeout=1)
                logging.info("GPT-SoVITS api_v2 ready on port %d", self._api_port)
                return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(0.5)
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode(errors="replace")
                    raise RuntimeError(f"GPT-SoVITS crashed: {stderr[-500:]}")
        raise RuntimeError("GPT-SoVITS api_v2 failed to start in 120s")

    def _unload_model(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("GPT-SoVITS unloaded")

    async def synthesize(self, text, voice="default", speed=1.0,
                         reference_audio="", output_path="") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            import soundfile as sf
            ref_path = reference_audio
            if not ref_path or not os.path.isfile(ref_path):
                dummy_sr = 32000
                t = np.arange(0, 3.0, 1.0 / dummy_sr)
                dummy_wav = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
                ref_path = os.path.join(OUTPUT_DIR, "_gptsovits_ref.wav")
                sf.write(ref_path, dummy_wav, dummy_sr)

            payload = json.dumps({
                "text": text, "text_lang": "zh",
                "ref_audio_path": ref_path,
                "prompt_text": "", "prompt_lang": "zh",
                "top_k": 15, "top_p": 1.0, "temperature": 1.0,
                "speed_factor": speed, "media_type": "wav",
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self._api_port}/tts",
                data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=120)
            audio_bytes = resp.read()

            if not output_path:
                output_path = os.path.join(OUTPUT_DIR, f"tts_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            data, sr = sf.read(output_path)
            return {
                "output_path": output_path,
                "duration_sec": round(len(data) / sr, 2),
                "sample_rate": sr, "backend": "gpt_sovits",
            }

        return await loop.run_in_executor(None, _synth)


class ChatterboxTrack(BaseTrack):
    """Track 2: Chatterbox-Turbo — English TTS."""
    name = "chatterbox"
    language = "en"
    vram_mb = 2000

    def _load_model(self) -> None:
        sys.path.insert(0, CHATTERBOX_ROOT)
        sys.path.insert(0, os.path.join(CHATTERBOX_ROOT, "src"))
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_snap = os.path.join(cache_dir, "models--ResembleAI--chatterbox-turbo", "snapshots")
        local_path = None
        if os.path.isdir(repo_snap):
            for snap in os.listdir(repo_snap):
                d = os.path.join(repo_snap, snap)
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "t3_turbo_v1.safetensors")):
                    local_path = d
                    break
        self._model = ChatterboxTurboTTS.from_local(local_path, "cuda") if local_path else \
            ChatterboxTurboTTS.from_pretrained("cuda")
        logging.info("Chatterbox-Turbo loaded")

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Chatterbox unloaded")

    async def synthesize(self, text, voice="default", speed=1.0,
                         reference_audio="", output_path="") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            if reference_audio and os.path.isfile(reference_audio):
                self._model.prepare_conditionals(reference_audio)
            wav = self._model.generate(text=text, temperature=0.8, top_k=1000, top_p=0.95)
            audio_np = wav.squeeze(0).numpy()
            sr = self._model.sr
            import soundfile as sf
            if not output_path:
                output_path = os.path.join(OUTPUT_DIR, f"tts_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_np, sr)
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            return {
                "output_path": output_path,
                "duration_sec": round(len(audio_np) / sr, 2),
                "sample_rate": sr, "backend": "chatterbox",
            }

        return await loop.run_in_executor(None, _synth)


# ═════════════════════════════════════════════════════════════════════════
# Track Manager
# ═════════════════════════════════════════════════════════════════════════

class TrackManager:
    def __init__(self, idle_timeout: float = 300.0):
        self._tracks = {
            "gpt_sovits": GPTSoVITSTrack(idle_timeout),
            "chatterbox": ChatterboxTrack(idle_timeout),
            "cosyvoice": CosyVoiceTrack(idle_timeout),
        }
        self._idle_timeout = idle_timeout
        self._reaper_task = None

    async def start(self):
        self._reaper_task = asyncio.create_task(self._idle_reaper())

    async def stop(self):
        if self._reaper_task:
            self._reaper_task.cancel()
        for t in self._tracks.values():
            await t.unload()

    def select_track(self, text, backend="auto", language="auto"):
        if backend != "auto" and backend in self._tracks:
            return backend
        if language == "auto":
            language = detect_language(text)
        return {"zh": "gpt_sovits", "en": "chatterbox", "auto": "cosyvoice"}.get(language, "cosyvoice")

    async def synthesize(self, text, voice="default", speed=1.0, backend="auto",
                         language="auto", reference_audio="", output_path=""):
        track_id = self.select_track(text, backend, language)
        track = self._tracks[track_id]
        try:
            result = await track.synthesize(text, voice, speed, reference_audio, output_path)
            result["track"] = track_id
            return result
        except Exception as e:
            if track_id != "cosyvoice":
                logging.warning("Track '%s' failed (%s), falling back to cosyvoice", track_id, e)
                try:
                    result = await self._tracks["cosyvoice"].synthesize(text, voice, speed, reference_audio, output_path)
                    result["track"] = "cosyvoice"
                    result["fallback_from"] = track_id
                    return result
                except Exception as e2:
                    return {"error": f"All tracks failed: {e}, cosyvoice: {e2}"}
            return {"error": str(e)}

    async def preload(self, track_id):
        track = self._tracks.get(track_id)
        if not track:
            return {"error": f"Unknown track: {track_id}"}
        try:
            await track.ensure_loaded()
            return {"status": "loaded", "track": track_id}
        except Exception as e:
            return {"error": str(e), "track": track_id}

    async def unload_track(self, track_id):
        track = self._tracks.get(track_id)
        if not track:
            return {"error": f"Unknown track: {track_id}"}
        await track.unload()
        return {"status": "unloaded", "track": track_id}

    def get_status(self):
        return {
            "idle_timeout": self._idle_timeout,
            "tracks": {tid: t.status() for tid, t in self._tracks.items()},
            "total_vram_mb": sum(t.vram_mb for t in self._tracks.values() if t.is_loaded),
        }

    async def _idle_reaper(self):
        while True:
            try:
                await asyncio.sleep(30)
                for tid, t in self._tracks.items():
                    if t.is_idle():
                        logging.info("Track '%s' idle > %.0fs, unloading", tid, self._idle_timeout)
                        await t.unload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error("Reaper error: %s", e)


# ═════════════════════════════════════════════════════════════════════════
# FastAPI Server
# ═════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tts-unified")

_manager: TrackManager = None
_args = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    _manager = TrackManager(_args.idle_timeout)
    await _manager.start()
    logger.info("Unified TTS server started on port %d", _args.port)
    yield
    await _manager.stop()


app = FastAPI(title="Unified TTS Server v2", version="2.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field("default")
    speed: float = Field(1.0, ge=0.3, le=3.0)
    backend: str = Field("auto")
    language: str = Field("auto")
    reference_audio: str = Field("")
    output_path: str = Field("")


class BatchTTSRequest(BaseModel):
    items: list[TTSRequest] = Field(..., min_length=1, max_length=100)


@app.post("/tts")
async def tts(req: TTSRequest):
    try:
        result = await _manager.synthesize(req.text, req.voice, req.speed,
                                           req.backend, req.language,
                                           req.reference_audio, req.output_path)
        if "error" in result:
            raise HTTPException(500, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/tts/batch")
async def tts_batch(req: BatchTTSRequest):
    results = []
    for item in req.items:
        try:
            r = await _manager.synthesize(item.text, item.voice, item.speed,
                                          item.backend, item.language,
                                          item.reference_audio, item.output_path)
            results.append(r)
        except Exception as e:
            results.append({"error": str(e), "text": item.text[:50]})
    return {"results": results, "total": len(results),
            "success": sum(1 for r in results if "error" not in r)}


@app.post("/tts/file")
async def tts_file(req: TTSRequest):
    try:
        result = await _manager.synthesize(req.text, req.voice, req.speed,
                                           req.backend, req.language,
                                           req.reference_audio, req.output_path)
        if "error" in result:
            raise HTTPException(500, result["error"])
        return FileResponse(result["output_path"], media_type="audio/wav",
                             filename=os.path.basename(result["output_path"]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    if not _manager:
        return JSONResponse({"status": "offline"}, status_code=503)
    s = _manager.get_status()
    return {"status": "healthy", "mode": "unified-lazy-load",
            "any_track_loaded": any(t["loaded"] for t in s["tracks"].values()), **s}


@app.get("/tracks")
async def tracks():
    return _manager.get_status() if _manager else {"error": "not initialized"}


@app.post("/tracks/load")
async def load_track(track_id: str):
    result = await _manager.preload(track_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/tracks/unload")
async def unload_track(track_id: str):
    result = await _manager.unload_track(track_id)
    return result


if __name__ == "__main__":
    import argparse
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "9880")))
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--idle-timeout", type=float, default=float(os.environ.get("TTS_IDLE_TIMEOUT", "300")))
    p.add_argument("--log-level", type=str, default="INFO")
    _args = p.parse_args()
    logging.getLogger().setLevel(getattr(logging, _args.log_level))
    uvicorn.run(app, host=_args.host, port=_args.port, log_level=_args.log_level.lower())
