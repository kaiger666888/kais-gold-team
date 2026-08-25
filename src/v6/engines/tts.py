"""Three-Track TTS Engine for kais-gold-team V6 — in-process, no HTTP layer.

Architecture (方案C — TrackManager embedded in TTSTracker):
    Track 1 (中文轨): GPT-SoVITS — Chinese TTS (voice_clone), ~4 GB VRAM
    Track 2 (英文轨): Chatterbox-Turbo — English TTS, ~2 GB VRAM
    Track 3 (双语轨): CosyVoice 300M — Bilingual TTS, ~2.5 GB VRAM

All three tracks share the gold-team 3090 GPU with lazy-load + idle-unload.
No HTTP intermediate — TrackManager runs in-process inside TTSTracker.

Lifecycle:
    - TTSTracker.start() creates TrackManager + idle reaper
    - submit() calls TrackManager.synthesize() synchronously → returns job_id immediately
    - Results cached in _results dict for poll()/get_output()
    - Tracks loaded on first request, unloaded after idle_timeout
"""
from __future__ import annotations

# ── Disable numba JIT BEFORE any import that touches librosa ──────────────
try:
    import numba  # noqa: E402 — must be before librosa
    numba.config.DISABLE_JIT = True
except ImportError:  # numba absent → no JIT to disable; librosa may not be installed either
    numba = None

import asyncio
import gc
import logging
import os
import re
import sys
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/mnt/agents/output")

# ── Model paths ─────────────────────────────────────────────────────────────
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")
GPTSOVITS_ROOT = os.environ.get("GPTSOVITS_ROOT", "/opt/GPT-SoVITS")
CHATTERBOX_ROOT = os.environ.get("CHATTERBOX_ROOT", "/opt/chatterbox")

# ── Language detection ─────────────────────────────────────────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


# ─── TTS Track Definition ──────────────────────────────────────────────────

class TTSTrack(str, Enum):
    ZH = "zh"                # GPT-SoVITS
    EN = "en"                # Chatterbox-Turbo
    BILINGUAL = "bilingual"  # CosyVoice


def detect_language(text: str) -> str:
    """Simple CJK-based language detection.

    Returns 'zh', 'en', or 'auto' (mixed).
    """
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and not has_latin:
        return "zh"
    if has_latin and not has_cjk:
        return "en"
    return "auto"


# ═══════════════════════════════════════════════════════════════════════════
# Track wrappers — lazy-load, auto-unload
# ═══════════════════════════════════════════════════════════════════════════

class BaseTrack:
    """Base class for a TTS track — manages model lifecycle."""

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
        """Load model if not already loaded (with dedup lock)."""
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
                logger.info("Track '%s' loaded successfully", self.name)
            except Exception as e:
                logger.error("Track '%s' failed to load: %s", self.name, e)
                raise
            finally:
                self._loading = False

    def _load_model(self) -> None:
        """Override in subclass — loads model to GPU."""
        raise NotImplementedError

    async def synthesize(self, text: str, voice: str = "default",
                         speed: float = 1.0, reference_audio: str = "",
                         output_path: str = "") -> dict:
        """Synthesize speech. Must be called after ensure_loaded."""
        raise NotImplementedError

    async def unload(self) -> None:
        """Unload model, free VRAM."""
        async with self._lock:
            if not self._loaded:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._unload_model)

    def _unload_model(self) -> None:
        """Override in subclass — frees GPU memory."""
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Track '%s' unloaded, VRAM freed", self.name)

    def is_idle(self) -> bool:
        return self._loaded and (time.monotonic() - self._last_used > self._idle_timeout)

    def status(self) -> dict:
        return {
            "name": self.name,
            "language": self.language,
            "loaded": self._loaded,
            "loading": self._loading,
            "vram_mb": self.vram_mb,
            "last_used": self._last_used,
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
        logger.info("CosyVoice-300M loaded")

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CosyVoice track unloaded")

    async def synthesize(self, text: str, voice: str = "default",
                         speed: float = 1.0, reference_audio: str = "",
                         output_path: str = "") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            try:
                if reference_audio and os.path.isfile(reference_audio):
                    result = self._model.inference_zero_shot(
                        text, "", reference_audio, stream=False, speed=speed,
                    )
                else:
                    import soundfile as sf
                    # No preset speakers → use cross_lingual with dummy ref
                    dummy_sr = 22050
                    t = np.arange(0, 10.0, 1.0 / dummy_sr)
                    dummy_wav = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
                    dummy_path = os.path.join(OUTPUT_ROOT, "_cosyvoice_ref.wav")
                    os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
                    sf.write(dummy_path, dummy_wav, dummy_sr)
                    result = self._model.inference_cross_lingual(
                        text, dummy_path, stream=False, speed=speed,
                    )
                import soundfile as sf
                all_audio = []
                sr = self._model.sample_rate
                for model_output in result:
                    tts_speech = model_output['tts_speech']
                    if hasattr(tts_speech, 'cpu'):
                        tts_speech = tts_speech.cpu().numpy()
                    if tts_speech.ndim == 3:
                        tts_speech = tts_speech.squeeze(0)
                    if tts_speech.ndim == 1:
                        tts_speech = tts_speech.unsqueeze(0)
                    all_audio.append(tts_speech)
                if not all_audio:
                    return {"error": "No audio generated"}
                audio_np = np.concatenate(all_audio, axis=1).squeeze(0) if len(all_audio) > 1 else all_audio[0].squeeze(0)
                if not output_path:
                    output_path = os.path.join(OUTPUT_ROOT, f"tts_{int(time.time())}.wav")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, audio_np, sr)
                duration = len(audio_np) / sr
                return {
                    "output_path": output_path,
                    "duration_sec": round(duration, 2),
                    "sample_rate": sr,
                    "backend": "cosyvoice",
                }
            except Exception as e:
                logger.error("CosyVoice synthesis error: %s", e)
                raise

        return await loop.run_in_executor(None, _synth)


class GPTSoVITSTrack(BaseTrack):
    """Track 1: GPT-SoVITS — Chinese voice cloning.

    Uses GPT-SoVITS api_v2.py as a subprocess (avoids massive import-time side effects).
    """

    name = "gpt_sovits"
    language = "zh"
    vram_mb = 4000
    _api_port: int = 9988
    _process: Optional[asyncio.subprocess.Process] = None

    def _load_model(self) -> None:
        """Start GPT-SoVITS api_v2.py as subprocess."""
        import subprocess
        script = os.path.join(GPTSOVITS_ROOT, "api_v2.py")
        config = os.path.join(GPTSOVITS_ROOT, "GPT_SoVITS", "configs", "tts_infer.yaml")

        python_bin = os.path.join(GPTSOVITS_ROOT, ".venv", "bin", "python")
        if not os.path.isfile(python_bin):
            python_bin = "python3"

        self._process = subprocess.Popen(
            [python_bin, script, "-a", "127.0.0.1", "-p", str(self._api_port), "-c", config],
            cwd=GPTSOVITS_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
        )
        import urllib.request
        import urllib.error
        for i in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._api_port}/", timeout=1)
                logger.info("GPT-SoVITS api_v2 started on port %d", self._api_port)
                return
            except (urllib.error.URLError, ConnectionRefusedError, OSError):
                time.sleep(0.5)
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode(errors='replace')
                    raise RuntimeError(f"GPT-SoVITS api_v2 crashed: {stderr[-500:]}")
        raise RuntimeError("GPT-SoVITS api_v2 failed to start within 120s")

    def _unload_model(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
            logger.info("GPT-SoVITS api_v2 process stopped")
        self._process = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPT-SoVITS track unloaded")

    async def synthesize(self, text: str, voice: str = "default",
                         speed: float = 1.0, reference_audio: str = "",
                         output_path: str = "") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            import urllib.request
            import json as _json
            import soundfile as sf

            ref_path = reference_audio
            if not ref_path or not os.path.isfile(ref_path):
                dummy_sr = 32000
                t = np.arange(0, 3.0, 1.0 / dummy_sr)
                dummy_wav = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
                ref_path = os.path.join(OUTPUT_ROOT, "_gpt_sovits_ref.wav")
                sf.write(ref_path, dummy_wav, dummy_sr)

            payload = _json.dumps({
                "text": text,
                "text_lang": "zh",
                "ref_audio_path": ref_path,
                "prompt_text": "",
                "prompt_lang": "zh",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "speed_factor": speed,
                "media_type": "wav",
            }).encode()

            req = urllib.request.Request(
                f"http://127.0.0.1:{self._api_port}/tts",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            audio_bytes = resp.read()

            if not output_path:
                output_path = os.path.join(OUTPUT_ROOT, f"tts_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            data, sr = sf.read(output_path)
            duration = len(data) / sr
            return {
                "output_path": output_path,
                "duration_sec": round(duration, 2),
                "sample_rate": sr,
                "backend": "gpt_sovits",
            }

        return await loop.run_in_executor(None, _synth)


class ChatterboxTrack(BaseTrack):
    """Track 2: Chatterbox-Turbo — English TTS (in-process on CUDA)."""

    name = "chatterbox"
    language = "en"
    vram_mb = 2000

    def _load_model(self) -> None:
        """Verify chatterbox is importable."""
        chatterbox_src = os.path.join(CHATTERBOX_ROOT, "src")
        if os.path.isdir(chatterbox_src):
            sys.path.insert(0, CHATTERBOX_ROOT)
            sys.path.insert(0, chatterbox_src)
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            logger.info("Chatterbox-Turbo importable")
        except ImportError as e:
            logger.warning("Chatterbox import failed: %s (track will fallback)", e)

    def _unload_model(self) -> None:
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Chatterbox track unloaded")

    async def synthesize(self, text: str, voice: str = "default",
                         speed: float = 1.0, reference_audio: str = "",
                         output_path: str = "") -> dict:
        await self.ensure_loaded()
        loop = asyncio.get_event_loop()

        def _synth():
            nonlocal output_path
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            # 优先从本地 CHATTERBOX_ROOT/models/turbo 加载，避免HF下载
            local_path = None
            ch_root = os.environ.get("CHATTERBOX_ROOT", "")
            if ch_root:
                turbo_dir = os.path.join(ch_root, "models", "turbo")
                if os.path.isdir(turbo_dir) and os.path.exists(os.path.join(turbo_dir, "t3_turbo_v1.safetensors")):
                    local_path = turbo_dir
                    logger.info(f"Chatterbox Turbo loading from local: {turbo_dir}")

            if not local_path:
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                repo_cache = os.path.join(cache_dir, "models--ResembleAI--chatterbox-turbo")
                snapshots_dir = os.path.join(repo_cache, "snapshots")
                if os.path.isdir(snapshots_dir):
                    for snap in os.listdir(snapshots_dir):
                        snap_dir = os.path.join(snapshots_dir, snap)
                        if os.path.isdir(snap_dir) and os.path.exists(os.path.join(snap_dir, "t3_turbo_v1.safetensors")):
                            local_path = snap_dir
                            break

            if local_path:
                model = ChatterboxTurboTTS.from_local(local_path, "cuda")
            else:
                logger.warning("Chatterbox: no local model found, downloading from HF (may be slow)")
                model = ChatterboxTurboTTS.from_pretrained("cuda")

            if reference_audio and os.path.isfile(reference_audio):
                model.prepare_conditionals(reference_audio)

            wav_tensor = model.generate(
                text=text,
                temperature=0.8,
                top_k=1000,
                top_p=0.95,
            )

            audio_np = wav_tensor.squeeze(0).numpy()
            sr = model.sr

            if not output_path:
                output_path = os.path.join(OUTPUT_ROOT, f"tts_{int(time.time())}.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            import soundfile as sf
            sf.write(output_path, audio_np, sr)
            duration = len(audio_np) / sr

            # Free model immediately (no caching between requests)
            del model
            torch.cuda.empty_cache()

            return {
                "output_path": output_path,
                "duration_sec": round(duration, 2),
                "sample_rate": sr,
                "backend": "chatterbox",
            }

        return await loop.run_in_executor(None, _synth)


# ═══════════════════════════════════════════════════════════════════════════
# Track Manager — orchestrates all three tracks
# ═══════════════════════════════════════════════════════════════════════════

class TrackManager:
    """Manages three TTS tracks with lazy loading and idle auto-unload."""

    def __init__(self, idle_timeout: float = 300.0):
        self._tracks: Dict[str, BaseTrack] = {
            "gpt_sovits": GPTSoVITSTrack(idle_timeout),
            "chatterbox": ChatterboxTrack(idle_timeout),
            "cosyvoice": CosyVoiceTrack(idle_timeout),
        }
        self._idle_timeout = idle_timeout
        self._reaper_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the idle reaper background task."""
        self._reaper_task = asyncio.create_task(self._idle_reaper())
        logger.info("TrackManager started (idle_timeout=%.0fs)", self._idle_timeout)

    async def stop(self) -> None:
        """Unload all tracks and stop reaper."""
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        for track in self._tracks.values():
            await track.unload()
        logger.info("TrackManager stopped, all tracks unloaded")

    def select_track(self, text: str, backend: str = "auto",
                     language: str = "auto") -> str:
        """Select the best track for the given text."""
        if backend != "auto" and backend in self._tracks:
            return backend

        if language == "auto":
            language = detect_language(text)

        lang_map = {"zh": "gpt_sovits", "en": "chatterbox", "auto": "cosyvoice"}
        return lang_map.get(language, "cosyvoice")

    async def synthesize(self, text: str, voice: str = "default",
                         speed: float = 1.0, backend: str = "auto",
                         language: str = "auto", reference_audio: str = "",
                         output_path: str = "") -> dict:
        """Route to the correct track and synthesize."""
        track_id = self.select_track(text, backend, language)
        track = self._tracks[track_id]

        try:
            result = await track.synthesize(
                text=text, voice=voice, speed=speed,
                reference_audio=reference_audio, output_path=output_path,
            )
            result["track"] = track_id
            return result
        except Exception as e:
            # Fallback to cosyvoice if primary fails
            if track_id != "cosyvoice":
                logger.warning("Track '%s' failed (%s), falling back to cosyvoice", track_id, e)
                try:
                    result = await self._tracks["cosyvoice"].synthesize(
                        text=text, voice=voice, speed=speed,
                        reference_audio=reference_audio, output_path=output_path,
                    )
                    result["track"] = "cosyvoice"
                    result["fallback_from"] = track_id
                    return result
                except Exception as e2:
                    return {"error": f"All tracks failed: {e}, cosyvoice: {e2}"}
            return {"error": str(e)}

    def get_status(self) -> dict:
        return {
            "idle_timeout": self._idle_timeout,
            "tracks": {tid: t.status() for tid, t in self._tracks.items()},
            "total_vram_mb": sum(t.vram_mb for t in self._tracks.values() if t.is_loaded),
            "available_vram_mb": sum(t.vram_mb for t in self._tracks.values()),
        }

    async def _idle_reaper(self) -> None:
        """Background task: unload idle tracks."""
        while True:
            try:
                await asyncio.sleep(30)
                for tid, track in self._tracks.items():
                    if track.is_idle():
                        logger.info("Track '%s' idle > %.0fs, unloading", tid, self._idle_timeout)
                        await track.unload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Idle reaper error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# TTSTracker — unified facade implementing BaseEngine
# ═══════════════════════════════════════════════════════════════════════════

class TTSTracker(BaseEngine):
    """Unified TTS facade — in-process TrackManager, no HTTP layer.

    Routing logic:
    - Explicit track specified → use that engine
    - language='zh' → Track 1 (GPT-SoVITS)
    - language='en' → Track 2 (Chatterbox-Turbo)
    - language='auto' or mixed → CosyVoice (bilingual)
    - Fallback: if target track fails, degrade to CosyVoice
    """

    def __init__(self, idle_timeout: float = 300.0,
                 output_root: str = OUTPUT_ROOT) -> None:
        self._output_root = output_root
        self._idle_timeout = idle_timeout
        self._manager = TrackManager(idle_timeout=idle_timeout)
        self._results: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "TTS Tracker (三轨调度 in-process)"

    @property
    def engine_id(self) -> str:
        return "tts-tracker"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["tts", "tts_zh", "tts_en", "tts_bilingual"],
            max_duration_sec=120.0,
            vram_total_mb=8500,  # 4G + 2G + 2.5G
            vram_available_mb=8500,
            models=["GPT-SoVITS", "Chatterbox-Turbo", "CosyVoice-300M"],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SUBPROCESS

    @property
    def manager(self) -> TrackManager:
        return self._manager

    async def start(self) -> None:
        await self._manager.start()
        logger.info("TTS Tracker started (in-process, %d tracks, idle=%.0fs)",
                     len(self._manager._tracks), self._idle_timeout)

    async def stop(self) -> None:
        await self._manager.stop()
        self._results.clear()

    def _pick_track_params(self, workflow: dict[str, Any]) -> tuple[str, str]:
        """Extract backend and language from workflow for TrackManager routing.

        Returns (backend, language) strings.
        """
        explicit = workflow.get("track", "")
        backend_map = {
            "zh": "gpt_sovits",
            "en": "chatterbox",
            "bilingual": "cosyvoice",
        }
        if explicit in backend_map:
            return backend_map[explicit], workflow.get("language", "auto")

        language = workflow.get("language", "auto").lower()
        return "auto", language

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit TTS — synchronously synthesizes, returns job_id immediately."""
        params = params or {}
        job_id = str(uuid.uuid4())[:12]
        task_id = params.get("task_id", job_id)

        text = workflow.get("text", "") or params.get("text", "")
        output_path = workflow.get("output_path", "") or f"{self._output_root}/{task_id}/voice.wav"

        backend, language = self._pick_track_params(workflow)

        logger.info(
            "TTS job %s: submit (text=%s, backend=%s, language=%s)",
            job_id, text[:60], backend, language,
        )

        # Synchronous TTS via TrackManager (in-process, no HTTP)
        result = await self._manager.synthesize(
            text=text,
            voice=workflow.get("voice", "default"),
            speed=workflow.get("speed", 1.0),
            backend=backend,
            language=language,
            reference_audio=workflow.get("reference_audio", ""),
            output_path=output_path,
        )

        if "error" in result:
            self._results[job_id] = {
                "status": "failed",
                "progress": 0.0,
                "error": result["error"],
                "output_path": "",
                "duration_sec": 0.0,
                "track": result.get("track", ""),
                "fallback_from": result.get("fallback_from", ""),
            }
            logger.error("TTS job %s failed: %s", job_id, result["error"])
        else:
            self._results[job_id] = {
                "status": "completed",
                "progress": 100.0,
                "error": "",
                "output_path": result.get("output_path", output_path),
                "duration_sec": result.get("duration_sec", 0.0),
                "track": result.get("track", ""),
                "fallback_from": result.get("fallback_from", ""),
                "sample_rate": result.get("sample_rate", 0),
                "backend": result.get("backend", ""),
            }
            logger.info(
                "TTS job %s completed (track=%s, duration=%.1fs, output=%s)",
                job_id, result.get("track", ""),
                result.get("duration_sec", 0), result.get("output_path", ""),
            )

        return job_id

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll — always completed/failed since TTS is synchronous."""
        result = self._results.get(engine_job_id)
        if result:
            return {
                "status": result["status"],
                "progress": result["progress"],
                "error": result.get("error", ""),
                "track": result.get("track", ""),
            }
        return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Return audio artifact for a completed job."""
        result = self._results.get(engine_job_id)
        if not result or result["status"] != "completed":
            return {"outputs": []}

        artifacts = [{
            "url": f"file://{result['output_path']}",
            "path": result["output_path"],
            "type": "audio",
            "format": "wav",
            "track": result.get("track", ""),
            "backend": result.get("backend", ""),
            "duration_sec": result.get("duration_sec", 0.0),
            "sample_rate": result.get("sample_rate", 0),
        }]
        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        """Cancel is meaningless — TTS completes synchronously."""
        return False

    async def health(self) -> dict[str, Any]:
        """Report health from TrackManager."""
        status = self._manager.get_status()
        any_loaded = any(t["loaded"] for t in status["tracks"].values())
        online_count = sum(1 for t in status["tracks"].values() if t["loaded"])
        return {
            "status": EngineStatus.ONLINE.value,
            "available": True,  # Always available (lazy-loads on demand)
            "mode": "in-process (lazy-load)",
            "tracks": status["tracks"],
            "online_tracks": online_count,
            "total_tracks": len(status["tracks"]),
            "total_vram_mb": status["total_vram_mb"],
            "idle_timeout": self._idle_timeout,
            "routing_summary": {
                "zh": "GPT-SoVITS (gpt_sovits)",
                "en": "Chatterbox-Turbo (chatterbox)",
                "bilingual": "CosyVoice-300M (cosyvoice)",
            },
        }
