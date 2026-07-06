#!/usr/bin/env python3
"""GPT-SoVITS TTS Server — Direct TTS class wrapper (bypasses buggy api_v2.py)

Uses GPT_SoVITS.TTS_infer_pack.TTS.TTS directly, avoiding api_v2.py's
Pydantic V2 incompatibilities.

Endpoints:
    POST /tts    — Synthesize Chinese speech → JSON {audio_path, duration_sec}
    GET  /health — Health check

Usage:
    python scripts/tts_gpt_sovits_direct.py --port 9880 --device cuda
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
import wave as wave_mod

import numpy as np
import torch

# Setup paths — GPT-SoVITS requires cwd to be GPT_SoVITS/ for relative imports
GPTSOVITS_ROOT = os.path.expanduser("~/GPT-SoVITS")
GPTSOVITS_PKG = os.path.join(GPTSOVITS_ROOT, "GPT_SoVITS")
# Change CWD to GPT_SoVITS for relative imports (eres2net, etc.)
os.chdir(GPTSOVITS_PKG)
sys.path.insert(0, GPTSOVITS_ROOT)
sys.path.insert(0, GPTSOVITS_PKG)
# eres2net uses bare imports (import pooling_layers), needs its dir on path
sys.path.insert(0, os.path.join(GPTSOVITS_PKG, "eres2net"))
sys.path.insert(0, os.path.join(GPTSOVITS_PKG, "module"))
# Do NOT add subdirectories to sys.path — they are found via package imports.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-gpt-sovits")

app = FastAPI(title="GPT-SoVITS TTS (Direct)", version="1.0")

_tts_pipeline = None
_sr = 32000
_device = "cuda"
_start_time = time.time()
LOAD_DIR = tempfile.mkdtemp(prefix="gpt_sovits_out_")


class TTSRequest(BaseModel):
    text: str = Field(..., description="中文文本")
    text_lang: str = Field("zh")
    ref_audio_path: str = Field("", description="参考音频路径(3-10秒)")
    prompt_text: str = Field("")
    prompt_lang: str = Field("zh")
    speed_factor: float = Field(1.0, ge=0.5, le=2.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(15)
    top_p: float = Field(1.0)
    seed: int = Field(-1)
    output_path: str = Field("")


class TTSResponse(BaseModel):
    audio_path: str
    duration_sec: float
    sample_rate: int
    service: str = "gpt-sovits"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "gpt-sovits"
    device: str = "cpu"
    model_loaded: bool = False
    vram_used_mb: float = 0.0
    uptime_sec: float = 0.0


def load_model(device: str):
    global _tts_pipeline, _sr, _device
    logger.info("Loading GPT-SoVITS TTS pipeline on %s...", device)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    config_path = os.path.join(GPTSOVITS_ROOT, "GPT_SoVITS", "configs", "tts_infer.yaml")
    tts_config = TTS_Config(config_path)
    _tts_pipeline = TTS(tts_config)
    _sr = 32000
    _device = device
    logger.info("GPT-SoVITS TTS loaded (sr=%d)", _sr)


@app.on_event("startup")
async def startup():
    try:
        load_model(_device)
    except Exception as e:
        logger.error("Failed to load GPT-SoVITS: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health():
    vram = 0.0
    if _device == "cuda" and torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 / 1024
    return HealthResponse(
        device=_device,
        model_loaded=_tts_pipeline is not None,
        vram_used_mb=round(vram, 1),
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    if _tts_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not req.ref_audio_path:
        raise HTTPException(
            status_code=400,
            detail="ref_audio_path required (3-10 second audio for voice cloning)"
        )

    t0 = time.time()

    try:
        # Run TTS in thread pool to avoid blocking asyncio
        loop = asyncio.get_event_loop()
        audio_chunks = []

        def _generate():
            chunks = []
            inference_dict = {
                "text": req.text,
                "text_lang": req.text_lang,
                "ref_audio_path": req.ref_audio_path,
                "prompt_text": req.prompt_text,
                "prompt_lang": req.prompt_lang,
                "speed_factor": req.speed_factor,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "seed": req.seed if req.seed >= 0 else -1,
                "streaming_mode": 0,
                "parallel_infer": False,
                "split_bucket": False,
                "text_split_method": "cut5",
                "batch_size": 1,
            }
            for chunk in _tts_pipeline.run(inference_dict):
                chunks.append(chunk)
            return chunks

        audio_chunks = await loop.run_in_executor(None, _generate)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Merge chunks
        audio = np.concatenate(audio_chunks, axis=0)
        sr = _sr

        elapsed = time.time() - t0

        # Save as wav
        if req.output_path:
            out_path = req.output_path
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        else:
            out_path = os.path.join(LOAD_DIR, f"tts_{int(time.time()*1000)}.wav")

        import soundfile as sf
        sf.write(out_path, audio, sr)

        duration = len(audio) / sr
        logger.info("GPT-SoVITS TTS: %.1fs audio in %.1fs", duration, elapsed)

        return TTSResponse(
            audio_path=out_path,
            duration_sec=round(duration, 2),
            sample_rate=sr,
        )

    except Exception as e:
        logger.error("GPT-SoVITS TTS failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS Server (Direct)")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
