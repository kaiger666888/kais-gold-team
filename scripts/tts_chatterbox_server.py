#!/usr/bin/env python3
"""Chatterbox-Turbo TTS Server (Track 2: 英文轨)

FastAPI wrapper around Chatterbox-Turbo for gold-team integration.

Endpoints:
    POST /tts   — Generate speech from text
    GET  /health — Health check

Usage:
    python scripts/tts_chatterbox_server.py [--port 9881] [--device cuda]

Model: Chatterbox-Turbo (~2GB VRAM)
GPU: RTX 3060Ti (shared with GPT-SoVITS)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Fix perth watermarker (NoneType on some systems)
import perth
if perth.PerthImplicitWatermarker is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

# Add chatterbox to path
CHATTERBOX_ROOT = os.path.expanduser("~/chatterbox")
if CHATTERBOX_ROOT not in sys.path:
    sys.path.insert(0, CHATTERBOX_ROOT)
sys.path.insert(0, os.path.join(CHATTERBOX_ROOT, "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-chatterbox")

app = FastAPI(title="Chatterbox-Turbo TTS", version="1.0")

# ─── Globals ───
_model = None
_sr = 24000
_device = "cuda" if torch.cuda.is_available() else "cpu"
_start_time = time.time()
LOAD_DIR = tempfile.mkdtemp(prefix="chatterbox_out_")


class TTSRequest(BaseModel):
    text: str = Field(..., description="English text (supports paralinguistic tags)")
    audio_prompt_path: str = Field("", description="Reference audio for voice cloning")
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(1000, ge=0, le=2000)
    seed: int = Field(0, description="0=random")
    output_path: str = Field("", description="Output file path (server-generated if empty)")


class TTSResponse(BaseModel):
    audio_path: str
    duration_sec: float
    sample_rate: int
    text: str
    service: str = "chatterbox-turbo"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "chatterbox-turbo"
    device: str = "cpu"
    model_loaded: bool = False
    vram_used_mb: float = 0.0
    uptime_sec: float = 0.0


def load_model(device: str):
    global _model, _sr, _device
    logger.info("Loading Chatterbox-Turbo on %s...", device)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    # Try from_local first (skip HuggingFace download), fallback to from_pretrained
    try:
        _model = ChatterboxTurboTTS.from_local(os.path.expanduser("~/chatterbox/models/turbo"), device)
        logger.info("Loaded Chatterbox-Turbo from local models/base")
    except Exception as e:
        logger.warning("from_local failed (%s), trying from_pretrained", e)
        _model = ChatterboxTurboTTS.from_pretrained(device)
    _sr = _model.sr
    _device = device
    logger.info("Chatterbox-Turbo loaded (sr=%d, device=%s)", _sr, _device)


@app.on_event("startup")
async def startup():
    try:
        load_model(_device)
    except Exception as e:
        logger.error("Failed to load Chatterbox-Turbo: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health():
    vram = 0.0
    if _device == "cuda" and torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 / 1024
    return HealthResponse(
        device=_device,
        model_loaded=_model is not None,
        vram_used_mb=round(vram, 1),
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.time()

    # Set seed
    if req.seed != 0:
        torch.manual_seed(req.seed)
        torch.cuda.manual_seed(req.seed)

    # Generate
    kwargs = {
        "text": req.text,
        "temperature": req.temperature,
        "min_p": 0.0,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": 1.2,
        "norm_loudness": True,
    }
    if req.audio_prompt_path:
        kwargs["audio_prompt_path"] = req.audio_prompt_path
    else:
        # Use default reference audio if none provided
        default_ref = os.path.join(tempfile.gettempdir(), "female_random_podcast.wav")
        if not os.path.exists(default_ref):
            import urllib.request as _urllib
            _urllib.urlretrieve("https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav", default_ref)
        if os.path.exists(default_ref):
            kwargs["audio_prompt_path"] = default_ref

    wav = _model.generate(**kwargs)
    elapsed = time.time() - t0

    # Save output
    if req.output_path:
        out_path = req.output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = os.path.join(LOAD_DIR, f"tts_{int(time.time()*1000)}.wav")

    import torchaudio as ta
    ta.save(out_path, wav, _sr)

    duration = wav.shape[-1] / _sr if wav.ndim > 1 else len(wav) / _sr

    logger.info(
        "TTS generated: %.1f sec audio in %.1fs (text=%s)",
        duration, elapsed, req.text[:50],
    )

    return TTSResponse(
        audio_path=out_path,
        duration_sec=round(duration, 2),
        sample_rate=_sr,
        text=req.text[:100],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatterbox-Turbo TTS Server")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
