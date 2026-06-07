#!/usr/bin/env python3
"""CosyVoice TTS Server (Track 3: 双语轨)

FastAPI wrapper around CosyVoice for gold-team integration.

Endpoints:
    POST /tts    — Generate speech (supports instruct2/zero_shot/cross_lingual/vc)
    GET  /health — Health check

Usage:
    python scripts/tts_cosyvoice_server.py [--port 9882] [--device cuda]

Model: CosyVoice-300M (~6GB VRAM)
GPU: RTX 3090
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

import numpy as np
import torch

# Add CosyVoice to path
COSYVOICE_ROOT = os.path.expanduser("~/CosyVoice")
if COSYVOICE_ROOT not in sys.path:
    sys.path.insert(0, COSYVOICE_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-cosyvoice")

app = FastAPI(title="CosyVoice TTS", version="1.0")

_model = None
_sr = 24000
_device = "cuda" if torch.cuda.is_available() else "cpu"
_start_time = time.time()
LOAD_DIR = tempfile.mkdtemp(prefix="cosyvoice_out_")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (zh/en)")
    mode: str = Field("instruct2", description="instruct2/zero_shot/cross_lingual/vc")
    speaker: str = Field("", description="Preset speaker (instruct2)")
    instruct_text: str = Field("", description="Instruction for voice control")
    ref_audio: str = Field("", description="Reference audio path")
    ref_text: str = Field("", description="Reference transcript (zero_shot)")
    language: str = Field("auto", description="zh/en/auto")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    output_path: str = Field("", description="Output file path")


class TTSResponse(BaseModel):
    audio_path: str
    duration_sec: float
    sample_rate: int
    mode: str
    language: str
    service: str = "cosyvoice"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "cosyvoice"
    device: str = "cpu"
    model_loaded: bool = False
    model_name: str = ""
    vram_used_mb: float = 0.0
    uptime_sec: float = 0.0


def load_model(device: str, model_dir: str = "pretrained_models/CosyVoice-300M"):
    global _model, _sr, _device
    logger.info("Loading CosyVoice on %s...", device)
    from cosyvoice.cli.cosyvoice import CosyVoice

    model_path = os.path.join(COSYVOICE_ROOT, model_dir)
    _model = CosyVoice(model_path, device)
    _sr = _model.sample_rate
    _device = device
    logger.info("CosyVoice loaded (sr=%d, device=%s)", _sr, _device)


@app.on_event("startup")
async def startup():
    try:
        load_model(_device)
    except Exception as e:
        logger.error("Failed to load CosyVoice: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health():
    vram = 0.0
    if _device == "cuda" and torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 / 1024
    return HealthResponse(
        device=_device,
        model_loaded=_model is not None,
        model_name="CosyVoice-300M" if _model else "",
        vram_used_mb=round(vram, 1),
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.text.strip() and req.mode != "vc":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.time()

    # Generate based on mode
    wav = None

    if req.mode == "instruct2":
        if req.speaker:
            # Use preset speaker with instruct
            for i, gen in enumerate(_model.instruct2_with_audio_prompt(
                req.text,
                req.instruct_text or "用自然的声音朗读",
                req.speaker,
            )):
                wav = gen
        else:
            # Instruct with default voice
            for i, gen in enumerate(_model.instruct2(req.text, req.instruct_text or "用自然的声音朗读")):
                wav = gen

    elif req.mode == "zero_shot":
        if not req.ref_audio or not req.ref_text:
            raise HTTPException(status_code=400, detail="zero_shot requires ref_audio and ref_text")
        for i, gen in enumerate(_model.zero_shot(req.text, req.ref_text, req.ref_audio)):
            wav = gen

    elif req.mode == "cross_lingual":
        if not req.ref_audio:
            raise HTTPException(status_code=400, detail="cross_lingual requires ref_audio")
        for i, gen in enumerate(_model.cross_lingual(req.text, req.ref_audio)):
            wav = gen

    elif req.mode == "vc":
        if not req.ref_audio:
            raise HTTPException(status_code=400, detail="vc requires ref_audio")
        for i, gen in enumerate(_model.vc(req.ref_audio)):
            wav = gen

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    if wav is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed = time.time() - t0

    # Save
    if req.output_path:
        out_path = req.output_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = os.path.join(LOAD_DIR, f"tts_{int(time.time()*1000)}.wav")

    import torchaudio as ta
    ta.save(out_path, wav, _sr)

    duration = wav.shape[-1] / _sr if wav.ndim > 1 else len(wav) / _sr

    logger.info(
        "CosyVoice TTS: %.1f sec audio in %.1fs (mode=%s, text=%s)",
        duration, elapsed, req.mode, req.text[:50],
    )

    return TTSResponse(
        audio_path=out_path,
        duration_sec=round(duration, 2),
        sample_rate=_sr,
        mode=req.mode,
        language=req.language,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
