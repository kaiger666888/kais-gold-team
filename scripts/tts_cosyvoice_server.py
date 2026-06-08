#!/usr/bin/env python3
"""CosyVoice TTS Server (Track 3: 双语轨)

FastAPI wrapper around CosyVoice-300M for gold-team integration.

Supported modes:
    - sft: Preset voice (预训练音色)
    - zero_shot: Voice cloning with reference audio + transcript
    - cross_lingual: Cross-language synthesis
    - vc: Voice conversion

Endpoints:
    POST /tts    — Generate speech
    GET  /health — Health check

Usage:
    python scripts/tts_cosyvoice_server.py [--port 9882] [--device cuda]

Model: CosyVoice-300M (~2.5GB VRAM)
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
    sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-cosyvoice")

app = FastAPI(title="CosyVoice TTS (Track 3)", version="1.0")

_model = None
_sr = 24000
_device = "cuda" if torch.cuda.is_available() else "cpu"
_start_time = time.time()
LOAD_DIR = tempfile.mkdtemp(prefix="cosyvoice_out_")
_model_dir = os.path.join(COSYVOICE_ROOT, "pretrained_models", "CosyVoice-300M")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (zh/en)")
    mode: str = Field("sft", description="sft/zero_shot/cross_lingual/vc")
    speaker: str = Field("", description="Preset speaker name (sft mode)")
    instruct_text: str = Field("", description="Instruction for voice control (ignored for 300M)")
    ref_audio: str = Field("", description="Reference audio path (zero_shot/cross_lingual/vc)")
    ref_text: str = Field("", description="Reference transcript (zero_shot mode)")
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
    available_speakers: list = []
    vram_used_mb: float = 0.0
    uptime_sec: float = 0.0


def load_model(device: str):
    global _model, _sr, _device
    logger.info("Loading CosyVoice-300M on %s...", device)
    from cosyvoice.cli.cosyvoice import CosyVoice
    _model = CosyVoice(_model_dir, device)
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
    speakers = []
    if _model:
        try:
            speakers = _model.list_available_spks()
        except Exception:
            pass
    return HealthResponse(
        device=_device,
        model_loaded=_model is not None,
        model_name="CosyVoice-300M" if _model else "",
        available_speakers=speakers,
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
    wav = None

    if req.mode == "sft":
        # Use preset speaker
        speakers = _model.list_available_spks()
        spk = req.speaker if req.speaker in speakers else (speakers[0] if speakers else None)
        if not spk:
            raise HTTPException(status_code=400, detail="No preset speakers available")
        for i, gen in enumerate(_model.inference_sft(req.text, spk)):
            wav = gen

    elif req.mode == "zero_shot":
        if not req.ref_audio or not req.ref_text:
            raise HTTPException(status_code=400, detail="zero_shot requires ref_audio and ref_text")
        for i, gen in enumerate(_model.inference_zero_shot(req.text, req.ref_text, req.ref_audio)):
            wav = gen

    elif req.mode == "cross_lingual":
        if not req.ref_audio:
            raise HTTPException(status_code=400, detail="cross_lingual requires ref_audio")
        for i, gen in enumerate(_model.inference_cross_lingual(req.text, req.ref_audio)):
            wav = gen

    elif req.mode == "vc":
        if not req.ref_audio:
            raise HTTPException(status_code=400, detail="vc requires ref_audio")
        for i, gen in enumerate(_model.inference_vc(req.ref_audio)):
            wav = gen

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

    if wav is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    # CosyVoice returns dict {'tts_speech': tensor} in some modes
    if isinstance(wav, dict):
        wav = wav.get('tts_speech', wav.get('output', None))

    if wav is None:
        raise HTTPException(status_code=500, detail="No audio tensor in response")

    elapsed = time.time() - t0

    # Save
    if req.output_path:
        out_path = req.output_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        out_path = os.path.join(LOAD_DIR, f"tts_{int(time.time()*1000)}.wav")

    import torchaudio as ta
    ta.save(out_path, wav, _sr)

    duration = wav.shape[-1] / _sr if wav.ndim > 1 else len(wav) / _sr

    logger.info("CosyVoice: %.1fs audio in %.1fs (mode=%s, text=%s)", duration, elapsed, req.mode, req.text[:50])

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
