#!/usr/bin/env python3
"""GPT-SoVITS TTS Server Manager (Track 1: 中文轨)

Manages the GPT-SoVITS native api_v2.py process.

GPT-SoVITS api_v2.py exposes:
    POST /tts → raw wav audio bytes (status 200) or JSON error (status 400)
    GET  /set_gpt_weights, /set_sovits_weights, /control

This script adds:
    GET /health → JSON health check

Usage:
    # Start GPT-SoVITS api_v2 on port 9880 with this wrapper:
    python scripts/tts_gpt_sovits_server.py --port 9880 --gpt-sovits-dir ~/GPT-SoVITS

    The native api_v2.py runs on an internal port (9880+100=9980),
    this wrapper runs on the public port (9880) and:
    - /tts   → proxies to native, returns JSON {audio_path, duration_sec}
    - /tts/raw → proxies to native, returns raw audio bytes (passthrough)
    - /health → JSON health status
    - /set_gpt_weights, /set_sovits_weights → proxied to native
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import wave as wave_mod

import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-gpt-sovits")

app = FastAPI(title="GPT-SoVITS TTS (Track 1)", version="1.0")

_start_time = time.time()
_native_port = 9980  # Internal port for api_v2.py
_native_url = ""
LOAD_DIR = tempfile.mkdtemp(prefix="gpt_sovits_out_")
_sr = 32000
_gpt_sovits_dir = os.path.expanduser("~/GPT-SoVITS")


class TTSRequest(BaseModel):
    text: str = Field(..., description="中文文本")
    text_lang: str = Field("zh", description="语言 zh/en/ja/ko")
    ref_audio_path: str = Field("", description="角色参考音频路径")
    prompt_text: str = Field("", description="参考音频对应文本")
    prompt_lang: str = Field("zh", description="参考音频语言")
    speed_factor: float = Field(1.0, ge=0.5, le=2.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(15, ge=1, le=100)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    seed: int = Field(-1, description="-1=random")
    output_path: str = Field("")
    text_split_method: str = Field("cut5")
    batch_size: int = Field(1)


class TTSResponse(BaseModel):
    audio_path: str
    duration_sec: float
    sample_rate: int
    text: str
    service: str = "gpt-sovits"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "gpt-sovits"
    native_api: str
    native_running: bool
    model_loaded: bool = False
    vram_used_mb: float = 0.0
    uptime_sec: float = 0.0


_native_process: subprocess.Popen | None = None


def start_native_api(gpt_dir: str, native_port: int):
    """Start GPT-SoVITS api_v2.py as a subprocess."""
    global _native_process
    api_script = os.path.join(gpt_dir, "api_v2.py")
    if not os.path.exists(api_script):
        logger.error("api_v2.py not found at %s", api_script)
        return False

    config_path = os.path.join(gpt_dir, "GPT_SoVITS", "configs", "tts_infer.yaml")

    log_path = os.path.join(os.path.dirname(__file__), "..", ".tts_logs", "gpt_sovits_native.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 3090 (CUDA index 0 = physical 3090)
    env["NLTK_DATA"] = "/home/kai/nltk_data"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Use GPT-SoVITS venv python
    gsv_venv_python = os.path.join(gpt_dir, ".venv", "bin", "python")
    if os.path.exists(gsv_venv_python):
        python_exe = gsv_venv_python
    else:
        python_exe = sys.executable

    logger.info("Starting GPT-SoVITS api_v2.py (port=%d, config=%s, python=%s)", native_port, config_path, python_exe)
    logger.info("Native log: %s", log_path)

    with open(log_path, "a") as lf:
        _native_process = subprocess.Popen(
            [
                python_exe, api_script,
                "-a", "127.0.0.1",
                "-p", str(native_port),
                "-c", config_path,
            ],
            stdout=lf,
            stderr=lf,
            env=env,
            cwd=gpt_dir,
            start_new_session=True,
        )

    logger.info("GPT-SoVITS native process started (PID %d)", _native_process.pid)
    return True


def check_native_health() -> bool:
    """Check if native api_v2 is responding."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{_native_port}/tts?text=test&text_lang=zh&ref_audio_path=test.wav&prompt_lang=zh", method="GET")
        try:
            urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError as e:
            # 400 means the server is running (just invalid params)
            if e.code >= 400:
                return True
            return False
        return True
    except (urllib.error.URLError, OSError, ConnectionError):
        return False


@app.on_event("startup")
async def startup():
    import urllib.request
    global _native_url
    _native_url = f"http://127.0.0.1:{_native_port}"

    # Check if native API is already running
    if not check_native_health():
        logger.info("Native GPT-SoVITS API not running, starting it...")
        start_native_api(_gpt_sovits_dir, _native_port)
        # Wait for it to start
        for i in range(60):
            time.sleep(2)
            if check_native_health():
                logger.info("Native GPT-SoVITS API started after %ds", (i + 1) * 2)
                break
        else:
            logger.warning("Native GPT-SoVITS API failed to start within 120s")
    else:
        logger.info("Native GPT-SoVITS API already running on port %d", _native_port)


@app.on_event("shutdown")
async def shutdown():
    if _native_process and _native_process.poll() is None:
        _native_process.terminate()
        logger.info("Native GPT-SoVITS process terminated")


@app.get("/health", response_model=HealthResponse)
async def health():
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024 / 1024

    native_ok = check_native_health()

    return HealthResponse(
        native_api=_native_url,
        native_running=native_ok,
        model_loaded=native_ok,
        vram_used_mb=round(vram, 1),
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    """Synthesize Chinese speech → JSON with audio file path."""
    import aiohttp

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.time()

    native_req = {
        "text": req.text,
        "text_lang": req.text_lang,
        "ref_audio_path": req.ref_audio_path,
        "prompt_text": req.prompt_text,
        "prompt_lang": req.prompt_lang,
        "speed_factor": req.speed_factor,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "seed": req.seed,
        "text_split_method": req.text_split_method,
        "batch_size": req.batch_size,
        "media_type": "wav",
        "streaming_mode": 0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_native_url}/tts",
                json=native_req,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise HTTPException(status_code=502, detail=f"Native API error: {error[:300]}")

                audio_bytes = await resp.read()
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"GPT-SoVITS API unreachable: {e}")

    # Save to file
    if req.output_path:
        out_path = req.output_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        out_path = os.path.join(LOAD_DIR, f"tts_{int(time.time()*1000)}.wav")

    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    # Get duration from wav header
    duration = 0.0
    try:
        with wave_mod.open(out_path, "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info("GPT-SoVITS TTS: %.1f sec in %.1fs", duration, elapsed)

    return TTSResponse(
        audio_path=out_path,
        duration_sec=round(duration, 2),
        sample_rate=_sr,
        text=req.text[:100],
    )


@app.post("/tts/raw")
async def tts_raw(req: TTSRequest):
    """Passthrough: synthesize and return raw wav bytes directly."""
    import aiohttp

    native_req = {
        "text": req.text,
        "text_lang": req.text_lang,
        "ref_audio_path": req.ref_audio_path,
        "prompt_text": req.prompt_text,
        "prompt_lang": req.prompt_lang,
        "speed_factor": req.speed_factor,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "seed": req.seed,
        "text_split_method": req.text_split_method,
        "batch_size": req.batch_size,
        "media_type": "wav",
        "streaming_mode": 0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_native_url}/tts",
            json=native_req,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise HTTPException(status_code=502, detail=error[:300])
            audio_bytes = await resp.read()

    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS Server (managed)")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--native-port", type=int, default=9980)
    parser.add_argument("--gpt-sovits-dir", type=str, default="~/GPT-SoVITS")
    args = parser.parse_args()

    _native_port = args.native_port
    _gpt_sovits_dir = os.path.expanduser(args.gpt_sovits_dir)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
