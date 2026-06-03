#!/usr/bin/env python3
"""
Wan2.1 GGUF ComfyUI API Server
================================
Lightweight FastAPI wrapper around a running ComfyUI instance (port 8188)
serving Wan2.1 GGUF models for T2V and I2V video generation.

Exposes:
  POST /generate  → submit T2V or I2V workflow to ComfyUI
  GET  /health    → health check (ComfyUI + GPU status)
  GET  /status/{task_id} → poll ComfyUI task status & outputs
  GET  /models    → list available models

Requires: ComfyUI running on 127.0.0.1:8188 with comfyui-wan custom nodes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Config ────────────────────────────────────────────────────────────

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8081"))
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/wan_output")

# ComfyUI model paths (inside comfyui-wan container / mapped volumes)
T5_ENCODER_PATH = os.getenv("T5_ENCODER_PATH", "models/umt5-xxl-enc-bf16.pth")
VAE_PATH = os.getenv("VAE_PATH", "models/wan_2.1_vae.safetensors")
T2V_MODEL_PATH = os.getenv("T2V_MODEL_PATH", "models/wan2.1-t2v-14b-Q8_0.gguf")
I2V_MODEL_PATH = os.getenv("I2V_MODEL_PATH", "models/wan2.1-i2v-480p-14b-Q4_K_M_city96.gguf")

# ComfyUI output served via its own HTTP
COMFYUI_BASE = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# Resolution presets
RESOLUTION_MAP = {
    "480p": {"width": 832, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "832x480": {"width": 832, "height": 480},
    "1280x720": {"width": 1280, "height": 720},
}

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 600.0  # 10 minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wan_gguf_api")

# ─── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(title="Wan2.1 GGUF API", version="1.0.0")

# ─── Pydantic models ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Unified generate request for both T2V and I2V."""
    # Core
    prompt: str = Field(default="", description="Text prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    task_type: str = Field(default="t2v", description="'t2v' or 'i2v'")

    # Video dimensions
    width: Optional[int] = Field(default=None, description="Video width (overrides resolution)")
    height: Optional[int] = Field(default=None, description="Video height (overrides resolution)")
    resolution: str = Field(default="480p", description="Preset: 480p, 720p, or WxH")
    num_frames: int = Field(default=81, description="Number of frames (must be 1+4n)")
    fps: int = Field(default=16, description="Output FPS for animated webp")

    # Sampling
    steps: int = Field(default=30, description="Sampling steps")
    cfg: float = Field(default=7.5, description="CFG scale")
    shift: float = Field(default=1.0, description="Noise schedule shift (PNDM/scheduler)")
    seed: int = Field(default=-1, description="Seed (-1 for random)")
    scheduler: str = Field(default="unipc", description="Scheduler: unipc, uni, dpmpp_2m, euler")
    sampler_name: str = Field(default="uni_pc", description="Sampler name")

    # Model selection
    model: Optional[str] = Field(default=None, description="GGUF model filename (auto for task_type)")
    precision: str = Field(default="bf16", description="Compute precision: bf16, fp16, fp32")

    # I2V specific
    reference_image: Optional[str] = Field(default=None, description="Reference image path or URL for I2V")
    noise_aug_strength: float = Field(default=0.02, description="I2V noise augmentation strength")
    start_latent_strength: float = Field(default=1.0, description="I2V start latent strength")
    end_latent_strength: float = Field(default=1.0, description="I2V end latent strength")

    # T2V specific (optional overrides for model paths)
    t5_encoder: Optional[str] = Field(default=None, description="Override T5 encoder path")
    vae: Optional[str] = Field(default=None, description="Override VAE path")

    # Output
    output_filename: Optional[str] = Field(default=None, description="Custom output filename")


class GenerateResponse(BaseModel):
    task_id: str
    task_type: str
    status: str = "submitted"


class StatusResponse(BaseModel):
    task_id: str
    status: str  # queued, running, completed, failed
    progress: float = 0.0
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    gpu: str = "NVIDIA GeForce RTX 3090 24GB"
    models: List[str] = []
    comfyui: Optional[Dict[str, Any]] = None


# ─── Workflow templates ────────────────────────────────────────────────

def build_t2v_workflow(req: GenerateRequest) -> dict:
    """Build ComfyUI API-format workflow for Wan2.1 T2V (GGUF)."""
    res = RESOLUTION_MAP.get(req.resolution, {"width": 832, "height": 480})
    width = req.width or res["width"]
    height = req.height or res["height"]

    t5_path = req.t5_encoder or T5_ENCODER_PATH
    vae_path_ = req.vae or VAE_PATH
    model_path = req.model or T2V_MODEL_PATH
    seed = req.seed if req.seed >= 0 else int(hashlib.md5(req.prompt.encode()).hexdigest(), 16) % (2**32)

    return {
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "t5": t5_path,
                "dtype": req.precision,
                "device": "offload_device",
            }
        },
        "4": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model_name": model_path,
                "dtype": req.precision,
                "device": "main_device",
                "attention_mode": "sdpa",
            }
        },
        "5": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "vae_name": vae_path_,
                "dtype": req.precision,
            }
        },
        "6": {
            "class_type": "WanVideoEmptyEmbeds",
            "inputs": {
                "width": width,
                "height": height,
                "num_frames": req.num_frames,
            }
        },
        "7": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "t5": ["3", 0],
                "text_enc": ["4", 0],
                "clip": ["4", 1],
                "empty_embeds": ["6", 0],
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "force_offload": True,
            }
        },
        "8": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "text_enc": ["7", 0],
                "text_enc_neg": ["7", 1],
                "transformer": ["4", 2],
                "empty_embeds": ["6", 1],
                "latent_image": ["6", 2],
                "steps": req.steps,
                "cfg": req.cfg,
                "shift": req.shift,
                "seed": seed,
                "force_offload": True,
                "scheduler": req.scheduler,
                "sampler_name": req.sampler_name,
            }
        },
        "9": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["5", 0],
                "latent": ["8", 0],
                "enable_vae_tiling": True,
            }
        },
        "10": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": req.output_filename or "wan_t2v",
                "fps": req.fps,
                "lossless": False,
                "quality": 80,
            }
        },
    }


def build_i2v_workflow(req: GenerateRequest, image_path: str) -> dict:
    """Build ComfyUI API-format workflow for Wan2.1 I2V (GGUF)."""
    res = RESOLUTION_MAP.get(req.resolution, {"width": 832, "height": 480})
    width = req.width or res["width"]
    height = req.height or res["height"]

    t5_path = req.t5_encoder or T5_ENCODER_PATH
    vae_path_ = req.vae or VAE_PATH
    model_path = req.model or I2V_MODEL_PATH
    seed = req.seed if req.seed >= 0 else int(hashlib.md5((req.prompt + image_path).encode()).hexdigest(), 16) % (2**32)

    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_path,
            }
        },
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "t5": t5_path,
                "dtype": req.precision,
                "device": "offload_device",
            }
        },
        "4": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model_name": model_path,
                "dtype": req.precision,
                "device": "main_device",
                "attention_mode": "sdpa",
            }
        },
        "5": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "vae_name": vae_path_,
                "dtype": req.precision,
            }
        },
        "6": {
            "class_type": "WanVideoEmptyEmbeds",
            "inputs": {
                "width": width,
                "height": height,
                "num_frames": req.num_frames,
            }
        },
        "7": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "t5": ["3", 0],
                "text_enc": ["4", 0],
                "clip": ["4", 1],
                "empty_embeds": ["6", 0],
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "force_offload": True,
            }
        },
        "11": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "image": ["1", 0],
                "transformer": ["4", 2],
                "empty_embeds": ["6", 2],
                "vae": ["5", 0],
                "clip": ["4", 1],
                "noise_aug_strength": req.noise_aug_strength,
                "start_latent_strength": req.start_latent_strength,
                "end_latent_strength": req.end_latent_strength,
            }
        },
        "8": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "text_enc": ["7", 0],
                "text_enc_neg": ["7", 1],
                "transformer": ["4", 2],
                "empty_embeds": ["6", 1],
                "latent_image": ["11", 0],
                "steps": req.steps,
                "cfg": req.cfg,
                "shift": req.shift,
                "seed": seed,
                "force_offload": True,
                "scheduler": req.scheduler,
                "sampler_name": req.sampler_name,
            }
        },
        "9": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["5", 0],
                "latent": ["8", 0],
                "enable_vae_tiling": True,
            }
        },
        "10": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": req.output_filename or "wan_i2v",
                "fps": req.fps,
                "lossless": False,
                "quality": 80,
            }
        },
    }


# ─── Helpers ───────────────────────────────────────────────────────────

async def comfyui_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    """Make an async request to ComfyUI."""
    url = f"{COMFYUI_BASE}{path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        resp = await client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()


async def upload_image_to_comfyui(image_source: str) -> str:
    """Upload image (local path or URL) to ComfyUI input folder, return filename.

    For local paths, copy/symlink into ComfyUI's input directory.
    For URLs, download then upload.
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # Download
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_source)
            resp.raise_for_status()
            content = resp.content
        filename = os.path.basename(image_source).split("?")[0] or f"img_{uuid.uuid4().hex[:8]}.png"
        local_path = os.path.join(OUTPUT_DIR, filename)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
    else:
        local_path = image_source
        filename = os.path.basename(local_path)

    # Upload to ComfyUI via /upload/image
    async with httpx.AsyncClient(timeout=30.0) as client:
        with open(local_path, "rb") as f:
            resp = await client.post(
                f"{COMFYUI_BASE}/upload/image",
                files={"image": (filename, f)},
                data={"overwrite": "true"},
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("name", filename)


def resolve_task_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Parse ComfyUI history outputs into a clean response."""
    result: Dict[str, Any] = {}
    for node_id, node_out in outputs.items():
        for img in node_out.get("images", []):
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            url = (
                f"{COMFYUI_BASE}/view?"
                f"filename={filename}&subfolder={subfolder}&type={img_type}"
            )
            ext = os.path.splitext(filename)[1].lower() or ".webp"
            if ext in (".webp", ".gif", ".mp4", ".avi"):
                result["video"] = url
            else:
                result["image"] = url
    return result


# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — verify ComfyUI is reachable and GPU is available."""
    comfyui_status = None
    models = [T2V_MODEL_PATH, I2V_MODEL_PATH]

    try:
        stats = await comfyui_request("GET", "/system_stats")
        devices = stats.get("devices", [])
        vram_info = []
        for d in devices:
            name = d.get("name", "unknown")
            vram_total = d.get("vram_total", 0) // (1024 * 1024)
            vram_free = d.get("vram_free", 0) // (1024 * 1024)
            vram_info.append(f"{name}: {vram_free}MB free / {vram_total}MB")
        comfyui_status = {
            "status": "connected",
            "vram": vram_info,
            "system_stats": stats.get("system_stats", {}),
        }
    except Exception as e:
        comfyui_status = {"status": "disconnected", "error": str(e)}

    return HealthResponse(
        status="ok" if comfyui_status and comfyui_status.get("status") == "connected" else "degraded",
        gpu="NVIDIA GeForce RTX 3090 24GB",
        models=models,
        comfyui=comfyui_status,
    )


@app.get("/models")
async def list_models():
    """List available GGUF models."""
    return {
        "t2v": T2V_MODEL_PATH,
        "i2v": I2V_MODEL_PATH,
        "t5_encoder": T5_ENCODER_PATH,
        "vae": VAE_PATH,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Submit a T2V or I2V generation task to ComfyUI."""
    # Validate task type
    if req.task_type not in ("t2v", "i2v"):
        raise HTTPException(400, f"Invalid task_type '{req.task_type}', must be 't2v' or 'i2v'")

    # I2V requires reference image
    if req.task_type == "i2v" and not req.reference_image:
        raise HTTPException(400, "I2V requires 'reference_image' (path or URL)")

    # Validate num_frames (must be 1 + 4n for Wan2.1)
    if (req.num_frames - 1) % 4 != 0:
        raise HTTPException(
            400,
            f"num_frames must be 1+4n (got {req.num_frames}); valid: 5, 9, 17, 21, 33, 41, 49, 81, 161"
        )

    # Validate resolution
    res = RESOLUTION_MAP.get(req.resolution)
    if res is None:
        # Try parsing as WxH
        try:
            w, h = req.resolution.lower().split("x")
            int(w), int(h)
        except ValueError:
            raise HTTPException(400, f"Invalid resolution '{req.resolution}', use 480p, 720p, or WxH")

    # Build workflow
    if req.task_type == "t2v":
        workflow = build_t2v_workflow(req)
    else:
        # Upload reference image to ComfyUI
        try:
            img_name = await upload_image_to_comfyui(req.reference_image)
        except Exception as e:
            raise HTTPException(400, f"Failed to load reference image: {e}")
        workflow = build_i2v_workflow(req, img_name)

    # Submit to ComfyUI
    client_id = str(uuid.uuid4())
    try:
        result = await comfyui_request("POST", "/prompt", json={
            "prompt": workflow,
            "client_id": client_id,
        })
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        logger.error("ComfyUI submit failed: %s", detail)
        raise HTTPException(502, f"ComfyUI rejected workflow: {detail}")
    except Exception as e:
        raise HTTPException(502, f"Cannot reach ComfyUI: {e}")

    task_id = result.get("prompt_id")
    if not task_id:
        raise HTTPException(502, f"ComfyUI returned no prompt_id: {result}")

    logger.info("Submitted %s task %s (client_id=%s)", req.task_type, task_id, client_id)
    return GenerateResponse(task_id=task_id, task_type=req.task_type, status="submitted")


@app.get("/status/{task_id}", response_model=StatusResponse)
async def status(task_id: str):
    """Poll ComfyUI task status."""
    try:
        history = await comfyui_request("GET", f"/history/{task_id}")
    except Exception as e:
        raise HTTPException(502, f"Cannot reach ComfyUI: {e}")

    if task_id not in history:
        return StatusResponse(task_id=task_id, status="queued", progress=0.0)

    item = history[task_id]

    # Check for errors
    status_info = item.get("status", {})
    if status_info.get("status_str") == "error":
        msgs = status_info.get("messages", [])
        error_detail = json.dumps(msgs) if msgs else "ComfyUI execution error"
        return StatusResponse(
            task_id=task_id,
            status="failed",
            error=error_detail,
        )

    # Check outputs
    outputs = item.get("outputs", {})
    if outputs:
        parsed = resolve_task_outputs(outputs)
        return StatusResponse(
            task_id=task_id,
            status="completed",
            progress=100.0,
            outputs=parsed,
        )

    # Still running
    return StatusResponse(task_id=task_id, status="running", progress=50.0)


@app.post("/cancel")
async def cancel():
    """Cancel the currently running ComfyUI task."""
    try:
        await comfyui_request("POST", "/interrupt")
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(502, f"Cancel failed: {e}")


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Starting Wan2.1 GGUF API on %s:%d", LISTEN_HOST, LISTEN_PORT)
    logger.info("ComfyUI target: %s:%d", COMFYUI_HOST, COMFYUI_PORT)
    logger.info("T2V model: %s", T2V_MODEL_PATH)
    logger.info("I2V model: %s", I2V_MODEL_PATH)
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
