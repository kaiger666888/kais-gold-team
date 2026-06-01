"""Workflow Builder — converts task params to engine-specific formats.

Supports:
  - ComfyUI txt2img workflows (via build_txt2img_workflow)
  - ComfyUI FLUX Dev FP8 workflows (via build_flux_dev_workflow)
  - TTS workflows (via build_tts_workflow) — subprocess-based, not ComfyUI
"""
from __future__ import annotations

import os
from typing import Any


def build_flux_dev_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg_scale: float = 3.5,
    seed: int | None = None,
    filename_prefix: str = "flux-dev",
    unet_name: str = "flux1-dev-fp8.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    clip_name1: str = "clip_l/model.safetensors",
    clip_name2: str = "t5xxl_fp16/model-00001-of-00002.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux_vae/diffusion_pytorch_model.safetensors",
) -> dict[str, Any]:
    """Build a FLUX Dev FP8 ComfyUI workflow via separate loaders.

    Uses UNETLoader + DualCLIPLoader + VAELoader + KSampler + VAEDecode + SaveImage.
    This avoids CheckpointLoaderSimple (which requires a single-file checkpoint)
    and instead loads FLUX components individually for better memory control.

    Args:
        prompt: Text prompt for image generation.
        negative_prompt: Negative prompt (FLUX doesn't use negative, but kept for API compat).
        width: Image width (default 1024, must be multiple of 16).
        height: Image height (default 1024, must be multiple of 16).
        steps: Number of inference steps (default 28 for Dev).
        cfg_scale: Guidance scale (default 3.5 for FLUX Dev).
        seed: Random seed. Random if None.
        filename_prefix: Output filename prefix.
        unet_name: UNET model filename in ComfyUI unet/ directory.
        weight_dtype: UNET weight dtype (fp8_e4m3fn or default).
        clip_name1: CLIP-L model filename (text_encoder_1).
        clip_name2: T5-XXL model filename (text_encoder_2).
        clip_type: DualCLIPLoader type ("flux").
        vae_name: VAE model filename.

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "1": {  # UNETLoader
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": weight_dtype,
            },
        },
        "2": {  # DualCLIPLoader
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_name1,
                "clip_name2": clip_name2,
                "type": clip_type,
            },
        },
        "3": {  # VAELoader
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "4": {  # CLIPTextEncode
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0],
            },
        },
        "5": {  # EmptySD3LatentImage (FLUX uses SD3 latent format)
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {  # KSampler
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],  # FLUX ignores negative prompt
                "latent_image": ["5", 0],
            },
        },
        "7": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
            },
        },
        "8": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["7", 0],
            },
        },
    }
    return workflow


def build_txt2img_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg_scale: float = 7.5,
    seed: int | None = None,
    checkpoint: str = "sd_xl_turbo_1.0_fp16.safetensors",
) -> dict[str, Any]:
    """Build a basic txt2img ComfyUI workflow.

    Uses the standard KSampler + CheckpointLoader + CLIPTextEncode + VAEDecode + SaveImage pipeline.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "3": {  # KSampler
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {  # CheckpointLoaderSimple
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint,
            },
        },
        "5": {  # EmptyLatentImage
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {  # CLIPTextEncode (positive)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],
            },
        },
        "7": {  # CLIPTextEncode (negative)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1],
            },
        },
        "8": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "kais-render",
                "images": ["8", 0],
            },
        },
    }
    return workflow


def build_tts_workflow(
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    backend: str = "auto",
    output_path: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    """Build a TTS workflow dict for the TTSEngine.

    Unlike ComfyUI workflows, this returns a parameter dict consumed by
    TTSEngine.submit() which invokes scripts/tts_infer.py via subprocess.

    Args:
        text: Text to synthesize.
        voice: Voice name — 'default', '中文女', '中文男', 'english_female',
               'english_male', or a full edge-tts voice ID.
        speed: Speech speed multiplier (1.0 = normal).
        backend: 'auto' (try CosyVoice → edge-tts), 'cosyvoice', or 'edge-tts'.
        output_path: Explicit output file path. Auto-generated if empty.
        task_id: Used for auto-generating output path.

    Returns:
        Dict with TTS parameters for TTSEngine.submit().
    """
    if not output_path:
        output_root = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")
        tid = task_id or "tts-unknown"
        output_path = os.path.join(output_root, tid, "voice.wav")

    return {
        "text": text,
        "voice": voice,
        "speed": speed,
        "backend": backend,
        "output_path": output_path,
    }
