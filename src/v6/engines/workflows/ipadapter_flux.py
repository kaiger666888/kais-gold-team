"""IP-Adapter + Flux text-to-image workflow builder.

Generates ComfyUI API-format workflows for character/style-consistent
image generation using Flux Dev FP8 + IP-Adapter.

When no reference image is provided, falls back to pure Flux txt2img.
"""

from __future__ import annotations

from typing import Any

logger = __import__("logging").getLogger(__name__)

# ─── Model constants ───

UNET_MODEL = "flux1-dev-fp8.safetensors"
UNET_WEIGHT_DTYPE = "default"

CLIP_T5 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_L = "clip_l.safetensors"
CLIP_TYPE = "flux"

VAE_MODEL = "flux1-ae.safetensors"

# IP-Adapter models (inside ComfyUI container)
IPADAPTER_MODEL = "instantx_flux1_dev_ip_adapter_fp16.safetensors"
CLIP_VISION_MODEL = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# ─── Sampler defaults ───

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 3.5


def build_flux_txt2img_workflow(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
) -> dict[str, Any]:
    """Pure Flux Dev FP8 text-to-image (no IP-Adapter).

    Use this for the first/anchor image that will serve as reference
    for subsequent IP-Adapter generations.

    Node graph:
        UNETLoader → DualCLIPLoader → VAELoader → CLIPTextEncodeFlux
        → EmptyFlux2LatentImage → KSampler → VAEDecode → SaveImage
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_MODEL,
                "weight_dtype": UNET_WEIGHT_DTYPE,
            },
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": CLIP_T5,
                "clip_name2": CLIP_L,
                "type": CLIP_TYPE,
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_MODEL},
        },
        "4": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
        },
        "5": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": guidance,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "latent_image": ["5", 0],
                "denoise": 1.0,
                "positive": ["4", 0],
                "negative": ["4", 0],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
            },
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "ipadapter_scene",
            },
        },
    }


def build_ipadapter_txt2img_workflow(
    prompt: str,
    reference_image: dict[str, str],
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    weight: float = 0.8,
    weight_start: float = 0.0,
    weight_end: float = 1.0,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    filename_prefix: str = "ipadapter_scene",
) -> dict[str, Any]:
    """IP-Adapter + Flux text-to-image with character/style consistency.

    Uses a reference image to maintain character identity across scenes.

    Args:
        prompt: Text prompt describing the target scene.
        reference_image: ComfyUI image reference dict with keys:
            - ``filename``: image filename in ComfyUI input directory
            - ``subfolder``: optional subfolder (default empty)
            - ``type``: input type (default "input")
        width/height: Output dimensions.
        seed: Random seed.
        steps: Denoising steps.
        guidance: CFG guidance scale.
        weight: IP-Adapter influence weight (0.0-1.0).
            Higher = more faithful to reference image.
        weight_start/end: IP-Adapter influence schedule (0.0-1.0).
        sampler_name: KSampler name.
        scheduler: KSampler scheduler.
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict (ready to wrap in ``{prompt: ...}``).

    Node graph:
        UNETLoader ──┐
        DualCLIPLoader ──┤
        VAELoader ──┤
        IPAdapterUnifiedLoader ──┤
        LoadImage ──┤
            │
        CLIPTextEncodeFlux (prompt) ──┤
        IPAdapter ──┤
            │
        EmptyFlux2LatentImage
            │
        KSampler
            │
        VAEDecode
            │
        SaveImage
    """
    # Build image ref dict with defaults
    img_ref = {
        "filename": reference_image["filename"],
        "subfolder": reference_image.get("subfolder", ""),
        "type": reference_image.get("type", "input"),
    }

    workflow: dict[str, Any] = {
        # 1. Load models
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_MODEL,
                "weight_dtype": UNET_WEIGHT_DTYPE,
            },
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": CLIP_T5,
                "clip_name2": CLIP_L,
                "type": CLIP_TYPE,
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_MODEL},
        },
        "4": {
            "class_type": "IPAdapterUnifiedLoader",
            "inputs": {
                "model_name": IPADAPTER_MODEL,
                "preset": "FLUX.1-dev (fp8)",
                "lora_strength": 1.0,
                "provider": "CPU",
            },
        },
        "5": {
            "class_type": "IPAdapterInsightFaceLoader",
            "inputs": {
                "provider": "CPU",
            },
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": img_ref,
        },

        # 2. Text encoding
        "7": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": ["2", 0],
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
        },

        # 3. IP-Adapter conditioning
        "8": {
            "class_type": "IPAdapterApply",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["4", 0],
                "clip": ["2", 0],
                "image": ["6", 0],
                "weight": weight,
                "weight_type": "linear",
                "start_at": weight_start,
                "end_at": weight_end,
                "positive": ["7", 0],
                "negative": ["7", 0],
            },
        },

        # 4. Latent + Sampling
        "9": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": steps,
                "cfg": guidance,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "latent_image": ["9", 0],
                "denoise": 1.0,
                "positive": ["8", 0],
                "negative": ["8", 0],
            },
        },

        # 5. Decode + Save
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0],
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["11", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }

    return workflow
