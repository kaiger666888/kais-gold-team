"""MiniMax H3 text-to-video / image-to-video workflow builder.

Generates ComfyUI API-format workflows for MiniMax H3 video generation,
a Flow Matching audiovisual model distilled to run on a single RTX 3090
via int8 (UNet) + NVFP4 (text encoder) quantization.

Two modes share one builder:
  * **T2V** — pure text → video (no frame conditioning).
  * **I2V** — first/last frame conditioning. Pass ``first_frame`` (and
    optionally ``last_frame``) as image filenames in ComfyUI's input
    directory; the corresponding ``LoadImage`` nodes and node-20 inputs
    are wired only when a frame is supplied.

The node graph below mirrors the ComfyUI graph validated locally on the
3090 (see the H3 integration brief). Node IDs are string numbers, as the
ComfyUI API format requires.
"""
from __future__ import annotations

import random
from typing import Any

logger = __import__("logging").getLogger(__name__)

# ─── Model files (inside ComfyUI container, /data/models/comfyui/ volume) ───

UNET_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"      # ~20GB int8
UNET_WEIGHT_DTYPE = "default"

CLIP_MODEL = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"          # ~15GB NVFP4
CLIP_TYPE = "minimax"

VAE_MODEL = "minimax_h3_video_vae_fp16.safetensors"                  # ~4.9GB fp16

# ─── Flow Matching sampler defaults ───
#
# H3 is a CFG-distilled Flow Matching model: cfg MUST stay 1.0 and the
# video sigma-shift MUST stay 12.0. The defaults below are the verified
# "fast preview" profile (15 steps); 36 steps is the high-quality profile,
# 50 steps is the official recommendation.

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_STEPS = 15
DEFAULT_CFG = 1.0
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

# Output geometry / duration
DEFAULT_WIDTH = 1344      # 16:9 landscape
DEFAULT_HEIGHT = 768
DEFAULT_LENGTH = 124      # frames; 124 = 17*7+5 ≈ 5.17s @ 24fps
DEFAULT_FPS = 24

# Frame counts must satisfy the model's 17k+5 alignment:
#   5, 22, 39, 56, 73, 90, 107, 124, ...
ALIGNED_LENGTHS = tuple(17 * k + 5 for k in range(0, 16))

# ─── TESpeed 残差缓存加速 (ComfyUI-TE-Speed-MiniMaxH3-OSS) ───
# 在 SigmaShift(14) 与 KSampler(30) 之间注入 block-cache 节点,
# 50 步实测 20m40s → 12m01s (-42%)。需要容器内已执行 patch_model.py
# (注入 ("block_loop", 0) 钩子); 未执行时节点静默返回未 patch 的 model。
TESPEED_ENABLED = True


def _resolve_seed(seed: int | None) -> int:
    """Return a concrete seed, drawing one at random when ``seed`` is None."""
    if seed is None:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def build_minimax_h3_workflow(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    steps: int = DEFAULT_STEPS,
    shift_video: float = DEFAULT_SHIFT_VIDEO,
    shift_audio: float = DEFAULT_SHIFT_AUDIO,
    cfg: float = DEFAULT_CFG,
    seed: int | None = None,
    fps: float = DEFAULT_FPS,
    first_frame: str | None = None,
    last_frame: str | None = None,
    filename_prefix: str = "minimax_h3",
) -> dict[str, Any]:
    """Build a ComfyUI API-format workflow for MiniMax H3 T2V/I2V.

    Args:
        prompt: Text prompt describing the video to generate.
        width/height: Output frame dimensions.
        length: Number of generated frames. Must satisfy the model's
            ``17k+5`` alignment (5, 22, 39, 56, 73, 90, 107, 124, ...).
        steps: KSampler denoising steps (15 fast / 36 HQ / 50 official).
        shift_video: Sigma shift for the video stream (Flow Matching).
            Must remain 12.0 for H3.
        shift_audio: Sigma shift for the audio stream (default 3.0).
        cfg: CFG scale. H3 is CFG-distilled, so this must remain 1.0.
        seed: Random seed. ``None`` draws one at random.
        fps: Frames per second for the output video (CreateVideo node).
        first_frame: Optional first-frame image filename (ComfyUI input
            dir) for I2V conditioning. Enables I2V mode when provided.
        last_frame: Optional last-frame image filename for I2V
            conditioning. Only wired when ``first_frame`` is also given.
        filename_prefix: Output filename prefix for the SaveVideo node.

    Returns:
        ComfyUI API-format workflow dict (ready to wrap in
        ``{prompt: ...}`` and POST to ``/prompt``).

    Node graph (T2V):

        10 UNETLoader ─────┐
        11 CLIPLoader ─────┤
        12 VAELoader ──────┤
            │              │
        14 MiniMaxH3SigmaShift ◀── model from 10
            │              │
        20 MiniMaxH3ImageToVideo ◀── clip 11, vae 12, prompt
            │ (outputs positive cond [20,0] + latent [20,1])
        21 CLIPTextEncode("") ── negative cond (clip 11)
            │
        30 KSampler ◀── model 14, positive 20, negative 21, latent 20
            │
        40 VAEDecode ◀── samples 30, vae 12
            │
        45 CreateVideo ◀── images 40, fps
            │
        50 SaveVideo ◀── video 45

    I2V additionally wires LoadImage nodes 15/16 into node 20's
    ``first_frame`` / ``last_frame`` inputs.
    """
    actual_seed = _resolve_seed(seed)

    if length not in ALIGNED_LENGTHS:
        logger.warning(
            "MiniMax H3 length=%d is not 17k+5-aligned; the model may "
            "reject it. Valid: %s",
            length, ", ".join(str(n) for n in ALIGNED_LENGTHS[:10]) + ", ...",
        )

    workflow: dict[str, Any] = {
        # ── Model loaders ──
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_MODEL,
                "weight_dtype": UNET_WEIGHT_DTYPE,
            },
        },
        "11": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_MODEL,
                "type": CLIP_TYPE,
            },
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_MODEL},
        },

        # ── Sigma shift for Flow Matching (video + audio streams) ──
        "14": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["10", 0],
                "shift_video": shift_video,
                "shift_audio": shift_audio,
            },
        },
    }

    # ── Optional I2V frame conditioning ──
    # LoadImage nodes are only emitted when a frame is supplied; node 20's
    # first_frame/last_frame inputs are wired accordingly.
    i2v_inputs: dict[str, Any] = {}
    if first_frame:
        workflow["15"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_frame},
        }
        i2v_inputs["first_frame"] = ["15", 0]
        if last_frame:
            workflow["16"] = {
                "class_type": "LoadImage",
                "inputs": {"image": last_frame},
            }
            i2v_inputs["last_frame"] = ["16", 0]

    # ── Conditioning + latent ──
    # MiniMaxH3ImageToVideo emits both the positive conditioning (output 0)
    # and the latent image (output 1), which KSampler consumes directly.
    workflow["20"] = {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["11", 0],
            "vae": ["12", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            **i2v_inputs,
        },
    }
    # Empty negative prompt — CFG-distilled (cfg=1.0), but KSampler still
    # requires a negative conditioning input.
    workflow["21"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",
            "clip": ["11", 0],
        },
    }

    # ── Sampling ──
    # TESpeed 加速: SigmaShift(14) → TESpeed(35) → KSampler(30)
    # (ComfyUI-TE-Speed-MiniMaxH3-OSS, 需要 patch_model.py 已执行)
    if TESPEED_ENABLED:
        workflow["35"] = {
            "class_type": "TESpeedMiniMaxH3",
            "inputs": {
                "model": ["14", 0],
                "processing_control_value": 0.12,
                "processing_percent_1": 0.1,
                "processing_percent_2": 0.9,
                "mcs": 2,
                "device": "auto",
                "cache_depth": 0.75,
            },
        }
    workflow["30"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["35", 0] if TESPEED_ENABLED else ["14", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["20", 1],
            "seed": actual_seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": DEFAULT_SAMPLER,
            "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0,
        },
    }

    # ── Decode → video → save ──
    workflow["40"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["30", 0],
            "vae": ["12", 0],
        },
    }
    workflow["45"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["40", 0],
            "fps": fps,
        },
    }
    workflow["50"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["45", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }

    return workflow
