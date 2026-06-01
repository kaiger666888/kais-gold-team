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


def build_flux_ipadapter_workflow(
    prompt: str,
    reference_image: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg_scale: float = 3.5,
    weight: float = 0.8,
    start_percent: float = 0.0,
    end_percent: float = 0.8,
    seed: int | None = None,
    filename_prefix: str = "flux-ipadapter",
    unet_name: str = "flux1-dev-fp8.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    clip_name1: str = "clip_l/model.safetensors",
    clip_name2: str = "t5xxl_fp16/model-00001-of-00002.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux_vae/diffusion_pytorch_model.safetensors",
    ipadapter_name: str = "ip-adapter.bin",
    clip_vision_name: str = "google/siglip-so400m-patch14-384",
) -> dict[str, Any]:
    """Build a FLUX Dev FP8 + IP-Adapter ComfyUI workflow.

    Loads a reference image via IP-Adapter to inject style/composition into generation.
    Uses the same FLUX Dev FP8 components as build_flux_dev_workflow, plus:
    - IPAdapterFluxLoader (IP-Adapter weights + SIGLIP vision encoder)
    - LoadImage (reference image)
    - ApplyIPAdapterFlux (applies IP-Adapter to UNET model)

    Args:
        prompt: Text prompt for image generation.
        reference_image: Filename of reference image in ComfyUI input/ directory.
        negative_prompt: Negative prompt (FLUX ignores, kept for API compat).
        width: Image width (default 1024).
        height: Image height (default 1024).
        steps: Number of inference steps (default 28 for Dev).
        cfg_scale: Guidance scale (default 3.5 for FLUX Dev).
        weight: IP-Adapter influence weight (0.0-1.0, default 0.8).
        start_percent: When IP-Adapter starts applying (0.0-1.0).
        end_percent: When IP-Adapter stops applying (0.0-1.0).
        seed: Random seed. Random if None.
        filename_prefix: Output filename prefix.
        unet_name: UNET model filename.
        weight_dtype: UNET weight dtype.
        clip_name1: CLIP-L model filename.
        clip_name2: T5-XXL model filename.
        clip_type: DualCLIPLoader type.
        vae_name: VAE model filename.
        ipadapter_name: IP-Adapter weights filename.
        clip_vision_name: SIGLIP vision encoder name.

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
        "5": {  # EmptySD3LatentImage
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {  # KSampler (uses IP-Adapter-modified model)
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["12", 0],  # ApplyIPAdapterFlux output
                "positive": ["4", 0],
                "negative": ["4", 0],
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
        "10": {  # IPAdapterFluxLoader
            "class_type": "IPAdapterFluxLoader",
            "inputs": {
                "ipadapter": ipadapter_name,
                "clip_vision": clip_vision_name,
                "provider": "cuda",
            },
        },
        "11": {  # LoadImage
            "class_type": "LoadImage",
            "inputs": {
                "image": reference_image,
            },
        },
        "12": {  # ApplyIPAdapterFlux
            "class_type": "ApplyIPAdapterFlux",
            "inputs": {
                "model": ["1", 0],
                "ipadapter_flux": ["10", 0],
                "image": ["11", 0],
                "weight": weight,
                "start_percent": start_percent,
                "end_percent": end_percent,
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


def build_trellis_image_to_3d_workflow(
    image_name: str,
    resolution: int = 1024,
    steps: int = 25,
    cfg_scale: float = 7.5,
    shape_guidance: float = 0.5,
    texture_resolution: int = 1024,
    pbr_channels: str = "full",
    remove_background: bool = True,
    output_format: str = "glb",
    seed: int | None = None,
    filename_prefix: str = "trellis_3d",
) -> dict[str, Any]:
    """Build a TRELLIS 2 image-to-3D ComfyUI workflow.

    Preprocessing chain: LoadImage -> Resize -> RMBG-2.0 -> TRELLIS ImageTo3D -> TextureBake -> Export

    Args:
        image_name: Input image filename (must exist in ComfyUI input/ directory).
        resolution: 3D voxel resolution (512, 1024, or 1536).
        steps: TRELLIS denoising steps.
        cfg_scale: CFG scale for TRELLIS.
        shape_guidance: Shape guidance strength.
        texture_resolution: Texture bake resolution.
        pbr_channels: "color" (color only) or "full" (color + normal + roughness + metallic).
        remove_background: Whether to use RMBG-2.0 for background removal.
        output_format: Output format (glb, ply, fbx).
        seed: Random seed. Random if None.
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    class_type = "TRELLISImageTo3D"
    nodes: dict[str, Any] = {}

    # Node 1: Load Image
    nodes["1"] = {
        "class_type": "LoadImage",
        "inputs": {"image": image_name},
    }

    # Node 2: Image Resize
    nodes["2"] = {
        "class_type": "ImageResize",
        "inputs": {
            "image": ["1", 0],
            "width": resolution,
            "height": resolution,
            "method": "lanczos",
            "crop": "center",
        },
    }

    # Node 3: RMBG-2.0 Background Removal (conditional)
    if remove_background:
        nodes["3"] = {
            "class_type": "RemoveBG",
            "inputs": {"image": ["2", 0]},
        }
        trellis_image_input = ["3", 0]
    else:
        trellis_image_input = ["2", 0]

    # Node 4: TRELLIS Image to 3D
    nodes["4"] = {
        "class_type": class_type,
        "inputs": {
            "image": trellis_image_input,
            "model": "microsoft/TRELLIS.2-4B",
            "resolution": resolution,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "shape_guidance": shape_guidance,
            "texture_resolution": texture_resolution,
            "pbr_channels": pbr_channels,
            "low_vram": False,
            "seed": seed,
            "fp16": True,
        },
    }

    # Node 5: Texture Bake (for full PBR)
    if pbr_channels == "full":
        nodes["5"] = {
            "class_type": "TRELLISTextureBake",
            "inputs": {
                "mesh": ["4", 0],
                "texture_image": ["4", 1],
                "texture_resolution": texture_resolution,
                "pbr_channels": "full",
            },
        }
        export_mesh_input = ["5", 0]
    else:
        export_mesh_input = ["4", 0]

    # Node 6: Export GLB
    nodes["6"] = {
        "class_type": "TRELLISExport",
        "inputs": {
            "mesh": export_mesh_input,
            "format": output_format,
            "embed_textures": True,
        },
    }

    # Node 7: Save output
    nodes["7"] = {
        "class_type": "Save",
        "inputs": {
            "data": ["6", 0],
            "filename_prefix": filename_prefix,
        },
    }

    return nodes


def build_flux_trellis_full_workflow(
    prompt: str,
    negative_prompt: str = "",
    flux_steps: int = 20,
    flux_cfg: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    trellis_resolution: int = 1024,
    trellis_steps: int = 25,
    trellis_cfg: float = 7.5,
    shape_guidance: float = 0.5,
    texture_resolution: int = 1024,
    pbr_channels: str = "full",
    output_format: str = "glb",
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a FLUX -> TRELLIS full pipeline ComfyUI workflow.

    Serial execution: FLUX generates image, releases VRAM, then TRELLIS converts to 3D.
    Total estimated time on RTX 3090: ~30-40s (FLUX 10-15s + TRELLIS 15-25s).

    Args:
        prompt: Text prompt for FLUX image generation.
        negative_prompt: Negative prompt.
        flux_steps: FLUX inference steps (default 20).
        flux_cfg: FLUX guidance scale (default 3.5).
        width/height: Image dimensions.
        trellis_resolution: TRELLIS 3D resolution (512/1024).
        trellis_steps: TRELLIS denoising steps (default 25).
        trellis_cfg: TRELLIS CFG scale.
        shape_guidance: TRELLIS shape guidance.
        texture_resolution: Texture resolution.
        pbr_channels: PBR channels (color/full).
        output_format: 3D output format.
        seed: Random seed.

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    flux_seed = seed
    trellis_seed = seed + 1 if seed > 0 else random.randint(0, 2**32 - 1)

    nodes: dict[str, Any] = {}

    # ── Stage A: FLUX Text-to-Image ──
    nodes["10"] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "flux1-dev-fp8.safetensors",
            "weight_dtype": "fp8_e4m3fn",
        },
    }
    nodes["11"] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "clip_l/model.safetensors",
            "clip_name2": "t5xxl_fp16/model-00001-of-00002.safetensors",
            "type": "flux",
        },
    }
    nodes["12"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": ["11", 0],
        },
    }
    nodes["13"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    }
    nodes["14"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["10", 0],
            "positive": ["12", 0],
            "negative": ["12", 0],
            "latent_image": ["13", 0],
            "seed": flux_seed,
            "steps": flux_steps,
            "cfg": flux_cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    nodes["15"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "flux_vae/diffusion_pytorch_model.safetensors"},
    }
    nodes["16"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["14", 0],
            "vae": ["15", 0],
        },
    }
    nodes["17"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["16", 0],
            "filename_prefix": "flux_trellis_input",
        },
    }

    # ── Stage B: TRELLIS Image-to-3D (serial after VRAM release) ──
    nodes["20"] = {
        "class_type": "ImageResize",
        "inputs": {
            "image": ["16", 0],
            "width": trellis_resolution,
            "height": trellis_resolution,
            "method": "lanczos",
            "crop": "center",
        },
    }
    nodes["21"] = {
        "class_type": "RemoveBG",
        "inputs": {"image": ["20", 0]},
    }
    nodes["22"] = {
        "class_type": "TRELLISImageTo3D",
        "inputs": {
            "image": ["21", 0],
            "model": "microsoft/TRELLIS.2-4B",
            "resolution": trellis_resolution,
            "steps": trellis_steps,
            "cfg_scale": trellis_cfg,
            "shape_guidance": shape_guidance,
            "texture_resolution": texture_resolution,
            "pbr_channels": pbr_channels,
            "low_vram": False,
            "seed": trellis_seed,
            "fp16": True,
        },
    }
    nodes["23"] = {
        "class_type": "TRELLISExport",
        "inputs": {
            "mesh": ["22", 0],
            "format": output_format,
            "embed_textures": True,
        },
    }
    nodes["24"] = {
        "class_type": "Save",
        "inputs": {
            "data": ["23", 0],
            "filename_prefix": "flux_trellis_3d",
        },
    }

    return nodes
