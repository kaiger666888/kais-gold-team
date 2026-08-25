"""Workflow Builder — converts task params to engine-specific formats.

Supports:
  - ComfyUI txt2img workflows (via build_txt2img_workflow)
  - ComfyUI FLUX Dev FP8 workflows (via build_flux_dev_workflow)
  - ComfyUI FLUX IP-Adapter workflows (via build_flux_ipadapter_workflow)
  - ComfyUI PuLID FLUX workflows (via build_pulid_flux_workflow)
  - ComfyUI ControlNet Depth workflows (via build_controlnet_depth_workflow)
  - ComfyUI Wan 2.1 I2V dual-stage workflows (via build_wan21_i2v_dual_stage_workflow)
  - ComfyUI Upscale workflows (via build_upscale_workflow)
  - ComfyUI Face Restore workflows (via build_face_restore_workflow)
  - TRELLIS image-to-3D workflows (via build_trellis_image_to_3d_workflow)
  - FLUX + TRELLIS full pipeline (via build_flux_trellis_full_workflow)
  - ComfyUI Lip Sync workflows (via build_lipsync_workflow) — LatentSync
  - ComfyUI Frame Interpolation workflows (via build_frame_interpolate_workflow) — RIFE VFI
  - TTS workflows (via build_tts_workflow) — subprocess-based, not ComfyUI
  - Hunyuan3D workflows (via build_hunyuan3d_workflow) — subprocess-based
  - Shot Analysis workflows (via build_shot_analysis_workflow) — per-shot cinematic
    deconstruction: ShotGeometryLK + AILab_QwenVL_Advanced (semantic) +
    SAM3+SubjectMotionResidual (subject), merged via ShotJSONMerge
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
    clip_name1: str = "clip_l.safetensors",
    clip_name2: str = "t5xxl_fp8_e4m3fn_scaled.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux1-ae.safetensors",
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
    clip_name1: str = "clip_l.safetensors",
    clip_name2: str = "t5xxl_fp8_e4m3fn_scaled.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux1-ae.safetensors",
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
    language: str = "auto",
    reference_audio: str = "",
) -> dict[str, Any]:
    """Build a TTS workflow dict for the TTSTracker.

    Returns a parameter dict consumed by TTSTracker.submit().

    Args:
        text: Text to synthesize.
        voice: Voice name.
        speed: Speech speed multiplier.
        backend: 'auto', 'gpt_sovits', 'chatterbox', 'cosyvoice'.
        language: 'zh', 'en', 'auto', 'bilingual'.
        output_path: Explicit output file path.
        task_id: Used for auto-generating output path.
        reference_audio: Optional reference audio for voice cloning.

    Returns:
        Dict with TTS parameters.
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
        "language": language,
        "reference_audio": reference_audio,
        "track": backend if backend in ("zh", "en", "bilingual", "gpt_sovits", "chatterbox", "cosyvoice") else "",
    }


def build_hunyuan3d_workflow(
    input_image: str,
    output_path: str = "",
    model: str = "full",
    device: str = "cuda:0",
    steps: int = 50,
    seed: int | None = None,
    model_dir: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    """Build a Hunyuan3D-2 parameter dict for Hunyuan3DEngine.submit().

    Like ``build_tts_workflow``, this is not a ComfyUI graph — it returns
    plain parameters consumed by ``Hunyuan3DEngine`` which invokes
    ``scripts/hunyuan3d_infer.py`` via subprocess.

    Args:
        input_image: Absolute path to source image (PNG/JPG).
        output_path: Output GLB path. Auto-generated under KAIS_OUTPUT_ROOT if empty.
        model: "mini" or "full" (default full = Hunyuan3D-2.1).
        device: Torch device (cuda:0). Remapped to CUDA_VISIBLE_DEVICES in script.
        steps: Inference steps (default 50; ~75s on RTX 3090 for full model).
        seed: Reproducibility seed. Pipeline default if None.
        model_dir: Override model checkpoint directory.
        task_id: Used for default output path naming.

    Returns:
        Dict consumed by Hunyuan3DEngine.submit().
    """
    if not output_path:
        output_root = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")
        tid = task_id or "hunyuan3d-unknown"
        output_path = os.path.join(output_root, tid, "model.glb")

    workflow: dict[str, Any] = {
        "input_image": input_image,
        "output_path": output_path,
        "model": model,
        "device": device,
        "steps": steps,
    }
    if seed is not None:
        workflow["seed"] = seed
    if model_dir:
        workflow["model_dir"] = model_dir
    return workflow


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
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
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
        "inputs": {"vae_name": "flux1-ae.safetensors"},
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


def build_pulid_flux_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg_scale: float = 3.5,
    weight: float = 1.0,
    seed: int | None = None,
    filename_prefix: str = "pulid_flux",
    unet_name: str = "flux1-dev-fp8.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    clip_name1: str = "clip_l.safetensors",
    clip_name2: str = "t5xxl_fp8_e4m3fn_scaled.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux1-ae.safetensors",
    pulid_model: str = "ipadapter_pulid_flux_v0.9.0.safetensors",
) -> dict[str, Any]:
    """Build a PuLID FLUX character-consistency injection ComfyUI workflow.

    Uses PuLID_ComfyUI nodes: PulidModelLoader → PulidEvaClipLoader →
    PulidInsightFaceLoader → ApplyPulid.  ApplyPulid requires eva_clip
    (not CLIPVisionLoader) and face_analysis from PulidInsightFaceLoader.

    Args:
        image_name: Reference image filename in ComfyUI input/ directory.
        prompt: Text prompt for image generation.
        negative_prompt: Negative prompt (FLUX ignores, kept for API compat).
        width: Image width.
        height: Image height.
        steps: Inference steps (default 28 for FLUX Dev).
        cfg_scale: Guidance scale (default 3.5 for FLUX).
        weight: PuLID influence weight (0.0-1.0, default 1.0).
        seed: Random seed.
        filename_prefix: Output filename prefix.
        unet_name: UNET model filename.
        weight_dtype: UNET weight dtype.
        clip_name1: CLIP-L model filename.
        clip_name2: T5-XXL model filename.
        clip_type: DualCLIPLoader type ("flux").
        vae_name: VAE model filename.
        pulid_model: PuLID adapter weights filename (loras/ directory).

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow: dict[str, Any] = {
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
        "6": {  # LoadImage (reference portrait/character)
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "7": {  # PulidModelLoader (param: pulid_file, loads from loras/ dir)
            "class_type": "PulidModelLoader",
            "inputs": {
                "pulid_file": pulid_model,
            },
        },
        "8": {  # PulidEvaClipLoader (no required inputs, loads EVA-CLIP)
            "class_type": "PulidEvaClipLoader",
            "inputs": {},
        },
        "9": {  # PulidInsightFaceLoader (face analysis for PuLID)
            "class_type": "PulidInsightFaceLoader",
            "inputs": {
                "provider": "CUDA",
            },
        },
        "10": {  # ApplyPulid (needs eva_clip + face_analysis, not clip_vision)
            "class_type": "ApplyPulid",
            "inputs": {
                "model": ["1", 0],
                "pulid": ["7", 0],
                "eva_clip": ["8", 0],
                "face_analysis": ["9", 0],
                "image": ["6", 0],
                "method": "fidelity",
                "weight": weight,
                "start_at": 0.0,
                "end_at": 1.0,
            },
        },
        "11": {  # KSampler (uses PuLID-modified model)
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["10", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "12": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0],
            },
        },
        "13": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["12", 0],
            },
        },
    }
    return workflow


def build_controlnet_depth_workflow(
    image_name: str,
    depth_image_name: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg_scale: float = 3.5,
    strength: float = 1.0,
    seed: int | None = None,
    filename_prefix: str = "controlnet_depth",
    unet_name: str = "flux1-dev-fp8.safetensors",
    weight_dtype: str = "fp8_e4m3fn",
    clip_name1: str = "clip_l.safetensors",
    clip_name2: str = "t5xxl_fp8_e4m3fn_scaled.safetensors",
    clip_type: str = "flux",
    vae_name: str = "flux1-ae.safetensors",
    controlnet_name: str = "flux-depth-controlnet-v3.safetensors",
) -> dict[str, Any]:
    """Build a ControlNet Depth geometry-lock ComfyUI workflow.

    Uses a depth map to constrain FLUX generation geometry.

    Args:
        image_name: Source image filename in ComfyUI input/ directory.
        depth_image_name: Depth map image filename (EXR/PNG).
        prompt: Text prompt.
        negative_prompt: Negative prompt.
        width: Output width.
        height: Output height.
        steps: Inference steps.
        cfg_scale: Guidance scale.
        strength: ControlNet strength (0.0-1.0, default 1.0).
        seed: Random seed.
        filename_prefix: Output filename prefix.
        unet_name: UNET model filename.
        weight_dtype: UNET weight dtype.
        clip_name1: CLIP-L model filename.
        clip_name2: T5-XXL model filename.
        clip_type: DualCLIPLoader type ("flux").
        vae_name: VAE model filename.
        controlnet_name: ControlNet model name.

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow: dict[str, Any] = {
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
        "6": {  # Load Image (source)
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "7": {  # Load Image (depth map)
            "class_type": "LoadImage",
            "inputs": {
                "image": depth_image_name,
            },
        },
        "8": {  # ControlNetLoader
            "class_type": "ControlNetLoader",
            "inputs": {
                "control_net_name": controlnet_name,
            },
        },
        "9": {  # ControlNet Apply Advanced
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["4", 0],
                "control_net": ["8", 0],
                "image": ["7", 0],
                "strength": strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "10": {  # KSampler
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["9", 0],
                "negative": ["9", 1],
                "latent_image": ["5", 0],
            },
        },
        "11": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0],
            },
        },
        "12": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["11", 0],
            },
        },
    }
    return workflow


def build_wan21_i2v_dual_stage_workflow(
    image_name: str,
    prompt: str,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    steps: int = 20,
    cfg: float = 3.5,
    shift: float = 8.0,
    high_noise_end: float = 10.0,
    seed: int | None = None,
    filename_prefix: str = "wan_i2v",
    diffusion_model_name: str = "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf",
    low_noise_model_name: str = "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf",
    vae_name: str = "wan_2.1_vae.safetensors",
    clip_vision_name: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    t5_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
) -> dict[str, Any]:
    """Build a Wan 2.2 I2V 14B dual-stage video generation ComfyUI workflow.

    Uses UnetLoaderGGUF (reads from diffusion_models/ dir) with two separate
    GGUF models (HighNoise + LowNoise) and two KSamplerAdvanced nodes for
    the official Wan 2.2 dual-stage pipeline.

    WanImageToVideo requires positive, negative, vae as inputs (not image).
    The start_image is passed as an optional input to WanImageToVideo.

    Args:
        image_name: Input image filename in ComfyUI input/ directory.
        prompt: Text prompt for video generation.
        width: Video width (default 832 for 480p).
        height: Video height (default 480).
        length: Number of frames (default 81 = ~5s at 16fps).
        steps: Total steps split across both KSamplers (default 20).
        cfg: Guidance scale (default 3.5).
        shift: ModelSamplingSD3 shift value (default 8.0).
        high_noise_end: Step at which high-noise stage ends (default 10).
        seed: Random seed (shared across both stages).
        filename_prefix: Output filename prefix.
        diffusion_model_name: Wan 2.2 I2V high-noise GGUF model filename.
        low_noise_model_name: Wan 2.2 I2V low-noise GGUF model filename.
        vae_name: Wan VAE model filename.
        clip_vision_name: CLIP vision encoder filename.
        t5_name: T5/UMT5 text encoder filename.

    Returns:
        ComfyUI API-format workflow dict.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow: dict[str, Any] = {
        "1": {  # Load High-Noise Diffusion Model via GGUF loader (diffusion_models/ dir)
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": diffusion_model_name,
            },
        },
        "1b": {  # Load Low-Noise Diffusion Model via GGUF loader
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": low_noise_model_name,
            },
        },
        "2": {  # Load VAE (Wan 2.1)
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "3": {  # Load CLIP Vision (for image encoding)
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": clip_vision_name,
            },
        },
        "4": {  # Load UMT5 text encoder
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": t5_name,
                "type": "wan",
            },
        },
        "5": {  # CLIPTextEncode (positive)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 0],
            },
        },
        "6": {  # Load Image
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "7": {  # CLIPVisionEncode (encode image for WanImageToVideo)
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["3", 0],
                "image": ["6", 0],
                "crop": "center",
            },
        },
        "8": {  # WanImageToVideo — requires positive, negative, vae; optional start_image + clip_vision_output
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["5", 0],
                "vae": ["2", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "start_image": ["6", 0],
                "clip_vision_output": ["7", 0],
            },
        },
        "9": {  # ModelSamplingSD3 (shift tuning for high-noise stage)
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["1", 0],
                "shift": shift,
            },
        },
        "10": {  # KSampler Advanced — High Noise (stage 1)
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["9", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2],
                "start_at_step": 0,
                "end_at_step": int(high_noise_end),
                "return_with_leftover_noise": "enable",
            },
        },
        "11": {  # ModelSamplingSD3 (shift tuning for low-noise stage)
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["1b", 0],
                "shift": shift,
            },
        },
        "12": {  # KSampler Advanced — Low Noise (stage 2, uses low-noise model)
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["11", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["10", 0],
                "start_at_step": int(high_noise_end),
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
            },
        },
        "13": {  # VAE Decode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["2", 0],
            },
        },
        "14": {  # VHS_VideoCombine (save as video, 16fps)
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["13", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    return workflow


def build_upscale_workflow(
    image_name: str,
    upscale_model_name: str = "4x-UltraSharp.pth",
    filename_prefix: str = "upscaled",
) -> dict[str, Any]:
    """Build a 4x upscale ComfyUI workflow for auxiliary GPU (3060 Ti).

    Simple pipeline: LoadImage -> UpscaleModelLoader -> ImageUpscaleWithModel -> SaveImage.

    Args:
        image_name: Input image filename in ComfyUI input/ directory.
        upscale_model_name: Upscale model filename (default 4x-UltraSharp.pth).
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict.
    """
    workflow: dict[str, Any] = {
        "1": {  # Load Image
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "2": {  # Load Upscale Model
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": upscale_model_name,
            },
        },
        "3": {  # Image Upscale With Model
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0],
            },
        },
        "4": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["3", 0],
            },
        },
    }
    return workflow


def build_face_restore_workflow(
    image_name: str,
    model_name: str = "4x-UltraSharp.pth",
    filename_prefix: str = "face_restored",
) -> dict[str, Any]:
    """Build a face restoration / image enhancement ComfyUI workflow.

    NOTE: CodeFormer mtb nodes not functional (mtb face enhance only supports
    GFPGAN/RestoreFormer models which are not installed). Falls back to
    ImageUpscaleWithModel for high-quality enhancement on primary (RTX 3090).

    Pipeline: LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage.

    Args:
        image_name: Input image filename in ComfyUI input/ directory.
        model_name: Upscale model filename (default 4x-UltraSharp.pth).
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict.
    """
    workflow: dict[str, Any] = {
        "1": {  # Load Image
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "2": {  # Load Upscale Model
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": model_name,
            },
        },
        "3": {  # Image Upscale With Model
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0],
            },
        },
        "4": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["3", 0],
            },
        },
    }
    return workflow


def build_image_refine_workflow(
    image_name: str,
    prompt: str = "",
    negative_prompt: str = "",
    strength: float = 0.5,
    steps: int = 28,
    cfg_scale: float = 3.5,
    seed: int | None = None,
    filename_prefix: str = "refined",
) -> dict[str, Any]:
    """Build an image refine (img2img) ComfyUI workflow using FLUX Dev FP8.

    Uses LoadImage → VAEEncode → KSampler (img2img denoise) → VAEDecode → SaveImage.
    FLUX uses separate UNet/CLIP/VAE loaders for better memory control.
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "1": {  # LoadImage
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
        "2": {  # VAEEncode
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["10", 0],
            },
        },
        "3": {  # KSampler (img2img)
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": strength,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["2", 0],
            },
        },
        "4": {  # UNETLoader (FLUX Dev FP8)
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-dev-fp8.safetensors",
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "6": {  # CLIPTextEncode (positive)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["9", 0],
            },
        },
        "7": {  # CLIPTextEncode (negative)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt or "",
                "clip": ["9", 0],
            },
        },
        "8": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["10", 0],
            },
        },
        "9": {  # DualCLIPLoader (FLUX text encoders)
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
            },
        },
        "10": {  # VAELoader
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "flux1-ae.safetensors",
            },
        },
        "11": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["8", 0],
            },
        },
    }
    return workflow


def build_lipsync_workflow(
    video_input: str,
    audio_input: str,
    seed: int | None = None,
    lips_expression: float = 1.5,
    inference_steps: int = 20,
    output_fps: int = 25,
    filename_prefix: str = "lipsync",
) -> dict[str, Any]:
    """Build a LatentSync lip synchronization ComfyUI workflow.

    Pipeline: VHS_LoadVideo -> LoadAudio -> LatentSyncNode -> VHS_VideoCombine.

    Args:
        video_input: Video filename in ComfyUI input/ directory.
        audio_input: Audio filename or URL (http/https allowed).
        seed: Random seed. Random if None.
        lips_expression: Lip expression intensity (1.0-3.0, default 1.5).
        inference_steps: LatentSync inference steps (default 20).
        output_fps: Output video frame rate (default 25 for LatentSync).
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict with 4 nodes.

    Raises:
        ValueError: If video_input or audio_input contains path traversal or absolute path.
    """
    # Input validation — reject path traversal and absolute paths
    if ".." in video_input or video_input.startswith("/"):
        raise ValueError(
            f"video_input contains invalid path: {video_input!r} "
            "(must not contain '..' or start with '/')"
        )
    if not (audio_input.startswith("http://") or audio_input.startswith("https://")):
        if ".." in audio_input or audio_input.startswith("/"):
            raise ValueError(
                f"audio_input contains invalid path: {audio_input!r} "
                "(must not contain '..' or start with '/')"
            )

    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow: dict[str, Any] = {
        "1": {  # VHS_LoadVideo — extract frames from video
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_input,
                "force_rate": 0,
                "custom_width": -1,
                "custom_height": -1,
                "frame_start": 0,
                "frame_end": -1,
            },
        },
        "2": {  # LoadAudio — load audio waveform
            "class_type": "LoadAudio",
            "inputs": {
                "audio": audio_input,
            },
        },
        "3": {  # LatentSyncNode — lip sync inference
            "class_type": "LatentSyncNode",
            "inputs": {
                "images": ["1", 0],
                "audio": ["2", 0],
                "seed": seed,
                "lips_expression": lips_expression,
                "inference_steps": inference_steps,
            },
        },
        "4": {  # VHS_VideoCombine — encode output as MP4
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": output_fps,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    return workflow


def build_frame_interpolate_workflow(
    video_input: str,
    interpolation_factor: int = 2,
    ckpt_name: str = "rife49.pth",
    output_fps: int | None = None,
    seed: int | None = None,
    filename_prefix: str = "frame_interp",
) -> dict[str, Any]:
    """Build a RIFE frame interpolation ComfyUI workflow.

    Pipeline: VHS_LoadVideo -> RIFE VFI -> VHS_VideoCombine.

    The RIFE multiplier equals interpolation_factor - 1 (e.g., 2x -> mult 1, 4x -> mult 3).

    Args:
        video_input: Video filename in ComfyUI input/ directory.
        interpolation_factor: Target frame multiplier (2, 4, or 8; default 2).
        ckpt_name: RIFE model checkpoint filename (default rife49.pth).
        output_fps: Output frame rate. Defaults to 30 if not specified.
        seed: Kept for API consistency; RIFE VFI does not use seed.
        filename_prefix: Output filename prefix.

    Returns:
        ComfyUI API-format workflow dict with 3 nodes.

    Raises:
        ValueError: If video_input contains path traversal or absolute path.
    """
    # Input validation — reject path traversal and absolute paths
    if ".." in video_input or video_input.startswith("/"):
        raise ValueError(
            f"video_input contains invalid path: {video_input!r} "
            "(must not contain '..' or start with '/')"
        )

    # RIFE multiplier = interpolation_factor - 1
    multiplier = interpolation_factor - 1

    # Default output FPS
    if output_fps is None:
        output_fps = 30

    workflow: dict[str, Any] = {
        "1": {  # VHS_LoadVideo — extract frames from video
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_input,
                "force_rate": 0,
                "custom_width": -1,
                "custom_height": -1,
                "frame_start": 0,
                "frame_end": -1,
            },
        },
        "2": {  # RIFE VFI — frame interpolation
            "class_type": "RIFE VFI",
            "inputs": {
                "images": ["1", 0],
                "ckpt_name": ckpt_name,
                "multiplier": multiplier,
            },
        },
        "3": {  # VHS_VideoCombine — encode output as MP4
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["2", 0],
                "frame_rate": output_fps,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    return workflow


# ─── Shot Analysis (per-shot cinematic deconstruction) ───
# QwenVL prompt 模板(v2: 锐化景别定义 + 显式教"看背景判相机运动")。
# 移植自已验证的 /tmp/shot_analysis_driver.py,原样保留语义。
SEMANTIC_PROMPT = (
    "你是摄影指导分析助手。这是同一镜头的采样帧。判断步骤:先看【背景是否整体位移】判断相机运动,"
    "再看主体。只输出一个JSON对象,禁止任何其他文字:\n"
    '{"shot_scale":"远景(人很小,环境为主)|全景(人物全身+环境)|中景(人物膝盖/腰部以上)|'
    '近景(人物胸部以上)|特写(面部或局部)|大特写 之一,按人物在画面占比判断",'
    '"camera_primitive":"看背景:背景整体左右平移=pan_left/pan_right,上下平移=tilt_up/tilt_down,'
    '径向放大缩小=zoom_in/zoom_out/dolly_in/dolly_out,弧线移动=arc_left/arc_right;若【背景静止、只有主体在动】则=static;'
    '画面持续跟随移动主体=follow;手持微晃=handheld 之一",'
    '"camera_speed":"slow|medium|fast",'
    '"subject_motion":"主体自身运动方向与方式(已扣除相机),不超过15字",'
    '"lens_feel":"wide|normal|telephoto",'
    '"lighting":"不超过3个关键词"}\n'
    "无法判断的字段填null。"
)


def build_shot_analysis_workflow(
    shot: dict[str, Any],
    video: str,
    *,
    semantic: bool = True,
    subject: bool = False,
    grid_n: int = 20,
    fps: float = 24.0,
    save_dir: str = "/mnt/agents/output/gpu1/shot_analysis",
    qwen_model: str = "Qwen3-VL-8B-Instruct",
    quant: str = "8-bit (Balanced)",
    frame_count: int = 16,
) -> dict[str, Any]:
    """构建单镜头运镜解构 ComfyUI 工作流(API prompt dict)。

    三层分析:
      - 几何层(ShotGeometryLK):Lucas-Kaneda 网格光流 → 相机几何运动
      - 语义层(AILab_QwenVL_Advanced,可选):Qwen3-VL 锐化景别 + 相机 primitive
      - 主体层(SAM3Segment + SubjectMotionResidual,可选):主体分割 → 残差运动
    经 ShotJSONMerge 汇总落盘到 ``{save_dir}/shot_{id:03d}.json``。

    移植自已验证的 ``shot_analysis_driver.build_prompt`` + ``SEMANTIC_PROMPT``
    (v2),节点接线与字段原样保留。

    Args:
        shot: ``{"id": int, "start_sec": float, "end_sec": float}``。
        video: 容器可见的视频绝对路径。
        semantic: 启用 QwenVL 语义层(默认 True)。
        subject: 启用 SAM3 主体层(默认 False)。
        grid_n: 几何光流网格密度(默认 20)。
        fps: 帧率,用于把秒换算成帧索引(默认 24.0)。
        save_dir: **绝对容器路径** —— ShotJSONMerge 用 os.path.join(_OUT_DIR, save_dir, ...),
            传绝对路径会覆盖默认 _OUT_DIR,使输出落到主机挂载的可见存储。
        qwen_model: QwenVL 模型名(默认 Qwen3-VL-8B-Instruct)。
        quant: QwenVL 量化配置(默认 8-bit (Balanced))。
        frame_count: QwenVL 采样帧数(默认 16)。

    Returns:
        ComfyUI API-format workflow dict({node_id: {class_type, inputs}})。
    """
    shot_id_str = f"shot_{int(shot['id']):03d}"
    start, end = float(shot["start_sec"]), float(shot["end_sec"])
    skip = int(round(start * fps))
    cap = max(2, int(round((end - start) * fps)))

    nodes: dict[str, Any] = {}
    nodes["load"] = {
        "class_type": "VHS_LoadVideoPath",
        "inputs": {
            "video": video, "force_rate": 0.0, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": cap, "skip_first_frames": skip, "select_every_nth": 1,
        },
    }
    nodes["geo"] = {
        "class_type": "ShotGeometryLK",
        "inputs": {"images": ["load", 0], "grid_n": grid_n},
    }
    nodes["viz_geo"] = {"class_type": "PreviewImage", "inputs": {"images": ["geo", 1]}}

    merge_inputs: dict[str, Any] = {
        "shot_id": shot_id_str, "save_dir": save_dir,
        "geometry_json": ["geo", 0],
    }

    if semantic:
        # AILab_QwenVL_Advanced: API prompt 必须显式提供全部 required 输入(UI 默认不自动填)
        nodes["qwen"] = {
            "class_type": "AILab_QwenVL_Advanced",
            "inputs": {
                "model_name": qwen_model, "quantization": quant,
                "attention_mode": "auto", "use_torch_compile": False, "device": "auto",
                "preset_prompt": "🖼️ Detailed Description",   # custom_prompt 会完全覆盖它
                "custom_prompt": SEMANTIC_PROMPT,
                "max_tokens": 256, "temperature": 0.1, "top_p": 0.9,
                "num_beams": 1, "repetition_penalty": 1.2,
                "frame_count": frame_count, "keep_model_loaded": True, "seed": 1,
                "video": ["load", 0],            # 帧序列作为 video 输入
            },
        }
        merge_inputs["semantic_json"] = ["qwen", 0]

    if subject:
        # 主体层:SAM3 文本提示分割(逐帧,无需外部 loader) → masks → SubjectMotionResidual
        # SAM3Segment 自带模型加载,prompt 填主体描述
        nodes["sam"] = {
            "class_type": "SAM3Segment",
            "inputs": {
                "image": ["load", 0],
                # 风格化动画:通用 "main subject" 不接地气,用更具体的概念词 + 降阈值
                "prompt": "character, person, creature, animal, or figure",
                "output_mode": "Merged", "confidence_threshold": 0.25,
                "device": "GPU",
            },
        }
        nodes["subj"] = {
            "class_type": "SubjectMotionResidual",
            "inputs": {"images": ["load", 0], "masks": ["sam", 1], "grid_n": grid_n},
        }
        nodes["viz_subj"] = {"class_type": "PreviewImage", "inputs": {"images": ["subj", 1]}}
        merge_inputs["subject_json"] = ["subj", 0]

    nodes["merge"] = {"class_type": "ShotJSONMerge", "inputs": merge_inputs}
    return nodes
