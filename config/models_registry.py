"""V3.6 Models Registry — all AI models with metadata.

Hardware: Single machine (192.168.71.166) with dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G — CUDA inference primary

Each model entry includes:
    - vram: peak VRAM usage in MB
    - precision: bf16 or fp16 (3060Ti always fp16 when it was used for inference)
    - category: Heavy or Light
    - combo_id: which 3060Ti Combo this model belongs to (DEPRECATED — no longer used)
    - weight_path: HuggingFace model path
    - runtime: which runtime container to use
    - triton_hash: hash for Triton compile cache isolation
"""

from __future__ import annotations

MODELS: dict[str, dict] = {
    # ── Heavy Models (3090 only, independent process, sys.exit(0)) ──
    "sdxl_lightning": {
        "vram": 6000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": "Combo-Image",
        "weight_path": "ByteDance/SDXL-Lightning",
        "runtime": "diffusion-runtime",
        "triton_hash": "sdxl_lightning_fp16_sm86",
    },
    "sd35_large": {
        "vram": 20000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "stabilityai/stable-diffusion-3.5-large",
        "runtime": "diffusion-runtime",
        "triton_hash": "sd35_large_fp16_sm86",
    },
    "flux_kontext": {
        "vram": 19000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "black-forest-labs/flux-kontext",
        "runtime": "diffusion-runtime",
        "triton_hash": "flux_kontext_fp16_sm86",
    },
    "wan14b_i2v": {
        "vram": 20000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "Wan-AI/Wan2.1-I2V-14B",
        "runtime": "video-runtime",
        "triton_hash": "wan14b_i2v_fp16_sm86",
    },
    "wan13b_i2v": {
        "vram": 10000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "Wan-AI/Wan2.1-I2V-1.3B",
        "runtime": "video-runtime",
        "triton_hash": "wan13b_i2v_fp16_sm86",
    },
    # ── TRELLIS 2 (Text/Image-to-3D) ──
    "trellis": {
        "vram": 12400,
        "precision": "fp16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "microsoft/TRELLIS.2-4B",
        "runtime": "comfyui-runtime",
        "triton_hash": "trellis_fp16_sm86",
    },
    "trellis_preview": {
        "vram": 6800,
        "precision": "fp16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "microsoft/TRELLIS.2-4B",
        "runtime": "comfyui-runtime",
        "triton_hash": "trellis_preview_fp16_sm86",
    },
    "hunyuan3d2": {
        "vram": 20000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "tencent/Hunyuan3D-2",
        "runtime": "3d-runtime",
        "triton_hash": "hunyuan3d2_fp16_sm86",
    },
    "hunyuan3d2_mini": {
        "vram": 6000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "tencent/Hunyuan3D-2-mini",
        "runtime": "3d-runtime",
        "triton_hash": "hunyuan3d2_mini_fp16_sm86",
    },
    "yue_7b": {
        "vram": 16000,
        "precision": "bf16",
        "category": "Heavy",
        "combo_id": None,
        "weight_path": "m-a-p/YuE-s1-7b",
        "runtime": "audio-runtime",
        "triton_hash": "yue_7b_fp16_sm86",
    },
    # ── Light Models (3090 Light pool / 3060Ti overflow) ──
    "cosyvoice": {
        "vram": 6000,
        "precision": "bf16",
        "category": "Light",
        "combo_id": "Combo-Audio-Full",
        "weight_path": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "runtime": "audio-runtime",
        "triton_hash": "cosyvoice3_rl_fp16_sm86",
    },
    "whisper": {
        "vram": 3000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Understand",
        "weight_path": "openai/whisper-large-v3",
        "runtime": "audio-runtime",
        "triton_hash": "whisper_fp16_sm86",
    },
    "wd14": {
        "vram": 2000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Understand",
        "weight_path": "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "runtime": "diffusion-runtime",
        "triton_hash": "wd14_fp16_sm86",
    },
    "rvc": {
        "vram": 2000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Sync",
        "weight_path": "RVC-Project/Retrieval-based-Voice-Conversion",
        "runtime": "audio-runtime",
        "triton_hash": "rvc_fp16_sm86",
    },
    "uvr5": {
        "vram": 2000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Understand",
        "weight_path": "ljn995/UVR5",
        "runtime": "audio-runtime",
        "triton_hash": "uvr5_fp16_sm86",
    },
    "gpt_sovits": {
        "vram": 4000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Audio-Full",
        "weight_path": "fishaudio/GPT-SoVITS",
        "runtime": "audio-runtime",
        "triton_hash": "gpt_sovits_fp16_sm86",
    },
    "musetalk": {
        "vram": 4000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Sync",
        "weight_path": "TMElyralab/MuseTalk",
        "runtime": "video-runtime",
        "triton_hash": "musetalk_fp16_sm86",
    },
    "rife": {
        "vram": 2000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Sync",
        "weight_path": "hzwer/RIFE",
        "runtime": "video-runtime",
        "triton_hash": "rife_fp16_sm86",
    },
    "stable_audio": {
        "vram": 7500,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-SFX",
        "weight_path": "stabilityai/stable-audio-open-1.5",
        "runtime": "audio-runtime",
        "triton_hash": "stable_audio_fp16_sm86",
    },
    "liveportrait": {
        "vram": 6000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Virtual",
        "weight_path": "KwaiVGI/LivePortrait",
        "runtime": "video-runtime",
        "triton_hash": "liveportrait_fp16_sm86",
    },
    "foleycrafter": {
        "vram": 6000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": "Combo-Foley",
        "weight_path": "foleycrafter/FoleyCrafter",
        "runtime": "audio-runtime",
        "triton_hash": "foleycrafter_fp16_sm86",
    },
    "sdxl_ipadapter": {
        "vram": 1000,
        "precision": "bf16",
        "category": "Light",
        "combo_id": "Combo-Image",
        "weight_path": "ByteDance/SDXL-Lightning-IPAdapter",
        "runtime": "diffusion-runtime",
        "triton_hash": "sdxl_ipadapter_fp16_sm86",
    },
    "motiongpt": {
        "vram": 2000,
        "precision": "fp16",
        "category": "Light",
        "combo_id": None,
        "weight_path": "OpenMotionLab/MotionGPT",
        "runtime": "video-runtime",
        "triton_hash": "motiongpt_fp16_sm86",
    },
}

# Derived lookups
LIGHT_MODELS: dict[str, dict] = {
    k: v for k, v in MODELS.items() if v["category"] == "Light"
}
HEAVY_MODELS: dict[str, dict] = {
    k: v for k, v in MODELS.items() if v["category"] == "Heavy"
}
