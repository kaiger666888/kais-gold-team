"""V3.6 Stage Configuration — 7 stages for 3090 dynamic VRAM partitioning.

Each stage defines:
    - heavy_model: the model that occupies the Heavy slot on 3090
    - heavy_vram: peak VRAM (MB) needed by the heavy model
    - light_pool_max: maximum MB available for Light resident pool
    - resident_light: list of model_ids to keep resident in Light pool
    - desc: human-readable description

Hard cap: 21G total on 3090 (heavy_vram + light_pool ≤ 21000 MB).
"""

from __future__ import annotations

STAGE_CONFIG: dict[str, dict] = {
    "image_draw": {
        "heavy_model": "sdxl_lightning",
        "heavy_vram": 6000,
        "light_pool_max": 15000,
        "resident_light": ["cosyvoice", "whisper", "wd14", "rvc", "uvr5"],
        "desc": "海量抽卡+音频并行",
    },
    "video_preview": {
        "heavy_model": "ltx_i2v",
        "heavy_vram": 12000,
        "light_pool_max": 9000,
        "resident_light": ["cosyvoice", "whisper", "gpt_sovits"],
        "desc": "动态预览+配音",
    },
    "image_refine": {
        "heavy_model": "sd35_large",
        "heavy_vram": 20000,
        "light_pool_max": 1000,
        "resident_light": ["wd14"],
        "desc": "精修专注",
    },
    "video_final": {
        "heavy_model": "wan14b_i2v",
        "heavy_vram": 20000,
        "light_pool_max": 1000,
        "resident_light": [],
        "desc": "终版独占",
    },
    "3d_character": {
        "heavy_model": "trellis",
        "heavy_vram": 16000,
        "light_pool_max": 5000,
        "resident_light": ["whisper", "wd14"],
        "desc": "角色生成+字幕",
    },
    "3d_scene": {
        "heavy_model": "hunyuan3d2",
        "heavy_vram": 20000,
        "light_pool_max": 1000,
        "resident_light": [],
        "desc": "场景独占",
    },
    "music_final": {
        "heavy_model": "yue_7b",
        "heavy_vram": 16000,
        "light_pool_max": 5000,
        "resident_light": ["whisper", "wd14"],
        "desc": "音乐+字幕",
    },
}

# Inverse mapping: heavy model_id → stage name
STAGE_CONFIG_INV: dict[str, str] = {
    cfg["heavy_model"]: stage
    for stage, cfg in STAGE_CONFIG.items()
}
