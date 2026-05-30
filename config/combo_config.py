"""V3.6 3060Ti Combo Configuration — 7 Combos for 8G hard constraint.

⚠️ DEPRECATED — 3060Ti no longer handles CUDA inference.
3060Ti is now used for display + NVENC/NVDEC + ffmpeg IO only.
All AI inference runs on RTX 3090.

This config is kept for backwards compatibility but should not be used.
"""

from __future__ import annotations

COMBO_3060TI: dict[str, dict] = {
    "Combo-Audio-Full": {
        "models": {"cosyvoice": 6000, "gpt_sovits": 4000},
        "strategy": "serial_swap",
        "desc": "TTS + 克隆串行",
    },
    "Combo-Sync": {
        "models": {"musetalk": 4000, "rvc": 2000, "rife": 2000},
        "strategy": "resident",
        "desc": "对口型+变声+补帧",
    },
    "Combo-Understand": {
        "models": {"whisper": 3000, "wd14": 2000, "uvr5": 2000},
        "strategy": "resident",
        "desc": "字幕+反推+分离",
    },
    "Combo-Image": {
        "models": {"sdxl_lightning": 6000, "sdxl_ipadapter": 1000},
        "strategy": "resident",
        "desc": "抽卡溢出",
    },
    "Combo-Virtual": {
        "models": {"liveportrait": 6000},
        "strategy": "resident",
        "desc": "虚拟人",
    },
    "Combo-SFX": {
        "models": {"stable_audio": 7500},
        "strategy": "resident",
        "desc": "BGM/SFX独占",
    },
    "Combo-Foley": {
        "models": {"foleycrafter": 6000},
        "strategy": "resident",
        "desc": "Foley独占",
    },
}
