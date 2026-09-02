"""V3.6 Routing Table — 35 nodes with stage-aware routing.

Hardware: Single machine (192.168.71.166) with dual GPU:
    - GPU 0: RTX 3060 Ti 8G — display + NVENC/NVDEC + ffmpeg IO (host)
    - GPU 1: RTX 3090 24G — CUDA inference primary

Each node defines GPU allocation per stage:
    - preview: 3090 Light pool large (9-15G available)
    - heavy: 3090 Light pool small/closed (0-3G available)
    - overflow: 3060Ti Combo overflow (DEPRECATED — no longer used)
    - cpu: CPU-only tasks (Blender, FFmpeg)

The routing table covers all 35 task nodes from the V3.6 architecture spec,
with 7 stages dynamically selecting the appropriate allocation column.

2026-09-02: nodes 16/18 (文生视频/图生视频低参预览, ltx_i2v) 已随 ltx_i2v 引擎退役
删除, 任务编号保留不重排以维持 task id 稳定性。
"""

from __future__ import annotations

from typing import Optional

ROUTING_TABLE: dict[int, dict] = {
    1: {
        "name": "文生图（低参抽卡）",
        "model_id": "sdxl_lightning",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "SDXL + Light pool full"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "SDXL"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Image"},
        "cpu": None,
    },
    2: {
        "name": "文生图（高参生产）",
        "model_id": "sd35_large",
        "preview": None,
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "SD3.5 Zero Resident"},
        "overflow": None,
        "cpu": None,
    },
    3: {
        "name": "图生文（反推标签）",
        "model_id": "wd14",
        "preview": {"gpu": "3090", "slot": "light", "note": "WD14 Light pool resident"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "WD14 if fits"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Understand"},
        "cpu": None,
    },
    4: {
        "name": "图生图（低参）",
        "model_id": "sdxl_lightning",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "SDXL + IPAdapter"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Image"},
        "cpu": None,
    },
    5: {
        "name": "图生图（高参精修）",
        "model_id": "sd35_large",
        "preview": None,
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "SD3.5 / FLUX"},
        "overflow": None,
        "cpu": None,
    },
    6: {
        "name": "视频生图（废片修复）",
        "model_id": "sd35_large",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "SD3.5 / SDXL"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "SD3.5"},
        "overflow": None,
        "cpu": None,
    },
    7: {
        "name": "文生动作",
        "model_id": "motiongpt",
        "preview": {"gpu": "3090", "slot": "light", "note": "MotionGPT 2G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": None, "note": "通用溢出"},
        "cpu": None,
    },
    8: {
        "name": "文生3D（角色原型）",
        "model_id": "hunyuan3d2_mini",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "Hunyuan3D-2mini 6G"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": None, "note": "通用溢出"},
        "cpu": None,
    },
    9: {
        "name": "图生3D（角色精模）",
        "model_id": "trellis",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS 16G + Whisper"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS 20G peak"},
        "overflow": None,
        "cpu": None,
    },
    10: {
        "name": "文生3D（场景高参）",
        "model_id": "hunyuan3d2",
        "preview": None,
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "Hunyuan3D-2 20G"},
        "overflow": None,
        "cpu": None,
    },
    11: {
        "name": "图生3D（场景）",
        "model_id": "hunyuan3d2",
        "preview": None,
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "Hunyuan3D-2"},
        "overflow": None,
        "cpu": None,
    },
    12: {
        "name": "图生3D（道具）",
        "model_id": "trellis",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS + Whisper"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS"},
        "overflow": None,
        "cpu": None,
    },
    13: {
        "name": "文生虚拟人",
        "model_id": "liveportrait",
        "preview": {"gpu": "3090", "slot": "light", "note": "LivePortrait 6G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Virtual"},
        "cpu": None,
    },
    14: {
        "name": "图生虚拟人",
        "model_id": "liveportrait",
        "preview": {"gpu": "3090", "slot": "light", "note": "LivePortrait"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Virtual"},
        "cpu": None,
    },
    15: {
        "name": "文生场景",
        "model_id": "sd35_large",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "SD3.5 + Light pool"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "SD3.5"},
        "overflow": None,
        "cpu": None,
    },
    # 16 (文生视频低参预览, ltx_i2v) 已退役 — 2026-09-02 ltx_i2v 引擎下线, 编号保留不重排
    17: {
        "name": "文生视频（高参确认）",
        "model_id": "wan13b_i2v",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "Wan1.3B 10G + Light pool"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "Wan1.3B"},
        "overflow": None,
        "cpu": None,
    },
    # 18 (图生视频低参预览, ltx_i2v) 已退役 — 2026-09-02 ltx_i2v 引擎下线, 编号保留不重排
    19: {
        "name": "图生视频（高参终版）",
        "model_id": "wan14b_i2v",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "Wan14B 20G + empty Light"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "Wan14B"},
        "overflow": None,
        "cpu": None,
    },
    20: {
        "name": "视频生视频",
        "model_id": "wan14b_i2v",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "Wan14B"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "Wan14B V2V"},
        "overflow": None,
        "cpu": None,
    },
    21: {
        "name": "视频延长/补帧",
        "model_id": "rife",
        "preview": {"gpu": "3090", "slot": "light", "note": "RIFE 2G Light"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "Wan14B-InP"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Sync"},
        "cpu": None,
    },
    22: {
        "name": "对口型",
        "model_id": "musetalk",
        "preview": {"gpu": "3090", "slot": "light", "note": "MuseTalk 4G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Sync"},
        "cpu": None,
    },
    23: {
        "name": "文生音乐（日常BGM）",
        "model_id": "stable_audio",
        "preview": {"gpu": "3090", "slot": "light", "note": "Stable Audio 7.5G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-SFX"},
        "cpu": None,
    },
    24: {
        "name": "文生音乐（主题曲/终版）",
        "model_id": "yue_7b",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "YuE 7B 16G + Whisper"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "YuE 7B"},
        "overflow": None,
        "cpu": None,
    },
    25: {
        "name": "音乐生音乐",
        "model_id": "stable_audio",
        "preview": {"gpu": "3090", "slot": "light", "note": "Stable Audio if fits"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-SFX"},
        "cpu": None,
    },
    26: {
        "name": "文生音效",
        "model_id": "foleycrafter",
        "preview": {"gpu": "3090", "slot": "light", "note": "FoleyCrafter 6G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Foley"},
        "cpu": None,
    },
    27: {
        "name": "视频生音频（Foley）",
        "model_id": "foleycrafter",
        "preview": {"gpu": "3090", "slot": "light", "note": "FoleyCrafter"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Foley"},
        "cpu": None,
    },
    28: {
        "name": "音频生音频（分离/修复）",
        "model_id": "uvr5",
        "preview": {"gpu": "3090", "slot": "light", "note": "UVR5 2G Light"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "UVR5 if fits"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Understand"},
        "cpu": None,
    },
    29: {
        "name": "文生语音 TTS",
        "model_id": "cosyvoice",
        "preview": {"gpu": "3090", "slot": "light", "note": "CosyVoice 6G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Audio-Full"},
        "cpu": None,
    },
    30: {
        "name": "语音克隆",
        "model_id": "gpt_sovits",
        "preview": {"gpu": "3090", "slot": "light", "note": "GPT-SoVITS 4G Light"},
        "heavy": None,
        "overflow": {"gpu": "3060ti", "combo": "Combo-Audio-Full"},
        "cpu": None,
    },
    31: {
        "name": "语音变声",
        "model_id": "rvc",
        "preview": {"gpu": "3090", "slot": "light", "note": "RVC 2G Light"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "RVC if fits"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Sync"},
        "cpu": None,
    },
    32: {
        "name": "视频生文/字幕",
        "model_id": "whisper",
        "preview": {"gpu": "3090", "slot": "light", "note": "Whisper 3G Light"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "Whisper if fits"},
        "overflow": {"gpu": "3060ti", "combo": "Combo-Understand"},
        "cpu": None,
    },
    33: {
        "name": "字幕合成/剪辑",
        "model_id": "ffmpeg_x264",
        "preview": None,
        "heavy": None,
        "overflow": None,
        "cpu": {"tool": "ffmpeg", "note": "CPU FFmpeg"},
    },
    34: {
        "name": "3D 资产后处理",
        "model_id": "blender_cycles",
        "preview": None,
        "heavy": None,
        "overflow": None,
        "cpu": {"tool": "blender", "note": "CPU Blender"},
    },
    35: {
        "name": "多机位参考图渲染",
        "model_id": "blender_cycles",
        "preview": None,
        "heavy": None,
        "overflow": None,
        "cpu": {"tool": "blender", "note": "CPU Blender"},
    },
    36: {
        "name": "深度/法线/AO 烘焙",
        "model_id": "blender_cycles",
        "preview": None,
        "heavy": None,
        "overflow": None,
        "cpu": {"tool": "blender", "note": "CPU Blender"},
    },
    # TRELLIS 2 文生3D
    37: {
        "name": "文/图生3D（生产）",
        "model_id": "trellis",
        "preview": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS 1024 prod"},
        "heavy": {"gpu": "3090", "slot": "heavy", "note": "TRELLIS 1024 prod"},
        "overflow": None,
        "cpu": None,
    },
    38: {
        "name": "文/图生3D（预览）",
        "model_id": "trellis_preview",
        "preview": {"gpu": "3090", "slot": "light", "note": "TRELLIS 512 preview"},
        "heavy": {"gpu": "3090", "slot": "light", "note": "TRELLIS 512 preview"},
        "overflow": {"gpu": "3060ti", "note": "TRELLIS 512 on 3060Ti"},
        "cpu": None,
    },
}


def build_routing_table() -> dict[int, dict]:
    """Build and validate the complete routing table.

    Returns:
        The ROUTING_TABLE dict after validation.

    Raises:
        ValueError: If any node is missing required fields.
    """
    for node_id, node in ROUTING_TABLE.items():
        required = {"name", "model_id"}
        missing = required - set(node.keys())
        if missing:
            raise ValueError(f"Node {node_id} missing fields: {missing}")
    return ROUTING_TABLE
