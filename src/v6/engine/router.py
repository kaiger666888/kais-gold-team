"""Engine Router — dual ComfyUI (primary 3090 + auxiliary 3060 Ti), degrade to cloud.

Hardware: Single machine with dual ComfyUI instances:
    - comfyui-primary (RTX 3090, 24GB, :8188) — video, 3D, FLUX, heavy tasks
    - comfyui-auxiliary (RTX 3060 Ti, 8GB/5.5GB usable, :8189) — upscale, face_restore, light refine
"""
from __future__ import annotations

import logging
from typing import Optional

from src.v6.models.task import EnginePool, GenerationTask, ModelPreference, TaskType

logger = logging.getLogger(__name__)

# ─── Dedicated engine mappings (type → engine_id) ───
# These task types have specialized engines that should always be preferred
# over the generic comfyui-primary fallback.
# NOTE: MUSIC/SFX removed in v1.5 — gold-team no longer hosts music generation.
# Music is now served by Node-layer routes (/api/v1/ace/generate → ComfyUI
# /prompt). Sending MUSIC/SFX to gold-team results in task FAILED with a
# redirect message (see src/v6/executor.py).
DEDICATED_ENGINES: dict[TaskType, str] = {
    TaskType.IMAGE_TO_3D: "hunyuan3d-local",
    TaskType.IMAGE_TO_3D_MV: "hunyuan3d-mv-local",
    TaskType.TTS: "tts-tracker",
    TaskType.TTS_ZH: "tts-tracker",
    TaskType.TTS_EN: "tts-tracker",
    TaskType.TTS_BILINGUAL: "tts-tracker",
    TaskType.COLOR_GRADE: "color-grade",
}

# ─── Light task types routed to auxiliary ───
# NOTE: face_restore removed — mtb face enhance not functional on either instance.
#   Routes to primary, uses simple upscale pipeline.
LIGHT_TASK_TYPES: set[TaskType] = {
    TaskType.UPSCALE,
    TaskType.IMAGE_REFINE,
}

# VRAM requirements by task type (estimates, GB)
VRAM_ESTIMATES: dict[TaskType, float] = {
    TaskType.VIDEO_FINAL: 22.0,
    TaskType.VIDEO_PREVIEW: 14.0,
    TaskType.IMAGE_DRAW: 8.0,
    TaskType.IMAGE_REFINE: 6.0,
    TaskType.TTS: 2.0,
    TaskType.TTS_ZH: 4.0,         # GPT-SoVITS on 3090 (CUDA=0)
    TaskType.TTS_EN: 2.0,         # Chatterbox-Turbo on 3090 (shared GPU)
    TaskType.TTS_BILINGUAL: 6.0,  # CosyVoice on 3090
    TaskType.MUSIC: 4.0,
    TaskType.SFX: 2.0,
    TaskType.UPSCALE: 2.0,
    TaskType.FACE_RESTORE: 1.5,
    TaskType.IMAGE_TO_3D: 10.0,
    TaskType.IMAGE_TO_3D_MV: 12.0,      # Hunyuan3D-2mv (multiview, heavier)
    TaskType.IMAGE_PULID: 16.0,          # FLUX + PuLID
    TaskType.IMAGE_DRAW_IPADAPTER: 16.0,  # FLUX + IP-Adapter
    TaskType.CONTROLNET_DEPTH: 18.0,     # FLUX + ControlNet
    TaskType.WAN_I2V: 20.0,              # Wan 2.1 14B
    TaskType.COLOR_GRADE: 0.0,           # CPU only — ffmpeg + LUT, no GPU VRAM
}

# Auxiliary VRAM cap — only accept tasks needing < 5 GB
AUX_VRAM_CAP_GB = 5.0

# Local-only task types (no cloud fallback)
LOCAL_ONLY_TYPES: set[TaskType] = set()

# Cloud-capable task types
CLOUD_CAPABLE: set[TaskType] = {
    TaskType.VIDEO_FINAL,
    TaskType.VIDEO_PREVIEW,
    TaskType.IMAGE_DRAW,
    TaskType.IMAGE_REFINE,
    TaskType.IMAGE_TO_3D,
    TaskType.WAN_I2V,
}

# Primary VRAM (RTX 3090)
LOCAL_VRAM_GB = 24.0
VRAM_HARD_CAP_GB = 23.5


class EngineRouter:
    """Decides which engine pool (local-primary / local-auxiliary / cloud) a task should run on."""

    def __init__(
        self,
        local_available: bool = True,
        local_vram_used_gb: float = 0.0,
        primary_available: bool = True,
        auxiliary_available: bool = True,
    ) -> None:
        self.local_available = local_available
        self.local_vram_used_gb = local_vram_used_gb
        self.primary_available = primary_available
        self.auxiliary_available = auxiliary_available

    def _vram_available(self) -> float:
        return max(0.0, VRAM_HARD_CAP_GB - self.local_vram_used_gb)

    def route(self, task: GenerationTask) -> tuple[EnginePool, str]:
        """Route a task to an engine pool.  Returns (pool, engine_id).

        Logic:
            1. Explicit CLOUD preference → cloud
            2. Explicit LOCAL preference → local (primary or auxiliary)
            3. AUTO:
               a. Light task + auxiliary healthy → auxiliary
               b. Otherwise → primary (if healthy)
               c. Primary down + cloud capable → cloud fallback
        """
        # ── Explicit preference ──
        if task.model_preference == ModelPreference.CLOUD:
            return EnginePool.CLOUD, self._pick_cloud_engine_id(task)

        if task.model_preference == ModelPreference.LOCAL:
            if self.local_available or self.auxiliary_available:
                return EnginePool.LOCAL, self._pick_local_engine_id(task)
            return EnginePool.CLOUD, "cloud-mock"

        # ── AUTO routing ──

        # Step 1: light tasks → auxiliary when available
        if task.type in LIGHT_TASK_TYPES and self.auxiliary_available:
            vram_needed = VRAM_ESTIMATES.get(task.type, 8.0)
            if vram_needed <= AUX_VRAM_CAP_GB:
                logger.info(
                    "Light task %s (type=%s, %.1fGB) → auxiliary",
                    task.task_id, task.type.value, vram_needed,
                )
                return EnginePool.LOCAL, "comfyui-auxiliary"

        # Step 2: heavy tasks → primary when available
        if self.local_available:
            vram_needed = VRAM_ESTIMATES.get(task.type, 8.0)
            vram_available = self._vram_available()
            if vram_needed <= vram_available:
                return EnginePool.LOCAL, self._pick_local_engine_id(task)

        # Step 3: primary unavailable / VRAM insufficient → try auxiliary for light tasks
        if task.type in LIGHT_TASK_TYPES and self.auxiliary_available:
            vram_needed = VRAM_ESTIMATES.get(task.type, 8.0)
            if vram_needed <= AUX_VRAM_CAP_GB:
                logger.info(
                    "Primary unavailable, routing light task %s → auxiliary",
                    task.task_id,
                )
                return EnginePool.LOCAL, "comfyui-auxiliary"

        # Step 4: cloud fallback for capable types
        if task.type in CLOUD_CAPABLE:
            logger.info(
                "Local engines unavailable/insufficient → cloud for task %s",
                task.task_id,
            )
            return EnginePool.CLOUD, self._pick_cloud_engine_id(task)

        # Step 5: no fallback — queue on primary anyway
        logger.warning(
            "No suitable engine for task %s (type=%s), queueing on primary",
            task.task_id, task.type.value,
        )
        return EnginePool.LOCAL, "comfyui-primary"

    def _pick_local_engine_id(self, task: GenerationTask) -> str:
        """Pick the best local engine ID for the task type."""
        # TRELLIS bypass: IMAGE_TO_3D + trellis/flux_trellis goes to comfyui-primary
        if task.type == TaskType.IMAGE_TO_3D:
            extra = task.params.get("extra", {})
            if extra.get("engine") == "trellis" or extra.get("mode") == "flux_trellis":
                return "comfyui-primary"
        # Dedicated engine mappings take priority
        if task.type in DEDICATED_ENGINES:
            return DEDICATED_ENGINES[task.type]
        if task.type in LIGHT_TASK_TYPES and self.auxiliary_available:
            return "comfyui-auxiliary"
        return "comfyui-primary"

    def _pick_cloud_engine_id(self, task: GenerationTask) -> str:
        """Pick the best cloud engine ID for the task type.

        cloud-jimeng only supports image generation (text2image +
        image2image). Video and other types fall back to cloud-mock or
        another configured engine.
        """
        task_type = task.type.value if hasattr(task.type, 'value') else str(task.type)
        if task_type in ("image_draw", "image_refine"):
            return "cloud-jimeng"
        # Video / audio / other types: not supported by dreamina engine
        return "cloud-mock"


# Singleton
_router: Optional[EngineRouter] = None


def get_engine_router() -> EngineRouter:
    global _router
    if _router is None:
        _router = EngineRouter(local_available=True, local_vram_used_gb=0.0)
    return _router
