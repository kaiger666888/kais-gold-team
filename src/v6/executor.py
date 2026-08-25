"""Task executor — picks tasks from queue and dispatches to engines."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from src.v6.callbacks import build_callback_payload, send_callback
from src.v6.engines.base import BaseEngine
from src.v6.engines.comfyui import ComfyUIEngine
from src.v6.engines.mock import MockEngine
from src.v6.models.task import (
    EnginePool,
    GenerationTask,
    TaskMetadata,
    TaskOutputs,
    TaskStatus,
    TaskType,
)
from src.v6.store import get_task_store

logger = logging.getLogger(__name__)


# Map TaskType → default output fields for mock/local
_TASK_OUTPUT_FIELDS: dict[TaskType, dict[str, str]] = {
    TaskType.VIDEO_FINAL: {"video": "final.mp4", "thumbnail": "thumb.jpg"},
    TaskType.VIDEO_PREVIEW: {"video": "preview.mp4", "thumbnail": "thumb.jpg"},
    TaskType.IMAGE_DRAW: {"image": "render.png", "thumbnail": "thumb.jpg"},
    TaskType.IMAGE_REFINE: {"image": "refined.png"},
    TaskType.TTS: {"audio": "voice.wav"},
    TaskType.MUSIC: {"audio": "bgm.wav"},
    TaskType.SFX: {"audio": "sfx.wav"},
    TaskType.UPSCALE: {"image": "upscaled.png"},
    TaskType.FACE_RESTORE: {"image": "face_restored.png"},
    TaskType.IMAGE_TO_3D: {"image": "model.glb"},
    TaskType.IMAGE_TO_3D_MV: {"image": "model_mv.glb"},
    TaskType.IMAGE_PULID: {"image": "pulid_flux.png", "thumbnail": "thumb.jpg"},
    TaskType.CONTROLNET_DEPTH: {"image": "controlnet_depth.png", "thumbnail": "thumb.jpg"},
    TaskType.WAN_I2V: {"video": "wan_i2v.mp4", "thumbnail": "thumb.jpg"},
    # SHOT_ANALYSIS output is a JSON sidecar (ShotJSONMerge-written, not a
    # ComfyUI media artifact) — used only as a fallback template; the real
    # output path is constructed in the SHOT_ANALYSIS dispatch branch.
    TaskType.SHOT_ANALYSIS: {"analysis": "shot.json"},
}


class TaskExecutor:
    """Background worker that pulls pending tasks and runs them through engines."""

    def __init__(self) -> None:
        self._engines: dict[str, BaseEngine] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def register_engine(self, engine: BaseEngine) -> None:
        """Register an engine by its engine_id."""
        self._engines[engine.engine_id] = engine
        logger.info("Executor: registered engine '%s' (%s)", engine.engine_id, engine.name)

    def get_engine(self, engine_id: str) -> Optional[BaseEngine]:
        return self._engines.get(engine_id)

    def list_engines(self) -> list[BaseEngine]:
        return list(self._engines.values())

    async def start(self) -> None:
        """Start all registered engines and the background worker loop."""
        for engine in self._engines.values():
            await engine.start()

        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("TaskExecutor started with %d engine(s)", len(self._engines))

    async def stop(self) -> None:
        """Stop worker and teardown engines."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        for engine in self._engines.values():
            await engine.stop()

    async def _worker_loop(self) -> None:
        """Continuously poll the task queue and dispatch."""
        store = get_task_store()

        while self._running:
            try:
                task_id = await asyncio.wait_for(store._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            task = await store.get(task_id)
            if not task or task.status == TaskStatus.CANCELLED:
                continue

            # Run task (per-task timeout: 防退化镜头/模型挂死阻塞整个队列)
            try:
                await asyncio.wait_for(self._execute_task(task), timeout=600)
            except asyncio.TimeoutError:
                logger.warning("Task %s timed out after 600s — marking FAILED so queue continues",
                               task.task_id)
                await store.update(task.task_id, status=TaskStatus.FAILED,
                                   error="task timed out after 600s")

    async def _execute_task(self, task: GenerationTask) -> None:
        """Execute a single task through the appropriate engine."""
        store = get_task_store()

        await store.update(task.task_id, status=TaskStatus.RUNNING, progress=0.0)

        engine_id = task.engine_id or "mock"
        engine = self._resolve_engine(engine_id, task)

        if not engine:
            await store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                error=f"No engine available for '{engine_id}'",
            )
            return

        try:
            # Build workflow from task params
            workflow = task.params.get("workflow")
            if not workflow or "__mock__" in workflow:
                # Route to appropriate workflow builder based on task type
                if task.type in (TaskType.TTS, TaskType.TTS_ZH, TaskType.TTS_EN, TaskType.TTS_BILINGUAL):
                    from src.v6.engines.workflow_builder import build_tts_workflow
                    # Map task type to language/track for TrackManager routing
                    tts_lang_map = {
                        TaskType.TTS: "auto",
                        TaskType.TTS_ZH: "zh",
                        TaskType.TTS_EN: "en",
                        TaskType.TTS_BILINGUAL: "auto",
                    }
                    tts_track_map = {
                        TaskType.TTS: None,
                        TaskType.TTS_ZH: "zh",
                        TaskType.TTS_EN: "en",
                        TaskType.TTS_BILINGUAL: "bilingual",
                    }
                    lang = tts_lang_map[task.type]
                    track = tts_track_map[task.type]
                    workflow = build_tts_workflow(
                        text=task.params.get("text", ""),
                        voice=task.params.get("voice", "default"),
                        speed=task.params.get("speed", 1.0),
                        backend=track or "auto",
                        language=lang,
                        task_id=task.task_id,
                        reference_audio=task.params.get("reference_audio", ""),
                    )
                    logger.info("Auto-built TTS workflow for task %s (lang=%s, track=%s)", task.task_id, lang, track)
                elif task.type == TaskType.IMAGE_TO_3D:
                    extra = task.params.get("extra", {})
                    extra_engine = extra.get("engine", "")
                    extra_mode = extra.get("mode", "")

                    if extra_mode == "flux_trellis":
                        # WFB-05: FLUX -> TRELLIS full pipeline
                        from src.v6.engines.workflow_builder import build_flux_trellis_full_workflow
                        workflow = build_flux_trellis_full_workflow(
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            flux_steps=task.params.get("flux_steps", 20),
                            flux_cfg=task.params.get("flux_cfg", 3.5),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            trellis_resolution=task.params.get("trellis_resolution", 1024),
                            trellis_steps=task.params.get("trellis_steps", 25),
                            trellis_cfg=task.params.get("trellis_cfg", 7.5),
                            shape_guidance=task.params.get("shape_guidance", 0.5),
                            texture_resolution=task.params.get("texture_resolution", 1024),
                            pbr_channels=task.params.get("pbr_channels", "full"),
                            output_format=task.params.get("output_format", "glb"),
                            seed=task.params.get("seed"),
                        )
                        logger.info("Auto-built FLUX+TRELLIS full pipeline workflow for task %s", task.task_id)
                    elif extra_engine == "trellis":
                        # WFB-04: TRELLIS image-to-3D
                        from src.v6.engines.workflow_builder import build_trellis_image_to_3d_workflow
                        input_image = task.params.get("input_image") or task.params.get("image", "")
                        if not input_image:
                            logger.error("TRELLIS IMAGE_TO_3D requires 'input_image' param, task %s", task.task_id)
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="TRELLIS IMAGE_TO_3D requires 'input_image' param",
                            )
                            return
                        workflow = build_trellis_image_to_3d_workflow(
                            image_name=input_image,
                            resolution=task.params.get("resolution", 1024),
                            steps=task.params.get("steps", 25),
                            cfg_scale=task.params.get("cfg_scale", 7.5),
                            shape_guidance=task.params.get("shape_guidance", 0.5),
                            texture_resolution=task.params.get("texture_resolution", 1024),
                            pbr_channels=task.params.get("pbr_channels", "full"),
                            remove_background=task.params.get("remove_background", True),
                            output_format=task.params.get("output_format", "glb"),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", "trellis_3d"),
                        )
                        logger.info("Auto-built TRELLIS image-to-3D workflow for task %s", task.task_id)
                    else:
                        # Default: Hunyuan3D (unchanged)
                        from src.v6.engines.workflow_builder import build_hunyuan3d_workflow
                        input_image = task.params.get("input_image") or task.params.get("image", "")
                        if not input_image:
                            logger.error("IMAGE_TO_3D requires 'input_image' param, task %s", task.task_id)
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="IMAGE_TO_3D requires 'input_image' param",
                            )
                            return
                        workflow = build_hunyuan3d_workflow(
                            input_image=input_image,
                            output_path=task.params.get("output_path", ""),
                            model=task.params.get("model", "full"),
                            device=task.params.get("device", "cuda:0"),
                            steps=task.params.get("steps", 50),
                            seed=task.params.get("seed"),
                            model_dir=task.params.get("model_dir", ""),
                            task_id=task.task_id,
                        )
                        logger.info("Auto-built Hunyuan3D workflow for task %s", task.task_id)
                elif task.type == TaskType.IMAGE_DRAW:
                    # -- IMAGE_DRAW routing: params.extra.mode first, then model-based fallback --
                    extra = task.params.get("extra", {})
                    extra_mode = extra.get("mode", "") if isinstance(extra, dict) else ""
                    model = task.params.get("model", "")

                    if extra_mode == "ipadapter":
                        # IP-Adapter character consistency via params.extra.mode
                        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
                        ref_img = task.params.get("reference_image", "")
                        if not ref_img:
                            logger.error("IMAGE_DRAW ipadapter requires 'reference_image' param, task %s", task.task_id)
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="IMAGE_DRAW ipadapter requires 'reference_image' param",
                            )
                            return
                        workflow = build_flux_ipadapter_workflow(
                            prompt=task.params.get("prompt", ""),
                            reference_image=ref_img,
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            weight=task.params.get("weight", 0.8),
                            start_percent=task.params.get("start_percent", 0.0),
                            end_percent=task.params.get("end_percent", 0.8),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", "flux-ipadapter"),
                        )
                        logger.info("Auto-built IP-Adapter workflow for IMAGE_DRAW task %s (extra.mode=ipadapter)", task.task_id)

                    elif extra_mode == "pulid":
                        # PuLID character consistency via params.extra.mode
                        from src.v6.engines.workflow_builder import build_pulid_flux_workflow
                        ref_img = task.params.get("image", "") or task.params.get("reference_image", "")
                        if not ref_img:
                            logger.error("IMAGE_DRAW pulid requires 'image' or 'reference_image' param, task %s", task.task_id)
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="IMAGE_DRAW pulid requires 'image' or 'reference_image' param",
                            )
                            return
                        workflow = build_pulid_flux_workflow(
                            image_name=ref_img,
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            weight=task.params.get("weight", 1.0),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", "pulid_flux"),
                        )
                        logger.info("Auto-built PuLID workflow for IMAGE_DRAW task %s (extra.mode=pulid)", task.task_id)

                    elif extra_mode == "instantid":
                        # InstantID character consistency (reuses IP-Adapter infrastructure)
                        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
                        ref_img = task.params.get("reference_image", "")
                        if not ref_img:
                            logger.error("IMAGE_DRAW instantid requires 'reference_image' param, task %s", task.task_id)
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="IMAGE_DRAW instantid requires 'reference_image' param",
                            )
                            return
                        workflow = build_flux_ipadapter_workflow(
                            prompt=task.params.get("prompt", ""),
                            reference_image=ref_img,
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            weight=task.params.get("weight", 0.8),
                            start_percent=task.params.get("start_percent", 0.0),
                            end_percent=task.params.get("end_percent", 0.8),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", "flux-instantid"),
                        )
                        logger.info("Auto-built InstantID workflow for IMAGE_DRAW task %s (extra.mode=instantid, reuses IP-Adapter)", task.task_id)

                    elif model == "flux-dev-ipa":
                        # Legacy: model="flux-dev-ipa" backward compatibility
                        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
                        ref_img = task.params.get("reference_image", "")
                        if not ref_img:
                            logger.error("flux-dev-ipa requires 'reference_image' param, task %s", task.task_id)
                            await store.update(task.task_id, status=TaskStatus.FAILED, error="flux-dev-ipa requires 'reference_image' param")
                            return
                        workflow = build_flux_ipadapter_workflow(
                            prompt=task.params.get("prompt", ""),
                            reference_image=ref_img,
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            weight=task.params.get("weight", 0.8),
                            start_percent=task.params.get("start_percent", 0.0),
                            end_percent=task.params.get("end_percent", 0.8),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", "flux-ipadapter"),
                        )
                        logger.info("Auto-built FLUX Dev + IP-Adapter workflow for task %s (legacy model=flux-dev-ipa)", task.task_id)

                    elif model == "flux-dev":
                        # Legacy: model="flux-dev"
                        from src.v6.engines.workflow_builder import build_flux_dev_workflow
                        workflow = build_flux_dev_workflow(
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            seed=task.params.get("seed"),
                        )
                        logger.info("Auto-built FLUX Dev workflow for task %s (model=flux-dev)", task.task_id)

                    else:
                        # Default: FLUX Dev FP8 for IMAGE_DRAW
                        from src.v6.engines.workflow_builder import build_flux_dev_workflow
                        workflow = build_flux_dev_workflow(
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            seed=task.params.get("seed"),
                        )
                        logger.info("Auto-built FLUX Dev workflow for IMAGE_DRAW task %s (default)", task.task_id)

                elif task.type == TaskType.IMAGE_PULID:
                    from src.v6.engines.workflow_builder import build_pulid_flux_workflow
                    ref_img = task.params.get("image", "") or task.params.get("reference_image", "")
                    if not ref_img:
                        logger.error("IMAGE_PULID requires 'image' param, task %s", task.task_id)
                        await store.update(task.task_id, status=TaskStatus.FAILED, error="IMAGE_PULID requires 'image' param")
                        return
                    workflow = build_pulid_flux_workflow(
                        image_name=ref_img,
                        prompt=task.params.get("prompt", ""),
                        negative_prompt=task.params.get("negative_prompt", ""),
                        width=task.params.get("width", 1024),
                        height=task.params.get("height", 1024),
                        steps=task.params.get("steps", 28),
                        cfg_scale=task.params.get("cfg_scale", 3.5),
                        weight=task.params.get("weight", 1.0),
                        seed=task.params.get("seed"),
                        filename_prefix=task.params.get("filename_prefix", "pulid_flux"),
                    )
                    logger.info("Auto-built PuLID FLUX workflow for task %s", task.task_id)
                elif task.type == TaskType.CONTROLNET_DEPTH:
                    from src.v6.engines.workflow_builder import build_controlnet_depth_workflow
                    src_img = task.params.get("image", "")
                    depth_img = task.params.get("depth_image", "") or task.params.get("depth_image_name", "")
                    if not src_img or not depth_img:
                        logger.error("CONTROLNET_DEPTH requires 'image' and 'depth_image' params, task %s", task.task_id)
                        await store.update(task.task_id, status=TaskStatus.FAILED, error="CONTROLNET_DEPTH requires 'image' and 'depth_image' params")
                        return
                    workflow = build_controlnet_depth_workflow(
                        image_name=src_img,
                        depth_image_name=depth_img,
                        prompt=task.params.get("prompt", ""),
                        negative_prompt=task.params.get("negative_prompt", ""),
                        width=task.params.get("width", 1024),
                        height=task.params.get("height", 1024),
                        steps=task.params.get("steps", 28),
                        cfg_scale=task.params.get("cfg_scale", 3.5),
                        strength=task.params.get("strength", 1.0),
                        seed=task.params.get("seed"),
                        filename_prefix=task.params.get("filename_prefix", "controlnet_depth"),
                    )
                    logger.info("Auto-built ControlNet Depth workflow for task %s", task.task_id)
                elif task.type == TaskType.WAN_I2V:
                    from src.v6.engines.workflow_builder import build_wan21_i2v_dual_stage_workflow
                    src_img = task.params.get("image", "")
                    if not src_img:
                        logger.error("WAN_I2V requires 'image' param, task %s", task.task_id)
                        await store.update(task.task_id, status=TaskStatus.FAILED, error="WAN_I2V requires 'image' param")
                        return
                    workflow = build_wan21_i2v_dual_stage_workflow(
                        image_name=src_img,
                        prompt=task.params.get("prompt", ""),
                        width=task.params.get("width", 832),
                        height=task.params.get("height", 480),
                        length=task.params.get("length", 81),
                        steps=task.params.get("steps", 20),
                        cfg=task.params.get("cfg", 3.5),
                        shift=task.params.get("shift", 8.0),
                        high_noise_end=task.params.get("high_noise_end", 10.0),
                        seed=task.params.get("seed"),
                        filename_prefix=task.params.get("filename_prefix", "wan_i2v"),
                    )
                    logger.info("Auto-built Wan 2.2 I2V dual-stage workflow for task %s", task.task_id)
                elif task.type == TaskType.VIDEO_FINAL or task.type == TaskType.VIDEO_PREVIEW:
                    extra_mode = task.params.get("extra", {}).get("mode", "")
                    if extra_mode == "lip_sync":
                        # -- Lip Sync via LatentSync --
                        from src.v6.engines.workflow_builder import build_lipsync_workflow
                        video_input = task.params.get("video", "")
                        audio_input = task.params.get("audio_input", "")
                        if not video_input or not audio_input:
                            logger.error(
                                "Lip sync requires 'video' and 'audio_input' params, task %s",
                                task.task_id,
                            )
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="Lip sync requires 'video' and 'audio_input' params",
                            )
                            return
                        workflow = build_lipsync_workflow(
                            video_input=video_input,
                            audio_input=audio_input,
                            seed=task.params.get("seed"),
                            lips_expression=task.params.get("lips_expression", 1.5),
                            inference_steps=task.params.get("inference_steps", 20),
                            filename_prefix=task.params.get("filename_prefix", f"lipsync_{task.task_id}"),
                        )
                        logger.info("Auto-built LipSync workflow for task %s", task.task_id)
                    else:
                        # -- VIDEO_FINAL/VIDEO_PREVIEW = wan_i2v (default) --
                        from src.v6.engines.workflow_builder import build_wan21_i2v_dual_stage_workflow
                        src_img = task.params.get("image", "")
                        if not src_img:
                            logger.error("VIDEO_FINAL/VIDEO_PREVIEW requires 'image' param, task %s", task.task_id)
                            await store.update(task.task_id, status=TaskStatus.FAILED, error="VIDEO_FINAL/VIDEO_PREVIEW requires 'image' param")
                            return
                        workflow = build_wan21_i2v_dual_stage_workflow(
                            image_name=src_img,
                            prompt=task.params.get("prompt", ""),
                            width=task.params.get("width", 832),
                            height=task.params.get("height", 480),
                            length=task.params.get("length", 81),
                            steps=task.params.get("steps", 20),
                            cfg=task.params.get("cfg", 3.5),
                            shift=task.params.get("shift", 8.0),
                            high_noise_end=task.params.get("high_noise_end", 10.0),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", f"{task.type.value}_{task.task_id}"),
                        )
                        logger.info("Auto-built VIDEO_FINAL/VIDEO_PREVIEW workflow (=wan_i2v) for task %s", task.task_id)
                elif task.type == TaskType.UPSCALE:
                    extra_mode = task.params.get("extra", {}).get("mode", "")
                    if extra_mode == "frame_interp":
                        # -- Frame Interpolation via RIFE --
                        from src.v6.engines.workflow_builder import build_frame_interpolate_workflow
                        video_input = task.params.get("video", "")
                        if not video_input:
                            logger.error(
                                "Frame interpolation requires 'video' param, task %s",
                                task.task_id,
                            )
                            await store.update(
                                task.task_id,
                                status=TaskStatus.FAILED,
                                error="Frame interpolation requires 'video' param",
                            )
                            return
                        workflow = build_frame_interpolate_workflow(
                            video_input=video_input,
                            interpolation_factor=task.params.get("interpolation_factor", 2),
                            ckpt_name=task.params.get("ckpt_name", "rife49.pth"),
                            output_fps=task.params.get("output_fps"),
                            seed=task.params.get("seed"),
                            filename_prefix=task.params.get("filename_prefix", f"frame_interp_{task.task_id}"),
                        )
                        logger.info("Auto-built Frame Interpolation workflow for task %s", task.task_id)
                    else:
                        # -- UPSCALE = image upscale (default) --
                        from src.v6.engines.workflow_builder import build_upscale_workflow
                        src_img = task.params.get("image", "")
                        if not src_img:
                            logger.error("UPSCALE requires 'image' param, task %s", task.task_id)
                            await store.update(task.task_id, status=TaskStatus.FAILED, error="UPSCALE requires 'image' param")
                            return
                        workflow = build_upscale_workflow(
                            image_name=src_img,
                            upscale_model_name=task.params.get("upscale_model_name", "4x-UltraSharp.pth"),
                            filename_prefix=task.params.get("filename_prefix", "upscaled"),
                        )
                        logger.info("Auto-built Upscale workflow for task %s", task.task_id)
                elif task.type == TaskType.FACE_RESTORE:
                    from src.v6.engines.workflow_builder import build_face_restore_workflow
                    src_img = task.params.get("image", "")
                    if not src_img:
                        logger.error("FACE_RESTORE requires 'image' param, task %s", task.task_id)
                        await store.update(task.task_id, status=TaskStatus.FAILED, error="FACE_RESTORE requires 'image' param")
                        return
                    workflow = build_face_restore_workflow(
                        image_name=src_img,
                        model_name=task.params.get("model_name", "4x-UltraSharp.pth"),
                        filename_prefix=task.params.get("filename_prefix", "face_restored"),
                    )
                    logger.info("Auto-built Face Restore workflow for task %s", task.task_id)
                elif task.type == TaskType.MUSIC or task.type == TaskType.SFX:
                    # v1.5: ACE-Step music generation migrated to Node-layer ComfyUI
                    # workflow (src/routes/v1/ace/generate.ts). gold-team no longer
                    # builds MUSIC/SFX workflows — reject so callers know to use
                    # /api/v1/ace/generate instead.
                    logger.error(
                        "MUSIC/SFX task %s rejected — gold-team no longer handles music "
                        "generation. Use POST /api/v1/ace/generate (ComfyUI workflow) instead.",
                        task.task_id,
                    )
                    await store.update(
                        task.task_id,
                        status=TaskStatus.FAILED,
                        error="MUSIC/SFX not supported by gold-team since v1.5; use /api/v1/ace/generate",
                    )
                    return
                elif task.type == TaskType.IMAGE_TO_3D_MV:
                    # Hunyuan3D-2mv multiview image-to-3D
                    images = task.params.get("images", [])
                    if not images:
                        front = task.params.get("image", "") or task.params.get("front_image", "")
                        if not front:
                            logger.error("IMAGE_TO_3D_MV requires 'images' or 'image' param, task %s", task.task_id)
                            await store.update(task.task_id, status=TaskStatus.FAILED, error="IMAGE_TO_3D_MV requires 'images' or 'image' param")
                            return
                        images = [front]
                    workflow = {
                        "front_image": images[0] if images else "",
                        "left_image": images[1] if len(images) > 1 else "",
                        "back_image": images[2] if len(images) > 2 else "",
                        "right_image": images[3] if len(images) > 3 else "",
                        **{k: v for k, v in task.params.items() if k not in ("images", "image")},
                    }
                    logger.info("Auto-built Hunyuan3D-2mv workflow for task %s", task.task_id)
                elif task.type == TaskType.IMAGE_REFINE:
                    # Image refine — use img2img with ControlNet (placeholder: txt2img for now)
                    from src.v6.engines.workflow_builder import build_image_refine_workflow
                    src_img = task.params.get("image", "")
                    if not src_img:
                        logger.error("IMAGE_REFINE requires 'image' param, task %s", task.task_id)
                        await store.update(task.task_id, status=TaskStatus.FAILED, error="IMAGE_REFINE requires 'image' param")
                        return
                    workflow = build_image_refine_workflow(
                        image_name=src_img,
                        prompt=task.params.get("prompt", ""),
                        negative_prompt=task.params.get("negative_prompt", ""),
                        strength=task.params.get("strength", 0.5),
                        steps=task.params.get("steps", 28),
                        cfg_scale=task.params.get("cfg_scale", 3.5),
                        seed=task.params.get("seed"),
                        filename_prefix=task.params.get("filename_prefix", "refined"),
                    )
                    logger.info("Auto-built Image Refine workflow for task %s", task.task_id)
                elif task.type == TaskType.COLOR_GRADE:
                    # ── COLOR_GRADE: CPU-only video color grading (ffmpeg + LUT) ──
                    # The ColorGradeEngine consumes task.params directly as its
                    # workflow (input_path, output_path, mode, preset/lut/cdl
                    # values, contrast, brightness, saturation_boost,
                    # copy_audio). No ComfyUI workflow needed — the engine
                    # spawns ffmpeg and returns a video artifact, which the
                    # common media collector below handles correctly.
                    workflow = dict(task.params)
                    logger.info(
                        "Auto-built color-grade workflow for task %s (mode=%s)",
                        task.task_id, task.params.get("mode", "preset"),
                    )
                elif task.type == TaskType.SHOT_ANALYSIS:
                    # ── SHOT_ANALYSIS: per-shot cinematic deconstruction ──
                    # Self-contained execution path: the workflow's real output is a
                    # JSON sidecar written by ShotJSONMerge to save_dir/shot_XXX.json
                    # (NOT a ComfyUI media artifact). We submit→poll→read JSON→update
                    # store→return here, so we never fall through to the common media
                    # collector below (which would mis-parse it as image/video/audio).
                    await self._execute_shot_analysis(task, engine)
                    return
                else:
                    # Default: FLUX Dev FP8 for image_draw (most common)
                    # Only use txt2img (CheckpointLoaderSimple) when an explicit SDXL model is given
                    explicit_model = task.params.get("model", "")
                    if explicit_model and "flux" not in explicit_model.lower() and "sd" in explicit_model.lower():
                        from src.v6.engines.workflow_builder import build_txt2img_workflow
                        workflow = build_txt2img_workflow(
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 20),
                            cfg_scale=task.params.get("cfg_scale", 7.5),
                            seed=task.params.get("seed"),
                            checkpoint=explicit_model,
                        )
                        logger.info("Auto-built txt2img workflow for task %s (model=%s)", task.task_id, explicit_model)
                    else:
                        from src.v6.engines.workflow_builder import build_flux_dev_workflow
                        workflow = build_flux_dev_workflow(
                            prompt=task.params.get("prompt", ""),
                            negative_prompt=task.params.get("negative_prompt", ""),
                            width=task.params.get("width", 1024),
                            height=task.params.get("height", 1024),
                            steps=task.params.get("steps", 28),
                            cfg_scale=task.params.get("cfg_scale", 3.5),
                            seed=task.params.get("seed"),
                        )
                        logger.info("Auto-built FLUX Dev workflow for task %s (default)", task.task_id)
            engine_params = {"task_id": task.task_id, "type": task.type.value}

            engine_job_id = await engine.submit(workflow, engine_params)

            # Poll until done
            while self._running:
                result = await engine.poll(engine_job_id)
                status = result.get("status", "running")
                progress = result.get("progress", 0.0)

                await store.update(task.task_id, progress=progress)

                if status == "completed":
                    output_data = await engine.get_output(engine_job_id)

                    # Download artifacts from engine URLs to local storage
                    output_data = await self._download_artifacts(task.task_id, output_data)

                    outputs = self._build_task_outputs(task, output_data)
                    metadata = TaskMetadata(
                        seed=task.params.get("seed", 42),
                        cost_usd=0.0,
                        inference_time_sec=3.0,
                        gpu_memory_peak_gb=8.0,
                        model_name=engine.name,
                    )
                    await store.update(
                        task.task_id,
                        status=TaskStatus.COMPLETED,
                        outputs=outputs,
                        metadata=metadata,
                        progress=100.0,
                    )
                    logger.info("Task %s completed via %s", task.task_id, engine.engine_id)
                    break

                if status == "failed":
                    error_msg = result.get("error", "Engine execution failed")
                    await store.update(
                        task.task_id,
                        status=TaskStatus.FAILED,
                        error=error_msg,
                    )
                    logger.error("Task %s failed: %s", task.task_id, error_msg)
                    break

                await asyncio.sleep(0.5)

            # Send callback if configured
            if task.callback_url:
                await send_callback(task, task.callback_url, task.callback_secret)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Task %s execution error", task.task_id)
            await store.update(
                task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )
            if task.callback_url:
                await send_callback(task, task.callback_url, task.callback_secret)

    async def _execute_shot_analysis(self, task: GenerationTask, engine: BaseEngine) -> None:
        """Run a SHOT_ANALYSIS task end-to-end on the given engine.

        Builds the ComfyUI shot-analysis workflow (geometry + optional semantic/
        subject layers), submits via the engine, polls to completion, then reads
        the JSON sidecar that ShotJSONMerge wrote to ``save_dir/shot_XXX.json``
        and stores it as the task output.

        Unlike the common media-output path, this does NOT call
        ``engine.get_output()`` — shot-analysis produces a JSON file on disk, not
        a ComfyUI media artifact, so the media collector cannot find it.
        """
        from src.v6.engines.workflow_builder import build_shot_analysis_workflow

        store = get_task_store()

        video = task.params.get("video", "")
        shot_id = task.params.get("shot_id")
        start_sec = task.params.get("start_sec")
        end_sec = task.params.get("end_sec")
        if not video or shot_id is None or start_sec is None or end_sec is None:
            missing = [
                k for k, v in (
                    ("video", video), ("shot_id", shot_id),
                    ("start_sec", start_sec), ("end_sec", end_sec),
                ) if not v and v != 0
            ]
            msg = f"SHOT_ANALYSIS requires params: {', '.join(missing)}"
            logger.error("%s, task %s", msg, task.task_id)
            await store.update(task.task_id, status=TaskStatus.FAILED, error=msg)
            if task.callback_url:
                await send_callback(task, task.callback_url, task.callback_secret)
            return

        save_dir = task.params.get("save_dir", "/mnt/agents/output/gpu1/shot_analysis")
        shot = {"id": shot_id, "start_sec": start_sec, "end_sec": end_sec}

        try:
            workflow = build_shot_analysis_workflow(
                shot=shot,
                video=video,
                semantic=task.params.get("semantic", True),
                subject=task.params.get("subject", False),
                grid_n=task.params.get("grid_n", 20),
                fps=task.params.get("fps", 24.0),
                save_dir=save_dir,
                qwen_model=task.params.get("qwen_model", "Qwen3-VL-8B-Instruct"),
                quant=task.params.get("quant", "8-bit (Balanced)"),
            )
        except Exception as e:
            logger.exception("SHOT_ANALYSIS workflow build failed, task %s", task.task_id)
            await store.update(task.task_id, status=TaskStatus.FAILED, error=f"workflow build failed: {e}")
            if task.callback_url:
                await send_callback(task, task.callback_url, task.callback_secret)
            return

        logger.info(
            "Auto-built shot-analysis workflow for task %s (shot_id=%s, save_dir=%s)",
            task.task_id, shot_id, save_dir,
        )

        engine_params = {"task_id": task.task_id, "type": task.type.value}

        try:
            engine_job_id = await engine.submit(workflow, engine_params)
        except Exception as e:
            logger.exception("SHOT_ANALYSIS submit failed, task %s", task.task_id)
            await store.update(task.task_id, status=TaskStatus.FAILED, error=f"submit failed: {e}")
            if task.callback_url:
                await send_callback(task, task.callback_url, task.callback_secret)
            return

        shot_json_path = os.path.join(save_dir, f"shot_{int(shot_id):03d}.json")

        # Poll until terminal state
        try:
            while self._running:
                result = await engine.poll(engine_job_id)
                status = result.get("status", "running")
                progress = result.get("progress", 0.0)
                await store.update(task.task_id, progress=progress)

                if status == "completed":
                    # ShotJSONMerge should have written the sidecar JSON.
                    try:
                        with open(shot_json_path, "r", encoding="utf-8") as f:
                            shot_data = json.load(f)
                    except Exception as e:
                        logger.error(
                            "SHOT_ANALYSIS task %s: ComfyUI completed but shot JSON "
                            "unreadable at %s: %s",
                            task.task_id, shot_json_path, e,
                        )
                        await store.update(
                            task.task_id,
                            status=TaskStatus.FAILED,
                            error=f"shot JSON unreadable at {shot_json_path}: {e}",
                        )
                        if task.callback_url:
                            await send_callback(task, task.callback_url, task.callback_secret)
                        return

                    # Carry the parsed shot payload through TaskOutputs.extra so
                    # callers polling /tasks/{id} receive the full deconstruction
                    # inline (not just a filesystem path).
                    outputs = TaskOutputs(analysis=shot_json_path, payload=shot_data)
                    metadata = TaskMetadata(
                        seed=task.params.get("seed", 42),
                        cost_usd=0.0,
                        inference_time_sec=0.0,
                        gpu_memory_peak_gb=0.0,
                        model_name=engine.name,
                    )
                    await store.update(
                        task.task_id,
                        status=TaskStatus.COMPLETED,
                        outputs=outputs,
                        metadata=metadata,
                        progress=100.0,
                    )
                    logger.info(
                        "Task %s (shot-analysis) completed via %s — shot JSON at %s",
                        task.task_id, engine.engine_id, shot_json_path,
                    )
                    break

                if status == "failed":
                    error_msg = result.get("error", "Engine execution failed")
                    await store.update(
                        task.task_id,
                        status=TaskStatus.FAILED,
                        error=error_msg,
                    )
                    logger.error("Task %s (shot-analysis) failed: %s", task.task_id, error_msg)
                    break

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Task %s (shot-analysis) execution error", task.task_id)
            await store.update(task.task_id, status=TaskStatus.FAILED, error=str(e))

        if task.callback_url:
            await send_callback(task, task.callback_url, task.callback_secret)

    async def _download_artifacts(self, task_id: str, output_data: dict[str, Any]) -> dict[str, Any]:
        """Download engine output artifacts to local storage."""
        import httpx
        import os

        artifacts = output_data.get("outputs", [])
        if not artifacts:
            return output_data

        output_dir = os.path.join("/mnt/agents/output", task_id)
        os.makedirs(output_dir, exist_ok=True)

        downloaded = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for a in artifacts:
                url = a.get("url", "")
                if not url:
                    downloaded.append(a)
                    continue

                # Determine local filename
                a_type = a.get("type", "image")
                fmt = a.get("format", "png")
                ext = f".{fmt}" if not fmt.startswith(".") else fmt
                filename = f"{task_id}_{a_type}{ext}"
                local_path = os.path.join(output_dir, filename)

                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    logger.info("Downloaded %s → %s (%d bytes)", url, local_path, len(resp.content))
                    # Replace URL with local path
                    downloaded.append({**a, "url": local_path, "local_path": local_path})
                except Exception as e:
                    logger.warning("Failed to download %s: %s", url, e)
                    downloaded.append(a)

        return {"outputs": downloaded}

    def _resolve_engine(self, engine_id: str, task: GenerationTask) -> Optional[BaseEngine]:
        """Resolve engine by ID, preferring real engines over mock.

        For dedicated-engine task types, always prefer the specialized engine
        regardless of what engine_id the router assigned.
        """
        # Dedicated engine routing — specialized engines override router
        # TRELLIS bypass: IMAGE_TO_3D + trellis/flux_trellis goes to comfyui-primary
        from src.v6.engine.router import DEDICATED_ENGINES
        extra = task.params.get("extra", {})
        is_trellis = (
            task.type == TaskType.IMAGE_TO_3D
            and (extra.get("engine") == "trellis" or extra.get("mode") == "flux_trellis")
        )
        dedicated_id = None if is_trellis else DEDICATED_ENGINES.get(task.type)
        if dedicated_id:
            dedicated_engine = self._engines.get(dedicated_id)
            if dedicated_engine:
                return dedicated_engine
            logger.warning("Dedicated engine '%s' for type %s not available, falling back",
                           dedicated_id, task.type.value)

        # Direct match
        if engine_id in self._engines:
            return self._engines[engine_id]

        # Cloud engine IDs (cloud-jimeng, cloud-kling, cloud-seedance)
        if engine_id and engine_id.startswith("cloud-"):
            provider = engine_id.replace("cloud-", "")
            for eid, engine in self._engines.items():
                if eid == provider or eid == f"cloud-{provider}":
                    if hasattr(engine, 'is_configured') and engine.is_configured:
                        return engine
            logger.warning("Cloud engine '%s' not configured, falling back to mock", engine_id)

        # For local/unset engine_id, prefer comfyui-primary > comfyui-local > comfyui-auxiliary over mock
        if engine_id is None or engine_id in ("local", "local-comfyui", "local-comfyui-mock"):
            for eid in ("comfyui-primary", "comfyui-local", "comfyui-auxiliary"):
                comfyui = self._engines.get(eid)
                if comfyui and comfyui.status().value == "online":
                    return comfyui
            return self._engines.get("mock")

        # Fallback to mock
        return self._engines.get("mock")

    def _build_task_outputs(self, task: GenerationTask, output_data: dict[str, Any]) -> TaskOutputs:
        """Build TaskOutputs from engine output data."""
        # If engine returned structured outputs with URLs, map them
        artifacts = output_data.get("outputs", [])

        video = None
        image = None
        audio = None
        thumbnail = None

        for a in artifacts:
            url = a.get("url", "")
            a_type = a.get("type", "")
            if a_type == "video" and not video:
                video = url
            elif a_type == "image" and not image:
                image = url
            elif a_type == "audio" and not audio:
                audio = url

        # Use first image as thumbnail if not set
        if not thumbnail and image:
            thumbnail = image

        # Fallback to template paths
        if not any([video, image, audio]):
            fields = _TASK_OUTPUT_FIELDS.get(task.type, {"image": "output.png"})
            paths = {
                k: f"/mnt/agents/output/{task.task_id}/{v}"
                for k, v in fields.items()
            }
            return TaskOutputs(**paths)

        return TaskOutputs(video=video, image=image, audio=audio, thumbnail=thumbnail)


# Singleton
_executor: Optional[TaskExecutor] = None


def get_executor() -> TaskExecutor:
    global _executor
    if _executor is None:
        _executor = TaskExecutor()
    return _executor
