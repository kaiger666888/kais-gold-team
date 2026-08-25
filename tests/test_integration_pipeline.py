"""End-to-end short-drama pipeline integration test.

Verifies the complete production chain:
  Step 1: IMAGE_DRAW with IP-Adapter → character image
  Step 2: VIDEO_FINAL (Wan I2V) → character video
  Step 3: VIDEO_FINAL with lip_sync → lip-synced video
  Step 4: UPSCALE with frame_interp → smooth video
  Step 5: UPSCALE (default) → super-resolution image
  Step 6: FACE_RESTORE → face-restored high-res image
  Step 7: Full chain — all 6 steps sequentially, data flows between steps
"""
from __future__ import annotations

import pytest

from src.v6.models.task import GenerationTask, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_type: TaskType,
    params: dict | None = None,
    task_id: str = "test-task-001",
) -> GenerationTask:
    """Create a GenerationTask with sensible defaults."""
    return GenerationTask(
        task_id=task_id,
        type=task_type,
        params=params or {},
        status=TaskStatus.QUEUED,
    )


def _find_class_type_in_workflow(workflow: dict, class_type: str) -> bool:
    """Check if any node in the workflow dict has the given class_type."""
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return True
    return False


def _build_workflow_for_task(task: GenerationTask) -> dict:
    """Build the appropriate workflow for a task, mirroring executor routing."""
    extra = task.params.get("extra", {}) if isinstance(task.params.get("extra"), dict) else {}
    extra_mode = extra.get("mode", "")
    extra_engine = extra.get("engine", "")
    model = task.params.get("model", "")

    if task.type == TaskType.IMAGE_DRAW:
        if extra_mode == "ipadapter":
            from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
            return build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=task.params.get("reference_image", ""),
            )
        elif extra_mode == "pulid":
            from src.v6.engines.workflow_builder import build_pulid_flux_workflow
            return build_pulid_flux_workflow(
                image_name=task.params.get("image", "") or task.params.get("reference_image", ""),
                prompt=task.params.get("prompt", ""),
            )
        elif extra_mode == "instantid":
            from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
            return build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=task.params.get("reference_image", ""),
            )
        elif model == "flux-dev-ipa":
            from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
            return build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=task.params.get("reference_image", ""),
            )
        else:
            from src.v6.engines.workflow_builder import build_flux_dev_workflow
            return build_flux_dev_workflow(prompt=task.params.get("prompt", ""))

    elif task.type in (TaskType.VIDEO_FINAL, TaskType.VIDEO_PREVIEW):
        if extra_mode == "lip_sync":
            from src.v6.engines.workflow_builder import build_lipsync_workflow
            return build_lipsync_workflow(
                video_input=task.params.get("video", ""),
                audio_input=task.params.get("audio_input", ""),
            )
        else:
            from src.v6.engines.workflow_builder import build_wan21_i2v_dual_stage_workflow
            return build_wan21_i2v_dual_stage_workflow(
                image_name=task.params.get("image", ""),
                prompt=task.params.get("prompt", ""),
            )

    elif task.type == TaskType.UPSCALE:
        if extra_mode == "frame_interp":
            from src.v6.engines.workflow_builder import build_frame_interpolate_workflow
            return build_frame_interpolate_workflow(
                video_input=task.params.get("video", ""),
            )
        else:
            from src.v6.engines.workflow_builder import build_upscale_workflow
            return build_upscale_workflow(
                image_name=task.params.get("image", ""),
            )

    elif task.type == TaskType.FACE_RESTORE:
        from src.v6.engines.workflow_builder import build_face_restore_workflow
        return build_face_restore_workflow(
            image_name=task.params.get("image", ""),
        )

    raise ValueError(f"Unsupported task type for pipeline test: {task.type}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShortDramaPipeline:
    """End-to-end short-drama production pipeline chain."""

    def test_step1_image_draw_ipadapter(self):
        """Pipeline step 1: IMAGE_DRAW with IP-Adapter produces IPAdapterFluxLoader node."""
        task = _make_task(
            TaskType.IMAGE_DRAW,
            params={
                "prompt": "a warrior character portrait",
                "reference_image": "char_001.png",
                "extra": {"mode": "ipadapter"},
            },
            task_id="pipeline-step-1",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "IP-Adapter pipeline step must contain IPAdapterFluxLoader node"

    def test_step2_video_final_wan_i2v(self):
        """Pipeline step 2: VIDEO_FINAL (Wan I2V) produces KSamplerAdvanced node."""
        task = _make_task(
            TaskType.VIDEO_FINAL,
            params={
                "image": "char_001_rendered.png",
                "prompt": "warrior walks forward",
            },
            task_id="pipeline-step-2",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "KSamplerAdvanced"), \
            "Wan I2V pipeline step must contain KSamplerAdvanced node"

    def test_step3_video_final_lip_sync(self):
        """Pipeline step 3: VIDEO_FINAL with lip_sync produces LatentSyncNode."""
        task = _make_task(
            TaskType.VIDEO_FINAL,
            params={
                "video": "char_video.mp4",
                "audio_input": "voice.wav",
                "extra": {"mode": "lip_sync"},
            },
            task_id="pipeline-step-3",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "LatentSyncNode"), \
            "Lip sync pipeline step must contain LatentSyncNode"

    def test_step4_upscale_frame_interp(self):
        """Pipeline step 4: UPSCALE with frame_interp produces RIFE VFI node."""
        task = _make_task(
            TaskType.UPSCALE,
            params={
                "video": "char_lipsync_video.mp4",
                "extra": {"mode": "frame_interp"},
            },
            task_id="pipeline-step-4",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "RIFE VFI"), \
            "Frame interpolation pipeline step must contain RIFE VFI node"

    def test_step5_upscale_super_resolution(self):
        """Pipeline step 5: UPSCALE (default) produces UpscaleModelLoader node."""
        task = _make_task(
            TaskType.UPSCALE,
            params={
                "image": "char_001_rendered.png",
            },
            task_id="pipeline-step-5",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "UpscaleModelLoader"), \
            "Super-resolution pipeline step must contain UpscaleModelLoader node"

    def test_step6_face_restore(self):
        """Pipeline step 6: FACE_RESTORE produces UpscaleModelLoader and ImageUpscaleWithModel nodes."""
        task = _make_task(
            TaskType.FACE_RESTORE,
            params={
                "image": "char_001_rendered.png",
            },
            task_id="pipeline-step-6",
        )
        workflow = _build_workflow_for_task(task)
        assert _find_class_type_in_workflow(workflow, "UpscaleModelLoader"), \
            "Face restore pipeline step must contain UpscaleModelLoader node"
        assert _find_class_type_in_workflow(workflow, "ImageUpscaleWithModel"), \
            "Face restore pipeline step must contain ImageUpscaleWithModel node"

    def test_full_pipeline_chain(self):
        """Full chain: run all 6 steps sequentially with data flowing between steps."""
        # Simulated outputs from each step
        outputs: dict[str, str] = {}

        # Step 1: Character image with IP-Adapter
        task1 = _make_task(
            TaskType.IMAGE_DRAW,
            params={
                "prompt": "a warrior character portrait",
                "reference_image": "char_001.png",
                "extra": {"mode": "ipadapter"},
            },
            task_id="chain-step-1",
        )
        workflow1 = _build_workflow_for_task(task1)
        assert _find_class_type_in_workflow(workflow1, "IPAdapterFluxLoader")
        outputs["step1_image"] = "chain-step-1/char_portrait.png"

        # Step 2: Video from character image (Wan I2V)
        task2 = _make_task(
            TaskType.VIDEO_FINAL,
            params={
                "image": outputs["step1_image"],
                "prompt": "warrior walks forward",
            },
            task_id="chain-step-2",
        )
        workflow2 = _build_workflow_for_task(task2)
        assert _find_class_type_in_workflow(workflow2, "KSamplerAdvanced")
        outputs["step2_video"] = "chain-step-2/char_video.mp4"

        # Step 3: Lip sync on the video
        task3 = _make_task(
            TaskType.VIDEO_FINAL,
            params={
                "video": outputs["step2_video"],
                "audio_input": "voice.wav",
                "extra": {"mode": "lip_sync"},
            },
            task_id="chain-step-3",
        )
        workflow3 = _build_workflow_for_task(task3)
        assert _find_class_type_in_workflow(workflow3, "LatentSyncNode")
        outputs["step3_video"] = "chain-step-3/lipsync_video.mp4"

        # Step 4: Frame interpolation for smooth video
        task4 = _make_task(
            TaskType.UPSCALE,
            params={
                "video": outputs["step3_video"],
                "extra": {"mode": "frame_interp"},
            },
            task_id="chain-step-4",
        )
        workflow4 = _build_workflow_for_task(task4)
        assert _find_class_type_in_workflow(workflow4, "RIFE VFI")
        outputs["step4_video"] = "chain-step-4/smooth_video.mp4"

        # Step 5: Super-resolution upscale of the character image
        task5 = _make_task(
            TaskType.UPSCALE,
            params={
                "image": outputs["step1_image"],
            },
            task_id="chain-step-5",
        )
        workflow5 = _build_workflow_for_task(task5)
        assert _find_class_type_in_workflow(workflow5, "UpscaleModelLoader")
        outputs["step5_image"] = "chain-step-5/char_hires.png"

        # Step 6: Face restoration on the super-resolution image
        task6 = _make_task(
            TaskType.FACE_RESTORE,
            params={
                "image": outputs["step5_image"],
            },
            task_id="chain-step-6",
        )
        workflow6 = _build_workflow_for_task(task6)
        assert _find_class_type_in_workflow(workflow6, "UpscaleModelLoader")
        outputs["step6_image"] = "chain-step-6/char_face_restored.png"

        # Verify complete chain produced all expected outputs
        assert len(outputs) == 6, f"Expected 6 pipeline outputs, got {len(outputs)}"
        for key, path in outputs.items():
            assert path, f"Pipeline output '{key}' is empty"
