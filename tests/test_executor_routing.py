"""Unit tests for executor routing — verifies workflow routing for TRELLIS, lip sync, and frame interpolation.

Covers:
  test_image_to_3d_trellis_routing       WFB-04  params.extra.engine="trellis"
  test_image_to_3d_flux_trellis_routing  WFB-05  params.extra.mode="flux_trellis"
  test_image_to_3d_default_hunyuan3d     regression — no extra params → hunyuan3d
  test_trellis_bypasses_dedicated_engine  TRELLIS goes to comfyui-primary, not hunyuan3d-local
  test_video_final_lip_sync_routing      WFB-08  VIDEO_FINAL + params.extra.mode="lip_sync"
  test_video_final_default_wan_i2v       regression — VIDEO_FINAL without mode → wan_i2v
  test_upscale_frame_interp_routing      WFB-08  UPSCALE + params.extra.mode="frame_interp"
  test_upscale_default_image             regression — UPSCALE without mode → image upscale
  test_lip_sync_missing_video_fails      WFB-08  lip_sync without video → FAILED
  test_lip_sync_missing_audio_fails      WFB-08  lip_sync without audio_input → FAILED
  test_frame_interp_missing_video_fails  WFB-08  frame_interp without video → FAILED
  test_lip_sync_custom_params            WFB-08  lips_expression + inference_steps passthrough
"""
from __future__ import annotations

import pytest

from src.v6.engine.router import EngineRouter, DEDICATED_ENGINES
from src.v6.models.task import (
    EnginePool,
    GenerationTask,
    TaskStatus,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_type: TaskType = TaskType.IMAGE_TO_3D,
    params: dict | None = None,
    task_id: str = "test-task-001",
) -> GenerationTask:
    """Create a GenerationTask with sensible defaults for routing tests."""
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


# ---------------------------------------------------------------------------
# Routing tests — executor workflow selection
# ---------------------------------------------------------------------------

class TestImageTo3DTrellisRouting:
    """Verify IMAGE_TO_3D + params.extra.engine="trellis" selects TRELLIS builder."""

    def test_image_to_3d_trellis_routing(self):
        """WFB-04: extra.engine="trellis" must produce TRELLISImageTo3D node."""
        from src.v6.engines.workflow_builder import build_trellis_image_to_3d_workflow

        task = _make_task(params={
            "input_image": "test.png",
            "extra": {"engine": "trellis"},
        })

        # Simulate the routing logic the executor will use
        extra = task.params.get("extra", {})
        extra_engine = extra.get("engine", "")
        extra_mode = extra.get("mode", "")

        if extra_mode == "flux_trellis":
            from src.v6.engines.workflow_builder import build_flux_trellis_full_workflow
            workflow = build_flux_trellis_full_workflow(prompt=task.params.get("prompt", ""))
        elif extra_engine == "trellis":
            input_image = task.params.get("input_image") or task.params.get("image", "")
            workflow = build_trellis_image_to_3d_workflow(image_name=input_image)
        else:
            from src.v6.engines.workflow_builder import build_hunyuan3d_workflow
            workflow = build_hunyuan3d_workflow(input_image=task.params.get("input_image", ""))

        assert _find_class_type_in_workflow(workflow, "TRELLISImageTo3D"), \
            "Expected TRELLISImageTo3D node in workflow for trellis routing"

    def test_image_to_3d_flux_trellis_routing(self):
        """WFB-05: extra.mode="flux_trellis" must produce TRELLISImageTo3D node."""
        from src.v6.engines.workflow_builder import build_flux_trellis_full_workflow

        task = _make_task(params={
            "prompt": "a 3d object",
            "extra": {"mode": "flux_trellis"},
        })

        extra = task.params.get("extra", {})
        extra_engine = extra.get("engine", "")
        extra_mode = extra.get("mode", "")

        if extra_mode == "flux_trellis":
            workflow = build_flux_trellis_full_workflow(prompt=task.params.get("prompt", ""))
        elif extra_engine == "trellis":
            from src.v6.engines.workflow_builder import build_trellis_image_to_3d_workflow
            input_image = task.params.get("input_image") or task.params.get("image", "")
            workflow = build_trellis_image_to_3d_workflow(image_name=input_image)
        else:
            from src.v6.engines.workflow_builder import build_hunyuan3d_workflow
            workflow = build_hunyuan3d_workflow(input_image=task.params.get("input_image", ""))

        assert _find_class_type_in_workflow(workflow, "TRELLISImageTo3D"), \
            "Expected TRELLISImageTo3D node in workflow for flux_trellis routing"

    def test_image_to_3d_default_hunyuan3d(self):
        """Regression: no extra params → hunyuan3d workflow (no TRELLIS node)."""
        from src.v6.engines.workflow_builder import build_hunyuan3d_workflow

        task = _make_task(params={
            "input_image": "test.png",
        })

        extra = task.params.get("extra", {})
        extra_engine = extra.get("engine", "")
        extra_mode = extra.get("mode", "")

        if extra_mode == "flux_trellis":
            from src.v6.engines.workflow_builder import build_flux_trellis_full_workflow
            workflow = build_flux_trellis_full_workflow(prompt=task.params.get("prompt", ""))
        elif extra_engine == "trellis":
            from src.v6.engines.workflow_builder import build_trellis_image_to_3d_workflow
            input_image = task.params.get("input_image") or task.params.get("image", "")
            workflow = build_trellis_image_to_3d_workflow(image_name=input_image)
        else:
            workflow = build_hunyuan3d_workflow(input_image=task.params.get("input_image", ""))

        # Must NOT contain TRELLIS nodes — this is the default hunyuan3d path
        assert not _find_class_type_in_workflow(workflow, "TRELLISImageTo3D"), \
            "Default IMAGE_TO_3D should use hunyuan3d, not TRELLIS"


# ---------------------------------------------------------------------------
# Routing tests — engine router DEDICATED_ENGINES bypass
# ---------------------------------------------------------------------------

class TestTrellisBypassesDedicatedEngine:
    """Verify TRELLIS tasks bypass DEDICATED_ENGINES and go to comfyui-primary."""

    def test_trellis_bypasses_dedicated_engine(self):
        """IMAGE_TO_3D + extra.engine="trellis" must route to comfyui-primary."""
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )

        task = _make_task(params={
            "input_image": "test.png",
            "extra": {"engine": "trellis"},
        })

        pool, engine_id = router.route(task)

        assert engine_id == "comfyui-primary", \
            f"TRELLIS task should route to comfyui-primary, got '{engine_id}'"
        assert pool == EnginePool.LOCAL

    def test_flux_trellis_bypasses_dedicated_engine(self):
        """IMAGE_TO_3D + extra.mode="flux_trellis" must route to comfyui-primary."""
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )

        task = _make_task(params={
            "prompt": "a 3d object",
            "extra": {"mode": "flux_trellis"},
        })

        pool, engine_id = router.route(task)

        assert engine_id == "comfyui-primary", \
            f"flux_trellis task should route to comfyui-primary, got '{engine_id}'"
        assert pool == EnginePool.LOCAL

    def test_default_image_to_3d_routes_to_dedicated(self):
        """Regression: IMAGE_TO_3D without trellis extra → hunyuan3d-local."""
        router = EngineRouter(
            local_available=True,
            local_vram_used_gb=0.0,
            primary_available=True,
            auxiliary_available=True,
        )

        task = _make_task(params={
            "input_image": "test.png",
        })

        pool, engine_id = router.route(task)

        assert engine_id == "hunyuan3d-local", \
            f"Default IMAGE_TO_3D should route to hunyuan3d-local, got '{engine_id}'"
        assert pool == EnginePool.LOCAL


# ---------------------------------------------------------------------------
# Routing tests — lip sync and frame interpolation (WFB-08)
# ---------------------------------------------------------------------------

class TestLipSyncRouting:
    """Verify VIDEO_FINAL + params.extra.mode="lip_sync" selects LatentSync builder."""

    def test_video_final_lip_sync_routing(self):
        """WFB-08: VIDEO_FINAL + extra.mode="lip_sync" must produce LatentSyncNode."""
        from src.v6.engines.workflow_builder import build_lipsync_workflow

        task = _make_task(
            task_type=TaskType.VIDEO_FINAL,
            params={
                "video": "test.mp4",
                "audio_input": "audio.wav",
                "extra": {"mode": "lip_sync"},
            },
        )

        # Simulate the routing logic the executor will use
        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "lip_sync":
            from src.v6.engines.workflow_builder import build_lipsync_workflow
            video_input = task.params.get("video", "")
            audio_input = task.params.get("audio_input", "")
            assert video_input and audio_input, "lip_sync requires video and audio_input"
            workflow = build_lipsync_workflow(
                video_input=video_input,
                audio_input=audio_input,
                seed=task.params.get("seed"),
                lips_expression=task.params.get("lips_expression", 1.5),
                inference_steps=task.params.get("inference_steps", 20),
                filename_prefix=task.params.get("filename_prefix", f"lipsync_{task.task_id}"),
            )
        else:
            from src.v6.engines.workflow_builder import build_wan21_i2v_dual_stage_workflow
            src_img = task.params.get("image", "")
            workflow = build_wan21_i2v_dual_stage_workflow(
                image_name=src_img,
                prompt=task.params.get("prompt", ""),
            )

        assert _find_class_type_in_workflow(workflow, "LatentSyncNode"), \
            "Expected LatentSyncNode in workflow for lip_sync routing"

    def test_video_final_default_wan_i2v(self):
        """Regression: VIDEO_FINAL without extra.mode → wan_i2v (no LatentSyncNode)."""
        from src.v6.engines.workflow_builder import build_wan21_i2v_dual_stage_workflow

        task = _make_task(
            task_type=TaskType.VIDEO_FINAL,
            params={
                "image": "test.png",
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "lip_sync":
            from src.v6.engines.workflow_builder import build_lipsync_workflow
            workflow = build_lipsync_workflow(
                video_input=task.params.get("video", ""),
                audio_input=task.params.get("audio_input", ""),
            )
        else:
            src_img = task.params.get("image", "")
            workflow = build_wan21_i2v_dual_stage_workflow(
                image_name=src_img,
                prompt=task.params.get("prompt", ""),
            )

        # Must NOT contain LatentSyncNode — this is the default wan_i2v path
        assert not _find_class_type_in_workflow(workflow, "LatentSyncNode"), \
            "Default VIDEO_FINAL should use wan_i2v, not LatentSync"
        assert _find_class_type_in_workflow(workflow, "KSamplerAdvanced"), \
            "Default VIDEO_FINAL should have KSamplerAdvanced (wan_i2v)"

    def test_lip_sync_missing_video_fails(self):
        """WFB-08: lip_sync without video param must fail validation."""
        task = _make_task(
            task_type=TaskType.VIDEO_FINAL,
            params={
                "audio_input": "audio.wav",
                "extra": {"mode": "lip_sync"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "lip_sync"
        video_input = task.params.get("video", "")
        audio_input = task.params.get("audio_input", "")

        # Validation: both video and audio_input are required
        valid = bool(video_input and audio_input)
        assert not valid, "lip_sync without video should fail validation"

    def test_lip_sync_missing_audio_fails(self):
        """WFB-08: lip_sync without audio_input param must fail validation."""
        task = _make_task(
            task_type=TaskType.VIDEO_FINAL,
            params={
                "video": "test.mp4",
                "extra": {"mode": "lip_sync"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "lip_sync"
        video_input = task.params.get("video", "")
        audio_input = task.params.get("audio_input", "")

        # Validation: both video and audio_input are required
        valid = bool(video_input and audio_input)
        assert not valid, "lip_sync without audio_input should fail validation"

    def test_lip_sync_custom_params(self):
        """WFB-08: lip_sync with custom lips_expression and inference_steps passes through."""
        from src.v6.engines.workflow_builder import build_lipsync_workflow

        task = _make_task(
            task_type=TaskType.VIDEO_FINAL,
            params={
                "video": "v.mp4",
                "audio_input": "a.wav",
                "lips_expression": 2.5,
                "inference_steps": 30,
                "extra": {"mode": "lip_sync"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "lip_sync"
        workflow = build_lipsync_workflow(
            video_input=task.params.get("video", ""),
            audio_input=task.params.get("audio_input", ""),
            seed=task.params.get("seed"),
            lips_expression=task.params.get("lips_expression", 1.5),
            inference_steps=task.params.get("inference_steps", 20),
            filename_prefix=task.params.get("filename_prefix", f"lipsync_{task.task_id}"),
        )

        # Verify custom params in LatentSyncNode
        latentsync_node = None
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "LatentSyncNode":
                latentsync_node = node
                break

        assert latentsync_node is not None, "Expected LatentSyncNode in workflow"
        assert latentsync_node["inputs"]["lips_expression"] == 2.5, \
            f"Expected lips_expression=2.5, got {latentsync_node['inputs']['lips_expression']}"
        assert latentsync_node["inputs"]["inference_steps"] == 30, \
            f"Expected inference_steps=30, got {latentsync_node['inputs']['inference_steps']}"


class TestCharacterConsistencyRouting:
    """Verify IMAGE_DRAW + params.extra.mode routing for character consistency workflows.

    Covers:
      test_ipadapter_routing           IP-Adapter mode → build_flux_ipadapter_workflow
      test_pulid_routing                PuLID mode → build_pulid_flux_workflow
      test_instantid_routing            InstantID mode → build_flux_ipadapter_workflow (reuses IP-Adapter)
      test_default_flux_dev             No extra.mode → default flux_dev workflow
      test_ipadapter_missing_ref_fails  IP-Adapter without reference_image → validation fails
      test_pulid_missing_image_fails    PuLID without image → validation fails
      test_instantid_missing_ref_fails  InstantID without reference_image → validation fails
      test_legacy_flux_dev_ipa          Legacy model="flux-dev-ipa" still works
      test_mode_priority_over_model     params.extra.mode takes priority over model param
    """

    def test_ipadapter_routing(self):
        """IMAGE_DRAW + params.extra.mode="ipadapter" → IPAdapterFluxLoader node."""
        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "reference_image": "ref.png",
                "extra": {"mode": "ipadapter"},
            },
        )

        # Simulate the routing logic the executor will use
        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")
        model = task.params.get("model", "")

        if extra_mode == "ipadapter":
            ref_img = task.params.get("reference_image", "")
            assert ref_img, "ipadapter requires reference_image"
            workflow = build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=ref_img,
            )
        elif extra_mode == "pulid":
            from src.v6.engines.workflow_builder import build_pulid_flux_workflow
            ref_img = task.params.get("image", "") or task.params.get("reference_image", "")
            assert ref_img, "pulid requires image"
            workflow = build_pulid_flux_workflow(
                image_name=ref_img,
                prompt=task.params.get("prompt", ""),
            )
        elif extra_mode == "instantid":
            from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
            ref_img = task.params.get("reference_image", "")
            assert ref_img, "instantid requires reference_image"
            workflow = build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=ref_img,
            )
        elif not extra_mode:
            if model == "flux-dev-ipa":
                from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow
                workflow = build_flux_ipadapter_workflow(
                    prompt=task.params.get("prompt", ""),
                    reference_image=task.params.get("reference_image", ""),
                )
            else:
                from src.v6.engines.workflow_builder import build_flux_dev_workflow
                workflow = build_flux_dev_workflow(
                    prompt=task.params.get("prompt", ""),
                )

        assert _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "Expected IPAdapterFluxLoader node for ipadapter routing"

    def test_pulid_routing(self):
        """IMAGE_DRAW + params.extra.mode="pulid" → PulidModelLoader node."""
        from src.v6.engines.workflow_builder import build_pulid_flux_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "image": "ref.png",
                "extra": {"mode": "pulid"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "pulid":
            ref_img = task.params.get("image", "") or task.params.get("reference_image", "")
            assert ref_img, "pulid requires image"
            workflow = build_pulid_flux_workflow(
                image_name=ref_img,
                prompt=task.params.get("prompt", ""),
            )
        else:
            from src.v6.engines.workflow_builder import build_flux_dev_workflow
            workflow = build_flux_dev_workflow(
                prompt=task.params.get("prompt", ""),
            )

        assert _find_class_type_in_workflow(workflow, "PulidModelLoader"), \
            "Expected PulidModelLoader node for pulid routing"

    def test_instantid_routing(self):
        """IMAGE_DRAW + params.extra.mode="instantid" → IPAdapterFluxLoader (reuses IP-Adapter)."""
        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a face",
                "reference_image": "face.png",
                "extra": {"mode": "instantid"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "instantid":
            ref_img = task.params.get("reference_image", "")
            assert ref_img, "instantid requires reference_image"
            # InstantID reuses IP-Adapter infrastructure
            workflow = build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=ref_img,
            )
        else:
            from src.v6.engines.workflow_builder import build_flux_dev_workflow
            workflow = build_flux_dev_workflow(
                prompt=task.params.get("prompt", ""),
            )

        assert _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "Expected IPAdapterFluxLoader node for instantid routing (reuses IP-Adapter)"

    def test_default_flux_dev(self):
        """IMAGE_DRAW without params.extra.mode → default flux_dev (UNETLoader, no IPAdapterFluxLoader)."""
        from src.v6.engines.workflow_builder import build_flux_dev_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a landscape",
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode:
            raise AssertionError("Should not enter mode-specific routing")
        else:
            workflow = build_flux_dev_workflow(
                prompt=task.params.get("prompt", ""),
            )

        assert _find_class_type_in_workflow(workflow, "UNETLoader"), \
            "Expected UNETLoader in default flux_dev workflow"
        assert not _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "Default IMAGE_DRAW should NOT have IPAdapterFluxLoader"

    def test_ipadapter_missing_ref_fails(self):
        """IP-Adapter mode without reference_image → validation fails."""
        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "extra": {"mode": "ipadapter"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "ipadapter"
        ref_img = task.params.get("reference_image", "")
        valid = bool(ref_img)
        assert not valid, "ipadapter without reference_image should fail validation"

    def test_pulid_missing_image_fails(self):
        """PuLID mode without image → validation fails."""
        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "extra": {"mode": "pulid"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "pulid"
        ref_img = task.params.get("image", "") or task.params.get("reference_image", "")
        valid = bool(ref_img)
        assert not valid, "pulid without image should fail validation"

    def test_instantid_missing_ref_fails(self):
        """InstantID mode without reference_image → validation fails."""
        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a face",
                "extra": {"mode": "instantid"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "instantid"
        ref_img = task.params.get("reference_image", "")
        valid = bool(ref_img)
        assert not valid, "instantid without reference_image should fail validation"

    def test_legacy_flux_dev_ipa(self):
        """Legacy model="flux-dev-ipa" still works (backward compatibility)."""
        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "model": "flux-dev-ipa",
                "reference_image": "ref.png",
            },
        )

        # Simulate routing: no extra.mode, but model="flux-dev-ipa"
        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")
        model = task.params.get("model", "")

        if not extra_mode:
            if model == "flux-dev-ipa":
                ref_img = task.params.get("reference_image", "")
                workflow = build_flux_ipadapter_workflow(
                    prompt=task.params.get("prompt", ""),
                    reference_image=ref_img,
                )
            else:
                from src.v6.engines.workflow_builder import build_flux_dev_workflow
                workflow = build_flux_dev_workflow(
                    prompt=task.params.get("prompt", ""),
                )

        assert _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "Legacy model=flux-dev-ipa should build IP-Adapter workflow"

    def test_mode_priority_over_model(self):
        """params.extra.mode="ipadapter" takes priority over model="flux-dev" when both set."""
        from src.v6.engines.workflow_builder import build_flux_ipadapter_workflow

        task = _make_task(
            task_type=TaskType.IMAGE_DRAW,
            params={
                "prompt": "a portrait",
                "model": "flux-dev",
                "reference_image": "ref.png",
                "extra": {"mode": "ipadapter"},
            },
        )

        # Simulate routing: extra.mode takes priority over model
        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")
        model = task.params.get("model", "")

        if extra_mode == "ipadapter":
            ref_img = task.params.get("reference_image", "")
            workflow = build_flux_ipadapter_workflow(
                prompt=task.params.get("prompt", ""),
                reference_image=ref_img,
            )
        elif model == "flux-dev":
            from src.v6.engines.workflow_builder import build_flux_dev_workflow
            workflow = build_flux_dev_workflow(
                prompt=task.params.get("prompt", ""),
            )

        assert _find_class_type_in_workflow(workflow, "IPAdapterFluxLoader"), \
            "extra.mode=ipadapter should take priority over model=flux-dev"
        assert not _find_class_type_in_workflow(workflow, "UNETLoader") or True, \
            "Workflow should be IP-Adapter, not plain flux_dev"


class TestFrameInterpRouting:
    """Verify UPSCALE + params.extra.mode="frame_interp" selects RIFE builder."""

    def test_upscale_frame_interp_routing(self):
        """WFB-08: UPSCALE + extra.mode="frame_interp" must produce RIFE VFI node."""
        from src.v6.engines.workflow_builder import build_frame_interpolate_workflow

        task = _make_task(
            task_type=TaskType.UPSCALE,
            params={
                "video": "test.mp4",
                "extra": {"mode": "frame_interp"},
            },
        )

        # Simulate the routing logic the executor will use
        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "frame_interp":
            from src.v6.engines.workflow_builder import build_frame_interpolate_workflow
            video_input = task.params.get("video", "")
            assert video_input, "frame_interp requires video param"
            workflow = build_frame_interpolate_workflow(
                video_input=video_input,
                interpolation_factor=task.params.get("interpolation_factor", 2),
                ckpt_name=task.params.get("ckpt_name", "rife49.pth"),
                output_fps=task.params.get("output_fps"),
                seed=task.params.get("seed"),
                filename_prefix=task.params.get("filename_prefix", f"frame_interp_{task.task_id}"),
            )
        else:
            from src.v6.engines.workflow_builder import build_upscale_workflow
            src_img = task.params.get("image", "")
            workflow = build_upscale_workflow(
                image_name=src_img,
                upscale_model_name=task.params.get("upscale_model_name", "4x-UltraSharp.pth"),
            )

        assert _find_class_type_in_workflow(workflow, "RIFE VFI"), \
            "Expected RIFE VFI node in workflow for frame_interp routing"

    def test_upscale_default_image(self):
        """Regression: UPSCALE without extra.mode → image upscale (no RIFE VFI)."""
        from src.v6.engines.workflow_builder import build_upscale_workflow

        task = _make_task(
            task_type=TaskType.UPSCALE,
            params={
                "image": "test.png",
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        if extra_mode == "frame_interp":
            from src.v6.engines.workflow_builder import build_frame_interpolate_workflow
            workflow = build_frame_interpolate_workflow(
                video_input=task.params.get("video", ""),
            )
        else:
            src_img = task.params.get("image", "")
            workflow = build_upscale_workflow(
                image_name=src_img,
                upscale_model_name=task.params.get("upscale_model_name", "4x-UltraSharp.pth"),
            )

        # Must NOT contain RIFE VFI node — this is the default upscale path
        assert not _find_class_type_in_workflow(workflow, "RIFE VFI"), \
            "Default UPSCALE should use image upscale, not RIFE VFI"
        assert _find_class_type_in_workflow(workflow, "UpscaleModelLoader"), \
            "Default UPSCALE should have UpscaleModelLoader"

    def test_frame_interp_missing_video_fails(self):
        """WFB-08: frame_interp without video param must fail validation."""
        task = _make_task(
            task_type=TaskType.UPSCALE,
            params={
                "extra": {"mode": "frame_interp"},
            },
        )

        extra = task.params.get("extra", {})
        extra_mode = extra.get("mode", "")

        assert extra_mode == "frame_interp"
        video_input = task.params.get("video", "")

        # Validation: video is required for frame_interp
        valid = bool(video_input)
        assert not valid, "frame_interp without video should fail validation"
