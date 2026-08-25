"""Unit tests for workflow_builder — verifies existing builders produce correct
ComfyUI workflow dicts before new builders are added in subsequent plans.

Covers:
  WFB-01  build_flux_dev_workflow
  WFB-02  build_flux_ipadapter_workflow
  WFB-03  build_hunyuan3d_workflow
  WFB-06  build_lipsync_workflow
  WFB-07  build_frame_interpolate_workflow
"""
from __future__ import annotations

import os

import pytest

from src.v6.engines.workflow_builder import (
    build_flux_dev_workflow,
    build_flux_ipadapter_workflow,
    build_hunyuan3d_workflow,
    build_lipsync_workflow,
    build_frame_interpolate_workflow,
)


# ---------------------------------------------------------------------------
# WFB-01: build_flux_dev_workflow
# ---------------------------------------------------------------------------

class TestBuildFluxDevWorkflow:
    """Verify FLUX Dev FP8 workflow node graph."""

    def test_build_flux_dev_workflow(self, sample_seed: int):
        wf = build_flux_dev_workflow(prompt="test prompt", seed=sample_seed)

        # Top-level is a dict with string keys
        assert isinstance(wf, dict)
        assert all(isinstance(k, str) for k in wf.keys())

        # Node 1: UNETLoader
        assert wf["1"]["class_type"] == "UNETLoader"

        # Node 4: CLIPTextEncode with the prompt
        assert wf["4"]["class_type"] == "CLIPTextEncode"
        assert wf["4"]["inputs"]["text"] == "test prompt"

        # Node 6: KSampler with expected defaults
        assert wf["6"]["class_type"] == "KSampler"
        assert wf["6"]["inputs"]["seed"] == sample_seed
        assert wf["6"]["inputs"]["steps"] == 28
        assert wf["6"]["inputs"]["cfg"] == 3.5

        # Node 8: SaveImage
        assert wf["8"]["class_type"] == "SaveImage"

        # Node 7: VAEDecode links to KSampler output and VAELoader output
        assert wf["7"]["class_type"] == "VAEDecode"
        assert wf["7"]["inputs"]["samples"] == ["6", 0]
        assert wf["7"]["inputs"]["vae"] == ["3", 0]

        # Node 5: EmptySD3LatentImage default dimensions
        assert wf["5"]["class_type"] == "EmptySD3LatentImage"
        assert wf["5"]["inputs"]["width"] == 1024
        assert wf["5"]["inputs"]["height"] == 1024

    def test_build_flux_dev_workflow_custom_params(self, sample_seed: int):
        wf = build_flux_dev_workflow(
            prompt="custom",
            width=512,
            height=768,
            steps=15,
            cfg_scale=7.0,
            seed=99,
        )

        # KSampler reflects custom params
        assert wf["6"]["inputs"]["steps"] == 15
        assert wf["6"]["inputs"]["cfg"] == 7.0
        assert wf["6"]["inputs"]["seed"] == 99

        # EmptySD3LatentImage reflects custom dimensions
        assert wf["5"]["inputs"]["width"] == 512
        assert wf["5"]["inputs"]["height"] == 768


# ---------------------------------------------------------------------------
# WFB-02: build_flux_ipadapter_workflow
# ---------------------------------------------------------------------------

class TestBuildFluxIPAdapterWorkflow:
    """Verify FLUX + IP-Adapter workflow node graph."""

    def test_build_flux_ipadapter_workflow(self, sample_seed: int):
        wf = build_flux_ipadapter_workflow(
            prompt="test",
            reference_image="ref.png",
            seed=sample_seed,
        )

        # Node 10: IPAdapterFluxLoader
        assert wf["10"]["class_type"] == "IPAdapterFluxLoader"

        # Node 11: LoadImage with reference
        assert wf["11"]["class_type"] == "LoadImage"
        assert wf["11"]["inputs"]["image"] == "ref.png"

        # Node 12: ApplyIPAdapterFlux links correctly
        assert wf["12"]["class_type"] == "ApplyIPAdapterFlux"
        assert wf["12"]["inputs"]["ipadapter_flux"] == ["10", 0]
        assert wf["12"]["inputs"]["image"] == ["11", 0]

        # KSampler (node 6) uses IPAdapter-modified model from node 12
        assert wf["6"]["inputs"]["model"] == ["12", 0]

    def test_build_flux_ipadapter_workflow_custom_weight(self, sample_seed: int):
        wf = build_flux_ipadapter_workflow(
            prompt="test",
            reference_image="ref.png",
            weight=0.5,
            start_percent=0.2,
            end_percent=0.6,
            seed=sample_seed,
        )

        assert wf["12"]["inputs"]["weight"] == 0.5
        assert wf["12"]["inputs"]["start_percent"] == 0.2
        assert wf["12"]["inputs"]["end_percent"] == 0.6


# ---------------------------------------------------------------------------
# WFB-03: build_hunyuan3d_workflow
# ---------------------------------------------------------------------------

class TestBuildHunyuan3dWorkflow:
    """Verify Hunyuan3D subprocess parameter dict."""

    def test_build_hunyuan3d_workflow(self):
        wf = build_hunyuan3d_workflow(
            input_image="/path/to/img.png",
            task_id="test-123",
        )

        # Flat dict, NOT a numbered-node ComfyUI graph
        assert isinstance(wf, dict)
        assert all(isinstance(k, str) for k in wf.keys())
        # No class_type keys — not a node graph
        for v in wf.values():
            assert not isinstance(v, dict) or "class_type" not in v

        assert wf["input_image"] == "/path/to/img.png"
        assert "output_path" in wf
        assert "test-123" in wf["output_path"]
        assert wf["model"] == "full"
        assert wf["steps"] == 50


# ---------------------------------------------------------------------------
# WFB-06: build_lipsync_workflow
# ---------------------------------------------------------------------------

class TestBuildLipsyncWorkflow:
    """Verify LatentSync lip sync workflow node graph."""

    def test_build_lipsync_workflow(self):
        """Test 1: Returns dict with exactly 4 top-level keys (nodes '1' through '4')."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert isinstance(wf, dict)
        assert len(wf) == 4
        assert set(wf.keys()) == {"1", "2", "3", "4"}

    def test_node1_vhs_load_video(self):
        """Test 2: Node '1' has class_type VHS_LoadVideo with video='test_video.mp4'."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert wf["1"]["class_type"] == "VHS_LoadVideo"
        assert wf["1"]["inputs"]["video"] == "test_video.mp4"

    def test_node2_load_audio(self):
        """Test 3: Node '2' has class_type LoadAudio with audio='test_audio.wav'."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert wf["2"]["class_type"] == "LoadAudio"
        assert wf["2"]["inputs"]["audio"] == "test_audio.wav"

    def test_node3_latentsync_links(self):
        """Test 4: Node '3' LatentSyncNode links to node '1' output 0 and node '2' output 0."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert wf["3"]["class_type"] == "LatentSyncNode"
        assert wf["3"]["inputs"]["images"] == ["1", 0]
        assert wf["3"]["inputs"]["audio"] == ["2", 0]

    def test_node4_vhs_videocombine(self):
        """Test 5: Node '4' VHS_VideoCombine links to node '3' output 0 with correct format."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert wf["4"]["class_type"] == "VHS_VideoCombine"
        assert wf["4"]["inputs"]["images"] == ["3", 0]
        assert wf["4"]["inputs"]["format"] == "video/h264-mp4"
        assert wf["4"]["inputs"]["frame_rate"] == 25

    def test_seed_deterministic(self):
        """Test 6: Seed is deterministic when provided (seed=42 -> LatentSyncNode seed=42)."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
        )
        assert wf["3"]["inputs"]["seed"] == 42

    def test_custom_params_passthrough(self):
        """Test 7: Custom params (lips_expression, inference_steps) passed to LatentSyncNode."""
        wf = build_lipsync_workflow(
            video_input="test_video.mp4",
            audio_input="test_audio.wav",
            seed=42,
            lips_expression=2.0,
            inference_steps=30,
        )
        assert wf["3"]["inputs"]["lips_expression"] == 2.0
        assert wf["3"]["inputs"]["inference_steps"] == 30

    def test_video_path_traversal_rejected(self):
        """Test 8: Path traversal rejection: video_input containing '..' raises ValueError."""
        with pytest.raises(ValueError, match="video_input"):
            build_lipsync_workflow(
                video_input="../etc/passwd",
                audio_input="test_audio.wav",
                seed=42,
            )

    def test_audio_path_traversal_rejected(self):
        """Test 9: Path traversal rejection: audio_input containing '..' raises ValueError."""
        with pytest.raises(ValueError, match="audio_input"):
            build_lipsync_workflow(
                video_input="test_video.mp4",
                audio_input="../secret.wav",
                seed=42,
            )


# ---------------------------------------------------------------------------
# WFB-07: build_frame_interpolate_workflow
# ---------------------------------------------------------------------------

class TestBuildFrameInterpolateWorkflow:
    """Verify RIFE frame interpolation workflow node graph."""

    def test_build_frame_interpolate_workflow(self):
        """Test 1: Returns dict with exactly 3 top-level keys (nodes '1' through '3')."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            seed=42,
        )
        assert isinstance(wf, dict)
        assert len(wf) == 3
        assert set(wf.keys()) == {"1", "2", "3"}

    def test_node1_vhs_load_video(self):
        """Test 2: Node '1' has class_type VHS_LoadVideo with video='test_video.mp4'."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            seed=42,
        )
        assert wf["1"]["class_type"] == "VHS_LoadVideo"
        assert wf["1"]["inputs"]["video"] == "test_video.mp4"

    def test_node2_rife_vfi_links(self):
        """Test 3: Node '2' RIFE VFI links to node '1' output 0."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            seed=42,
        )
        assert wf["2"]["class_type"] == "RIFE VFI"
        assert wf["2"]["inputs"]["images"] == ["1", 0]

    def test_node3_vhs_videocombine(self):
        """Test 4: Node '3' VHS_VideoCombine links to node '2' output 0."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            seed=42,
        )
        assert wf["3"]["class_type"] == "VHS_VideoCombine"
        assert wf["3"]["inputs"]["images"] == ["2", 0]
        assert wf["3"]["inputs"]["format"] == "video/h264-mp4"

    def test_default_multiplier(self):
        """Test 5: Default interpolation_factor=2 maps to RIFE multiplier=1."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            interpolation_factor=2,
        )
        assert wf["2"]["inputs"]["multiplier"] == 1

    def test_4x_multiplier(self):
        """Test 6: interpolation_factor=4 maps to multiplier=3."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            interpolation_factor=4,
        )
        assert wf["2"]["inputs"]["multiplier"] == 3

    def test_8x_multiplier(self):
        """Test 7: interpolation_factor=8 maps to multiplier=7."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            interpolation_factor=8,
        )
        assert wf["2"]["inputs"]["multiplier"] == 7

    def test_custom_ckpt_name(self):
        """Test 8: Custom ckpt_name='rife49.pth' passed to RIFE VFI node."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            ckpt_name="rife49.pth",
        )
        assert wf["2"]["inputs"]["ckpt_name"] == "rife49.pth"

    def test_custom_output_fps(self):
        """Test 9: Custom output_fps=60 passed to VHS_VideoCombine."""
        wf = build_frame_interpolate_workflow(
            video_input="test_video.mp4",
            output_fps=60,
        )
        assert wf["3"]["inputs"]["frame_rate"] == 60

    def test_video_path_traversal_rejected(self):
        """Test 10: Path traversal rejection: video_input containing '..' raises ValueError."""
        with pytest.raises(ValueError, match="video_input"):
            build_frame_interpolate_workflow(
                video_input="../etc/passwd",
            )
