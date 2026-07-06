"""NVIDIA Kimodo Motion Generation Engine — in-process, lazy-load, idle-unload.

Mirrors the TTSTracker pattern: model loads directly into the gold-team
Python process on first request, unloads after 5 min idle to free VRAM
for other engines on the shared 3090. Tasks are serialized — when Kimodo
runs, it has the full 24GB available (~17GB used).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import uuid
from typing import Any

import numpy as np
import torch

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/mnt/agents/output")


class KimodoEngine(BaseEngine):
    """NVIDIA Kimodo text-to-motion engine — in-process, lazy-load."""

    IDLE_TIMEOUT = 300.0       # 5 min idle → unload
    VRAM_MB = 17000            # full GPU load (diffusion + LLM2Vec text encoder)
    REAPER_INTERVAL = 30.0     # check idle every 30 s
    MAX_RESULTS = 100          # keep last 100 results

    def __init__(
        self,
        idle_timeout: float = IDLE_TIMEOUT,
        output_root: str = OUTPUT_ROOT,
    ) -> None:
        self._output_root = output_root
        self._idle_timeout = idle_timeout
        self._model_name = os.environ.get("KIMODO_MODEL", "kimodo-smplx-rp")
        self._device = os.environ.get("KIMODO_DEVICE", "cuda:0")

        self._model: Any = None
        self._loaded: bool = False
        self._loading: bool = False
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._results: dict[str, dict] = {}
        self._reaper_task: asyncio.Task | None = None

    # ── BaseEngine properties ──────────────────────────────────────────────
    @property
    def name(self) -> str: return "Kimodo Motion Generation"

    @property
    def engine_id(self) -> str: return "kimodo-local"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["motion_generate"],
            max_duration_sec=30.0,
            vram_total_mb=self.VRAM_MB,
            vram_available_mb=self.VRAM_MB,
            models=["Kimodo-SMPLX-RP-v1", "Kimodo-SOMA-RP-v1.1"],
        )

    @property
    def backend_type(self) -> BackendType: return BackendType.SUBPROCESS

    # ── Lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(self._idle_reaper())
        logger.info(
            "KimodoEngine started (model=%s, device=%s, idle_timeout=%.0fs)",
            self._model_name, self._device, self._idle_timeout,
        )

    async def stop(self) -> None:
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        await self._unload()
        self._results.clear()
        logger.info("KimodoEngine stopped")

    # ── Model lifecycle (lazy-load / idle-unload) ─────────────────────────
    async def _ensure_loaded(self) -> None:
        """Load model if not already loaded (with dedup lock)."""
        async with self._lock:
            if self._loaded:
                self._last_used = time.monotonic()
                return
            if self._loading:
                while self._loading:
                    await asyncio.sleep(0.5)
                return
            self._loading = True
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)
                self._loaded = True
                self._last_used = time.monotonic()
                logger.info("Kimodo model '%s' loaded successfully", self._model_name)
            except Exception as e:
                logger.error("Kimodo model '%s' failed to load: %s", self._model_name, e)
                raise
            finally:
                self._loading = False

    def _load_model(self) -> None:
        """Load Kimodo model into GPU. Runs in executor thread."""
        # Pre-download LLM2Vec adapters + patch base_model to NousResearch mirror
        # (avoids gated meta-llama repo access issues)
        self._patch_llama_mirror()

        from kimodo import load_model
        self._model = load_model(self._model_name, device=self._device)

    @staticmethod
    def _patch_llama_mirror() -> None:
        """Ensure MNTP adapter_config.json points to open NousResearch mirror.

        The adapter is downloaded at runtime by huggingface_hub; we patch its
        base_model_name_or_path before Kimodo's LLM2Vec tries to load it.
        Idempotent — safe to call on every load.
        """
        import json
        import pathlib

        try:
            from huggingface_hub import snapshot_download
            # Force-download the MNTP adapter so we can patch it
            adapter_path = snapshot_download(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            )
            config_file = pathlib.Path(adapter_path) / "adapter_config.json"
            if not config_file.exists():
                logger.warning("MNTP adapter_config.json not found at %s", config_file)
                return
            cfg = json.loads(config_file.read_text())
            if cfg.get("base_model_name_or_path") != "NousResearch/Meta-Llama-3-8B-Instruct":
                cfg["base_model_name_or_path"] = "NousResearch/Meta-Llama-3-8B-Instruct"
                config_file.write_text(json.dumps(cfg, indent=2))
                logger.info("Patched MNTP adapter base_model → NousResearch/Meta-Llama-3-8B-Instruct")
            else:
                logger.debug("MNTP adapter already patched to NousResearch")
        except Exception as e:
            logger.warning("LLaMA mirror patch skipped: %s", e)

    async def _unload(self) -> None:
        async with self._lock:
            if not self._loaded:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._unload_model)

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Kimodo model unloaded, VRAM freed")

    def _is_idle(self) -> bool:
        return self._loaded and (time.monotonic() - self._last_used > self._idle_timeout)

    async def _idle_reaper(self) -> None:
        """Background task: unload idle model."""
        while True:
            try:
                await asyncio.sleep(int(self.REAPER_INTERVAL))
                if self._is_idle():
                    logger.info("Kimodo idle > %.0fs, unloading", self._idle_timeout)
                    await self._unload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Kimodo idle reaper error: %s", e)

    # ── Generation ─────────────────────────────────────────────────────────
    def _generate(self, workflow: dict[str, Any], task_id: str) -> dict:
        """Actual Kimodo call. Runs in executor thread (blocking).

        Supported workflow params:
            prompt: text description of motion (required)
            duration_sec: motion length in seconds (default 5.0)
            num_denoising_steps: DDIM steps (default 100, 150-200 for quality)
            num_samples: number of samples (default 1)
            heading_angle_deg: initial body facing direction in degrees
                (0=+Z forward, 90=+X right, 180=-Z backward, -90=-X left)
            cfg_weight: classifier-free guidance [text_cfg, constraint_cfg]
                (default [2.0, 2.0])
            cfg_type: CFG strategy override (e.g. "separated")
            multi_prompt: treat prompts as sequential segments (default False)
            root_margin: post-process root correction margin in meters (default 0.04)

            output_format: "npz" (default) or "bvh" (also exports motion.bvh)
            root_path: list of [X, Z] waypoints in meters; auto-spread across frames
            root_heading: optional list of heading radians matching root_path
            prompts: list of strings (or pipe-delimited str) for multi_prompt
            durations_sec: list of per-segment durations (multi_prompt only)
            num_transition_frames: blend frames between segments (default 5)
        """
        prompt = workflow.get("prompt", "") or workflow.get("text", "")
        if not prompt:
            raise ValueError("Kimodo workflow missing 'prompt'")

        duration_sec = float(workflow.get("duration_sec", 5.0))
        num_denoising_steps = int(workflow.get("num_denoising_steps", 100))
        num_samples = int(workflow.get("num_samples", 1))

        # Direction control: heading angle (degrees → radians tensor)
        heading_deg = workflow.get("heading_angle_deg", None)
        first_heading_angle = None
        if heading_deg is not None:
            heading_rad = float(heading_deg) * 3.141592653589793 / 180.0
            first_heading_angle = torch.tensor([heading_rad], device=self._device)
            logger.info("Kimodo heading_angle=%.1f° (%.4f rad)", heading_deg, heading_rad)

        # CFG control
        cfg_weight = workflow.get("cfg_weight", [2.0, 2.0])
        cfg_type = workflow.get("cfg_type", None)
        multi_prompt = bool(workflow.get("multi_prompt", False))
        root_margin = float(workflow.get("root_margin", 0.04))

        # Output format (Feature 1)
        output_format = str(workflow.get("output_format", "npz")).lower()

        fps = getattr(self._model, "fps", 30)
        num_frames = max(1, int(duration_sec * fps))

        # ── Feature 3: multi_prompt full support ────────────────────────────
        num_frames_kw = num_frames
        if multi_prompt:
            prompts_input = workflow.get("prompts", None)
            if isinstance(prompts_input, list) and prompts_input:
                prompt = [str(p) for p in prompts_input]
            elif isinstance(prompts_input, str) and prompts_input:
                prompt = [p.strip() for p in prompts_input.split("|") if p.strip()]
            elif isinstance(prompt, str) and "|" in prompt:
                prompt = [p.strip() for p in prompt.split("|") if p.strip()]
            # else: leave prompt as-is (single string treated as 1-segment)

            durations = workflow.get("durations_sec", None)
            if isinstance(durations, list) and durations:
                num_frames_kw = [max(1, int(float(d) * fps)) for d in durations]

        logger.info(
            "Kimodo generate (task=%s, frames=%s, fps=%s, steps=%d, heading=%s, "
            "multi_prompt=%s, output_format=%s)",
            task_id, num_frames_kw if multi_prompt else num_frames, fps,
            num_denoising_steps,
            f"{heading_deg}°" if heading_deg is not None else "default",
            multi_prompt, output_format,
        )

        # Build kwargs — only pass optional params that are set
        call_kwargs = dict(
            prompts=prompt,
            num_frames=num_frames_kw,
            num_denoising_steps=num_denoising_steps,
            num_samples=num_samples,
            multi_prompt=multi_prompt,
            return_numpy=True,
            post_processing=True,
            cfg_weight=cfg_weight,
            root_margin=root_margin,
        )
        if first_heading_angle is not None:
            call_kwargs["first_heading_angle"] = first_heading_angle
        if cfg_type is not None:
            call_kwargs["cfg_type"] = cfg_type

        if multi_prompt:
            call_kwargs["num_transition_frames"] = int(
                workflow.get("num_transition_frames", 5)
            )

        # ── Feature 2: Root2DConstraintSet path constraint ──────────────────
        root_path = workflow.get("root_path", None)
        constraint_lst = []
        if root_path and isinstance(root_path, list) and len(root_path) >= 2:
            from kimodo.constraints import Root2DConstraintSet

            skeleton = self._resolve_skeleton()
            num_path_points = len(root_path)
            frame_indices = torch.linspace(
                0, num_frames - 1, num_path_points, dtype=torch.long,
                device=self._device,
            )
            # Root XZ plane: indices [0, 2] in Y-up. Drop any Y values.
            xz_pairs = [
                (pt[0], pt[2]) if len(pt) >= 3 else (pt[0], pt[1])
                for pt in root_path
            ]
            smooth_root_2d = torch.tensor(
                xz_pairs, dtype=torch.float32, device=self._device,
            )

            constraint_kwargs = dict(
                skeleton=skeleton,
                frame_indices=frame_indices,
                smooth_root_2d=smooth_root_2d,
            )
            root_heading = workflow.get("root_heading", None)
            if root_heading and isinstance(root_heading, list):
                constraint_kwargs["global_root_heading"] = torch.tensor(
                    root_heading, dtype=torch.float32, device=self._device,
                )

            constraint_lst.append(Root2DConstraintSet(**constraint_kwargs))
            logger.info(
                "Kimodo Root2D path constraint: %d waypoints across %d frames",
                num_path_points, num_frames,
            )

        if constraint_lst:
            call_kwargs["constraint_lst"] = constraint_lst

        output = self._model(**call_kwargs)

        out_dir = os.path.join(self._output_root, task_id)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "motion.npz")

        from kimodo.exports.motion_io import save_kimodo_npz
        save_kimodo_npz(output_path, output)

        # ── Feature 1: BVH export ───────────────────────────────────────────
        bvh_path: str | None = None
        if output_format == "bvh":
            try:
                from kimodo.exports import save_motion_bvh
                bvh_path = os.path.join(out_dir, "motion.bvh")

                local_rot_mats = output["local_rot_mats"]
                root_positions = output["root_positions"]
                # Ensure torch tensors on CPU (model may return GPU tensors or numpy)
                if not isinstance(local_rot_mats, torch.Tensor):
                    local_rot_mats = torch.from_numpy(np.asarray(local_rot_mats))
                local_rot_mats = local_rot_mats.detach().cpu()
                if not isinstance(root_positions, torch.Tensor):
                    root_positions = torch.from_numpy(np.asarray(root_positions))
                root_positions = root_positions.detach().cpu()

                skeleton = self._resolve_skeleton()
                # standard_tpose=False requires global_rot_offsets + bvh_neutral_joints
                # (SOMA-only assets). Fall back to standard T-pose for SMPLX etc.
                standard_tpose = not (
                    hasattr(skeleton, "global_rot_offsets")
                    and hasattr(skeleton, "bvh_neutral_joints")
                )

                save_motion_bvh(
                    bvh_path,
                    local_rot_mats,
                    root_positions,
                    skeleton=skeleton,
                    fps=fps,
                    standard_tpose=standard_tpose,
                )
                logger.info(
                    "Kimodo BVH exported to %s (standard_tpose=%s)",
                    bvh_path, standard_tpose,
                )
            except Exception as e:
                logger.error("Kimodo BVH export failed (NPZ still saved): %s", e)
                bvh_path = None

        result = {
            "output_path": output_path,
            "duration_sec": round(num_frames / fps, 2),
            "fps": fps,
            "num_frames": num_frames,
            "prompt": prompt if isinstance(prompt, str) else " | ".join(prompt),
        }
        if bvh_path:
            result["bvh_path"] = bvh_path
        return result

    def _resolve_skeleton(self):
        """Return the model's skeleton object, tolerating either attribute path."""
        if hasattr(self._model, "skeleton"):
            return self._model.skeleton
        return self._model.motion_rep.skeleton

    # ── BaseEngine interface ───────────────────────────────────────────────
    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit motion generation — runs in executor, returns job_id immediately."""
        params = params or {}
        job_id = str(uuid.uuid4())[:12]
        task_id = params.get("task_id", job_id)

        logger.info("Kimodo job %s: submit (prompt=%s)", job_id, str(workflow.get("prompt", ""))[:60])
        self._results[job_id] = {
            "status": "running", "progress": 10.0, "error": "",
            "output_path": "", "task_id": task_id,
        }

        try:
            await self._ensure_loaded()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate, workflow, task_id)
            self._results[job_id] = {
                "status": "completed", "progress": 100.0, "error": "",
                "task_id": task_id, **result,
            }
            logger.info(
                "Kimodo job %s completed (frames=%d, output=%s)",
                job_id, result["num_frames"], result["output_path"],
            )
        except Exception as e:
            self._results[job_id] = {
                "status": "failed", "progress": 0.0, "error": str(e),
                "output_path": "", "task_id": task_id,
            }
            logger.error("Kimodo job %s failed: %s", job_id, e)

        self._trim_results()
        return job_id

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        result = self._results.get(engine_job_id)
        if result:
            return {
                "status": result["status"],
                "progress": result["progress"],
                "error": result.get("error", ""),
            }
        return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        result = self._results.get(engine_job_id)
        if not result or result["status"] != "completed":
            return {"outputs": []}
        outputs = [{
            "url": f"file://{result['output_path']}",
            "path": result["output_path"],
            "type": "motion",
            "format": "npz",
            "duration_sec": result.get("duration_sec", 0.0),
            "fps": result.get("fps", 0),
            "num_frames": result.get("num_frames", 0),
            "prompt": result.get("prompt", ""),
        }]
        bvh = result.get("bvh_path")
        if bvh:
            outputs.append({
                "url": f"file://{bvh}",
                "path": bvh,
                "type": "motion",
                "format": "bvh",
                "duration_sec": result.get("duration_sec", 0.0),
                "fps": result.get("fps", 0),
                "num_frames": result.get("num_frames", 0),
                "prompt": result.get("prompt", ""),
            })
        return {"outputs": outputs}

    async def cancel(self, engine_job_id: str) -> bool:
        return False  # synchronous generation, can't cancel mid-run

    async def health(self) -> dict[str, Any]:
        # Probe importability without loading weights — graceful when kimodo
        # package isn't installed yet.
        try:
            import kimodo  # noqa: F401
            kimodo_available = True
        except ImportError:
            kimodo_available = False

        return {
            "status": EngineStatus.ONLINE.value if kimodo_available else EngineStatus.OFFLINE.value,
            "available": kimodo_available,
            "mode": "in-process (lazy-load)",
            "model_name": self._model_name,
            "device": self._device,
            "loaded": self._loaded,
            "loading": self._loading,
            "vram_mb": self.VRAM_MB,
            "idle_timeout": self._idle_timeout,
            "idle_seconds": (time.monotonic() - self._last_used) if self._loaded else None,
        }

    # ── Internal ───────────────────────────────────────────────────────────
    def _trim_results(self) -> None:
        """Keep only the most recent MAX_RESULTS entries (insertion-ordered)."""
        while len(self._results) > self.MAX_RESULTS:
            self._results.pop(next(iter(self._results)), None)
