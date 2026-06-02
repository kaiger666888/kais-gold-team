"""Hunyuan3D-2 Engine — runs Tencent Hunyuan3D-2 via subprocess for gold-team V6.

Mirrors the fire-and-forget pattern in ``tts.py``: the engine spawns
``scripts/hunyuan3d_infer.py`` as a child process and tracks it via asyncio.
The subprocess prints a single JSON line on stdout when done, which we parse
to extract the output GLB path and stats.

Lifecycle:
    submit()     → spawn subprocess, return job_id immediately
    poll()       → report status (queued/running/completed/failed) + progress
    get_output() → return GLB artifact URL after completion
    cancel()     → SIGKILL the subprocess
    health()     → verify model directory + script exist
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from src.v6.engines.base import BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "hunyuan3d_infer.py")

DEFAULT_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_MODEL_DIR",
    "/data/models/tencent/Hunyuan3D-2",
)

# Regex for parsing progress percentages from stderr (e.g. "45%|████▌     |")
_PROGRESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class Hunyuan3DJob:
    """Tracks a single Hunyuan3D subprocess job."""

    def __init__(self, job_id: str, params: dict) -> None:
        self.job_id = job_id
        self.params = params
        self.status: str = "queued"  # queued | running | completed | failed
        self.progress: float = 0.0
        self.output_path: str = ""
        self.error: str = ""
        self.vertices: int = 0
        self.faces: int = 0
        self.elapsed_load_sec: float = 0.0
        self.elapsed_inference_sec: float = 0.0
        self.started_at: float = 0.0
        self.process: Optional[asyncio.subprocess.Process] = None


class Hunyuan3DEngine(BaseEngine):
    """Subprocess-based engine for Tencent Hunyuan3D-2 image-to-3D.

    Spawns ``scripts/hunyuan3d_infer.py`` per task. Heavy model load (~28s)
    and inference (~75s on full model, 50 steps) happen in the child process,
    so the engine itself stays lightweight.
    """

    def __init__(
        self,
        output_root: str = OUTPUT_ROOT,
        model_dir: str = DEFAULT_MODEL_DIR,
    ) -> None:
        self._output_root = output_root
        self._model_dir = model_dir
        self._jobs: dict[str, Hunyuan3DJob] = {}
        self._script_path = os.path.abspath(SCRIPT_PATH)
        self._python = os.environ.get("KAIS_HUNYUAN3D_PYTHON", "python3")

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Hunyuan3D-2 Engine (image→GLB)"

    @property
    def engine_id(self) -> str:
        return "hunyuan3d-local"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["image_to_3d"],
            max_duration_sec=600.0,
            vram_total_mb=24000,
            vram_available_mb=24000,
            models=["Hunyuan3D-2.1"],
        )

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not os.path.isfile(self._script_path):
            logger.warning("Hunyuan3D script not found at %s", self._script_path)
        else:
            logger.info("Hunyuan3D engine ready, script: %s", self._script_path)

    async def stop(self) -> None:
        for job in self._jobs.values():
            if job.process and job.process.returncode is None:
                job.process.kill()
        self._jobs.clear()

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit an image-to-3D task.

        Workflow keys:
            input_image (str, required): Path to source image.
            output_path (str, optional): GLB output path. Auto-generated if empty.
            model (str, optional): "mini" or "full" (default: full).
            device (str, optional): e.g. "cuda:0" (default). Mapped to
                CUDA_VISIBLE_DEVICES before subprocess import.
            steps (int, optional): Inference steps (default 50).
            seed (int, optional): Reproducibility seed.
            model_dir (str, optional): Override model checkpoint directory.
        """
        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        input_image = workflow.get("input_image") or workflow.get("image") or ""
        if not input_image:
            raise ValueError("Hunyuan3D requires 'input_image' parameter")

        output_path = workflow.get("output_path", "")
        if not output_path:
            output_path = os.path.join(self._output_root, task_id, "model.glb")

        model_variant = workflow.get("model", "full")
        device = workflow.get("device", "cuda:0")
        steps = int(workflow.get("steps", 50))
        seed = workflow.get("seed")
        model_dir = workflow.get("model_dir", self._model_dir)
        subfolder = workflow.get("subfolder")

        job = Hunyuan3DJob(
            job_id=job_id,
            params={
                "input_image": input_image,
                "output_path": output_path,
                "model": model_variant,
                "device": device,
                "steps": steps,
                "model_dir": model_dir,
                "task_id": task_id,
                **({"seed": seed} if seed is not None else {}),
            },
        )
        self._jobs[job_id] = job

        cmd = [
            self._python, self._script_path,
            "--input", input_image,
            "--output", output_path,
            "--model", model_variant,
            "--device", device,
            "--steps", str(steps),
            "--model-dir", model_dir,
        ]
        if subfolder:
            cmd.extend(["--subfolder", subfolder])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(
            "Hunyuan3D job %s: submitting (model=%s, device=%s, steps=%d, input=%s)",
            job_id, model_variant, device, steps, input_image,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            job.process = process
            job.status = "running"
            job.started_at = time.monotonic()
            asyncio.create_task(self._watch_job(job))
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Hunyuan3D job %s failed to start: %s", job_id, e)

        return job_id

    # ── background watcher ────────────────────────────────────────────────

    async def _watch_job(self, job: Hunyuan3DJob) -> None:
        """Read stdout/stderr concurrently, parse progress + final JSON."""
        assert job.process is not None
        try:
            stdout_buf = bytearray()
            stderr_buf = bytearray()

            async def _read(stream, buf: bytearray) -> None:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    # Update progress from latest stderr line containing %
                    text = chunk.decode(errors="replace")
                    m = None
                    for line in text.splitlines()[::-1]:
                        m = _PROGRESS_RE.search(line)
                        if m:
                            break
                    if m:
                        try:
                            pct = float(m.group(1))
                            # Clamp; inference % covers the post-load phase (~80% of total)
                            job.progress = max(job.progress, min(95.0, pct * 0.95))
                        except ValueError:
                            pass

            await asyncio.gather(
                _read(job.process.stdout, stdout_buf),
                _read(job.process.stderr, stderr_buf),
            )
            rc = await job.process.wait()

            if rc == 0:
                # Last non-empty line of stdout is the JSON result
                last_line = ""
                for line in stdout_buf.decode(errors="replace").splitlines()[::-1]:
                    if line.strip():
                        last_line = line.strip()
                        break
                try:
                    result = json.loads(last_line)
                except (json.JSONDecodeError, ValueError):
                    result = {}

                job.status = "completed"
                job.progress = 100.0
                job.output_path = result.get("output_path", job.params.get("output_path", ""))
                job.vertices = int(result.get("vertices", 0))
                job.faces = int(result.get("faces", 0))
                job.elapsed_load_sec = float(result.get("elapsed_load_sec", 0.0))
                job.elapsed_inference_sec = float(result.get("elapsed_inference_sec", 0.0))
                logger.info(
                    "Hunyuan3D job %s completed (load=%.1fs, infer=%.1fs, verts=%d, faces=%d, out=%s)",
                    job.job_id, job.elapsed_load_sec, job.elapsed_inference_sec,
                    job.vertices, job.faces, job.output_path,
                )
            else:
                err_text = stderr_buf.decode(errors="replace").strip()
                job.status = "failed"
                job.error = (err_text or stdout_buf.decode(errors="replace"))[:500]
                logger.error(
                    "Hunyuan3D job %s failed (rc=%d): %s",
                    job.job_id, rc, job.error[:200],
                )
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Hunyuan3D job %s watch error: %s", job.job_id, e)

    # ── poll / get_output / cancel / health ───────────────────────────────

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}

        result: dict[str, Any] = {
            "status": job.status,
            "progress": job.progress,
        }
        if job.status == "failed":
            result["error"] = job.error
        return result

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job or job.status != "completed":
            return {"outputs": []}

        output_path = job.output_path
        if not output_path or not os.path.isfile(output_path):
            return {"outputs": []}

        artifact = {
            "url": f"file://{output_path}",
            "path": output_path,
            "type": "model",
            "format": "glb",
            "model": job.params.get("model", "full"),
            "vertices": job.vertices,
            "faces": job.faces,
            "elapsed_load_sec": job.elapsed_load_sec,
            "elapsed_inference_sec": job.elapsed_inference_sec,
        }
        return {"outputs": [artifact]}

    async def cancel(self, engine_job_id: str) -> bool:
        job = self._jobs.get(engine_job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        if job.process and job.process.returncode is None:
            job.process.kill()
        job.status = "failed"
        job.error = "Cancelled"
        logger.info("Hunyuan3D job %s cancelled", engine_job_id)
        return True

    async def health(self) -> dict[str, Any]:
        script_ok = os.path.isfile(self._script_path)
        model_ok = os.path.isdir(self._model_dir)
        # Quick check: do the expected subdirs exist?
        sub_ok = all(
            os.path.isdir(os.path.join(self._model_dir, s))
            for s in ("hy3dshape", "hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1")
        ) if model_ok else False

        available = script_ok and sub_ok
        status = EngineStatus.ONLINE if available else EngineStatus.OFFLINE
        return {
            "status": status.value,
            "available": available,
            "script_path": self._script_path,
            "model_dir": self._model_dir,
            "model_ok": model_ok,
            "subdirs_ok": sub_ok,
        }
