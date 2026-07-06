"""Hunyuan3D-2mv Engine — multiview image-to-3D via subprocess for gold-team V6.

Extension of the hunyuan3d.py engine pattern. Accepts 1-4 reference images
(front/left/back/right) and runs ``scripts/hunyuan3d_mv_infer.py`` as a child
process, tracking progress via asyncio.

The multiview variant produces more accurate 3D shapes when multiple angle
references are provided (front is always required; left/back/right optional).

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

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "hunyuan3d_mv_infer.py")

DEFAULT_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_2MV_MODEL_DIR",
    "/data/models/tencent/Hunyuan3D-2mv",
)

DEFAULT_CODE_DIR = os.environ.get(
    "HUNYUAN3D_CODE_DIR",
    "/data/models/tencent/Hunyuan3D-2",
)

_PROGRESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class Hunyuan3DMvJob:
    """Tracks a single Hunyuan3D-2mv subprocess job."""

    def __init__(self, job_id: str, params: dict) -> None:
        self.job_id = job_id
        self.params = params
        self.status: str = "queued"
        self.progress: float = 0.0
        self.output_path: str = ""
        self.error: str = ""
        self.vertices: int = 0
        self.faces: int = 0
        self.elapsed_load_sec: float = 0.0
        self.elapsed_inference_sec: float = 0.0
        self.views: list[str] = []
        self.started_at: float = 0.0
        self.process: Optional[asyncio.subprocess.Process] = None


class Hunyuan3DMvEngine(BaseEngine):
    """Subprocess-based engine for Tencent Hunyuan3D-2mv multiview image-to-3D.

    Spawns ``scripts/hunyuan3d_mv_infer.py`` per task.
    """

    def __init__(
        self,
        output_root: str = OUTPUT_ROOT,
        model_dir: str = DEFAULT_MODEL_DIR,
        code_dir: str = DEFAULT_CODE_DIR,
    ) -> None:
        self._output_root = output_root
        self._model_dir = model_dir
        self._code_dir = code_dir
        self._jobs: dict[str, Hunyuan3DMvJob] = {}
        self._script_path = os.path.abspath(SCRIPT_PATH)
        self._python = os.environ.get("KAIS_HUNYUAN3D_PYTHON", "python3")

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Hunyuan3D-2mv Engine (multiview image→GLB)"

    @property
    def engine_id(self) -> str:
        return "hunyuan3d-mv-local"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["image_to_3d_mv"],
            max_duration_sec=600.0,
            vram_total_mb=24000,
            vram_available_mb=24000,
            models=["Hunyuan3D-2mv"],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SUBPROCESS

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not os.path.isfile(self._script_path):
            logger.warning("Hunyuan3D-2mv script not found at %s", self._script_path)
        else:
            logger.info("Hunyuan3D-2mv engine ready, script: %s", self._script_path)

    async def stop(self) -> None:
        for job in self._jobs.values():
            if job.process and job.process.returncode is None:
                job.process.kill()
        self._jobs.clear()

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a multiview image-to-3D task.

        Workflow keys:
            front_image (str, required): Path to front view image.
            left_image (str, optional): Path to left view.
            back_image (str, optional): Path to back view.
            right_image (str, optional): Path to right view.
            output_path (str, optional): GLB output path.
            device (str, optional): e.g. "cuda:0".
            steps (int, optional): Inference steps (default 50).
            seed (int, optional): Reproducibility seed.
        """
        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        front_image = (
            workflow.get("front_image")
            or workflow.get("front")
            or workflow.get("input_image")
            or workflow.get("image")
            or ""
        )
        if not front_image:
            raise ValueError("Hunyuan3D-2mv requires 'front_image' parameter")

        left_image = workflow.get("left_image") or workflow.get("left", "")
        back_image = workflow.get("back_image") or workflow.get("back", "")
        right_image = workflow.get("right_image") or workflow.get("right", "")

        output_path = workflow.get("output_path", "")
        if not output_path:
            output_path = os.path.join(self._output_root, task_id, "model.glb")

        device = workflow.get("device", "cuda:0")
        steps = int(workflow.get("steps", 50))
        seed = workflow.get("seed")
        model_dir = workflow.get("model_dir", self._model_dir)
        code_dir = workflow.get("code_dir", self._code_dir)

        views = {"front": front_image}
        if left_image:
            views["left"] = left_image
        if back_image:
            views["back"] = back_image
        if right_image:
            views["right"] = right_image

        job = Hunyuan3DMvJob(
            job_id=job_id,
            params={
                "front_image": front_image,
                "left_image": left_image,
                "back_image": back_image,
                "right_image": right_image,
                "output_path": output_path,
                "device": device,
                "steps": steps,
                "model_dir": model_dir,
                "code_dir": code_dir,
                "task_id": task_id,
            },
        )
        self._jobs[job_id] = job

        cmd = [
            self._python, self._script_path,
            "--front", front_image,
            "--output", output_path,
            "--model-dir", model_dir,
            "--code-dir", code_dir,
            "--device", device,
            "--steps", str(steps),
        ]
        if left_image:
            cmd.extend(["--left", left_image])
        if back_image:
            cmd.extend(["--back", back_image])
        if right_image:
            cmd.extend(["--right", right_image])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(
            "Hunyuan3D-2mv job %s: submitting (views=%s, device=%s, steps=%d)",
            job_id, list(views.keys()), device, steps,
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
            logger.error("Hunyuan3D-2mv job %s failed to start: %s", job_id, e)

        return job_id

    # ── background watcher ────────────────────────────────────────────────

    async def _watch_job(self, job: Hunyuan3DMvJob) -> None:
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
                    text = chunk.decode(errors="replace")
                    m = None
                    for line in text.splitlines()[::-1]:
                        m = _PROGRESS_RE.search(line)
                        if m:
                            break
                    if m:
                        try:
                            pct = float(m.group(1))
                            job.progress = max(job.progress, min(95.0, pct * 0.95))
                        except ValueError:
                            pass

            await asyncio.gather(
                _read(job.process.stdout, stdout_buf),
                _read(job.process.stderr, stderr_buf),
            )
            rc = await job.process.wait()

            if rc == 0:
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
                job.views = result.get("views", [])
                logger.info(
                    "Hunyuan3D-2mv job %s completed (load=%.1fs, infer=%.1fs, verts=%d, faces=%d, views=%s)",
                    job.job_id, job.elapsed_load_sec, job.elapsed_inference_sec,
                    job.vertices, job.faces, job.views,
                )
            else:
                err_text = stderr_buf.decode(errors="replace").strip()
                job.status = "failed"
                job.error = (err_text or stdout_buf.decode(errors="replace"))[:500]
                logger.error(
                    "Hunyuan3D-2mv job %s failed (rc=%d): %s",
                    job.job_id, rc, job.error[:200],
                )
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Hunyuan3D-2mv job %s watch error: %s", job.job_id, e)

    # ── poll / get_output / cancel / health ───────────────────────────────

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        job = self._jobs.get(engine_job_id)
        if not job:
            return {"status": "failed", "progress": 0.0, "error": "Unknown job ID"}
        result: dict[str, Any] = {
            "status": job.status,
            "progress": job.progress,
            "views": job.views,
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
            "model": "hunyuan3d-2mv",
            "vertices": job.vertices,
            "faces": job.faces,
            "elapsed_load_sec": job.elapsed_load_sec,
            "elapsed_inference_sec": job.elapsed_inference_sec,
            "views": job.views,
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
        logger.info("Hunyuan3D-2mv job %s cancelled", engine_job_id)
        return True

    async def health(self) -> dict[str, Any]:
        script_ok = os.path.isfile(self._script_path)
        model_ok = os.path.isdir(self._model_dir)
        model_file = os.path.isfile(os.path.join(self._model_dir, "model.fp16.safetensors"))
        available = script_ok and model_ok and model_file
        status = EngineStatus.ONLINE if available else EngineStatus.OFFLINE
        return {
            "status": status.value,
            "available": available,
            "script_path": self._script_path,
            "model_dir": self._model_dir,
            "model_file_ok": model_file,
        }
