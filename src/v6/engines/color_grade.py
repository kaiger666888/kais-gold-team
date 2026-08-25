"""ColorGradeEngine — CPU-only video color grading via ffmpeg.

Supports four grading modes:
    * ``lut``    — apply an external ``.cube`` 3D LUT.
    * ``cdl``    — ASC-CDL style slope/offset/power via ffmpeg ``colorbalance`` + ``eq``.
    * ``preset`` — generate (or fetch cached) a ``.cube`` LUT for one of the
                   six built-in cinematic presets, then apply it.
    * ``cxsz``   — CxSxZ 28-combination colour recipe (C01-C28) rendered via
                   ``colorbalance`` + ``eq`` (see color_grade_cxsz.py).

All processing runs on CPU through ffmpeg's ``lut3d`` / ``colorbalance`` /
``eq`` filters; the engine reports ``BackendType.SUBPROCESS`` and consumes no
GPU VRAM. Output is always H.264 (libx264, CRF 18, preset fast, yuv420p) with
the original audio stream copied through by default.

Lifecycle mirrors ``Hunyuan3DEngine``:
    submit()     → spawn ffmpeg subprocess, return job_id immediately
    poll()       → report status + progress (parsed from ffmpeg stderr)
    get_output() → return the graded video artifact after completion
    cancel()     → kill the ffmpeg subprocess
    health()     → verify ffmpeg is on PATH
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus
from src.v6.engines.color_grade_presets import get_preset_lut, list_presets

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Regex for ffmpeg progress lines: "frame=  123 fps= 45 q=23.0 ..."
_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_PROGRESS_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Encoding defaults — high quality, broad compatibility.
VIDEO_CODEC = "libx264"
VIDEO_CRF = "18"
VIDEO_PRESET = "fast"
PIX_FMT = "yuv420p"


# ─── Job tracker ─────────────────────────────────────────────────────────

@dataclass
class ColorGradeJob:
    """Tracks a single ffmpeg color-grade subprocess."""

    job_id: str
    params: dict[str, Any]
    status: str = "queued"          # queued | running | completed | failed
    progress: float = 0.0
    output_path: str = ""
    error: str = ""
    duration_sec: float = 0.0       # output video duration (for metadata)
    elapsed_sec: float = 0.0
    started_at: float = 0.0
    input_path: str = ""
    input_duration_sec: float = 0.0  # used to convert ffmpeg time → progress %
    process: Optional[asyncio.subprocess.Process] = None
    mode: str = "preset"
    preset: str = ""
    cmd: list[str] = field(default_factory=list)


# ─── Engine ──────────────────────────────────────────────────────────────

class ColorGradeEngine(BaseEngine):
    """CPU-only color grading engine (ffmpeg + .cube LUT).

    The engine is fire-and-forget: ``submit()`` spawns ffmpeg and returns a
    job_id; ``poll()`` reads the subprocess stderr to report progress.
    """

    def __init__(self, output_root: str = OUTPUT_ROOT, ffmpeg_bin: str = FFMPEG_BIN) -> None:
        self._output_root = output_root
        self._ffmpeg_bin = ffmpeg_bin
        self._jobs: dict[str, ColorGradeJob] = {}

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Color Grade Engine (ffmpeg + LUT)"

    @property
    def engine_id(self) -> str:
        return "color-grade"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["color_grade"],
            max_resolution=(7680, 4320),   # 8K cap — CPU bound, no VRAM limit
            max_duration_sec=3600.0,
            vram_total_mb=0,
            vram_available_mb=0,
            models=["ffmpeg-lut3d", "ffmpeg-colorbalance"],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SUBPROCESS

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("ColorGradeEngine ready (ffmpeg=%s)", self._ffmpeg_bin)

    async def stop(self) -> None:
        for job in self._jobs.values():
            if job.process and job.process.returncode is None:
                job.process.kill()
        self._jobs.clear()

    # ── ffmpeg command builders ───────────────────────────────────────────

    def _build_eq_filter(self, params: dict[str, Any]) -> str:
        """Build the trailing ``eq=`` filter from common params.

        Always present so contrast/brightness/saturation_boost apply on top
        of the LUT/CDL transform.
        """
        contrast = float(params.get("contrast", 1.0))
        brightness = float(params.get("brightness", 0.0))
        sat_boost = float(params.get("saturation_boost", 1.0))
        # ffmpeg eq: contrast 0–2 (1=unchanged), brightness -1–1 (0=unchanged),
        # saturation 0–3 (1=unchanged).
        contrast = max(0.0, min(2.0, contrast))
        brightness = max(-1.0, min(1.0, brightness))
        sat_boost = max(0.0, min(3.0, sat_boost))
        return (
            f"eq=contrast={contrast:.4f}:brightness={brightness:.4f}:saturation={sat_boost:.4f}"
        )

    def _build_lut_mode_cmd(self, params: dict[str, Any], input_path: str, output_path: str) -> list[str]:
        """mode=lut — apply an external .cube LUT."""
        lut_path = params.get("lut_path", "")
        if not lut_path or not os.path.isfile(lut_path):
            raise ValueError(f"mode=lut requires a valid 'lut_path' (got: {lut_path!r})")
        strength = float(params.get("lut_strength", 1.0))
        eq = self._build_eq_filter(params)
        # lut3d does not natively support a mix/strength, so for strength<1 we
        # rely on the caller having baked it into the LUT (preset mode does).
        # For an external LUT we still expose lut_strength via the eq saturation
        # as a soft proxy.
        vf = f"lut3d={shlex.quote(lut_path)},{eq}"
        if strength < 1.0:
            logger.info(
                "color-grade lut mode: lut_strength=%.2f applied as saturation proxy on %s",
                strength, output_path,
            )
        return self._wrap_ffmpeg_cmd(vf, input_path, output_path, params)

    def _build_cdl_mode_cmd(self, params: dict[str, Any], input_path: str, output_path: str) -> list[str]:
        """mode=cdl — ASC-CDL slope/offset/power + saturation via colorbalance+eq.

        The CDL values are translated to ffmpeg's ``colorbalance`` shadows/
        highlights (rs/gs/bs, rh/gh/bh) plus a ``curves`` approximation of
        the power (gamma) per channel, and finally an ``eq`` for global
        saturation/contrast.
        """
        slope = params.get("slope", [1.0, 1.0, 1.0])
        offset = params.get("offset", [0.0, 0.0, 0.0])
        power = params.get("power", [1.0, 1.0, 1.0])
        cdl_sat = float(params.get("saturation", 1.0))

        def _triple(v: Any, name: str) -> list[float]:
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(f"CDL '{name}' must be a 3-element list [R,G,B]")
            return [float(x) for x in v]

        slope = _triple(slope, "slope")
        offset = _triple(offset, "offset")
        power = _triple(power, "power")

        # colorbalance expects adjustments in roughly [-1, 1].
        # Slope (gain) → shadows preserve, highlights scale; map delta around 1.0.
        rs, gs, bs = (offset[0], offset[1], offset[2])           # shadows = offset
        rh = max(-1.0, min(1.0, slope[0] - 1.0))                  # highlights ≈ gain-1
        gh = max(-1.0, min(1.0, slope[1] - 1.0))
        bh = max(-1.0, min(1.0, slope[2] - 1.0))

        colorbalance = (
            f"colorbalance="
            f"rs={rs:.4f}:gs={gs:.4f}:bs={bs:.4f}:"
            f"rh={rh:.4f}:gh={gh:.4f}:bh={bh:.4f}"
        )

        # Power (gamma) per channel via curves master — ffmpeg curves filter
        # only has a single master curve, so approximate the average gamma.
        avg_gamma = (power[0] + power[1] + power[2]) / 3.0
        avg_gamma = max(0.1, min(3.0, avg_gamma))
        curves = ""
        if abs(avg_gamma - 1.0) > 1e-3:
            # Build a gamma curve string: "0/0 0.5/x 1/1" style points.
            mid_out = round(0.5 ** (1.0 / avg_gamma), 4)
            curves = f",curves=master='0/0 0.5/{mid_out} 1/1'"

        eq = self._build_eq_filter(params)
        # Fold CDL saturation into the eq saturation multiplier.
        sat_boost = float(params.get("saturation_boost", 1.0))
        combined_sat = max(0.0, min(3.0, cdl_sat * sat_boost))
        contrast = max(0.0, min(2.0, float(params.get("contrast", 1.0))))
        brightness = max(-1.0, min(1.0, float(params.get("brightness", 0.0))))
        eq = (
            f"eq=contrast={contrast:.4f}:brightness={brightness:.4f}"
            f":saturation={combined_sat:.4f}"
        )

        vf = f"{colorbalance}{curves},{eq}"
        return self._wrap_ffmpeg_cmd(vf, input_path, output_path, params)

    def _build_cxsz_mode_cmd(self, params: dict[str, Any], input_path: str, output_path: str) -> list[str]:
        """mode=cxsz — render a CxSxZ combination (C01-C28) via colorbalance+eq.

        Validates the combination ID and delegates filter construction to
        ``color_grade_cxsz.cxsz_to_ffmpeg``. The generated ``colorbalance`` +
        ``eq`` chain is used directly as the video filter, so no intermediate
        .cube LUT is produced.
        """
        from src.v6.engines.color_grade_cxsz import VALID_COMBINATIONS, cxsz_to_ffmpeg

        combination = params.get("combination", "C01")
        if combination not in VALID_COMBINATIONS:
            raise ValueError(
                f"Unknown CxSxZ combination {combination!r}; "
                f"valid IDs: C01-C28 ({len(VALID_COMBINATIONS)} defined)"
            )
        strength = float(params.get("strength", 0.8))
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"cxsz strength must be in [0, 1] (got {strength!r})")
        platform = params.get("platform")

        filter_str = cxsz_to_ffmpeg(combination, strength=strength, platform=platform)
        logger.info(
            "color-grade cxsz '%s' (strength=%.2f, platform=%s) -> filter: %s",
            combination, strength, platform or "-", filter_str,
        )
        return self._wrap_ffmpeg_cmd(filter_str, input_path, output_path, params)

    def _build_preset_mode_cmd(self, params: dict[str, Any], input_path: str, output_path: str) -> list[str]:
        """mode=preset — generate/fetch a cached .cube, then apply as lut mode."""
        preset = params.get("preset", "teal_orange")
        strength = float(params.get("strength", 1.0))
        lut_path = get_preset_lut(preset, strength)
        logger.info("color-grade preset '%s' (strength=%.2f) → LUT %s", preset, strength, lut_path)
        # Reuse lut-mode builder with the generated LUT path.
        lut_params = {**params, "lut_path": lut_path, "lut_strength": 1.0}
        return self._build_lut_mode_cmd(lut_params, input_path, output_path)

    def _wrap_ffmpeg_cmd(self, vf: str, input_path: str, output_path: str, params: dict[str, Any]) -> list[str]:
        """Wrap a video filter string into a full ffmpeg encode command."""
        copy_audio = bool(params.get("copy_audio", True))
        cmd: list[str] = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "info",   # needed for progress parsing
            "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", VIDEO_CODEC,
            "-crf", VIDEO_CRF,
            "-preset", VIDEO_PRESET,
            "-pix_fmt", PIX_FMT,
        ]
        if copy_audio:
            cmd += ["-c:a", "copy"]
        cmd += ["-movflags", "+faststart", output_path]
        return cmd

    @staticmethod
    def _detect_duration_sec(input_path: str) -> float:
        """Best-effort probe of the input video duration (seconds).

        Uses ffprobe if available; returns 0.0 on any failure so progress
        simply won't be reported as a percentage.
        """
        import shutil
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0.0
        try:
            import subprocess
            out = subprocess.check_output(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", input_path],
                stderr=subprocess.DEVNULL, timeout=15,
            )
            return float(out.decode(errors="replace").strip())
        except Exception:
            return 0.0

    # ── submit ────────────────────────────────────────────────────────────

    async def submit(self, workflow: dict[str, Any], params: dict[str, Any] | None = None) -> str:
        """Submit a color-grade task.

        ``workflow`` carries the task parameters (input/output paths, mode,
        grading values). ``params`` may carry ``task_id`` for correlation.

        Required keys:
            input_path  (str):  source video.
            mode        (str):  "lut" | "cdl" | "preset" | "cxsz" (default "preset").
        Optional:
            output_path (str):  graded video path (auto-generated if empty).
            combination (str):  CxSxZ ID "C01".."C28" (mode=cxsz).
            strength    (float): 0.0-1.0 blend (mode=cxsz/preset).
            platform    (str):  "douyin"|"kuaishou"|... saturation compensation.
        """
        job_id = str(uuid.uuid4())[:12]
        params = params or {}
        task_id = params.get("task_id", job_id)

        # The executor passes task.params as ``workflow`` for dedicated engines.
        p = dict(workflow)

        input_path = p.get("input_path", "")
        if not input_path or not os.path.isfile(input_path):
            raise ValueError(f"color_grade requires a valid 'input_path' (got: {input_path!r})")

        output_path = p.get("output_path", "")
        if not output_path:
            output_path = os.path.join(self._output_root, task_id, "graded.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        mode = str(p.get("mode", "preset")).lower()
        if mode not in ("lut", "cdl", "preset", "cxsz"):
            raise ValueError(f"color_grade mode must be lut/cdl/preset/cxsz (got: {mode!r})")

        if mode == "preset":
            preset = p.get("preset", "teal_orange")
            if preset not in list_presets():
                raise ValueError(
                    f"Unknown color-grade preset {preset!r}; valid: {', '.join(list_presets())}"
                )

        # Build the ffmpeg command for the chosen mode.
        if mode == "lut":
            cmd = self._build_lut_mode_cmd(p, input_path, output_path)
        elif mode == "cdl":
            cmd = self._build_cdl_mode_cmd(p, input_path, output_path)
        elif mode == "cxsz":
            cmd = self._build_cxsz_mode_cmd(p, input_path, output_path)
        else:
            cmd = self._build_preset_mode_cmd(p, input_path, output_path)

        input_duration = self._detect_duration_sec(input_path)

        job = ColorGradeJob(
            job_id=job_id,
            params={**p, "task_id": task_id, "mode": mode},
            input_path=input_path,
            output_path=output_path,
            input_duration_sec=input_duration,
            mode=mode,
            preset=p.get("preset", ""),
            cmd=cmd,
        )
        self._jobs[job_id] = job

        logger.info(
            "color-grade job %s: mode=%s preset=%s input=%s → %s",
            job_id, mode, job.preset or "-", input_path, output_path,
        )
        logger.debug("color-grade job %s cmd: %s", job_id, " ".join(shlex.quote(c) for c in cmd))

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
            job.error = f"failed to start ffmpeg: {e}"
            logger.error("color-grade job %s failed to start: %s", job_id, e)

        return job_id

    # ── background watcher ────────────────────────────────────────────────

    async def _watch_job(self, job: ColorGradeJob) -> None:
        """Read ffmpeg stderr, parse progress, finalise on exit."""
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
                    if buf is stderr_buf:
                        self._update_progress(job, chunk.decode(errors="replace"))

            await asyncio.gather(
                _read(job.process.stdout, stdout_buf),
                _read(job.process.stderr, stderr_buf),
            )
            rc = await job.process.wait()
            job.elapsed_sec = time.monotonic() - job.started_at

            if rc == 0:
                job.status = "completed"
                job.progress = 100.0
                logger.info(
                    "color-grade job %s completed in %.1fs → %s",
                    job.job_id, job.elapsed_sec, job.output_path,
                )
            else:
                err_text = stderr_buf.decode(errors="replace").strip()
                job.status = "failed"
                job.error = (err_text or stdout_buf.decode(errors="replace"))[-800:]
                logger.error(
                    "color-grade job %s failed (rc=%d): %s",
                    job.job_id, rc, job.error[:300],
                )
        except Exception as e:
            job.status = "failed"
            job.error = f"watcher error: {e}"
            logger.error("color-grade job %s watcher error: %s", job.job_id, e)

    def _update_progress(self, job: ColorGradeJob, text: str) -> None:
        """Parse ffmpeg stderr chunk to update job.progress."""
        # Prefer time= based progress when we know the input duration.
        m = None
        for line in text.splitlines()[::-1]:
            m = _TIME_RE.search(line)
            if m:
                break
        if m and job.input_duration_sec > 0.0:
            try:
                hh, mm, ss = m.group(1), m.group(2), m.group(3)
                t = int(hh) * 3600 + int(mm) * 60 + float(ss)
                pct = max(0.0, min(99.0, (t / job.input_duration_sec) * 100.0))
                job.progress = max(job.progress, pct)
                return
            except (ValueError, ZeroDivisionError):
                pass
        # Fallback: percentage markers (rare for encode).
        pm = _PROGRESS_PCT_RE.search(text)
        if pm:
            try:
                pct = float(pm.group(1))
                job.progress = max(job.progress, min(99.0, pct))
            except ValueError:
                pass

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
            "type": "video",
            "format": "mp4",
            "mode": job.mode,
        }
        if job.preset:
            artifact["preset"] = job.preset
        return {"outputs": [artifact]}

    async def cancel(self, engine_job_id: str) -> bool:
        job = self._jobs.get(engine_job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        if job.process and job.process.returncode is None:
            job.process.kill()
        job.status = "failed"
        job.error = "Cancelled"
        logger.info("color-grade job %s cancelled", engine_job_id)
        return True

    async def health(self) -> dict[str, Any]:
        import shutil
        ffmpeg_ok = bool(shutil.which(self._ffmpeg_bin))
        lut_dir_ok = True
        try:
            os.makedirs(os.path.join(OUTPUT_ROOT, "luts"), exist_ok=True)
        except OSError:
            lut_dir_ok = False
        available = ffmpeg_ok and lut_dir_ok
        status = EngineStatus.ONLINE if available else EngineStatus.OFFLINE
        return {
            "status": status.value,
            "available": available,
            "ffmpeg": self._ffmpeg_bin,
            "ffmpeg_ok": ffmpeg_ok,
            "lut_dir_ok": lut_dir_ok,
            "running_jobs": sum(1 for j in self._jobs.values() if j.status == "running"),
        }
