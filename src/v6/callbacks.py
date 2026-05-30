"""V6.0 standard callback with HMAC-SHA256 signing — GpuTaskCallback compliant."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from src.v6.models.task import GenerationTask, TaskStatus

logger = logging.getLogger(__name__)


def _sign_payload(payload: dict[str, Any], secret: str) -> str:
    """Generate HMAC-SHA256 signature for callback payload."""
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"sha256:{sig}"


def build_callback_payload(task: GenerationTask) -> dict[str, Any]:
    """Build GpuTaskCallback-compliant payload from a completed task."""
    # Map internal status to callback status
    status_map = {
        TaskStatus.COMPLETED: "success",
        TaskStatus.FAILED: "failed",
        TaskStatus.CANCELLED: "partial",
    }
    callback_status = status_map.get(task.status, "failed")

    # Map internal TaskType to callback task_type
    type_map = {
        "video_final": "video_generation",
        "video_preview": "video_generation",
        "image_draw": "image_generation",
        "image_refine": "image_generation",
        "tts": "tts_synthesis",
        "music": "bgm_generation",
        "sfx": "audio_generation",
        "upscale": "style_transfer",
        "face_restore": "face_fusion",
        "image_to_3d": "image_generation",
    }
    callback_task_type = type_map.get(task.type.value, "video_generation")

    outputs: list[dict[str, Any]] = []
    if task.outputs:
        if task.outputs.video:
            outputs.append({
                "url": task.outputs.video,
                "type": "video",
                "format": "mp4",
            })
        if task.outputs.image:
            outputs.append({
                "url": task.outputs.image,
                "type": "image",
                "format": "png",
            })
        if task.outputs.audio:
            outputs.append({
                "url": task.outputs.audio,
                "type": "audio",
                "format": "wav",
            })
        if task.outputs.thumbnail:
            outputs.append({
                "url": task.outputs.thumbnail,
                "type": "image",
                "format": "jpg",
            })

    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "job_id": task.params.get("job_id", ""),
        "pipeline_id": task.params.get("pipeline_id"),
        "phase": task.params.get("phase"),
        "shot_id": task.params.get("shot_id"),
        "task_type": callback_task_type,
        "status": callback_status,
        "outputs": outputs,
        "metrics": {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if task.metadata:
        payload["metrics"] = {
            "gpu_time_seconds": task.metadata.inference_time_sec,
            "total_time_seconds": task.metadata.inference_time_sec,
            "gpu_memory_peak_mb": (
                task.metadata.gpu_memory_peak_gb * 1024
                if task.metadata.gpu_memory_peak_gb
                else None
            ),
            "engine": task.engine_id or "mock",
        }
        if task.metadata.cost_usd is not None:
            payload["metrics"]["cost_usd"] = task.metadata.cost_usd

    if task.status == TaskStatus.FAILED and task.error:
        payload["error"] = {
            "code": "MODEL_ERROR",
            "message": task.error,
            "retryable": True,
        }

    return payload


async def send_callback(
    task: GenerationTask,
    callback_url: str,
    callback_secret: Optional[str] = None,
) -> bool:
    """Send V6.0 standard callback with optional HMAC signing."""
    payload = build_callback_payload(task)

    if callback_secret:
        signature = _sign_payload(payload, callback_secret)
        payload["signature"] = signature

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(callback_url, json=payload, headers=headers)
            if resp.status_code < 300:
                logger.info("Callback sent for task %s → %s", task.task_id, resp.status_code)
                return True
            else:
                logger.warning(
                    "Callback failed for task %s → %s: %s",
                    task.task_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return False
    except Exception as e:
        logger.error("Callback error for task %s: %s", task.task_id, e)
        return False
