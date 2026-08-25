"""Pydantic models for V6.0 task lifecycle — strict OpenAPI compliance."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enums ───

class TaskType(str, enum.Enum):
    VIDEO_FINAL = "video_final"
    VIDEO_PREVIEW = "video_preview"
    IMAGE_DRAW = "image_draw"
    IMAGE_REFINE = "image_refine"
    TTS = "tts"
    TTS_ZH = "tts_zh"              # GPT-SoVITS 中文语音合成
    TTS_EN = "tts_en"              # Chatterbox-Turbo 英文语音合成
    TTS_BILINGUAL = "tts_bilingual"  # CosyVoice 双语语音合成
    MUSIC = "music"
    SFX = "sfx"
    UPSCALE = "upscale"
    FACE_RESTORE = "face_restore"
    IMAGE_TO_3D = "image_to_3d"
    IMAGE_PULID = "image_pulid"            # PuLID character-consistency injection (FLUX + PuLID)
    IMAGE_DRAW_IPADAPTER = "image_draw_ipadapter"  # IP-Adapter character/style consistency (FLUX + IP-Adapter)
    CONTROLNET_DEPTH = "controlnet_depth"  # ControlNet Depth geometry lock (FLUX + ControlNet)
    WAN_I2V = "wan_i2v"                    # Wan 2.1 I2V dual-stage video generation
    IMAGE_TO_3D_MV = "image_to_3d_mv"          # Hunyuan3D-2mv multiview image-to-3D
    SHOT_ANALYSIS = "shot_analysis"            # 逐镜头运镜解构(几何+语义+主体)
    COLOR_GRADE = "color_grade"                # 视频调色 (LUT + CDL + ffmpeg, CPU only)


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, enum.Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ModelPreference(str, enum.Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


class EnginePool(str, enum.Enum):
    LOCAL = "local"
    CLOUD = "cloud"


# ─── Request Models ───

class TaskCreateRequest(BaseModel):
    task_id: str = Field(..., description="Caller-assigned unique task identifier")
    type: TaskType
    model_preference: ModelPreference = ModelPreference.AUTO
    params: dict[str, Any] = Field(..., description="Type-specific generation parameters")
    callback_url: Optional[str] = Field(None, format="uri")
    callback_secret: Optional[str] = None
    priority: Priority = Priority.NORMAL


class BatchCreateRequest(BaseModel):
    tasks: list[TaskCreateRequest] = Field(..., min_length=1, max_length=100)
    fail_fast: bool = False


# ─── Response Models ───

class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: str = "queued"  # queued | scheduled
    engine_target: str = "pending"  # local | cloud | pending
    queue_position: Optional[int] = None
    estimated_start_sec: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskOutputs(BaseModel):
    video: Optional[str] = None
    thumbnail: Optional[str] = None
    audio: Optional[str] = None
    image: Optional[str] = None

    model_config = {"extra": "allow"}


class TaskMetadata(BaseModel):
    seed: Optional[int] = None
    cost_usd: Optional[float] = None
    inference_time_sec: Optional[float] = None
    gpu_memory_peak_gb: Optional[float] = None
    model_name: Optional[str] = None
    cloud_task_id: Optional[str] = None

    model_config = {"extra": "allow"}


class TaskDetailResponse(BaseModel):
    task_id: str
    type: TaskType
    status: TaskStatus
    priority: Priority = Priority.NORMAL
    model_preference: ModelPreference = ModelPreference.AUTO
    engine_used: Optional[EnginePool] = None
    engine_id: Optional[str] = None
    params: dict[str, Any] = {}
    callback_url: Optional[str] = None
    outputs: Optional[TaskOutputs] = None
    metadata: Optional[TaskMetadata] = None
    error: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0, le=100)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskDetailResponse]
    total: int
    limit: int
    offset: int


class TaskCancelResponse(BaseModel):
    task_id: str
    status: str = "cancelled"
    message: str = "Task cancelled successfully"


# ─── Batch Response ───

class BatchTaskResult(BaseModel):
    task_id: str
    status: str  # queued | rejected
    error: Optional[str] = None
    queue_position: Optional[int] = None


class BatchCreateResponse(BaseModel):
    batch_id: str
    accepted: int
    rejected: int
    results: list[BatchTaskResult]


# ─── SSE Event ───

class SSEEventData(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    outputs: Optional[TaskOutputs] = None
    metadata: Optional[TaskMetadata] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SSEEvent(BaseModel):
    event: str  # queued | started | progress | completed | failed | cancelled
    data: SSEEventData


# ─── Error ───

class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[dict[str, Any]] = None


# ─── Internal Task Store Model ───

class GenerationTask(BaseModel):
    """Full task record stored in the task store."""
    task_id: str
    type: TaskType
    status: TaskStatus = TaskStatus.QUEUED
    priority: Priority = Priority.NORMAL
    model_preference: ModelPreference = ModelPreference.AUTO
    params: dict[str, Any]
    callback_url: Optional[str] = None
    callback_secret: Optional[str] = None
    engine_used: Optional[EnginePool] = None
    engine_id: Optional[str] = None
    outputs: Optional[TaskOutputs] = None
    metadata: Optional[TaskMetadata] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_detail(self) -> TaskDetailResponse:
        return TaskDetailResponse(
            task_id=self.task_id,
            type=self.type,
            status=self.status,
            priority=self.priority,
            model_preference=self.model_preference,
            engine_used=self.engine_used,
            engine_id=self.engine_id,
            params=self.params,
            callback_url=self.callback_url,
            outputs=self.outputs,
            metadata=self.metadata,
            error=self.error,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )
