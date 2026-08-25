"""Engine abstraction layer — pluggable GPU engine interfaces."""
import logging

from src.v6.engines.base import BaseEngine, EngineStatus, EngineCapabilities
from src.v6.engines.comfyui import ComfyUIEngine
from src.v6.engines.color_grade import ColorGradeEngine
from src.v6.engines.hunyuan3d import Hunyuan3DEngine
from src.v6.engines.hunyuan3d_mv import Hunyuan3DMvEngine
from src.v6.engines.mock import MockEngine
from src.v6.engines.tts_http import TripleTrackTTSEngine

# In-process TTS tracker pulls the heavy local TTS stack (numba/torch/
# librosa/...). Not installed in the slim image — degrade gracefully; TTS
# still works via TripleTrackTTSEngine (HTTP) above.
try:
    from src.v6.engines.tts import TTSTracker, TTSTrack
except Exception as _tts_exc:  # pragma: no cover — depends on image flavour
    TTSTracker = None
    TTSTrack = None
    logging.getLogger(__name__).warning(
        "In-process TTS tracker unavailable (heavy deps not installed): %s",
        _tts_exc,
    )

__all__ = [
    "BaseEngine",
    "ColorGradeEngine",
    "ComfyUIEngine",
    "EngineStatus",
    "EngineCapabilities",
    "Hunyuan3DEngine",
    "Hunyuan3DMvEngine",
    "MockEngine",
    "TTSTracker",
    "TTSTrack",
    "TripleTrackTTSEngine",
]
