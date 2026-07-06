"""JoyCaption image captioning engine — uses ComfyUI JC_GGUF node via HTTP."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from src.v6.engines.base import BackendType, BaseEngine, EngineCapabilities, EngineStatus

logger = logging.getLogger(__name__)

DEFAULT_COMFYUI_HOST = "127.0.0.1"
DEFAULT_COMFYUI_PORT = 8188
POLL_INTERVAL_SEC = 0.5
POLL_TIMEOUT_SEC = 120.0  # 2 min max for image captioning

# Default output directory — ComfyUI output inside the container
COMFYUI_OUTPUT_DIR = "/workspace/ComfyUI/output"


class JoyCaptionEngine(BaseEngine):
    """JoyCaption image captioning engine via ComfyUI JC_GGUF node.

    Uses the same ComfyUI instance as ComfyUIEngine but specializes
    in image-to-text (captioning). Workflows:
      1. Upload image to ComfyUI
      2. Build LoadImage → JC_GGUF → CaptionSaver workflow
      3. Submit & poll
      4. Read caption text from output .txt file or logs
    """

    def __init__(
        self,
        host: str = DEFAULT_COMFYUI_HOST,
        port: int = DEFAULT_COMFYUI_PORT,
        output_dir: str = COMFYUI_OUTPUT_DIR,
        poll_interval: float = POLL_INTERVAL_SEC,
        poll_timeout: float = POLL_TIMEOUT_SEC,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = str(uuid.uuid4())
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._base_url = f"http://{host}:{port}"
        self._output_dir = output_dir
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "JoyCaption Local"

    @property
    def engine_id(self) -> str:
        return "joycaption-local"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supported_types=["image_caption", "image_draw", "image_refine"],
            max_resolution=(2048, 2048),
            max_duration_sec=10.0,
            vram_total_mb=24576,
            vram_available_mb=24576,
            models=[
                "JoyCaption Alpha Two (Q6_K)",
            ],
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.COMFYUI

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        logger.info("JoyCaption engine client started → %s", self._base_url)

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ─── Convenience: caption ───

    async def caption(
        self,
        image_url: str,
        prompt_style: str = "Descriptive",
        processing_mode: str = "GPU",
        caption_length: str = "any",
        extra_prompt: str = "",
        temperature: float = 0.6,
    ) -> dict[str, Any]:
        """High-level: caption an image and return the text.

        Args:
            image_url: URL or local path to the image.
            prompt_style: JoyCaption prompt style (Descriptive, Detailed, etc.).
            processing_mode: GPU or CPU.
            caption_length: any, short, medium, long.
            extra_prompt: Additional instructions appended to the system prompt.
            temperature: Generation temperature.

        Returns:
            ``{"status": "completed"|"failed", "caption": str|None, ...}``
        """
        workflow = self._build_workflow(
            prompt_style=prompt_style,
            processing_mode=processing_mode,
            caption_length=caption_length,
            extra_prompt=extra_prompt,
            temperature=temperature,
        )
        result = await self.submit_and_wait(image_url, workflow)

        if result.get("status") == "completed":
            caption = result.get("caption", "")
            return {
                "status": "completed",
                "caption": caption,
                "engine_job_id": result.get("engine_job_id"),
            }
        return result

    # ─── Core API (BaseEngine interface) ───

    async def submit(
        self,
        workflow: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> str:
        """Submit a pre-built ComfyUI workflow for captioning.

        Note: For image captioning, use ``submit_and_wait`` or ``caption`` instead,
        as the workflow needs image upload first.

        Returns:
            ComfyUI prompt_id.
        """
        assert self._http is not None, "Engine not started"

        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }

        resp = await self._http.post("/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()

        prompt_id: str = data["prompt_id"]
        logger.info("JoyCaption workflow submitted: %s", prompt_id)
        return prompt_id

    async def submit_and_wait(
        self,
        image_url: str,
        workflow: dict[str, Any],
    ) -> dict[str, Any]:
        """Upload image, build workflow, submit, and wait for caption.

        Args:
            image_url: URL or local path of the image to caption.
            workflow: ComfyUI API workflow JSON (may have placeholder filenames).

        Returns:
            ``{"status": ..., "caption": str|None, ...}``
        """
        assert self._http is not None, "Engine not started"

        # 1. Download / read image data
        image_data, image_name = await self._load_image(image_url)

        # 2. Upload to ComfyUI
        uploaded_name, image_subfolder = await self._upload_image(image_data, image_name)

        # 3. Inject filename into workflow
        workflow = self._inject_image_filename(workflow, uploaded_name, image_subfolder)

        # 4. Submit
        prompt_id = await self.submit(workflow)

        # 5. Poll until done
        elapsed = 0.0
        while elapsed < self._poll_timeout:
            result = await self.poll(prompt_id)
            status = result["status"]

            if status == "completed":
                # 6. Extract caption text
                caption = await self._extract_caption(uploaded_name, prompt_id)
                return {
                    "status": "completed",
                    "engine_job_id": prompt_id,
                    "caption": caption,
                }
            if status == "failed":
                return {
                    "status": "failed",
                    "engine_job_id": prompt_id,
                    "error": result.get("error"),
                }

            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        # Timeout
        await self.cancel(prompt_id)
        return {"status": "failed", "engine_job_id": prompt_id, "error": "Caption timed out"}

    async def poll(self, engine_job_id: str) -> dict[str, Any]:
        """Poll ComfyUI execution status via /history endpoint."""
        assert self._http is not None

        resp = await self._http.get(f"/history/{engine_job_id}")
        if resp.status_code == 404:
            return {"status": "queued", "progress": 0.0}

        resp.raise_for_status()
        history = resp.json()

        if engine_job_id not in history:
            return {"status": "queued", "progress": 0.0}

        item = history[engine_job_id]

        # Check for error
        if "status" in item:
            status_data = item["status"]
            if status_data.get("status_str") == "error":
                messages = status_data.get("messages", [])
                error_msg = "ComfyUI execution error"
                for msg in messages:
                    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                        if msg[0] == "execution_error" and isinstance(msg[1], dict):
                            error_msg = msg[1].get("exception_message", error_msg)
                            break
                    elif isinstance(msg, str):
                        error_msg = msg
                        break
                return {"status": "failed", "progress": 0.0, "error": error_msg}

        # Check outputs → completed
        outputs = item.get("outputs", {})
        if outputs:
            return {"status": "completed", "progress": 100.0, "outputs": outputs}

        return {"status": "running", "progress": 50.0}

    async def get_output(self, engine_job_id: str) -> dict[str, Any]:
        """Get output artifacts from ComfyUI history."""
        assert self._http is not None

        resp = await self._http.get(f"/history/{engine_job_id}")
        resp.raise_for_status()
        history = resp.json()

        item = history.get(engine_job_id, {})
        outputs = item.get("outputs", {})

        artifacts: list[dict[str, Any]] = []
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                img_type = img.get("type", "output")
                url = (
                    f"{self._base_url}/view?"
                    f"filename={filename}&subfolder={subfolder}&type={img_type}"
                )
                artifacts.append({"url": url, "type": "image", "format": "png", "node": node_id})

        return {"outputs": artifacts}

    async def cancel(self, engine_job_id: str) -> bool:
        """Interrupt ComfyUI execution."""
        assert self._http is not None
        try:
            resp = await self._http.post("/interrupt")
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("JoyCaption cancel failed: %s", e)
            return False

    async def health(self) -> dict[str, Any]:
        """Check ComfyUI availability (shared instance)."""
        if not self._http:
            return {"status": EngineStatus.OFFLINE.value, "available": False}

        try:
            resp = await self._http.get("/system_stats", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()

            devices = data.get("devices", [])
            vram_total = 0
            vram_free = 0
            for d in devices:
                vram_total += d.get("vram_total", 0)
                vram_free += d.get("vram_free", 0)

            return {
                "status": EngineStatus.ONLINE.value,
                "available": True,
                "vram_total_mb": vram_total // (1024 * 1024),
                "vram_available_mb": vram_free // (1024 * 1024),
                "devices": devices,
            }
        except Exception as e:
            logger.warning("JoyCaption health check failed: %s", e)
            return {
                "status": EngineStatus.OFFLINE.value,
                "available": False,
                "error": str(e),
            }

    # ─── Internal helpers ───

    def _build_workflow(
        self,
        prompt_style: str = "Descriptive",
        processing_mode: str = "GPU",
        caption_length: str = "any",
        extra_prompt: str = "",
        temperature: float = 0.6,
    ) -> dict[str, Any]:
        """Build ComfyUI API workflow JSON for JoyCaption.

        Uses placeholder filename "PLACEHOLDER_IMAGE.png" — actual
        filename is injected after upload.
        """
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": "PLACEHOLDER_IMAGE.png",
                },
            },
            "2": {
                "class_type": "JC_GGUF",
                "inputs": {
                    "image": ["1", 0],
                    "model": "JoyCaption Alpha Two (Q6_K)",
                    "processing_mode": processing_mode,
                    "prompt_style": prompt_style,
                    "caption_length": caption_length,
                    "extra_prompt": extra_prompt,
                    "temperature": temperature,
                    "memory_management": "Keep in Memory",
                },
            },
            "3": {
                "class_type": "CaptionSaver",
                "inputs": {
                    "string": ["2", 0],
                    "image_path": "PLACEHOLDER_IMAGE.png",
                    "image": ["1", 0],
                },
            },
        }

    def _inject_image_filename(
        self,
        workflow: dict[str, Any],
        filename: str,
        subfolder: str,
    ) -> dict[str, Any]:
        """Replace placeholder image filenames in workflow with actual uploaded name."""
        # ComfyUI uses filename without subfolder in the input field when subfolder is ""
        display_name = filename if not subfolder else f"{subfolder}/{filename}"

        import copy
        wf = copy.deepcopy(workflow)

        # LoadImage node (id "1")
        if "1" in wf and "inputs" in wf["1"]:
            wf["1"]["inputs"]["image"] = display_name

        # CaptionSaver node (id "3") — image_path field
        if "3" in wf and "inputs" in wf["3"]:
            wf["3"]["inputs"]["image_path"] = display_name

        return wf

    async def _load_image(self, image_url: str) -> tuple[bytes, str]:
        """Load image from URL or local file path."""
        if image_url.startswith(("http://", "https://")):
            assert self._http is not None
            resp = await self._http.get(image_url)
            resp.raise_for_status()
            return resp.content, image_url.rsplit("/", 1)[-1].split("?")[0] or "image.png"
        else:
            # Local file path
            path = Path(image_url)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_url}")
            data = path.read_bytes()
            return data, path.name

    async def _upload_image(
        self,
        image_data: bytes,
        image_name: str,
        subfolder: str = "",
        image_type: str = "input",
        overwrite: bool = True,
    ) -> tuple[str, str]:
        """Upload image to ComfyUI /upload/image endpoint.

        Returns:
            (filename, subfolder) as recognized by ComfyUI.
        """
        assert self._http is not None

        # Ensure proper extension
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
            image_name += ".png"

        data = {
            "subfolder": subfolder,
            "type": image_type,
            "overwrite": str(overwrite).lower(),
        }
        files = {
            "image": (image_name, image_data),
        }

        resp = await self._http.post("/upload/image", data=data, files=files)
        resp.raise_for_status()
        result = resp.json()

        # ComfyUI may rename the file to avoid conflicts
        uploaded_name = result.get("name", image_name)
        uploaded_subfolder = result.get("subfolder", subfolder)
        logger.info("Image uploaded to ComfyUI: %s (subfolder=%s)", uploaded_name, uploaded_subfolder)
        return uploaded_name, uploaded_subfolder

    async def _extract_caption(
        self,
        image_name: str,
        prompt_id: str,
    ) -> str:
        """Extract caption text from ComfyUI output.

        Strategy:
        1. Try reading the output .txt file from ComfyUI output dir
        2. Fall back to parsing ComfyUI node output text
        3. Fall back to ComfyUI logs
        """
        # Strategy 1: Read .txt file from output directory
        # CaptionSaver saves to {image_name}.txt in the output folder
        base_name = Path(image_name).stem
        txt_path = Path(self._output_dir) / f"{base_name}.txt"

        if txt_path.exists():
            caption = txt_path.read_text().strip()
            if caption:
                logger.info("Caption extracted from file: %s", txt_path)
                return caption

        # Strategy 2: Check node outputs for text
        try:
            output = await self.get_output(prompt_id)
            # Some custom nodes put text in outputs directly
            for node_id, node_data in output.items():
                if isinstance(node_data, dict) and "text" in node_data:
                    text = node_data["text"]
                    if isinstance(text, list) and text:
                        return text[0].get("text", "")
        except Exception:
            pass

        logger.warning("Could not extract caption text for prompt %s", prompt_id)
        return ""
