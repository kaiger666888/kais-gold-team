#!/usr/bin/env python3
"""Standalone Hunyuan3D-2 inference script for kais-gold-team V6.

Two-stage pipeline:
  Stage 1 (shape): Hunyuan3D-2mini generates high-quality geometry from input image.
  Stage 2 (texture, optional): Hunyuan3D-2.0 multiview diffusion paints PBR textures
          onto the geometry from the input image.

Designed to be invoked as a subprocess by ``src/v6/engines/hunyuan3d.py``
(fire-and-forget pattern, identical to ``scripts/tts_infer.py``).

Usage (geometry only):
    python scripts/hunyuan3d_infer.py \\
        --input /path/to/image.png \\
        --output /mnt/agents/output/{task_id}/model.glb \\
        --model mini \\
        --device cuda:0 \\
        --steps 50

Usage (geometry + PBR texture):
    python scripts/hunyuan3d_infer.py \\
        --input /path/to/image.png \\
        --output /mnt/agents/output/{task_id}/model.glb \\
        --model mini \\
        --texture-mode texture \\
        --device cuda:0 \\
        --steps 50

Outputs a single JSON line on stdout (other output goes to stderr) with:
    {
      "output_path": "/path/to/model.glb",
      "vertices": 12345,
      "faces": 24000,
      "elapsed_load_sec": 28.1,
      "elapsed_inference_sec": 75.2,
      "elapsed_texture_sec": 477.3,
      "texture_mode": "texture",
      "model": "mini",
      "device": "cuda:0"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Default model locations — can be overridden via --model-dir or HUNYUAN3D_MODEL_DIR env.
DEFAULT_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_MODEL_DIR",
    "/data/models/tencent/Hunyuan3D-2",
)
# PBR paint model directory (v2-0 non-PBR, compatible with pip hy3dgen package)
DEFAULT_PAINT_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_PAINT_MODEL_DIR",
    "/data/models/Hunyuan3D-2.1",
)


def _setup_paths(model_dir: str, paint_model_dir: str = "") -> None:
    """Add hy3dshape / hy3dpaint submodules to sys.path so their packages import.

    The Hunyuan3D-2 release ships its python code under ``hy3dshape/`` and
    ``hy3dpaint/`` subdirectories. The reference ``run_shape.py`` inserts
    these into sys.path; we mirror that here.

    Mini-model directories don't include the code, so we also add the
    full-model directory and paint model directory as fallbacks.
    """
    search_dirs = [model_dir]
    if paint_model_dir and paint_model_dir not in search_dirs:
        search_dirs.append(paint_model_dir)
    if DEFAULT_MODEL_DIR not in search_dirs:
        search_dirs.append(DEFAULT_MODEL_DIR)
    for base in search_dirs:
        for sub in ("hy3dshape", "hy3dpaint"):
            sub_path = os.path.join(base, sub)
            if os.path.isdir(sub_path) and sub_path not in sys.path:
                sys.path.insert(0, sub_path)


def _apply_torchvision_fix() -> None:
    """Apply optional torchvision compatibility shim shipped with Hunyuan3D-2."""
    try:
        from torchvision_fix import apply_fix  # type: ignore
        apply_fix()
    except Exception:
        # Optional — only present in some distributions.
        pass


def _patch_config_targets(model_dir: str) -> None:
    """Patch config.yaml files that use ``hy3dgen.shapegen`` → ``hy3dshape``.

    Mini-model configs (e.g. Hunyuan3D-2mini) reference ``hy3dgen.shapegen``
    but the installed code package is ``hy3dshape``.  This rewrites the config
    on disk so ``instantiate_from_config`` can import the correct classes.
    """
    for root, _dirs, files in os.walk(model_dir):
        for fname in files:
            if fname != "config.yaml":
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r") as fp:
                    content = fp.read()
                if "hy3dgen.shapegen" not in content:
                    continue
                patched = content.replace("hy3dgen.shapegen", "hy3dshape")
                with open(path, "w") as fp:
                    fp.write(patched)
                print(
                    f"[hunyuan3d] patched {path}: hy3dgen.shapegen → hy3dshape",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[hunyuan3d] warning: could not patch {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 image-to-3D inference (shape + optional PBR texture)",
    )
    parser.add_argument("--input", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Path to output GLB file")
    parser.add_argument(
        "--model",
        choices=["mini", "full"],
        default="mini",
        help="Shape model variant (mini=2mini recommended, default: mini)",
    )
    parser.add_argument(
        "--texture-mode",
        choices=["none", "texture"],
        default="none",
        help="Texture mode: none=geometry only, texture=Hunyuan3D-2.0 multiview PBR paint (default: none)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (e.g. cuda:0). Must be set BEFORE torch import — "
             "use CUDA_VISIBLE_DEVICES instead for strict isolation.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Local path to Hunyuan3D shape model directory",
    )
    parser.add_argument(
        "--paint-model-dir",
        default=DEFAULT_PAINT_MODEL_DIR,
        help="Local path to Hunyuan3D paint/PBR model directory (default: /data/models/Hunyuan3D-2.1)",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Model subfolder (default: auto-detect from --model)",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=1024,
        help="PBR texture render resolution (default: 1024; use 512 for low VRAM)",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="PBR texture output resolution (default: 1024; use 512 for low VRAM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None = pipeline default)",
    )
    args = parser.parse_args()

    # Validate inputs early
    if not os.path.isfile(args.input):
        _err(f"Input image not found: {args.input}")
        return 2
    Path(os.path.dirname(os.path.abspath(args.output)) or ".").mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(args.model_dir):
        _err(f"Model directory not found: {args.model_dir}")
        return 3

    # CUDA_VISIBLE_DEVICES must be set before importing torch; honor device flag
    # by translating cuda:N into N if user passed it that way.
    if args.device.startswith("cuda:") and ":" in args.device:
        idx = args.device.split(":", 1)[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", idx)
        args.device = "cuda:0"  # remapped after masking

    # Critical for multi-GPU machines: PCI_BUS_ID ensures device ordering matches
    # physical layout (3060Ti=0, 3090=1) instead of default fastest-first.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    _setup_paths(args.model_dir, args.paint_model_dir)
    _patch_config_targets(args.model_dir)

    # Heavy imports happen AFTER env setup
    import torch  # noqa: F401  (imported for side effects / CUDA ctx)
    from PIL import Image

    _apply_torchvision_fix()

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    # Resolve subfolder: explicit override > auto-detect from model variant
    if args.subfolder:
        subfolder = args.subfolder
    else:
        subfolder = "hunyuan3d-dit-v2-1" if args.model == "full" else "hunyuan3d-dit-v2-mini"

    # ── Stage 1: Shape Generation ──
    t_load_start = time.monotonic()
    print(
        f"[hunyuan3d] loading shape pipeline from {args.model_dir} "
        f"(model={args.model}, subfolder={subfolder})",
        file=sys.stderr,
    )
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_dir, subfolder=subfolder,
    )
    pipeline.to(args.device)
    t_load_end = time.monotonic()
    elapsed_load = round(t_load_end - t_load_start, 2)
    print(f"[hunyuan3d] shape pipeline loaded in {elapsed_load}s", file=sys.stderr)

    image = Image.open(args.input).convert("RGBA")
    print(f"[hunyuan3d] input image: {image.size} {image.mode}", file=sys.stderr)

    # Build kwargs — only pass seed if explicitly provided (pipeline has its own default)
    gen_kwargs: dict = {}
    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.steps:
        gen_kwargs.setdefault("num_inference_steps", args.steps)

    t_inf_start = time.monotonic()
    mesh = pipeline(image=image, **gen_kwargs)[0]
    t_inf_end = time.monotonic()
    elapsed_inf = round(t_inf_end - t_inf_start, 2)
    print(f"[hunyuan3d] shape generated in {elapsed_inf}s", file=sys.stderr)

    # Free shape pipeline VRAM before texture stage
    del pipeline
    torch.cuda.empty_cache()

    # Export geometry GLB (intermediate, replaced if texture stage runs)
    output_path = os.path.abspath(args.output)
    mesh.export(output_path)
    file_size = os.path.getsize(output_path)

    vertices = int(getattr(mesh, "vertices", []).shape[0]) if hasattr(getattr(mesh, "vertices", []), "shape") else 0
    faces = int(getattr(mesh, "faces", []).shape[0]) if hasattr(getattr(mesh, "faces", []), "shape") else 0

    elapsed_tex = 0.0
    tex_mode = "none"

    # ── Stage 2: PBR Texture Painting (optional) ──
    if args.texture_mode == "texture":
        print(
            f"[hunyuan3d] starting PBR texture painting "
            f"(render_size={args.render_size}, texture_size={args.texture_size})",
            file=sys.stderr,
        )
        elapsed_tex = _run_texture_painting(
            geo_path=output_path,
            input_image=args.input,
            output_path=output_path,
            paint_model_dir=args.paint_model_dir,
            device=args.device,
            render_size=args.render_size,
            texture_size=args.texture_size,
        )
        tex_mode = "texture"

        # Re-read GLB to get textured vertex/face counts
        try:
            import trimesh
            scene = trimesh.load(output_path, force="scene")
            total_v = sum(len(g.vertices) for g in scene.geometry.values())
            total_f = sum(len(g.faces) for g in scene.geometry.values())
            if total_v > 0:
                vertices = total_v
            if total_f > 0:
                faces = total_f
            file_size = os.path.getsize(output_path)
        except Exception:
            pass  # keep shape-stage counts

    result = {
        "output_path": output_path,
        "vertices": vertices,
        "faces": faces,
        "file_size_bytes": file_size,
        "elapsed_load_sec": elapsed_load,
        "elapsed_inference_sec": elapsed_inf,
        "elapsed_texture_sec": round(elapsed_tex, 2),
        "texture_mode": tex_mode,
        "model": args.model,
        "device": args.device,
    }
    # Single JSON line on stdout — engine parses this.
    print(json.dumps(result))
    return 0


def _run_texture_painting(
    geo_path: str,
    input_image: str,
    output_path: str,
    paint_model_dir: str,
    device: str,
    render_size: int,
    texture_size: int,
) -> float:
    """Run Hunyuan3D-2.0 multiview PBR texture painting on a geometry GLB.

    Uses the pip ``hy3dgen`` package's PaintPipeline with the v2-0 non-PBR model
    (fully compatible, unlike v2-1 PBR which has a different forward signature).

    Key patches applied:
      1. Multiview_Diffusion_Net.__init__: add trust_remote_code=True for custom pipeline
      2. Hunyuan3DPaintPipeline.load_models: skip delight model (not needed)

    Returns elapsed seconds.
    """
    import torch
    import time as _time

    # Patch multiview loader BEFORE importing the pipeline
    import hy3dgen.texgen.utils.multiview_utils as mv_utils

    original_init = mv_utils.Multiview_Diffusion_Net.__init__

    def patched_init(self, config):
        import os as _os
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        # The pip package ships a custom hunyuanpaint pipeline
        custom_pipeline_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(mv_utils.__file__))),
            "hunyuanpaint",
        )
        print(
            f"[hunyuan3d-tex] custom pipeline: {custom_pipeline_path}",
            file=sys.stderr,
        )

        pipeline = DiffusionPipeline.from_pretrained(
            multiview_ckpt_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing",
        )
        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline.to(self.device)
        print(
            f"[hunyuan3d-tex] multiview pipeline loaded on {self.device}",
            file=sys.stderr,
        )

    mv_utils.Multiview_Diffusion_Net.__init__ = patched_init

    # Patch PaintPipeline to skip delight model
    import hy3dgen.texgen.pipelines as pipelines

    def patched_load_models(self):
        torch.cuda.empty_cache()
        from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net
        self.models["multiview_model"] = Multiview_Diffusion_Net(self.config)
        self.models["delight_model"] = None
        print("[hunyuan3d-tex] models loaded (delight skipped)", file=sys.stderr)

    pipelines.Hunyuan3DPaintPipeline.load_models = patched_load_models

    # Now create config and run
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig

    multiview_path = os.path.join(paint_model_dir, "hunyuan3d-paintpbr-v2-1")
    if not os.path.isdir(multiview_path):
        # Fallback: try hunyuan3d-paint-v2-0
        multiview_path = os.path.join(paint_model_dir, "hunyuan3d-paint-v2-0")
    print(f"[hunyuan3d-tex] multiview model path: {multiview_path}", file=sys.stderr)

    conf = Hunyuan3DTexGenConfig(
        light_remover_ckpt_path="/nonexistent",  # skipped
        multiview_ckpt_path=multiview_path,
    )
    conf.render_size = render_size
    conf.texture_size = texture_size

    t0 = _time.monotonic()
    print("[hunyuan3d-tex] loading paint pipeline...", file=sys.stderr)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    try:
        paint_pipeline.enable_model_cpu_offload()
        print("[hunyuan3d-tex] CPU offload enabled", file=sys.stderr)
    except Exception as e:
        print(f"[hunyuan3d-tex] CPU offload: {e}", file=sys.stderr)

    torch.cuda.empty_cache()
    print(
        f"[hunyuan3d-tex] VRAM: allocated="
        f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
        file=sys.stderr,
    )

    print("[hunyuan3d-tex] painting textures...", file=sys.stderr)
    try:
        paint_pipeline(
            mesh_path=geo_path,
            image_path=input_image,
            output_mesh_path=output_path,
        )
        elapsed = _time.monotonic() - t0
        print(f"[hunyuan3d-tex] texture painting done in {elapsed:.1f}s", file=sys.stderr)
        return elapsed
    except Exception as e:
        # Retry with smaller resolution
        print(
            f"[hunyuan3d-tex] error at {render_size}x{texture_size}: {e}",
            file=sys.stderr,
        )
        del paint_pipeline
        torch.cuda.empty_cache()

        fallback_rs = max(512, render_size // 2)
        fallback_ts = max(512, texture_size // 2)
        print(
            f"[hunyuan3d-tex] retry with {fallback_rs}x{fallback_ts}",
            file=sys.stderr,
        )
        conf.render_size = fallback_rs
        conf.texture_size = fallback_ts
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        try:
            paint_pipeline.enable_model_cpu_offload()
        except Exception:
            pass

        paint_pipeline(
            mesh_path=geo_path,
            image_path=input_image,
            output_mesh_path=output_path,
        )
        elapsed = _time.monotonic() - t0
        print(
            f"[hunyuan3d-tex] texture painting done (fallback) in {elapsed:.1f}s",
            file=sys.stderr,
        )
        return elapsed


def _err(msg: str) -> None:
    print(f"[hunyuan3d] ERROR: {msg}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
