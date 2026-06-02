#!/usr/bin/env python3
"""Standalone Hunyuan3D-2 inference script for kais-gold-team V6.

Runs the Tencent Hunyuan3D-2 shape generation pipeline on a single input image
and exports a GLB mesh. Designed to be invoked as a subprocess by
``src/v6/engines/hunyuan3d.py`` (fire-and-forget pattern, identical to
``scripts/tts_infer.py``).

Usage:
    python scripts/hunyuan3d_infer.py \\
        --input /path/to/image.png \\
        --output /mnt/agents/output/{task_id}/model.glb \\
        --model full \\
        --device cuda:0 \\
        --steps 50

Outputs a single JSON line on stdout (other output goes to stderr) with:
    {
      "output_path": "/path/to/model.glb",
      "vertices": 12345,
      "faces": 24000,
      "elapsed_load_sec": 28.1,
      "elapsed_inference_sec": 75.2,
      "model": "full",
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


# Default model location — can be overridden via --model-dir or HUNYUAN3D_MODEL_DIR env.
DEFAULT_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_MODEL_DIR",
    "/data/models/tencent/Hunyuan3D-2",
)


def _setup_paths(model_dir: str) -> None:
    """Add hy3dshape / hy3dpaint submodules to sys.path so their packages import.

    The Hunyuan3D-2 release ships its python code under ``hy3dshape/`` and
    ``hy3dpaint/`` subdirectories. The reference ``run_shape.py`` inserts
    these into sys.path; we mirror that here.

    Mini-model directories don't include the code, so we also add the default
    full-model directory as a fallback.
    """
    search_dirs = [model_dir]
    if DEFAULT_MODEL_DIR != model_dir:
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
        description="Hunyuan3D-2 image-to-3D inference",
    )
    parser.add_argument("--input", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Path to output GLB file")
    parser.add_argument(
        "--model",
        choices=["mini", "full"],
        default="full",
        help="Model variant (only 'full' v2.1 is wired by default)",
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
        help="Local path to Hunyuan3D-2 model directory",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help="Model subfolder (default: auto-detect from --model)",
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

    _setup_paths(args.model_dir)
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
        subfolder = "hunyuan3d-dit-v2-1" if args.model == "full" else "hunyuan3d-dit-v2-mini-fast"

    t_load_start = time.monotonic()
    print(
        f"[hunyuan3d] loading pipeline from {args.model_dir} "
        f"(model={args.model}, subfolder={subfolder})",
        file=sys.stderr,
    )
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_dir, subfolder=subfolder,
    )
    pipeline.to(args.device)
    t_load_end = time.monotonic()
    elapsed_load = round(t_load_end - t_load_start, 2)
    print(f"[hunyuan3d] pipeline loaded in {elapsed_load}s", file=sys.stderr)

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

    # Export GLB
    output_path = os.path.abspath(args.output)
    mesh.export(output_path)
    file_size = os.path.getsize(output_path)

    vertices = int(getattr(mesh, "vertices", []).shape[0]) if hasattr(getattr(mesh, "vertices", []), "shape") else 0
    faces = int(getattr(mesh, "faces", []).shape[0]) if hasattr(getattr(mesh, "faces", []), "shape") else 0

    result = {
        "output_path": output_path,
        "vertices": vertices,
        "faces": faces,
        "file_size_bytes": file_size,
        "elapsed_load_sec": elapsed_load,
        "elapsed_inference_sec": elapsed_inf,
        "model": args.model,
        "device": args.device,
    }
    # Single JSON line on stdout — engine parses this.
    print(json.dumps(result))
    return 0


def _err(msg: str) -> None:
    print(f"[hunyuan3d] ERROR: {msg}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
