#!/usr/bin/env python3
"""Hunyuan3D-2mv inference script — multiview image-to-3D.

Supports 1-4 reference images (front/left/back/right views) to generate
a 3D GLB mesh with better shape accuracy than single-view inference.

Usage:
    # Single image (front only, same as 2.1):
    python scripts/hunyuan3d_mv_infer.py \
        --front /path/to/front.png \
        --output /mnt/agents/output/task/model.glb

    # Multi-view (recommended for best results):
    python scripts/hunyuan3d_mv_infer.py \
        --front front.png --left left.png --back back.png --right right.png \
        --output /mnt/agents/output/task/model.glb

    # With seed:
    python scripts/hunyuan3d_mv_infer.py \
        --front front.png --seed 42 --steps 50 \
        --output /mnt/agents/output/task/model.glb

Outputs a single JSON line on stdout with result info.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reuse the same model dir structure as Hunyuan3D-2 for shared pipeline code
_DEFAULT_2MV_MODEL_DIR = os.environ.get(
    "HUNYUAN3D_2MV_MODEL_DIR",
    "/data/models/tencent/Hunyuan3D-2mv",
)
# Shared pipeline code lives in the 2.1 installation
_CODE_DIR = os.environ.get(
    "HUNYUAN3D_CODE_DIR",
    "/data/models/tencent/Hunyuan3D-2",
)


def _setup_paths(code_dir: str = None) -> None:
    """Add hy3dshape submodule to sys.path."""
    if code_dir is None:
        code_dir = _CODE_DIR
    for sub in ("hy3dshape", "hy3dpaint"):
        sub_path = os.path.join(_CODE_DIR, sub)
        if os.path.isdir(sub_path) and sub_path not in sys.path:
            sys.path.insert(0, sub_path)


def _patch_config(model_dir: str) -> None:
    """Patch config.yaml: hy3dgen.shapegen → hy3dshape."""
    path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        content = f.read()
    if "hy3dgen.shapegen" not in content:
        return
    patched = content.replace("hy3dgen.shapegen", "hy3dshape")
    with open(path, "w") as f:
        f.write(patched)
    print(f"[hunyuan3d-mv] patched config: hy3dgen.shapegen → hy3dshape", file=sys.stderr)


def _err(msg: str) -> None:
    print(f"[hunyuan3d-mv] ERROR: {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2mv multiview image-to-3D inference",
    )
    parser.add_argument("--front", required=True, help="Front view image path")
    parser.add_argument("--left", default=None, help="Left view image path")
    parser.add_argument("--back", default=None, help="Back view image path")
    parser.add_argument("--right", default=None, help="Right view image path")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--model-dir", default=_DEFAULT_2MV_MODEL_DIR)
    parser.add_argument("--code-dir", default=_CODE_DIR,
                        help="Shared pipeline code directory (hy3dshape/)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv",
                        help="Model subfolder within model-dir")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Validate front image
    if not os.path.isfile(args.front):
        _err(f"Front image not found: {args.front}")
        return 2

    # Validate optional views
    for name, path in [("left", args.left), ("back", args.back), ("right", args.right)]:
        if path and not os.path.isfile(path):
            _err(f"{name} image not found: {path}")
            return 2

    # Create output dir
    Path(os.path.dirname(os.path.abspath(args.output)) or ".").mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(args.model_dir):
        _err(f"Model directory not found: {args.model_dir}")
        return 3

    # CUDA device mapping
    if args.device.startswith("cuda:") and ":" in args.device:
        idx = args.device.split(":", 1)[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", idx)
        args.device = "cuda:0"

    code_dir = args.code_dir
    _setup_paths(code_dir)
    _patch_config(args.model_dir)

    # Heavy imports after path setup
    import torch
    from PIL import Image

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except ImportError:
        pass

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    t_load_start = time.monotonic()
    print(f"[hunyuan3d-mv] loading pipeline from {args.model_dir} (subfolder={args.subfolder})", file=sys.stderr)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_dir, subfolder=args.subfolder, use_safetensors=True,
    )
    pipeline.to(args.device)
    t_load_end = time.monotonic()
    elapsed_load = round(t_load_end - t_load_start, 2)
    print(f"[hunyuan3d-mv] pipeline loaded in {elapsed_load}s", file=sys.stderr)

    # Build image dict for MVImageProcessorV2
    image_dict = {"front": args.front}
    if args.left:
        image_dict["left"] = args.left
    if args.back:
        image_dict["back"] = args.back
    if args.right:
        image_dict["right"] = args.right

    print(
        f"[hunyuan3d-mv] input views: {list(image_dict.keys())} "
        f"({len(image_dict)} view(s))",
        file=sys.stderr,
    )

    gen_kwargs = {}
    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.steps:
        gen_kwargs["num_inference_steps"] = args.steps

    t_inf_start = time.monotonic()
    mesh = pipeline(image=image_dict, **gen_kwargs)[0]
    t_inf_end = time.monotonic()
    elapsed_inf = round(t_inf_end - t_inf_start, 2)
    print(f"[hunyuan3d-mv] shape generated in {elapsed_inf}s", file=sys.stderr)

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
        "model": "hunyuan3d-2mv",
        "device": args.device,
        "views": list(image_dict.keys()),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
