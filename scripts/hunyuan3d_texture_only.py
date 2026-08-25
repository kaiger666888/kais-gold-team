#!/usr/bin/env python3
"""Standalone Hunyuan3D texture painting - takes a geometry GLB + reference image,
outputs textured GLB with proper PBR material."""
import sys, os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import trimesh
import numpy as np
from PIL import Image

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", required=True, help="Geometry GLB path")
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output textured GLB path")
    parser.add_argument("--paint-model-dir", default="/data/models/Hunyuan3D-2.1")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--texture-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda:0"
    t0 = time.monotonic()

    # Load geometry mesh
    mesh = trimesh.load(args.geo, force='mesh')
    print(f"[tex] loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", file=sys.stderr)

    # Patch multiview loader
    import hy3dgen.texgen.utils.multiview_utils as mv_utils
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    def patched_init(self, config):
        import os as _os
        self.device = config.device
        self.view_size = 512
        multiview_ckpt_path = config.multiview_ckpt_path

        custom_pipeline_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(mv_utils.__file__))),
            "hunyuanpaint",
        )
        print(f"[tex] custom pipeline: {custom_pipeline_path}", file=sys.stderr)

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
        print(f"[tex] multiview pipeline loaded on {self.device}", file=sys.stderr)

    mv_utils.Multiview_Diffusion_Net.__init__ = patched_init

    # Patch delight model to no-op
    import hy3dgen.texgen.pipelines as pipelines

    def patched_load_models(self):
        torch.cuda.empty_cache()
        self.models["multiview_model"] = mv_utils.Multiview_Diffusion_Net(self.config)
        class _NoOp:
            def __call__(self, img): return img
        self.models["delight_model"] = _NoOp()
        print("[tex] models loaded (delight skipped)", file=sys.stderr)

    pipelines.Hunyuan3DPaintPipeline.load_models = patched_load_models

    # Create config
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig

    multiview_path = os.path.join(args.paint_model_dir, "hunyuan3d-paintpbr-v2-1")
    if not os.path.isdir(multiview_path):
        multiview_path = os.path.join(args.paint_model_dir, "hunyuan3d-paint-v2-0")
    print(f"[tex] multiview model path: {multiview_path}", file=sys.stderr)

    conf = Hunyuan3DTexGenConfig(
        light_remover_ckpt_path="/nonexistent",
        multiview_ckpt_path=multiview_path,
    )
    conf.render_size = args.render_size
    conf.texture_size = args.texture_size

    # Create paint pipeline
    print("[tex] loading paint pipeline...", file=sys.stderr)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    try:
        paint_pipeline.enable_model_cpu_offload()
        print("[tex] CPU offload enabled", file=sys.stderr)
    except Exception as e:
        print(f"[tex] CPU offload: {e}", file=sys.stderr)

    torch.cuda.empty_cache()
    print(f"[tex] VRAM: allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB", file=sys.stderr)

    # Run texture painting
    print("[tex] painting textures...", file=sys.stderr)
    textured_mesh = paint_pipeline(mesh, args.image)
    
    t_elapsed = time.monotonic() - t0
    print(f"[tex] painting done in {t_elapsed:.1f}s", file=sys.stderr)
    print(f"[tex] result type: {type(textured_mesh).__name__}", file=sys.stderr)
    print(f"[tex] result visual: {type(textured_mesh.visual).__name__}", file=sys.stderr)
    
    if hasattr(textured_mesh.visual, 'uv') and textured_mesh.visual.uv is not None:
        print(f"[tex] UV: {textured_mesh.visual.uv.shape}", file=sys.stderr)
    if hasattr(textured_mesh.visual, 'image') and textured_mesh.visual.image:
        print(f"[tex] texture image: {textured_mesh.visual.image.size}", file=sys.stderr)

    # Export with proper PBR material
    # The pipeline's save_mesh() uses SimpleMaterial, which loses texture in GLB export
    # Convert to PBRMaterial with baseColorTexture
    if hasattr(textured_mesh.visual, 'image') and textured_mesh.visual.image:
        tex_img = textured_mesh.visual.image
        uv = textured_mesh.visual.uv if hasattr(textured_mesh.visual, 'uv') else None
        pbr_mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        textured_mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr_mat)
        print(f"[tex] converted to PBR material", file=sys.stderr)

    textured_mesh.export(args.output, file_type='glb')
    file_size = os.path.getsize(args.output)
    print(f"[tex] exported {file_size/1024/1024:.1f}MB to {args.output}", file=sys.stderr)

    # Output JSON result
    import json
    result = {
        "output_path": args.output,
        "vertices": len(textured_mesh.vertices),
        "faces": len(textured_mesh.faces),
        "elapsed_texture_sec": t_elapsed,
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
