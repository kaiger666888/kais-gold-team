#!/usr/bin/env python3
"""
FLUX.1-dev fp8 Image Generation via ComfyUI API

Usage:
  python flux_generate.py --prompt "a cute cat" --output output.png
  python flux_generate.py --prompt "a cute cat" --width 1024 --height 768 --steps 28
  python flux_generate.py --prompt "a cute cat" --seed random
  python flux_generate.py --prompt "a cute cat" --cfg 4.0 --sampler euler
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import random

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

# Default model paths (inside ComfyUI container)
DEFAULT_UNET = "flux1-dev-fp8.safetensors"
DEFAULT_CLIP_L = "clip_l/model.safetensors"
DEFAULT_T5 = "t5xxl_fp16/model-00001-of-00002.safetensors"
DEFAULT_VAE = "flux_vae/diffusion_pytorch_model.safetensors"


def build_workflow(prompt, negative_prompt="", width=1024, height=1024,
                  steps=28, cfg=3.5, seed=None, sampler_name="euler",
                  scheduler="simple", unet_name=None, clip_l=None,
                  t5_name=None, vae_name=None):
    """Build ComfyUI API workflow JSON for FLUX.1-dev fp8."""
    if seed is None or seed == "random":
        seed = random.randint(0, 2**32 - 1)

    unet = unet_name or DEFAULT_UNET
    clip_l = clip_l or DEFAULT_CLIP_L
    t5 = t5_name or DEFAULT_T5
    vae = vae_name or DEFAULT_VAE

    workflow = {
        # Load UNET (flux fp8)
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet,
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        # Load CLIP (clip_l + t5xxl for FLUX)
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l,
                "clip_name2": t5,
                "type": "flux"
            }
        },
        # Encode positive prompt
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 0]
            }
        },
        # Encode negative prompt (empty for FLUX)
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0]
            }
        },
        # Empty latent
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # KSampler
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        # VAE Decoder
        "8": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["1", 0],
                "vae": ["8", 0]
            }
        },
        # Save
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "flux-fp8",
                "images": ["9", 0]
            }
        }
    }

    return workflow, seed


def submit_and_wait(workflow, timeout=300, poll_interval=3):
    """Submit workflow to ComfyUI and wait for completion."""
    # Submit
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Submit failed ({e.code}): {body}")

    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in response: {data}")

    # Poll for completion
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(poll_interval)
        try:
            hist_req = urllib.request.Request(
                f"{COMFYUI_URL}/history/{prompt_id}"
            )
            hist_resp = urllib.request.urlopen(hist_req, timeout=10)
            hist = json.loads(hist_resp.read())

            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", {})
                    raise RuntimeError(f"Execution error: {json.dumps(msgs, indent=2)}")

                outputs = hist[prompt_id].get("outputs", {})
                images = []
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        for img in node_out["images"]:
                            images.append({
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                            })
                return images
        except urllib.error.HTTPError:
            pass  # Not ready yet

    raise TimeoutError(f"Generation timed out after {timeout}s")


def download_image(filename, subfolder="", output_path=None):
    """Download generated image from ComfyUI."""
    params = f"filename={urllib.parse.quote(filename)}"
    if subfolder:
        params += f"&subfolder={urllib.parse.quote(subfolder)}"

    req = urllib.request.Request(
        f"{COMFYUI_URL}/view?{params}"
    )
    resp = urllib.request.urlopen(req, timeout=60)
    image_data = resp.read()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)
        return output_path
    return image_data


def generate(prompt, negative_prompt="", width=1024, height=1024,
             steps=28, cfg=3.5, seed=None, sampler_name="euler",
             scheduler="simple", output=None, **kwargs):
    """
    Generate an image with FLUX.1-dev fp8.

    Args:
        prompt: Text prompt
        negative_prompt: Negative prompt (usually empty for FLUX)
        width: Image width (default 1024)
        height: Image height (default 1024)
        steps: Sampling steps (default 28)
        cfg: Guidance scale (default 3.5)
        seed: Random seed or "random"
        sampler_name: Sampler (default "euler")
        scheduler: Scheduler (default "simple")
        output: Output file path. If None, returns raw bytes.

    Returns:
        (output_path, seed) if output is set
        (bytes, seed) if output is None
    """
    workflow, actual_seed = build_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        sampler_name=sampler_name,
        scheduler=scheduler,
        **kwargs
    )

    print(f"🚀 Generating FLUX.1-dev fp8: {width}x{height}, {steps} steps, seed={actual_seed}")
    print(f"   Prompt: {prompt[:80]}{'...' if len(prompt)>80 else ''}")

    images = submit_and_wait(workflow)
    if not images:
        raise RuntimeError("No images generated")

    img_info = images[0]
    result = download_image(img_info["filename"], img_info["subfolder"], output)

    if output:
        size_kb = os.path.getsize(output) / 1024
        print(f"✅ Saved: {output} ({size_kb:.0f}KB)")
        return output, actual_seed
    else:
        print(f"✅ Generated: {len(result)/1024:.0f}KB")
        return result, actual_seed


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1-dev fp8 Image Generation via ComfyUI API"
    )
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--negative", "-n", default="", help="Negative prompt")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: auto)")
    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)
    parser.add_argument("--steps", "-s", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--seed", default="random",
                        help="Seed number or 'random'")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--comfyui-url", default=COMFYUI_URL)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Auto output path
    if not args.output:
        ts = int(time.time())
        args.output = f"flux-{ts}.png"

    try:
        output_path, seed = generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            sampler_name=args.sampler,
            scheduler=args.scheduler,
            output=args.output
        )
        print(f"   Seed: {seed}")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
