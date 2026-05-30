#!/usr/bin/env python3
"""Standalone TTS inference script for kais-gold-team V6.

Supports two backends:
  - CosyVoice2: High-quality Chinese TTS via CosyVoice2-0.5B (GPU, if installed)
  - edge-tts: Microsoft Edge TTS fallback (always available)

Usage:
    python scripts/tts_infer.py \
        --text "你好世界" \
        --output /mnt/agents/output/{task_id}/voice.wav \
        --voice default \
        --speed 1.0 \
        --backend auto

Output: WAV file at --output path. Prints JSON status to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path


def ensure_output_dir(output_path: str) -> None:
    """Create output directory if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def _setup_env() -> None:
    """Set up environment variables for CosyVoice."""
    os.environ.setdefault('COSYVOICE_ROOT', '/opt/CosyVoice')
    os.environ.setdefault('COSYVOICE_MODEL_DIR',
        os.path.join(os.environ['COSYVOICE_ROOT'], 'pretrained_models', 'iic', 'CosyVoice2-0.5B'))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


def infer_cosyvoice(text: str, output_path: str, voice: str, speed: float) -> dict:
    """Run CosyVoice2 inference.

    Expects CosyVoice repo cloned at COSYVOICE_ROOT with CosyVoice2-0.5B model.
    Uses CosyVoice2.inference_instruct2 for zero-shot synthesis.
    """
    cosy_root = os.environ.get(
        "COSYVOICE_ROOT",
        "/opt/CosyVoice",
    )
    if not os.path.isdir(cosy_root):
        return {
            "status": "error",
            "error": f"CosyVoice not found at {cosy_root}. Set COSYVOICE_ROOT env var.",
            "backend": "cosyvoice",
        }

    try:
        sys.path.insert(0, cosy_root)
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore

        model_dir = os.environ.get(
            "COSYVOICE_MODEL_DIR",
            os.path.join(cosy_root, "pretrained_models", "iic", "CosyVoice2-0.5B"),
        )

        # Resolve symlink if needed (ModelScope creates symlinks)
        model_dir = os.path.realpath(model_dir)
        if not os.path.isdir(model_dir):
            return {
                "status": "error",
                "error": f"CosyVoice2 model not found at {model_dir}. Set COSYVOICE_MODEL_DIR.",
                "backend": "cosyvoice",
            }

        cosy = CosyVoice2(model_dir)

        ensure_output_dir(output_path)

        import torchaudio

        # CosyVoice2 uses instruct2 mode — provide natural language instruction
        # Voice instructions map
        instruct_map = {
            "default": "用标准的普通话朗读。",
            "中文女": "用温柔的女性声音朗读。",
            "中文男": "用沉稳的男性声音朗读。",
            "english_female": "Read in a warm female voice.",
            "english_male": "Read in a deep male voice.",
            "japanese_female": "日本語の女性の声で読んでください。",
        }
        instruct_text = instruct_map.get(voice, instruct_map["default"])

        # Generate a short silent reference WAV for inference_instruct2
        import torch as _torch
        ref_wav_path = os.path.join("/tmp", "cosyvoice_ref.wav")
        if not os.path.exists(ref_wav_path):
            _silence = _torch.zeros(1, 24000)  # 1 second silence at 24kHz
            torchaudio.save(ref_wav_path, _silence, 24000)

        # Collect all streaming chunks and concatenate
        all_wav = None
        sample_rate = cosy.sample_rate
        for result in cosy.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_wav=ref_wav_path,
            stream=False,
            speed=speed,
        ):
            wav = result["tts_speech"]
            if all_wav is None:
                all_wav = wav
            else:
                import torch
                all_wav = torch.cat([all_wav, wav], dim=1)

        if all_wav is not None:
            torchaudio.save(output_path, all_wav, sample_rate)
        else:
            return {
                "status": "error",
                "error": "CosyVoice2 produced no output",
                "backend": "cosyvoice",
            }

        return {
            "status": "ok",
            "output_path": output_path,
            "backend": "cosyvoice",
            "sample_rate": sample_rate,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "status": "error",
            "error": f"CosyVoice inference failed: {e}",
            "traceback": tb,
            "backend": "cosyvoice",
        }


async def infer_edge_tts(text: str, output_path: str, voice: str, speed: float) -> dict:
    """Run edge-tts inference (Microsoft Edge TTS)."""
    try:
        import edge_tts
    except ImportError:
        return {
            "status": "error",
            "error": "edge-tts not installed. Run: pip install edge-tts",
            "backend": "edge-tts",
        }

    ensure_output_dir(output_path)

    # Voice mapping: short names → edge-tts voice IDs
    voice_map = {
        "default": "zh-CN-XiaoxiaoNeural",
        "中文女": "zh-CN-XiaoxiaoNeural",
        "中文男": "zh-CN-YunxiNeural",
        "english_female": "en-US-JennyNeural",
        "english_male": "en-US-GuyNeural",
        "japanese_female": "ja-JP-NanamiNeural",
    }
    edge_voice = voice_map.get(voice, voice if "-" in voice else "zh-CN-XiaoxiaoNeural")

    # Speed format for edge-tts: "+0%", "-50%", etc.
    speed_str = f"{int((speed - 1.0) * 100):+d}%"

    communicate = edge_tts.Communicate(text, edge_voice, rate=speed_str)
    await communicate.save(output_path)

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        return {
            "status": "error",
            "error": "edge-tts produced empty output",
            "backend": "edge-tts",
        }

    return {
        "status": "ok",
        "output_path": output_path,
        "backend": "edge-tts",
        "voice_used": edge_voice,
    }


async def main():
    _setup_env()
    parser = argparse.ArgumentParser(description="TTS inference for kais-gold-team")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output WAV/MP3 file path")
    parser.add_argument("--voice", default="default", help="Voice name or ID")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (1.0 = normal)")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cosyvoice", "edge-tts"],
        help="TTS backend to use",
    )
    args = parser.parse_args()

    start = time.monotonic()

    if args.backend == "cosyvoice":
        result = infer_cosyvoice(args.text, args.output, args.voice, args.speed)
    elif args.backend == "edge-tts":
        result = await infer_edge_tts(args.text, args.output, args.voice, args.speed)
    else:
        # Auto: try CosyVoice first, fall back to edge-tts
        result = infer_cosyvoice(args.text, args.output, args.voice, args.speed)
        if result["status"] != "ok":
            print(f"CosyVoice unavailable ({result.get('error', '')}), falling back to edge-tts", file=sys.stderr)
            result = await infer_edge_tts(args.text, args.output, args.voice, args.speed)

    elapsed = time.monotonic() - start
    result["duration_sec"] = round(elapsed, 2)
    result["text_length"] = len(args.text)

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    asyncio.run(main())
