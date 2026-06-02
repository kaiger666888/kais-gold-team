#!/usr/bin/env python3
"""Standalone TTS inference script for kais-gold-team V6.

Supports two backends:
  - CosyVoice3: High-quality TTS via Fun-CosyVoice3-0.5B-2512_RL (GPU)
  - edge-tts: Microsoft Edge TTS fallback (always available)

CosyVoice3 推理模式:
  - instruct2: 自然语言指令控制（默认），支持方言/情绪/语速/细粒度控制标签
  - zero_shot: 声音克隆，传参考音频+对应文本，克隆任意音色
  - cross_lingual: 跨语言合成（参考音频语言≠合成文本语言）
  - vc: 语音转换（改变参考音频的说话内容）

Usage:
    # instruct2 模式（默认）
    python scripts/tts_infer.py \
        --text "你好世界" \
        --output /mnt/agents/output/{task_id}/voice.wav

    # zero_shot 声音克隆
    python scripts/tts_infer.py \
        --text "今天天气真好" \
        --mode zero_shot \
        --ref-audio /path/to/ref.wav \
        --ref-text "参考音频对应的文字" \
        --output /mnt/agents/output/{task_id}/voice.wav

    # 细粒度控制标签
    python scripts/tts_infer.py \
        --text "大家好<strong>我是小明</strong>" \
        --instruct "用四川话朗读，带有<laughter>开心的语气" \
        --emotion happy \
        --output /mnt/agents/output/{task_id}/voice.wav

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
        os.path.join(os.environ['COSYVOICE_ROOT'], 'pretrained_models', 'FunAudioLLM', 'Fun-CosyVoice3-0.5B-2512'))
    os.environ.setdefault('COSYVOICE_LLM_WEIGHT', 'llm.rl.pt')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


# ============================================================
# CosyVoice3 推理模式映射
# ============================================================

# 情绪 → instruct 前缀
EMOTION_INSTRUCT_MAP = {
    "happy": "开心地",
    "sad": "悲伤地",
    "angry": "愤怒地",
    "neutral": "",
}

# 语言标识 → instruct 后缀
LANG_INSTRUCT_MAP = {
    "zh": "用中文",
    "en": "用英语",
    "ja": "用日语",
    "ko": "用韩语",
    "de": "用德语",
    "fr": "用法语",
    "es": "用西班牙语",
    "ru": "用俄语",
    "ar": "用阿拉伯语",
    "hi": "用印地语",
    "it": "用意大利语",
    "pt": "用葡萄牙语",
    "th": "用泰语",
    "vi": "用越南语",
    "id": "用印尼语",
    "ms": "用马来语",
    "nl": "用荷兰语",
    "pl": "用波兰语",
    "tr": "用土耳其语",
    "uk": "用乌克兰语",
    "bg": "用保加利亚语",
    "cs": "用捷克语",
    "da": "用丹麦语",
    "el": "用希腊语",
    "fi": "用芬兰语",
    "he": "用希伯来语",
    "hu": "用匈牙利语",
    "no": "用挪威语",
    "ro": "用罗马尼亚语",
    "sv": "用瑞典语",
}

# 预设音色 → instruct 描述
VOICE_INSTRUCT_MAP = {
    "default": "用标准的普通话",
    "中文女": "用温柔的女性声音",
    "中文男": "用沉稳的男性声音",
    "english_female": "Read in a warm female voice.",
    "english_male": "Read in a deep male voice.",
    "japanese_female": "日本語の女性の声で",
}

# 流式/非流式
STREAM_DEFAULT = False


def _build_instruct_text(
    voice: str = "default",
    instruct: str = "",
    emotion: str = "neutral",
    language: str = "auto",
    speed: float = 1.0,
) -> str:
    """构建 instruct2 的 instruct_text。

    Args:
        voice: 预设音色名（VOICE_INSTRUCT_MAP 中的 key）
        instruct: 自定义自然语言指令（优先级最高）
        emotion: 情绪（happy/sad/angry/neutral）
        language: 语言代码（auto 自动检测）
        speed: 语速（仅影响 instruct 描述，实际语速由 speed 参数控制）

    Returns:
        instruct_text 字符串（不含 <|endofprompt|>）
    """
    if instruct:
        # 用户直接指定了完整指令
        return instruct

    # 从预设音色构建
    parts = []

    # 1. 情绪前缀
    emotion_prefix = EMOTION_INSTRUCT_MAP.get(emotion, "")
    if emotion_prefix:
        parts.append(emotion_prefix)

    # 2. 音色/风格描述
    voice_desc = VOICE_INSTRUCT_MAP.get(voice, VOICE_INSTRUCT_MAP["default"])
    parts.append(voice_desc)

    # 3. 语速描述
    if speed > 1.3:
        parts.append("语速较快")
    elif speed < 0.7:
        parts.append("语速较慢")

    instruction = "".join(parts)
    if not instruction.endswith("。") and not instruction.endswith("."):
        instruction += "朗读。"

    return instruction


def _load_cosyvoice3(model_dir: str, llm_weight: str = "llm.pt"):
    """加载 CosyVoice3 模型。"""
    cosy_root = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")
    sys.path.insert(0, cosy_root)
    from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore

    cosy = CosyVoice3(model_dir)

    # RL 权重替换
    use_rl = llm_weight != "llm.pt"
    if use_rl:
        rl_path = os.path.join(model_dir, llm_weight)
        if os.path.isfile(rl_path):
            import torch as _torch
            rl_state = _torch.load(rl_path, map_location="cpu", weights_only=True)
            cosy.model.llm.load_state_dict(rl_state)
            del rl_state
            print(f"CosyVoice3: loaded RL weight {llm_weight}")
        else:
            print(f"CosyVoice3: RL weight {llm_weight} not found, using base llm.pt")

    return cosy


def _save_wav(all_wav, sample_rate: int, output_path: str) -> dict:
    """保存音频并返回结果。"""
    import torchaudio

    if all_wav is not None and all_wav.shape[1] > 0:
        torchaudio.save(output_path, all_wav, sample_rate)
        duration = all_wav.shape[1] / sample_rate
        return {
            "status": "ok",
            "output_path": output_path,
            "backend": "cosyvoice",
            "sample_rate": sample_rate,
            "duration": round(duration, 2),
        }
    else:
        return {
            "status": "error",
            "error": "CosyVoice3 produced no output",
            "backend": "cosyvoice",
        }


# ============================================================
# 推理模式实现
# ============================================================

def infer_cosyvoice_instruct2(
    cosy,
    text: str,
    output_path: str,
    voice: str = "default",
    instruct: str = "",
    emotion: str = "neutral",
    language: str = "auto",
    speed: float = 1.0,
    stream: bool = False,
    prompt_wav: str = "",
) -> dict:
    """CosyVoice3 instruct2 模式 — 自然语言指令控制。

    Args:
        cosy: 已加载的 CosyVoice3 实例
        text: 要合成的文本（支持控制标签如 <strong>, [laughter] 等）
        output_path: 输出 WAV 路径
        voice: 预设音色（VOICE_INSTRUCT_MAP key）
        instruct: 自定义自然语言指令（覆盖 voice/emotion/language）
        emotion: 情绪（happy/sad/angry/neutral）
        language: 语言代码（auto 自动检测）
        speed: 语速（0.5-2.0）
        stream: 是否流式输出
        prompt_wav: 参考音频路径（可选，用于 voice stamping）
    """
    import torch
    import torchaudio

    instruct_text = _build_instruct_text(voice, instruct, emotion, language, speed)
    instruct_text += "<|endofprompt|>"

    # 参考音频：优先用户指定，否则用仓库自带的
    cosy_root = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")
    if prompt_wav and os.path.isfile(prompt_wav):
        ref = prompt_wav
    else:
        ref = os.path.join(cosy_root, "asset", "zero_shot_prompt.wav")
        if not os.path.isfile(ref):
            # 兜底：生成 1 秒静音
            ref = os.path.join("/tmp", "cosyvoice_ref.wav")
            if not os.path.exists(ref):
                silence = torch.zeros(1, 24000)
                torchaudio.save(ref, silence, 24000)

    ensure_output_dir(output_path)
    all_wav = None
    sample_rate = cosy.sample_rate

    try:
        for result in cosy.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_wav=ref,
            stream=stream,
            speed=speed,
        ):
            wav = result["tts_speech"]
            if all_wav is None:
                all_wav = wav
            else:
                all_wav = torch.cat([all_wav, wav], dim=1)
    except Exception as e:
        return {
            "status": "error",
            "error": f"CosyVoice3 instruct2 inference failed: {e}",
            "backend": "cosyvoice",
            "mode": "instruct2",
        }

    result = _save_wav(all_wav, sample_rate, output_path)
    result["mode"] = "instruct2"
    return result


def infer_cosyvoice_zero_shot(
    cosy,
    text: str,
    output_path: str,
    ref_audio: str,
    ref_text: str,
    speed: float = 1.0,
    stream: bool = False,
) -> dict:
    """CosyVoice3 zero_shot 模式 — 声音克隆。

    Args:
        cosy: 已加载的 CosyVoice3 实例
        text: 要合成的文本
        output_path: 输出 WAV 路径
        ref_audio: 参考音频路径（要克隆的目标音色）
        ref_text: 参考音频对应的文字内容
        speed: 语速（0.5-2.0）
        stream: 是否流式输出
    """
    import torch

    if not os.path.isfile(ref_audio):
        return {
            "status": "error",
            "error": f"Reference audio not found: {ref_audio}",
            "backend": "cosyvoice",
            "mode": "zero_shot",
        }

    if not ref_text:
        return {
            "status": "error",
            "error": "zero_shot mode requires --ref-text",
            "backend": "cosyvoice",
            "mode": "zero_shot",
        }

    ensure_output_dir(output_path)
    all_wav = None
    sample_rate = cosy.sample_rate

    try:
        for result in cosy.inference_zero_shot(
            tts_text=text,
            prompt_text=ref_text,
            prompt_wav=ref_audio,
            stream=stream,
            speed=speed,
        ):
            wav = result["tts_speech"]
            if all_wav is None:
                all_wav = wav
            else:
                all_wav = torch.cat([all_wav, wav], dim=1)
    except Exception as e:
        return {
            "status": "error",
            "error": f"CosyVoice3 zero_shot inference failed: {e}",
            "backend": "cosyvoice",
            "mode": "zero_shot",
        }

    result = _save_wav(all_wav, sample_rate, output_path)
    result["mode"] = "zero_shot"
    return result


def infer_cosyvoice_cross_lingual(
    cosy,
    text: str,
    output_path: str,
    ref_audio: str,
    speed: float = 1.0,
    stream: bool = False,
) -> dict:
    """CosyVoice3 cross_lingual 模式 — 跨语言合成。

    Args:
        cosy: 已加载的 CosyVoice3 实例
        text: 要合成的文本（可以与参考音频不同语言）
        output_path: 输出 WAV 路径
        ref_audio: 参考音频路径（提供音色，语言可不同）
        speed: 语速（0.5-2.0）
        stream: 是否流式输出
    """
    import torch

    if not os.path.isfile(ref_audio):
        return {
            "status": "error",
            "error": f"Reference audio not found: {ref_audio}",
            "backend": "cosyvoice",
            "mode": "cross_lingual",
        }

    ensure_output_dir(output_path)
    all_wav = None
    sample_rate = cosy.sample_rate

    try:
        for result in cosy.inference_cross_lingual(
            tts_text=text,
            prompt_wav=ref_audio,
            stream=stream,
            speed=speed,
        ):
            wav = result["tts_speech"]
            if all_wav is None:
                all_wav = wav
            else:
                all_wav = torch.cat([all_wav, wav], dim=1)
    except Exception as e:
        return {
            "status": "error",
            "error": f"CosyVoice3 cross_lingual inference failed: {e}",
            "backend": "cosyvoice",
            "mode": "cross_lingual",
        }

    result = _save_wav(all_wav, sample_rate, output_path)
    result["mode"] = "cross_lingual"
    return result


def infer_cosyvoice_vc(
    cosy,
    output_path: str,
    ref_audio: str,
    prompt_audio: str,
    speed: float = 1.0,
    stream: bool = False,
) -> dict:
    """CosyVoice3 vc 模式 — 语音转换。

    Args:
        cosy: 已加载的 CosyVoice3 实例
        output_path: 输出 WAV 路径
        ref_audio: 参考音频（目标音色）
        prompt_audio: 源音频（要转换的内容）
        speed: 语速（0.5-2.0）
        stream: 是否流式输出
    """
    import torch

    if not os.path.isfile(ref_audio):
        return {
            "status": "error",
            "error": f"Reference audio not found: {ref_audio}",
            "backend": "cosyvoice",
            "mode": "vc",
        }
    if not os.path.isfile(prompt_audio):
        return {
            "status": "error",
            "error": f"Prompt audio not found: {prompt_audio}",
            "backend": "cosyvoice",
            "mode": "vc",
        }

    ensure_output_dir(output_path)
    all_wav = None
    sample_rate = cosy.sample_rate

    try:
        for result in cosy.inference_vc(
            prompt_wav=ref_audio,
            prompt_wav_16k=prompt_audio,
            stream=stream,
            speed=speed,
        ):
            wav = result["tts_speech"]
            if all_wav is None:
                all_wav = wav
            else:
                all_wav = torch.cat([all_wav, wav], dim=1)
    except Exception as e:
        return {
            "status": "error",
            "error": f"CosyVoice3 vc inference failed: {e}",
            "backend": "cosyvoice",
            "mode": "vc",
        }

    result = _save_wav(all_wav, sample_rate, output_path)
    result["mode"] = "vc"
    return result


# ============================================================
# 统一 CosyVoice 入口
# ============================================================

def infer_cosyvoice(text: str, output_path: str, voice: str, speed: float, **kwargs) -> dict:
    """CosyVoice3 统一入口，根据 mode 路由到不同推理模式。

    Args:
        text: 要合成的文本（vc 模式可选）
        output_path: 输出 WAV 路径
        voice: 预设音色名
        speed: 语速
        **kwargs: 额外参数（mode, instruct, emotion, language, ref_audio, ref_text, prompt_audio, stream）

    Returns:
        dict with status, output_path, etc.
    """
    mode = kwargs.get("mode", "instruct2")
    cosy_root = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")

    if not os.path.isdir(cosy_root):
        return {
            "status": "error",
            "error": f"CosyVoice not found at {cosy_root}. Set COSYVOICE_ROOT env var.",
            "backend": "cosyvoice",
        }

    try:
        model_dir = os.environ.get(
            "COSYVOICE_MODEL_DIR",
            os.path.join(cosy_root, "pretrained_models", "FunAudioLLM", "Fun-CosyVoice3-0.5B-2512"),
        )
        model_dir = os.path.realpath(model_dir)
        if not os.path.isdir(model_dir):
            return {
                "status": "error",
                "error": f"CosyVoice3 model not found at {model_dir}. Set COSYVOICE_MODEL_DIR.",
                "backend": "cosyvoice",
            }

        llm_weight = os.environ.get("COSYVOICE_LLM_WEIGHT", "llm.pt")
        cosy = _load_cosyvoice3(model_dir, llm_weight)

        if mode == "zero_shot":
            ref_audio = kwargs.get("ref_audio", "")
            ref_text = kwargs.get("ref_text", "")
            return infer_cosyvoice_zero_shot(
                cosy, text, output_path,
                ref_audio=ref_audio,
                ref_text=ref_text,
                speed=speed,
                stream=kwargs.get("stream", False),
            )

        elif mode == "cross_lingual":
            ref_audio = kwargs.get("ref_audio", "")
            return infer_cosyvoice_cross_lingual(
                cosy, text, output_path,
                ref_audio=ref_audio,
                speed=speed,
                stream=kwargs.get("stream", False),
            )

        elif mode == "vc":
            ref_audio = kwargs.get("ref_audio", "")
            prompt_audio = kwargs.get("prompt_audio", "")
            return infer_cosyvoice_vc(
                cosy, output_path,
                ref_audio=ref_audio,
                prompt_audio=prompt_audio,
                speed=speed,
                stream=kwargs.get("stream", False),
            )

        else:  # default: instruct2
            return infer_cosyvoice_instruct2(
                cosy, text, output_path,
                voice=voice,
                instruct=kwargs.get("instruct", ""),
                emotion=kwargs.get("emotion", "neutral"),
                language=kwargs.get("language", "auto"),
                speed=speed,
                stream=kwargs.get("stream", False),
                prompt_wav=kwargs.get("prompt_wav", ""),
            )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "status": "error",
            "error": f"CosyVoice3 inference failed: {e}",
            "traceback": tb,
            "backend": "cosyvoice",
        }


# ============================================================
# edge-tts
# ============================================================

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


# ============================================================
# CLI 入口
# ============================================================

async def main():
    _setup_env()
    parser = argparse.ArgumentParser(description="TTS inference for kais-gold-team V6 — CosyVoice3 + edge-tts")

    # === 必填参数 ===
    parser.add_argument("--text", required=False, default="", help="要合成的文本（vc 模式可选）")
    parser.add_argument("--output", required=True, help="输出 WAV 文件路径")

    # === 模式选择 ===
    parser.add_argument(
        "--mode",
        default="instruct2",
        choices=["instruct2", "zero_shot", "cross_lingual", "vc"],
        help="推理模式: instruct2=指令控制, zero_shot=声音克隆, cross_lingual=跨语言, vc=语音转换 (默认: instruct2)",
    )

    # === instruct2 模式参数 ===
    parser.add_argument(
        "--voice",
        default="default",
        help="预设音色: default/中文女/中文男/english_female/english_male/japanese_female (默认: default)",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="自定义自然语言指令，如 '用四川话开心地朗读'。设置后覆盖 voice/emotion/language (默认: 自动构建)",
    )
    parser.add_argument(
        "--emotion",
        default="neutral",
        choices=["happy", "sad", "angry", "neutral"],
        help="情绪: happy=开心, sad=悲伤, angry=愤怒, neutral=平静 (默认: neutral)",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="语言代码: zh/en/ja/ko/de/fr/es/ru 等 31 种语言，auto=自动检测 (默认: auto)",
    )
    parser.add_argument(
        "--prompt-wav",
        default="",
        help="instruct2 参考音频路径（可选，用于 voice stamping）",
    )

    # === zero_shot / cross_lingual 模式参数 ===
    parser.add_argument(
        "--ref-audio",
        default="",
        help="参考音频路径（zero_shot: 克隆目标音色 / cross_lingual: 提供音色）",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="参考音频对应的文字内容（zero_shot 必填）",
    )

    # === vc 模式参数 ===
    parser.add_argument(
        "--prompt-audio",
        default="",
        help="源音频路径（vc 模式：要转换的内容，必填）",
    )

    # === 通用参数 ===
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="语速 (0.5-2.0, 默认: 1.0)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="启用流式输出 (默认: 关闭)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cosyvoice", "edge-tts"],
        help="TTS 后端 (默认: auto，优先 CosyVoice)",
    )

    args = parser.parse_args()

    start = time.monotonic()

    # 构建额外参数
    cosy_kwargs = {
        "mode": args.mode,
        "instruct": args.instruct,
        "emotion": args.emotion,
        "language": args.language,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "prompt_audio": args.prompt_audio,
        "stream": args.stream,
        "prompt_wav": args.prompt_wav,
    }

    if args.backend == "cosyvoice":
        result = infer_cosyvoice(args.text, args.output, args.voice, args.speed, **cosy_kwargs)
    elif args.backend == "edge-tts":
        result = await infer_edge_tts(args.text, args.output, args.voice, args.speed)
    else:  # auto
        result = infer_cosyvoice(args.text, args.output, args.voice, args.speed, **cosy_kwargs)
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
