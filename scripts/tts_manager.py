#!/usr/bin/env python3
"""TTS Service Manager — manage three-track TTS services.

Usage:
    python scripts/tts_manager.py status         # Check all services
    python scripts/tts_manager.py start [track]  # Start all or specific track
    python scripts/tts_manager.py stop [track]   # Stop all or specific track
    python scripts/tts_manager.py restart [track]
    python scripts/tts_manager.py test [track]   # Quick TTS test
    python scripts/tts_manager.py logs [track]    # Tail logs

Tracks: zh (GPT-SoVITS), en (Chatterbox), bilingual (CosyVoice)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PID_DIR = os.path.join(REPO_ROOT, ".tts_pids")
LOG_DIR = os.path.join(REPO_ROOT, ".tts_logs")

os.makedirs(PID_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRACKS = {
    "zh": {
        "name": "GPT-SoVITS",
        "port": 9880,
        "script": os.path.join(REPO_ROOT, "scripts", "tts_gpt_sovits_server.py"),
        "native_api": "~/GPT-SoVITS/api_v2.py",
        "python": os.path.expanduser("~/GPT-SoVITS/.venv/bin/python"),
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
        "vram": 4.0,
        "gpu": "3060Ti",
    },
    "en": {
        "name": "Chatterbox-Turbo",
        "port": 9881,
        "script": os.path.join(REPO_ROOT, "scripts", "tts_chatterbox_server.py"),
        "python": os.path.expanduser("~/chatterbox/.venv/bin/python"),
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
        "vram": 2.0,
        "gpu": "3060Ti",
    },
    "bilingual": {
        "name": "CosyVoice-3.0",
        "port": 9882,
        "script": os.path.join(REPO_ROOT, "scripts", "tts_cosyvoice_server.py"),
        "python": os.path.expanduser("~/CosyVoice/.venv/bin/python"),
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": os.path.expanduser("~/CosyVoice") + ":" + os.path.expanduser("~/CosyVoice/third_party/Matcha-TTS"),
        },
        "vram": 6.0,
        "gpu": "3090",
    },
}


def pid_file(track: str) -> str:
    return os.path.join(PID_DIR, f"{track}.pid")


def log_file(track: str) -> str:
    return os.path.join(LOG_DIR, f"{track}.log")


def is_running(track: str) -> bool:
    pf = pid_file(track)
    if not os.path.exists(pf):
        return False
    try:
        pid = int(open(pf).read().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def check_health(track: str) -> dict:
    port = TRACKS[track]["port"]
    url = f"http://127.0.0.1:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "offline", "error": str(e)[:80]}


def start_track(track: str) -> bool:
    if track not in TRACKS:
        print(f"  ❌ Unknown track: {track}")
        return False

    if is_running(track):
        print(f"  ✅ {TRACKS[track]['name']} already running (port {TRACKS[track]['port']})")
        return True

    config = TRACKS[track]
    script = config["script"]
    python_bin = config["python"]

    if not os.path.exists(script):
        print(f"  ❌ Script not found: {script}")
        return False

    if not os.path.exists(python_bin):
        python_bin = sys.executable  # fallback to system python

    env = os.environ.copy()
    env.update(config.get("env", {}))

    log = log_file(track)
    pf = pid_file(track)

    print(f"  🚀 Starting {config['name']} on port {config['port']} ({config['gpu']}, ~{config['vram']}GB)...")
    print(f"     Python: {python_bin}")
    print(f"     Log: {log}")

    with open(log, "a") as lf:
        lf.write(f"\n=== TTS Manager starting {config['name']} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        proc = subprocess.Popen(
            [python_bin, script, "--port", str(config["port"]), "--host", "0.0.0.0"],
            stdout=lf,
            stderr=lf,
            env=env,
            start_new_session=True,
            cwd=REPO_ROOT,
        )

    with open(pf, "w") as f:
        f.write(str(proc.pid))

    time.sleep(3)
    if is_running(track):
        health = check_health(track)
        status = health.get("status", "unknown")
        icon = "🟢" if status == "ok" else "🟡"
        print(f"  {icon} {config['name']} started (PID {proc.pid}, port {config['port']}, health: {status})")
        return True
    else:
        print(f"  ❌ {config['name']} process died — check {log}")
        return False


def stop_track(track: str) -> bool:
    pf = pid_file(track)
    if not os.path.exists(pf):
        print(f"  ⏭️  {track} not running")
        return True

    try:
        pid = int(open(pf).read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        print(f"  🛑 Stopped {track} (PID {pid})")
    except (ValueError, ProcessLookupError, PermissionError) as e:
        print(f"  ⚠️  Error stopping {track}: {e}")

    try:
        os.unlink(pf)
    except OSError:
        pass
    return True


def test_track(track: str) -> bool:
    """Quick TTS synthesis test."""
    port = TRACKS[track]["port"]
    url = f"http://127.0.0.1:{port}/tts"

    if track == "zh":
        payload = json.dumps({
            "text": "你好，这是一个语音合成测试。",
            "text_lang": "zh",
            "ref_audio_path": "",
            "prompt_text": "测试",
            "prompt_lang": "zh",
            "output_path": os.path.join(tempfile.gettempdir(), "tts_test_zh.wav"),
            "speed_factor": 1.0,
        }).encode()
    elif track == "en":
        payload = json.dumps({
            "text": "Hello, this is a voice synthesis test.",
            "temperature": 0.8,
            "output_path": os.path.join(tempfile.gettempdir(), "tts_test_en.wav"),
        }).encode()
    else:  # bilingual
        payload = json.dumps({
            "text": "这是一个双语测试。This is a bilingual test.",
            "mode": "instruct2",
            "instruct_text": "用自然的声音朗读",
            "output_path": os.path.join(tempfile.gettempdir(), "tts_test_bilingual.wav"),
        }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            audio_path = result.get("audio_path", "")
            duration = result.get("duration_sec", 0)
            print(f"  ✅ Test passed! Duration: {duration}s, Output: {audio_path}")
            if os.path.exists(audio_path):
                size = os.path.getsize(audio_path)
                print(f"     File size: {size} bytes")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  ❌ HTTP {e.code}: {body}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def show_status():
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║                TTS 三轨服务状态 (Three-Track TTS)                 ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║ Track      │ Service          │ Port │ GPU    │ VRAM  │ Status    ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")

    for track, config in TRACKS.items():
        running = is_running(track)
        health = check_health(track) if running else {}

        status = health.get("status", "offline")
        vram = health.get("vram_used_mb", f"~{config['vram']}GB")
        icon = "🟢" if status == "ok" else ("🟡" if running else "🔴")

        print(f"║ {track:9s} │ {config['name']:16s} │ {config['port']:4d} │ {config['gpu']:6s} │ {str(vram):5s} │ {icon} {status:8s} ║")

    print("╚═══════════════════════════════════════════════════════════════════╝\n")


def main():
    parser = argparse.ArgumentParser(description="TTS Service Manager")
    parser.add_argument("action", choices=["status", "start", "stop", "restart", "test", "logs"])
    parser.add_argument("track", nargs="?", help="Specific track (zh/en/bilingual) or all")
    args = parser.parse_args()

    tracks = [args.track] if args.track and args.track in TRACKS else list(TRACKS.keys())

    if args.action == "status":
        show_status()

    elif args.action == "start":
        for t in tracks:
            start_track(t)
        print()

    elif args.action == "stop":
        for t in tracks:
            stop_track(t)

    elif args.action == "restart":
        for t in tracks:
            stop_track(t)
            time.sleep(2)
            start_track(t)

    elif args.action == "test":
        for t in tracks:
            if is_running(t):
                print(f"🧪 Testing {TRACKS[t]['name']}...")
                test_track(t)
            else:
                print(f"⏭️  {t} not running, skipping test")

    elif args.action == "logs":
        for t in tracks:
            lf = log_file(t)
            if os.path.exists(lf):
                print(f"\n=== {TRACKS[t]['name']} ({lf}) ===")
                subprocess.run(["tail", "-30", lf])


if __name__ == "__main__":
    main()
