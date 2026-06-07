#!/usr/bin/env python3
"""TTS Service Manager — manage three-track TTS services.

Usage:
    python scripts/tts_manager.py status         # Check all services
    python scripts/tts_manager.py start [track]  # Start all or specific track
    python scripts/tts_manager.py stop [track]   # Stop all or specific track
    python scripts/tts_manager.py restart [track]
    python scripts/tts_manager.py logs [track]    # Tail logs

Tracks: zh (GPT-SoVITS), en (Chatterbox), bilingual (CosyVoice)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

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
        "gpu": 1,  # 3060Ti
        "vram": 4.0,
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
    },
    "en": {
        "name": "Chatterbox-Turbo",
        "port": 9881,
        "script": os.path.join(REPO_ROOT, "scripts", "tts_chatterbox_server.py"),
        "gpu": 1,  # 3060Ti
        "vram": 2.0,
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
    },
    "bilingual": {
        "name": "CosyVoice-3.0",
        "port": 9882,
        "script": os.path.join(REPO_ROOT, "scripts", "tts_cosyvoice_server.py"),
        "gpu": 0,  # 3090
        "vram": 6.0,
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
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
        os.kill(pid, 0)  # Check if process exists
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def start_track(track: str) -> bool:
    if track not in TRACKS:
        print(f"❌ Unknown track: {track}. Available: {', '.join(TRACKS.keys())}")
        return False

    if is_running(track):
        print(f"✅ {TRACKS[track]['name']} already running (port {TRACKS[track]['port']})")
        return True

    config = TRACKS[track]
    script = config["script"]

    if not os.path.exists(script):
        print(f"❌ Script not found: {script}")
        return False

    env = os.environ.copy()
    env.update(config.get("env", {}))

    log = log_file(track)
    pf = pid_file(track)

    print(f"🚀 Starting {config['name']} on port {config['port']} (GPU {config['gpu']}, ~{config['vram']}GB)...")
    print(f"   Log: {log}")

    with open(log, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, script, "--port", str(config["port"]), "--host", "0.0.0.0"],
            stdout=lf,
            stderr=lf,
            env=env,
            start_new_session=True,
        )

    with open(pf, "w") as f:
        f.write(str(proc.pid))

    # Wait a moment and check
    time.sleep(2)
    if is_running(track):
        print(f"✅ {config['name']} started (PID {proc.pid}, port {config['port']})")
        return True
    else:
        print(f"❌ {config['name']} failed to start — check {log}")
        return False


def stop_track(track: str) -> bool:
    pf = pid_file(track)
    if not os.path.exists(pf):
        print(f"⏭️  {track} not running (no PID file)")
        return True

    try:
        pid = int(open(pf).read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"🛑 Stopped {track} (PID {pid})")
    except (ValueError, ProcessLookupError, PermissionError) as e:
        print(f"⚠️  Error stopping {track}: {e}")

    try:
        os.unlink(pf)
    except OSError:
        pass
    return True


def check_health(track: str) -> dict:
    import urllib.request
    import json

    config = TRACKS[track]
    url = f"http://127.0.0.1:{config['port']}/health"

    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def show_status():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║              TTS 三轨服务状态 (Three-Track TTS)              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ Track  │ Service          │ Port │ GPU    │ VRAM │ Status    ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    for track, config in TRACKS.items():
        running = is_running(track)
        health = check_health(track) if running else {}

        gpu_str = f"3090 #{config['gpu']}" if config['gpu'] == 0 else f"3060Ti #{config['gpu']}"
        status = health.get("status", "offline")
        vram = health.get("vram_used_mb", f"~{config['vram']}GB")
        icon = "🟢" if status == "ok" else ("🟡" if running else "🔴")

        print(f"║ {track:6s} │ {config['name']:16s} │ {config['port']:4d} │ {gpu_str:6s} │ {str(vram):5s} │ {icon} {status:8s} ║")

    print("╚══════════════════════════════════════════════════════════════╝\n")


def main():
    parser = argparse.ArgumentParser(description="TTS Service Manager")
    parser.add_argument("action", choices=["status", "start", "stop", "restart", "logs"])
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
            time.sleep(1)
            start_track(t)

    elif args.action == "logs":
        for t in tracks:
            lf = log_file(t)
            if os.path.exists(lf):
                print(f"\n=== {TRACKS[t]['name']} logs ({lf}) ===")
                subprocess.run(["tail", "-20", lf])


if __name__ == "__main__":
    main()
