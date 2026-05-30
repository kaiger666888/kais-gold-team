# Engine Status Report

Generated: 2026-05-29 15:12

## Local Docker Images (15 total)

| Image | Size | Port | Status |
|-------|------|------|--------|
| kais-wan | 19.1GB | 8081 | ✅ Running |
| kais-ltx | 18.9GB | 8083 | ✅ Running |
| kais-rife | 19.1GB | 8086 | ✅ Running |
| kais-sdxl | 18.8GB | 8087 | ✅ Running |
| kais-flux | 18.8GB | 8080 | ✅ Running |
| kais-liveportrait | 20.1GB | 8085 | ✅ Running |
| kais-gpt_sovits | 20.4GB | 8094 | ⚠️ Running (model loading) |
| kais-yue | 18.8GB | 8092 | ✅ Running |
| kais-uvr5 | 18.9GB | 8093 | ✅ Running |
| kais-facefusion | 18.8GB | 7860 | ✅ Running |
| kais-acestep | 19.9GB | 8009 | ⚠️ Running (downloading config) |
| kais-forge | 47.1GB | 7870 | ⬜ Not started |
| kais-cosyvoice2 | 35.8GB | 9880 | ⬜ Not started |
| kais-blender | 5.82GB | N/A | ⬜ Not started |
| kais-gold-team:cosyvoice2 | 35.8GB | 9880 | ⬜ Not started |

## Health Check Results

| Engine | Container | Port | API Health |
|--------|-----------|------|------------|
| Wan2.1 (T2V/I2V) | kais-engine-wan | 8081 | ✅ Healthy |
| LTX-Video (I2V) | kais-engine-ltx | 8083 | ✅ Healthy |
| RIFE (Interpolation) | kais-engine-rife | 8086 | ✅ Healthy |
| SDXL (T2I) | kais-engine-sdxl | 8087 | ✅ Healthy |
| FLUX (T2I) | kais-engine-flux | 8080 | ✅ Healthy |
| LivePortrait (Virtual Human) | kais-engine-liveportrait | 8085 | ✅ Healthy |
| Yue (Music) | kais-engine-yue | 8092 | ✅ Healthy |
| UVR5 (Audio Sep) | kais-engine-uvr5 | 8093 | ✅ Healthy |
| FaceFusion (Face Swap) | kais-engine-facefusion | 7860 | ✅ Healthy |
| GPT-SoVITS (Voice Clone) | kais-engine-gpt_sovits | 8094 | ⚠️ Model loading |
| AceStep (Audio) | kais-engine-acestep | 8009 | ⚠️ Config download |

## Engine YAML Definitions (15 files in engines/)

| YAML | Image | Task Types |
|------|-------|------------|
| wan14b.yaml | kais-wan | wan14b_t2v_preview, wan14b_t2v_final |
| wan13b.yaml | kais-wan | wan13b_t2v_preview |
| wan14b_i2v.yaml | kais-wan | wan14b_i2v_preview, wan14b_i2v_final |
| ltx_i2v.yaml | kais-ltx | ltx_i2v_preview |
| rife.yaml | kais-rife | rife_interpolation |
| sd35_large.yaml | kais-sdxl | sd35_draw |
| sdxl_lightning.yaml | kais-sdxl | sdxl_draw |
| flux.yaml | kais-flux | flux_draw |
| liveportrait.yaml | kais-liveportrait | liveportrait_generate |
| gpt_sovits.yaml | kais-gpt_sovits | voice_clone |
| yue.yaml | kais-yue | music_generate |
| uvr5.yaml | kais-uvr5 | audio_separate |
| facefusion.yaml | kais-facefusion | face_swap |
| acestep.yaml | kais-acestep | audio_generate |
| cosyvoice.yaml | kais-gold-team:cosyvoice2 | cosyvoice_tts |

## Available Task Types (16 total)

1. wan14b_t2v_preview — 文生视频预览 (480p, 5s)
2. wan14b_t2v_final — 文生视频最终版 (720p, 5s)
3. wan13b_t2v_preview — 快速视频预览 (480p, 3s)
4. wan14b_i2v_preview — 图生视频预览 (480p)
5. wan14b_i2v_final — 图生视频最终版 (720p)
6. ltx_i2v_preview — LTX 图生视频
7. rife_interpolation — RIFE 帧插值
8. sd35_draw — SD3.5 文生图
9. sdxl_draw — SDXL 快速文生图
10. flux_draw — FLUX 文生图
11. liveportrait_generate — LivePortrait 虚拟人
12. voice_clone — GPT-SoVITS 语音克隆
13. music_generate — Yue 音乐生成
14. audio_separate — UVR5 音频分离
15. face_swap — FaceFusion 换脸
16. cosyvoice_tts — CosyVoice TTS
17. audio_generate — AceStep 音效生成

## Notes

- All engines share GPU 0 (RTX 3090 24GB) — concurrent execution may cause OOM
- kais-forge (47.1GB) and kais-blender not started due to size/complexity
- gpt_sovits needs external API service to be fully ready
- acestep host port mapped to 8009 (8001 conflicts with kais-movie-agent)
- SDXL and SD3.5 share the same image (kais-sdxl) — only one container needed
- Video generation priority: wan14b > ltx > rife
