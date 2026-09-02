# Engine Status Report

> ⚠️ **退役说明（2026-09-02）**
> 本文档原为 2026-05-29 的引擎舰队状态快照，所载 "✅ Running / ✅ Healthy" 状态均已失真，
> 不代表当前实况。2026-09-02 经 `docker ps -a` / `docker images` 核实：
> kais-ltx / kais-wan / kais-rife 等独立引擎容器的容器与镜像均已不存在于本机。
> 视频生成已由 KMC 全量切换 H3 路径（P11a preview-lock / P11b lightx2v-8-768p），
> ltx_i2v 引擎配置面已于同日自本仓整体退役（engines/ltx_i2v.yaml 删除，
> routing_table / models_registry / stage_config / engine_registry 同步清理）。
> 本页改写后仅保留经核实的当前真值，不再维护逐容器健康状态——请以 `docker ps` 实测为准。

Generated: 2026-05-29 15:12（历史快照，已失真）
Rewritten: 2026-09-02（退役真值改写）

## Docker 实况（2026-09-02 `docker ps -a` 核实）

| 容器 | 镜像 | 状态 |
|------|------|------|
| kais-aigc-platform-gold-team-1 | kais-gold-team:real | Up (healthy) |
| comfyui-primary | yanwk/comfyui-boot:cu130-megapak-pt211 | Up (healthy) |
| comfyui-auxiliary | yanwk/comfyui-boot:cu130-megapak-pt211 | Up (healthy) |
| kais-gold-team | kais-aigc-platform-kais-gold-team | Created（从未启动） |
| **kais-ltx / kais-wan / kais-rife** | — | **不存在**（容器与镜像均已清除） |

旧引擎舰队镜像（kais-ltx / kais-wan / kais-rife / kais-sdxl / kais-flux /
kais-liveportrait / kais-yue / kais-uvr5 / kais-facefusion / kais-acestep 等）
已不在本机 docker images 中。`engines/*.yaml` 中仍声明这些 `docker_image` 的条目
属于陈旧配置，按需逐引擎退役（ltx_i2v 已于 2026-09-02 完成）。

## Engine YAML Definitions (16 files in engines/, 2026-09-02)

| YAML | Image | Task Types |
|------|-------|------------|
| acestep.yaml | kais-acestep:latest | audio_generate |
| chatterbox.yaml | none（native service） | tts_en |
| cosyvoice.yaml | none（native service） | tts_bilingual |
| facefusion.yaml | kais-facefusion:latest | face_swap |
| flux.yaml | kais-flux:latest | flux_draw |
| gpt_sovits.yaml | none（native service） | tts_zh |
| liveportrait.yaml | kais-liveportrait:latest | liveportrait_generate |
| rife.yaml | kais-rife:latest | rife_interpolation |
| sd35_large.yaml | kais-sdxl:latest | sd35_draw |
| sdxl_lightning.yaml | kais-sdxl:latest | sdxl_draw |
| trellis.yaml | kais-comfyui:latest | text_to_3d, image_to_3d, preview_3d |
| uvr5.yaml | kais-uvr5:latest | audio_separate |
| wan13b.yaml | kais-wan:latest | wan13b_t2v_preview |
| wan14b_i2v.yaml | kais-wan:latest | wan14b_i2v_preview, wan14b_i2v_final |
| wan14b.yaml | kais-wan:latest | wan14b_t2v_preview, wan14b_t2v_final |
| yue.yaml | kais-yue:latest | music_generate |

~~ltx_i2v.yaml（kais-ltx / ltx_i2v_preview）~~ — 2026-09-02 随 ltx_i2v 引擎退役删除。

注：上表 Image 列为 YAML 声明值，非容器实况；除 comfyui 系与 kais-gold-team:real 外，
对应容器当前均不存在（见上方 Docker 实况）。

## Available Task Types (19 total)

1. audio_generate — AceStep 音效生成
2. tts_en — Chatterbox 英文 TTS
3. tts_bilingual — CosyVoice 双语 TTS
4. face_swap — FaceFusion 换脸
5. flux_draw — FLUX 文生图
6. tts_zh — GPT-SoVITS 中文 TTS
7. liveportrait_generate — LivePortrait 虚拟人
8. rife_interpolation — RIFE 帧插值
9. sd35_draw — SD3.5 文生图
10. sdxl_draw — SDXL 快速文生图
11. text_to_3d — TRELLIS 文生3D
12. image_to_3d — TRELLIS 图生3D
13. preview_3d — TRELLIS 3D 预览
14. audio_separate — UVR5 音频分离
15. wan13b_t2v_preview — 快速视频预览 (480p, 3s)
16. wan14b_i2v_preview — 图生视频预览 (480p)
17. wan14b_i2v_final — 图生视频最终版 (720p)
18. wan14b_t2v_preview — 文生视频预览 (480p, 5s)
19. wan14b_t2v_final — 文生视频最终版 (720p, 5s)

~~ltx_i2v_preview — LTX 图生视频~~ — 2026-09-02 随 ltx_i2v 引擎退役删除。

## Notes

- 旧注 "Video generation priority: wan14b > ltx > rife" 已失效——ltx 已退役，
  视频预览由 KMC H3 路径承接；本仓 wan/rife YAML 仍保留，是否退役由后续 lane 决定
- 旧快照的逐容器健康检查表（11 引擎 "Healthy"）已整体作废删除，勿引用
- TRELLIS 引擎走 comfyui 路径（kais-comfyui），与 comfyui-primary/auxiliary 容器同栈
- SDXL 与 SD3.5 共用同一镜像声明（kais-sdxl）——仅一个容器的声明，当前容器不存在
- chatterbox / cosyvoice / gpt_sovits 为 native service（无 docker 镜像），
  由宿主进程直接拉起，见 scripts/tts_*.py
