"""Programmatic .cube 3D-LUT generator for color-grade presets.

Generates a standard Adobe ``.cube`` 3D LUT (33^3 grid) for each of six
cinematic presets. Each preset encodes:
    * shadow / highlight tint separation (warm vs. cool split),
    * an S-curve contrast boost (or fade for low-contrast presets),
    * a saturation multiplier,
and blends the result with the original colours at the requested ``strength``
(0.0 = original, 1.0 = fully graded).

LUTs are cached under ``/mnt/agents/output/luts/{preset}_s{strength}.cube`` so
that repeated renderings of the same style incur zero regeneration cost.

The maths is intentionally simple (per-channel, in [0,1] sRGB) — ffmpeg's
``lut3d`` filter will interpolate the 33^3 grid smoothly at runtime.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────

LUT_SIZE = 33  # 33^3 3D LUT grid — standard for .cube files
OUTPUT_ROOT = os.environ.get("KAIS_OUTPUT_ROOT", "/mnt/agents/output")
LUT_CACHE_DIR = os.environ.get(
    "COLOR_GRADE_LUT_DIR",
    os.path.join(OUTPUT_ROOT, "luts"),
)


# ─── Preset definitions ──────────────────────────────────────────────────

@dataclass(frozen=True)
class PresetConfig:
    """Parameters describing one cinematic colour preset.

    Tint vectors are additive RGB deltas in the range roughly [-0.3, 0.3].
    They are applied to the shadow region (``shadow_*``) and highlight region
    (``highlight_*``) independently, with the split driven by a luminance
    weight so mid-tones are barely affected.

    Attributes:
        shadow:       (dR, dG, dB) applied to shadows.
        highlight:    (dR, dG, dB) applied to highlights.
        saturation:   multiplicative saturation factor (1.0 = unchanged).
        contrast:     S-curve steepness (0 ≈ no contrast change, >1 boosts).
        contrast_bias:centre of the S-curve (usually 0.5).
    """

    shadow: tuple[float, float, float]
    highlight: tuple[float, float, float]
    saturation: float
    contrast: float
    contrast_bias: float = 0.5


# (dR, dG, dB) deltas:
#   teal    ≈ (-0.18, +0.04, +0.10)
#   orange  ≈ (+0.12, +0.03, -0.10)
#   sepia   ≈ (-0.05, -0.03, -0.08)   (brown-ish)
#   gold    ≈ (+0.10, +0.05, -0.06)
#   deep blue ≈ (-0.08, -0.04, +0.16)
#   cyan      ≈ (-0.12, +0.02, +0.08)
#   mid-grey  ≈ (0, 0, 0)
#   pure white shift ≈ (0, 0, 0)
#   yellow-green ≈ (-0.05, +0.03, -0.14)
#   warm yellow  ≈ (+0.12, +0.06, -0.10)
PRESETS: dict[str, PresetConfig] = {
    "teal_orange": PresetConfig(
        shadow=(-0.18, 0.04, 0.10),   # teal shadows
        highlight=(0.12, 0.03, -0.10),  # orange highlights
        saturation=1.25,
        contrast=1.5,
    ),
    "warm_cinematic": PresetConfig(
        shadow=(-0.05, -0.03, -0.08),  # sepia / brown
        highlight=(0.10, 0.05, -0.06),  # gold
        saturation=1.15,
        contrast=1.25,
    ),
    "cold_thriller": PresetConfig(
        shadow=(-0.08, -0.04, 0.16),   # deep blue
        highlight=(-0.12, 0.02, 0.08),  # cyan
        saturation=0.90,               # -10%
        contrast=1.8,                  # high contrast
    ),
    "film_noir": PresetConfig(
        shadow=(0.0, 0.0, 0.0),        # mid-grey → desaturated
        highlight=(0.0, 0.0, 0.0),     # white highlights
        saturation=0.40,               # -60%
        contrast=2.4,                  # extreme contrast
    ),
    "vintage": PresetConfig(
        shadow=(-0.05, 0.03, -0.14),   # yellow-green
        highlight=(0.12, 0.06, -0.10),  # warm yellow
        saturation=1.20,               # +20%
        contrast=0.7,                  # faded (low contrast)
    ),
    "high_contrast": PresetConfig(
        shadow=(0.0, 0.0, 0.0),        # pure black
        highlight=(0.0, 0.0, 0.0),     # pure white
        saturation=1.10,
        contrast=2.0,                  # strong S-curve
    ),
}

PRESET_NAMES: list[str] = list(PRESETS.keys())


# ─── Colour helpers ──────────────────────────────────────────────────────

def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _luma(r: float, g: float, b: float) -> float:
    """Perceptual luminance (Rec. 709 weights)."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _s_curve(x: float, contrast: float, bias: float = 0.5) -> float:
    """Apply a smooth S-curve contrast around ``bias``.

    ``contrast`` = 1 → identity; >1 → steeper (more contrast); <1 → flatter.
    Uses a sigmoid-style mapping remapped to [0,1].
    """
    if contrast <= 0.0:
        return x
    # Normalise around the bias, then apply a tanh-based S.
    t = (x - bias) / max(bias if bias >= 0.5 else (1.0 - bias), 1e-6)
    # Steepness grows with contrast; contrast=1 should ≈ identity.
    k = (contrast - 1.0) * 4.0
    shaped = math.tanh(t * k) / math.tanh(k) if k != 0.0 else t
    out = bias + shaped * (bias if x < bias else (1.0 - bias))
    return _clamp(out)


def _apply_saturation(r: float, g: float, b: float, sat: float) -> tuple[float, float, float]:
    """Push colours toward (sat>1) or away from (sat<1) the grey axis."""
    l = _luma(r, g, b)
    return (
        _clamp(l + (r - l) * sat),
        _clamp(l + (g - l) * sat),
        _clamp(l + (b - l) * sat),
    )


def _transform_pixel(
    r: float, g: float, b: float,
    cfg: PresetConfig,
) -> tuple[float, float, float]:
    """Apply one preset's full transform to a single RGB sample in [0,1]."""
    # 1. Luminance-driven shadow/highlight tint split.
    #    shadow_weight: 1 in deep shadows → 0 in highlights
    #    highlight_weight: 0 in shadows → 1 in highlights
    l = _luma(r, g, b)
    shadow_w = _clamp(1.0 - l * 2.0)        # full effect below l=0.5
    highlight_w = _clamp((l - 0.5) * 2.0)   # full effect above l=0.5

    r += cfg.shadow[0] * shadow_w + cfg.highlight[0] * highlight_w
    g += cfg.shadow[1] * shadow_w + cfg.highlight[1] * highlight_w
    b += cfg.shadow[2] * shadow_w + cfg.highlight[2] * highlight_w
    r, g, b = _clamp(r), _clamp(g), _clamp(b)

    # 2. S-curve contrast per channel (keeps hues, shifts tonality).
    r = _s_curve(r, cfg.contrast, cfg.contrast_bias)
    g = _s_curve(g, cfg.contrast, cfg.contrast_bias)
    b = _s_curve(b, cfg.contrast, cfg.contrast_bias)

    # 3. Saturation boost / cut.
    r, g, b = _apply_saturation(r, g, b, cfg.saturation)
    return _clamp(r), _clamp(g), _clamp(b)


def _blend(orig: tuple[float, float, float], graded: tuple[float, float, float], strength: float) -> tuple[float, float, float]:
    """Linear blend between original and graded colours (0..1)."""
    s = max(0.0, min(1.0, strength))
    return (
        orig[0] + (graded[0] - orig[0]) * s,
        orig[1] + (graded[1] - orig[1]) * s,
        orig[2] + (graded[2] - orig[2]) * s,
    )


# ─── .cube writer ────────────────────────────────────────────────────────

def _build_lut_rows(cfg: PresetConfig, strength: float) -> list[str]:
    """Compute all 33^3 LUT rows as formatted ``R G B`` strings."""
    n = LUT_SIZE
    step = 1.0 / (n - 1)
    rows: list[str] = []
    for bi in range(n):
        b = bi * step
        for gi in range(n):
            g = gi * step
            for ri in range(n):
                r = ri * step
                gr, gg, gb = _transform_pixel(r, g, b, cfg)
                fr, fg, fb = _blend((r, g, b), (gr, gg, gb), strength)
                rows.append(f"{fr:.6f} {fg:.6f} {fb:.6f}")
    return rows


def _write_cube(path: str, cfg: PresetConfig, strength: float, rows: list[str]) -> None:
    """Write a standard Adobe ``.cube`` file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        f"# kais-aigc color-grade preset LUT",
        f"# preset: generated programmatically",
        f"# strength: {strength:.3f}",
        f"LUT_3D_SIZE {LUT_SIZE}",
        f"DOMAIN_MIN 0.0 0.0 0.0",
        f"DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n".join(rows))
        f.write("\n")


# ─── Public API ──────────────────────────────────────────────────────────

def _lut_path(preset: str, strength: float) -> str:
    return os.path.join(LUT_CACHE_DIR, f"{preset}_s{strength:.2f}.cube")


def get_preset_lut(preset: str, strength: float = 1.0) -> str:
    """Return the path to a cached ``.cube`` for ``preset``/``strength``.

    Generates the LUT on first use and caches it on disk; subsequent calls
    with the same arguments return the cached file immediately.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown color-grade preset '{preset}'. "
            f"Valid presets: {', '.join(PRESET_NAMES)}"
        )
    # Quantise strength to 2dp so cache keys are stable.
    strength = round(max(0.0, min(1.0, strength)), 2)
    path = _lut_path(preset, strength)
    if os.path.isfile(path):
        logger.debug("Color-grade LUT cache hit: %s", path)
        return path

    logger.info("Generating color-grade LUT for preset '%s' (strength=%.2f)", preset, strength)
    cfg = PRESETS[preset]
    rows = _build_lut_rows(cfg, strength)
    _write_cube(path, cfg, strength, rows)
    logger.info("Color-grade LUT written → %s (%d entries)", path, len(rows))
    return path


def list_presets() -> list[str]:
    """Return the list of available preset names."""
    return PRESET_NAMES
