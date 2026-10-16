"""Colour segmentation of red / blue pen marks in CIELAB.

Works on chroma relative to the estimated paper colour, so tinted or unevenly
lit scans do not trigger false detections.  Raw pixel masks are then refined by
stroke_analyzer (shape / context) -- we never blindly delete "all red pixels".
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Settings, profile_for

RED_RANGE = ((0.0, 55.0), (320.0, 360.0))     # Lab hue angle (deg)
BLUE_RANGE = ((215.0, 315.0),)


def lab_chroma(bgr: np.ndarray):
    """Return (L, a_rel, b_rel, chroma, hue_deg) with paper tint removed."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    step = max(1, int(np.sqrt(L.size / 400_000)))
    Ls = L[::step, ::step]
    sel = Ls >= max(150, int(np.percentile(Ls, 60)))
    if sel.any():
        a -= float(np.median(a[::step, ::step][sel]))
        b -= float(np.median(b[::step, ::step][sel]))
    chroma, hue = cv2.cartToPolar(a, b, angleInDegrees=True)
    return L, a, b, chroma, hue


def _in_ranges(hue: np.ndarray, ranges) -> np.ndarray:
    out = np.zeros(hue.shape, bool)
    for lo, hi in ranges:
        out |= (hue >= lo) & (hue <= hi)
    return out


def chroma_threshold(sens: float, scale: float) -> float:
    sens = float(np.clip(sens, 0.0, 1.0))
    return (36.0 - 26.0 * sens) * scale


def detect_pen_pixels(bgr: np.ndarray, settings: Settings):
    """Pixel-level candidates.  Returns (high_conf, low_conf) uint8 0/255 masks."""
    prof = profile_for(settings)
    _, _, _, chroma, hue = lab_chroma(bgr)
    high = np.zeros(chroma.shape, bool)
    low = np.zeros(chroma.shape, bool)
    for enabled, sens, ranges in (
        (settings.detect_red, settings.sens_red, RED_RANGE),
        (settings.detect_blue, settings.sens_blue, BLUE_RANGE),
    ):
        if not enabled:
            continue
        thr = chroma_threshold(sens, prof.thr_scale)
        in_h = _in_ranges(hue, ranges)
        high |= in_h & (chroma > thr)
        low |= in_h & (chroma > 0.55 * thr)
    return high.astype(np.uint8) * 255, low.astype(np.uint8) * 255


def detect_mask(bgr: np.ndarray, settings: Settings) -> np.ndarray:
    """Full detection: colour -> stroke analysis -> expansion. Returns 0/255 mask."""
    from core import stroke_analyzer

    high, low = detect_pen_pixels(bgr, settings)
    if not high.any():
        return np.zeros(high.shape, np.uint8)
    mask = stroke_analyzer.refine(high, low, bgr, settings)
    return stroke_analyzer.expand(mask, settings)
