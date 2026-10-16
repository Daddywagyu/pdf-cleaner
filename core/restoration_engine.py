"""Background restoration after the pen mask is known.

Per connected region the engine picks a method:
  * flat paper around the stroke  -> background estimate (+ matched grain)
  * translucent ink on flat paper (advanced) -> transmittance un-mixing, which
    keeps print that shows through the ink
  * anything else (text / lines / photos nearby) -> OpenCV Telea / Navier-Stokes
A residual-colour pass then re-treats pixels that are still tinted.
Hidden detail cannot be recovered; the engine only propagates evidence from
the surrounding pixels and never synthesises new text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from config import Settings, profile_for
from core import color_detector


@dataclass
class RestoreResult:
    image: np.ndarray
    residual_ratio: float      # remaining tinted pixels / original mask pixels
    regions: int
    methods: dict


def _ellipse(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _background_fill(roi: np.ndarray, comp: np.ndarray, valid: np.ndarray, sigma: float):
    """Normalised-convolution background estimate; None if it cannot cover the hole."""
    img = roi.astype(np.float32)
    v = valid.astype(np.float32)
    est = np.zeros_like(img)
    done = np.zeros(comp.shape, bool)
    s = sigma
    for _ in range(5):
        den = cv2.GaussianBlur(v, (0, 0), s)
        num = cv2.GaussianBlur(img * v[..., None], (0, 0), s)
        ok = (den > 0.08) & ~done
        est[ok] = num[ok] / den[ok][:, None]
        done |= ok
        if done[comp > 0].all():
            return est
        s *= 2.5
    return None


def _dark(px: np.ndarray, paper_lum: float) -> np.ndarray:
    return px.astype(np.float32).mean(axis=-1) < min(120.0, paper_lum - 90.0)


def _reconnect_rules(img, orig, comp, allmask, k: float, paper_lum: float):
    """Straight horizontal / vertical rules (table lines, graph axes) that the pen crossed are
    re-drawn by linear interpolation, but only when a long dark run exists on BOTH sides
    (so ordinary text strokes are never turned into bars)."""
    L = max(8, int(round(25 * k)))
    for axis in (0, 1):                      # 0: horizontal rules (scan rows), 1: vertical rules
        o = orig if axis == 0 else orig.transpose(1, 0, 2)
        c = comp if axis == 0 else comp.T
        a = allmask if axis == 0 else allmask.T
        out = img if axis == 0 else img.transpose(1, 0, 2)          # views -> in-place writes OK
        H, W = c.shape
        for y in np.where(c.any(axis=1))[0]:
            row = c[y] > 0
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            for s0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):   # run [s0, e0)
                if e0 - s0 > 6 * L or s0 - L < 0 or e0 + L > W:
                    continue
                left, right = o[y, s0 - L:s0], o[y, e0:e0 + L]
                if a[y, s0 - L:s0].any() or a[y, e0:e0 + L].any():
                    continue
                if _dark(left, paper_lum).all() and _dark(right, paper_lum).all():
                    t = np.linspace(0, 1, e0 - s0 + 2)[1:-1, None]
                    out[y, s0:e0] = (left[-1] * (1 - t) + right[0] * t).astype(np.uint8)


def _restore_region(img, orig, comp, allmask, settings, rng, counts):
    """img/orig/comp/allmask are ROI views; img is modified in place."""
    k = settings.scale
    radius = max(1, int(round(settings.inpaint_radius * k)))
    method = settings.restore_method
    prof = profile_for(settings)
    rw = max(4, 2 * radius + 2)
    ring = cv2.bitwise_and(cv2.dilate(comp, _ellipse(rw)), cv2.bitwise_not(allmask)) > 0
    valid = (allmask == 0)

    paper_lum = float(np.percentile(orig.mean(axis=2), 80))

    def telea(flag=cv2.INPAINT_TELEA):
        img[:] = cv2.inpaint(img, comp, radius, flag)
        _reconnect_rules(img, orig, comp, allmask, k, paper_lum)

    if method in ("telea", "ns"):
        counts[method] = counts.get(method, 0) + 1
        return telea(cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS)
    if ring.sum() < 12:
        counts["telea"] = counts.get("telea", 0) + 1
        return telea()

    ring_px = orig[ring].astype(np.float32)
    std = ring_px.std(axis=0)
    flat = float(std.max()) < 12.0
    # paper colour = median of the brighter ring pixels (robust against print nearby)
    ring_lum = ring_px.mean(axis=1)
    bright_ring = ring_px[ring_lum >= np.percentile(ring_lum, 60)]
    paper = np.median(bright_ring, axis=0)
    paper_flat = float(bright_ring.std(axis=0).max()) < 10.0

    if prof.unmix and method == "auto" and paper_flat:
        m = comp > 0
        comp_px = orig[m].astype(np.float32)
        lum = comp_px.mean(axis=1)
        bright = comp_px[lum >= np.median(lum)]
        t = np.clip(np.median(bright, axis=0) / np.maximum(paper, 1.0), 0.01, 1.0)
        if 0.30 <= float(t.min()) < 0.97:                          # translucent ink only
            res = np.minimum(comp_px / t, paper + 6.0)
            img[m] = np.clip(res, 0, 255).astype(np.uint8)
            counts["unmix"] = counts.get("unmix", 0) + 1
            return
    if method == "background" or flat:
        est = _background_fill(orig, comp, valid, sigma=max(3.0, rw))
        if est is not None:
            m = comp > 0
            noise = rng.normal(0.0, 1.0, size=(int(m.sum()), 3)).astype(np.float32) * (std * 0.7)
            img[m] = np.clip(est[m] + noise, 0, 255).astype(np.uint8)
            counts["background"] = counts.get("background", 0) + 1
            return
    counts["telea"] = counts.get("telea", 0) + 1
    telea()


def restore(bgr: np.ndarray, mask: np.ndarray, settings: Settings,
            cancel: Callable[[], bool] | None = None) -> RestoreResult:
    out = bgr.copy()
    m = (mask > 0).astype(np.uint8) * 255
    total = int(np.count_nonzero(m))
    if total == 0:
        return RestoreResult(out, 0.0, 0, {})
    k = settings.scale
    rng = np.random.default_rng(1234)          # deterministic output
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    pad = int(round(40 * k))
    H, W = m.shape
    counts: dict = {}
    for i in range(1, n):
        if cancel and cancel():
            break
        x, y, w, h, _a = stats[i]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        comp = ((labels[y0:y1, x0:x1] == i).astype(np.uint8)) * 255
        _restore_region(out[y0:y1, x0:x1], bgr[y0:y1, x0:x1], comp, m[y0:y1, x0:x1],
                        settings, rng, counts)

    # residual colour clean-up (limited to the neighbourhood of the mask)
    residual = _residual_mask(out, m, settings)
    if residual.any():
        r = max(1, int(round(settings.inpaint_radius * k)))
        residual = cv2.dilate(residual, _ellipse(1))
        out = cv2.inpaint(out, residual, r, cv2.INPAINT_TELEA)
        residual = _residual_mask(out, m, settings)
    ratio = float(np.count_nonzero(residual)) / total
    return RestoreResult(out, ratio, n - 1, counts)


def _residual_mask(img: np.ndarray, mask: np.ndarray, settings: Settings) -> np.ndarray:
    k = settings.scale
    band = cv2.dilate(mask, _ellipse(max(1, int(round(3 * k)))))
    high, _ = color_detector.detect_pen_pixels(img, settings)
    return cv2.bitwise_and(high, band)
