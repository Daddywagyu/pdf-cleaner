"""Shape / context analysis of candidate colour pixels.

Goal: keep real pen strokes, reject coloured *print* (small coloured text,
logos, coloured boxes).  Uses connected components, distance transform
(stroke width), text-line detection and hysteresis growth.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Settings, profile_for


def _lut_mask(labels: np.ndarray, selected: np.ndarray) -> np.ndarray:
    lut = np.zeros(len(selected), np.uint8)
    lut[selected] = 255
    lut[0] = 0
    return lut[labels]


def _text_like(stats: np.ndarray, k: float) -> np.ndarray:
    """Flag components that look like characters in a printed text line."""
    n = len(stats)
    flags = np.zeros(n, bool)
    w, h = stats[:, 2], stats[:, 3]
    cand = np.where(
        (h >= 6 * k) & (h <= 50 * k) & (w >= 2 * k) & (w <= 50 * k) & (np.arange(n) > 0)
    )[0]
    if len(cand) < 4 or len(cand) > 4000:
        return flags
    cx = stats[cand, 0] + w[cand] / 2.0
    cy = stats[cand, 1] + h[cand] / 2.0
    hh = h[cand].astype(np.float32)
    for j in range(len(cand)):
        near = (
            (np.abs(cy - cy[j]) < 0.5 * hh[j])
            & (np.abs(cx - cx[j]) < 4.0 * hh[j])
            & (np.abs(hh - hh[j]) < 0.5 * hh[j] + 3)
        )
        if near.sum() - 1 >= 3:
            flags[cand[j]] = True
    return flags


def _photo_body(clean: np.ndarray, k: float) -> np.ndarray:
    """Very thick coloured areas (photos, solid graphics): morphological opening with a
    ~50 px disc (computed at 1/4 resolution for speed).  Pen strokes are far thinner."""
    f = 0.25
    small = cv2.resize(clean, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    small = (small > 127).astype(np.uint8)
    r = max(2, int(round(25 * k * f)))
    body = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if not body.any():
        return np.zeros_like(clean)
    body = cv2.resize(body * 255, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_NEAREST)
    g = max(1, int(round(3 * k)))
    return cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))


def refine(high: np.ndarray, low: np.ndarray, bgr: np.ndarray, settings: Settings) -> np.ndarray:
    prof = profile_for(settings)
    k = settings.scale
    page_area = high.shape[0] * high.shape[1]

    clean = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if not clean.any():
        return np.zeros_like(high)

    body = _photo_body(clean, k) if prof.reject_graphics else np.zeros_like(clean)
    attached = None
    if body.any():
        n0, lab0 = cv2.connectedComponents(clean, connectivity=8)
        has_body = np.zeros(n0, bool)
        has_body[np.unique(lab0[body > 0])] = True
        has_body[0] = False
        attached = has_body[lab0]                      # pixels whose parent blob contains a photo body
        del lab0
        sm = cv2.resize(body, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        sm = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(10 * k)) + 1,) * 2))
        wide = cv2.resize(sm, (body.shape[1], body.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        attached = attached | wide                     # ...or lies in the surroundings of a photo
        near_body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(14 * k) + 1,) * 2))
        clean = cv2.bitwise_and(clean, cv2.bitwise_not(body))
        low = cv2.bitwise_and(low, cv2.bitwise_not(body))
        if not clean.any():
            return np.zeros_like(clean)
    else:
        near_body = None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    dt = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
    text_flags = _text_like(stats, k)

    keep = np.zeros(n, bool)
    small_ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        longest = max(w, h)
        sub = labels[y:y + h, x:x + w] == i
        thick = 2.0 * float(dt[y:y + h, x:x + w][sub].max())
        fill = area / float(w * h)
        if prof.reject_graphics:
            if thick >= 50 * k or (min(w, h) >= 120 * k and fill > 0.6):
                continue                               # too thick / big solid block = photo or graphic
            if fill > 0.85 and max(w, h) < 3.5 * min(w, h) and min(w, h) >= 10 * k:
                continue                               # compact solid box (legend swatch, logo)
        if text_flags[i] and thick < 7 * k:
            continue                                   # coloured printed text
        if attached is not None and attached[y:y + h, x:x + w][sub].any():
            # piece of a blob that contains a photo/graphic: accept only clean, elongated strokes
            if not (4 * k <= thick <= 30 * k and longest >= max(3 * thick, 80 * k) and area >= prof.min_area * k * k):
                continue
            if (near_body[y:y + h, x:x + w][sub] > 0).mean() > 0.25:          # hugging the photo edge = shard
                continue
        if area >= prof.min_area * k * k and (longest >= prof.min_dim * k or thick >= 6 * k):
            if prof.unmix and area > 0.01 * page_area and fill > 0.5:
                reg = bgr[y:y + h, x:x + w][sub]       # photos have varied colour
                if reg.std(axis=0).max() > 28:
                    continue
            keep[i] = True
        elif area >= 6 * k * k and not text_flags[i]:
            small_ok[i] = True

    kept = _lut_mask(labels, keep)
    if not kept.any():
        return kept

    result = kept.copy()
    if prof.link_gap > 0 and small_ok.any():           # re-attach broken stroke fragments
        g = int(round(prof.link_gap * k))
        near = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        touch = np.unique(labels[(near > 0) & (labels > 0)])
        sel = np.zeros(n, bool)
        sel[touch] = True
        sel &= small_ok
        result |= _lut_mask(labels, sel)
    if prof.hysteresis:                                # faint anti-aliased edges / halos
        r = max(2, int(round(4 * k)))
        halo = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        low_c = cv2.bitwise_and(low, halo)
        n2, lab2 = cv2.connectedComponents(low_c, connectivity=8)
        sel2 = np.zeros(n2, bool)
        sel2[np.unique(lab2[result > 0])] = True
        result |= _lut_mask(lab2, sel2)
    if near_body is not None:                          # never touch the surroundings of photos/graphics
        result = cv2.bitwise_and(result, cv2.bitwise_not(near_body))
    return result


def expand(mask: np.ndarray, settings: Settings) -> np.ndarray:
    """Grow the mask (adjustable) so coloured fringes are covered."""
    prof = profile_for(settings)
    r = int(round(min(settings.mask_expand, prof.max_expand) * settings.scale))
    if r <= 0:
        return mask
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
