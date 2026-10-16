"""Glue: detection -> manual edits -> restoration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import Settings
from core import color_detector, restoration_engine
from core.mask_editor import MaskEditor


@dataclass
class PageResult:
    auto_mask: np.ndarray
    final_mask: np.ndarray
    restored: np.ndarray | None
    residual_ratio: float = 0.0
    methods: dict = field(default_factory=dict)


def detect_only(bgr: np.ndarray, settings: Settings) -> np.ndarray:
    return color_detector.detect_mask(bgr, settings)


def process_page(bgr: np.ndarray, settings: Settings, editor: MaskEditor | None = None,
                 auto_mask: np.ndarray | None = None,
                 cancel: Callable[[], bool] | None = None) -> PageResult:
    auto = auto_mask if auto_mask is not None else detect_only(bgr, settings)
    final = editor.final_mask(auto) if editor is not None and editor.shape == auto.shape else auto
    res = restoration_engine.restore(bgr, final, settings, cancel)
    return PageResult(auto, final, res.image, res.residual_ratio, res.methods)
