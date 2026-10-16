"""Manual correction layers + undo/redo (bit-packed snapshots to save RAM)."""
from __future__ import annotations

import cv2
import numpy as np


class MaskEditor:
    """final = (auto | added) & ~removed.   Brush 'add' / 'restore'."""

    MAX_UNDO = 40

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.added = np.zeros(shape, np.uint8)
        self.removed = np.zeros(shape, np.uint8)
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []

    # ---- snapshots -------------------------------------------------
    def _snap(self):
        return (np.packbits(self.added > 0), np.packbits(self.removed > 0))

    def _load(self, snap):
        n = self.shape[0] * self.shape[1]
        self.added = (np.unpackbits(snap[0])[:n].reshape(self.shape) * 255).astype(np.uint8)
        self.removed = (np.unpackbits(snap[1])[:n].reshape(self.shape) * 255).astype(np.uint8)

    def begin_stroke(self):
        self._undo.append(self._snap())
        if len(self._undo) > self.MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snap())
        self._load(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snap())
        self._load(self._redo.pop())
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ---- editing ---------------------------------------------------
    def paint(self, p0, p1, radius: int, mode: str):
        """mode: 'add' (force removal) or 'restore' (protect from removal)."""
        r = max(1, int(radius))
        a, b = (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))
        on, off = (self.added, self.removed) if mode == "add" else (self.removed, self.added)
        cv2.line(on, a, b, 255, 2 * r + 1, cv2.LINE_8)
        cv2.circle(on, b, r, 255, -1)
        cv2.line(off, a, b, 0, 2 * r + 1, cv2.LINE_8)
        cv2.circle(off, b, r, 0, -1)

    def reset(self):
        self.begin_stroke()
        self.added[:] = 0
        self.removed[:] = 0

    def has_edits(self) -> bool:
        return bool(self.added.any() or self.removed.any())

    def final_mask(self, auto: np.ndarray | None) -> np.ndarray:
        base = auto if auto is not None else np.zeros(self.shape, np.uint8)
        m = cv2.bitwise_or(base, self.added)
        return cv2.bitwise_and(m, cv2.bitwise_not(self.removed))

    def layers(self):
        return self.added.copy(), self.removed.copy()

    def set_layers(self, added, removed):
        if added.shape == self.shape and removed.shape == self.shape:
            self.added, self.removed = added.copy(), removed.copy()
