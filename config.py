"""Central configuration / settings (no global mutable state)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

APP_NAME = "Local PDF Pen Remover"
APP_VERSION = "1.0.0"

MODES = ("conservative", "balanced", "advanced")
RESTORE_METHODS = ("auto", "telea", "ns", "background")
BASE_DPI = 200.0          # all pixel-based thresholds are defined at 200 DPI
MAX_PIXELS = 150_000_000  # memory guard for one rendered page


@dataclass
class Settings:
    mode: str = "balanced"            # conservative | balanced | advanced
    detect_red: bool = True
    detect_blue: bool = True
    sens_red: float = 0.5             # 0..1 (higher = more sensitive)
    sens_blue: float = 0.5
    mask_expand: int = 2              # px @200DPI
    inpaint_radius: int = 3           # px @200DPI
    restore_method: str = "auto"      # auto | telea | ns | background
    process_dpi: int = 200            # 200 / 300 / 400
    keep_text_layer: bool = True      # re-add invisible text layer on export
    image_format: str = "png"         # png (lossless) | jpeg
    jpeg_quality: int = 95

    @property
    def scale(self) -> float:
        return self.process_dpi / BASE_DPI

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModeProfile:
    thr_scale: float      # multiplier on colour-chroma threshold (>1 = stricter)
    min_area: int         # px^2 @200DPI
    min_dim: int          # px @200DPI (longest bbox side)
    hysteresis: bool      # grow into faint stroke edges
    link_gap: int         # px @200DPI to re-attach small fragments of kept strokes
    max_expand: int       # cap for mask expansion
    reject_graphics: bool
    unmix: bool           # allow transmittance un-mixing for translucent ink


PROFILES = {
    "conservative": ModeProfile(1.25, 250, 50, False, 0, 1, True, False),
    "balanced": ModeProfile(1.0, 60, 24, True, 8, 8, True, False),
    "advanced": ModeProfile(0.9, 40, 18, True, 12, 10, True, True),
}


def profile_for(settings: Settings) -> ModeProfile:
    return PROFILES.get(settings.mode, PROFILES["balanced"])
