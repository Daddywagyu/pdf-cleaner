import cv2
import numpy as np

from config import Settings
from core import color_detector
from tests import synthetic as syn


def recall(mask, gt):
    return float(((mask > 0) & (gt > 0)).sum() / (gt > 0).sum())


def test_red_pen_on_white(clean_img):
    dirty, gt = syn.add_pens(clean_img, "red")
    assert recall(color_detector.detect_mask(dirty, Settings()), gt) > 0.9


def test_blue_pen_on_white(clean_img):
    dirty, gt = syn.add_pens(clean_img, "blue")
    assert recall(color_detector.detect_mask(dirty, Settings()), gt) > 0.9


def test_colour_switches(clean_img):
    dirty, gt = syn.add_pens(clean_img, "all")
    no_red = color_detector.detect_mask(dirty, Settings(detect_red=False))
    only_blue_gt = syn.add_pens(clean_img, "blue")[1]
    assert recall(no_red, only_blue_gt) > 0.9
    red_gt = syn.add_pens(clean_img, "red")[1]
    assert recall(no_red, red_gt) < 0.05


def test_clean_page_has_no_detection(clean_img):
    """Printed coloured text / swatches / a colourful photo must not be detected."""
    for mode in ("conservative", "balanced", "advanced"):
        m = color_detector.detect_mask(clean_img, Settings(mode=mode))
        assert int((m > 0).sum()) < 50, mode


def test_uneven_tinted_scan_no_false_positive(clean_img):
    h, w = clean_img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 0.9 + 0.1 * np.sin(xx / 300.0) * np.cos(yy / 400.0)
    tint = np.array([0.88, 0.95, 1.0], np.float32)            # yellowish paper (BGR)
    img = np.clip(clean_img.astype(np.float32) * shade[..., None] * tint, 0, 255).astype(np.uint8)
    # strip the photo+swatches so only paper/text remain
    img[560 * 1654 // 595:, :] = 255
    m = color_detector.detect_mask(img, Settings())
    assert int((m > 0).sum()) < 50


def test_sensitivity_controls_light_pen(clean_img):
    dirty, gt = syn.add_pens(clean_img, "light")
    low = color_detector.detect_mask(dirty, Settings(sens_red=0.0, sens_blue=0.0))
    high = color_detector.detect_mask(dirty, Settings(sens_red=0.9, sens_blue=0.9))
    assert recall(high, gt) > 0.9
    assert recall(low, gt) < recall(high, gt)


def test_mode_ordering(pen_page):
    _, dirty, _ = pen_page
    c = (color_detector.detect_mask(dirty, Settings(mode="conservative")) > 0).sum()
    b = (color_detector.detect_mask(dirty, Settings(mode="balanced")) > 0).sum()
    assert 0 < c <= b


def test_mask_expansion_grows(pen_page):
    _, dirty, _ = pen_page
    a = (color_detector.detect_mask(dirty, Settings(mask_expand=0)) > 0).sum()
    b = (color_detector.detect_mask(dirty, Settings(mask_expand=6)) > 0).sum()
    assert b > a


def test_pen_over_text_is_not_confused_with_text(pen_page):
    clean, dirty, gt = pen_page
    m = color_detector.detect_mask(dirty, Settings())
    far = cv2.dilate(gt, np.ones((31, 31), np.uint8)) == 0
    assert int(((m > 0) & far).sum()) == 0
