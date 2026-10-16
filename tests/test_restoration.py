import cv2
import numpy as np

from config import Settings
from core import processing_pipeline as pp
from core.mask_editor import MaskEditor
from tests import synthetic as syn


def metrics(clean, dirty, gt, result):
    gtb = gt > 0
    far = cv2.dilate(gt, np.ones((31, 31), np.uint8)) == 0
    changed_far = int((np.abs(result.restored.astype(int) - dirty.astype(int)).max(axis=2)[far] > 0).sum())
    dark = (clean.max(axis=2) < 90) & far                       # black print away from the pen
    text_intact = float((np.abs(result.restored.astype(int) - clean.astype(int)).max(axis=2)[dark] < 40).mean())
    cov = float((result.final_mask[gtb] > 0).mean())
    return dict(residual=result.residual_ratio, coverage=cov, changed_far=changed_far, text_intact=text_intact)


def test_balanced_removes_pen_without_collateral(pen_page):
    clean, dirty, gt = pen_page
    r = pp.process_page(dirty, Settings())
    m = metrics(clean, dirty, gt, r)
    assert m["coverage"] > 0.9
    assert m["residual"] < 0.02
    assert m["changed_far"] == 0              # nothing changed away from the pen
    assert m["text_intact"] > 0.999


def test_no_coloured_pen_pixels_remain(pen_page):
    from core import color_detector
    _, dirty, gt = pen_page
    r = pp.process_page(dirty, Settings())
    high, _ = color_detector.detect_pen_pixels(r.restored, Settings())
    near_pen = cv2.dilate(gt, np.ones((9, 9), np.uint8)) > 0          # printed colour elsewhere is allowed
    assert int(((high > 0) & near_pen).sum()) < 0.002 * (gt > 0).sum()


def test_all_modes_and_methods_run(pen_page):
    _, dirty, gt = pen_page
    for mode in ("conservative", "balanced", "advanced"):
        for meth in ("auto", "telea", "ns", "background"):
            r = pp.process_page(dirty, Settings(mode=mode, restore_method=meth))
            assert r.restored.shape == dirty.shape


def test_pen_on_blank_paper_is_restored_to_paper():
    img = np.full((600, 800, 3), 245, np.uint8)
    cv2.line(img, (100, 300), (700, 320), (40, 40, 200), 11, cv2.LINE_AA)
    cv2.line(img, (100, 400), (700, 380), (200, 70, 30), 11, cv2.LINE_AA)
    r = pp.process_page(img, Settings())
    assert np.abs(r.restored.astype(int) - 245).max() <= 12


def test_advanced_unmix_preserves_text_under_translucent_ink(clean_img):
    dirty, gt = syn.add_pens(clean_img, "light")
    bal = pp.process_page(dirty, Settings(mode="balanced"))
    adv = pp.process_page(dirty, Settings(mode="advanced"))
    g = gt > 0
    err = lambda r: np.abs(r.restored.astype(int) - clean_img.astype(int)).max(axis=2)[g].mean()
    assert err(adv) < err(bal)
    assert "unmix" in adv.methods


def test_pen_over_grid_line_keeps_line():
    img = np.full((500, 700, 3), 255, np.uint8)
    cv2.line(img, (50, 250), (650, 250), (0, 0, 0), 2)                 # table rule
    cv2.line(img, (330, 100), (350, 400), (30, 30, 210), 12, cv2.LINE_AA)  # pen crosses it
    r = pp.process_page(img, Settings())
    assert r.restored[250, 340:360].max() < 120                         # line still dark where crossed


def test_deterministic(pen_page):
    _, dirty, _ = pen_page
    a = pp.process_page(dirty, Settings()).restored
    b = pp.process_page(dirty, Settings()).restored
    assert np.array_equal(a, b)


def test_mask_editor_add_restore_undo_redo():
    e = MaskEditor((100, 100))
    auto = np.zeros((100, 100), np.uint8)
    auto[40:60, 40:60] = 255
    e.begin_stroke(); e.paint((10, 10), (30, 10), 3, "add")
    assert e.final_mask(auto)[10, 20] == 255
    e.begin_stroke(); e.paint((50, 40), (50, 60), 5, "restore")
    assert e.final_mask(auto)[50, 50] == 0 and e.final_mask(auto)[45, 42] == 255
    assert e.undo() and e.final_mask(auto)[50, 50] == 255
    assert e.redo() and e.final_mask(auto)[50, 50] == 0
    e.reset()
    assert not e.has_edits() and e.undo()


def test_manual_edit_overrides_detection(pen_page):
    clean, dirty, gt = pen_page
    e = MaskEditor(dirty.shape[:2])
    e.begin_stroke()
    e.paint((0, 0), (dirty.shape[1] - 1, 0), 3, "restore")
    e.removed[:] = 255                                                     # protect everything
    r = pp.process_page(dirty, Settings(), e)
    assert int((r.final_mask > 0).sum()) == 0
    assert np.array_equal(r.restored, dirty)
