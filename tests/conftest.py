import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from tests import synthetic as syn


@pytest.fixture(scope="session")
def clean_pdf(tmp_path_factory):
    return syn.build_clean_pdf(tmp_path_factory.mktemp("pdf") / "clean.pdf", pages=3)


@pytest.fixture(scope="session")
def clean_img(clean_pdf):
    return syn.render(clean_pdf, 0, 200)


@pytest.fixture(scope="session")
def pen_page(clean_img):
    dirty, gt = syn.add_pens(clean_img, "all")
    return clean_img, dirty, gt
