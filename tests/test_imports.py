from __future__ import annotations

import sys
from pathlib import Path


def test_can_import_package() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import text_classification  # noqa: F401
