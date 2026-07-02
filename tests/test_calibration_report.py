"""Tests for the single-file survival-overview PDF."""

import numpy as np

from SRtools import auto_initial_guess as aig
from SRtools.calibration_report import plot_survival_overview


def _result(species, status="ok"):
    t = np.linspace(0, 100, 50)
    s = np.clip(1 - t / 110.0, 0, 1)
    return aig.FitResult(
        species=species, theta=[1e-3, 0.5, 1.0, 17.0], status=status,
        median_dev=0.05 if status == "ok" else 0.4,
        steep_dev=0.07 if status == "ok" else 0.3,
        data_survival=(t, s), guess_survival=(t, np.clip(s + 0.03, 0, 1)),
    )


def test_overview_one_file_expected_pages(tmp_path):
    results = [_result(f"sp_{i}", "ok" if i % 2 else "best_effort") for i in range(7)]
    out = tmp_path / "survival_overview.pdf"
    pages = plot_survival_overview(results, str(out), rows=2, cols=3)  # 6 per page
    assert pages == 2          # 7 results -> 2 pages
    assert out.exists() and out.stat().st_size > 0


def test_overview_handles_missing_curves(tmp_path):
    r = aig.FitResult(species="nodata", status="insufficient_data",
                      data_survival=None, guess_survival=None)
    out = tmp_path / "x.pdf"
    pages = plot_survival_overview([r], str(out), rows=1, cols=1)
    assert pages == 1 and out.exists()
