"""Unit tests for SRtools.auto_initial_guess (fast: pure functions + synthetic)."""

import math

import numpy as np
import pandas as pd
import pytest

from SRtools import auto_initial_guess as aig
from SRtools.deathTimesDataSet import Dataset


# --- compute_timing: pure math --------------------------------------------

def test_compute_timing_basic():
    assert aig.compute_timing(2000, target_dt=0.25, nsteps=5000) == (5000, 2)
    assert aig.compute_timing(500, target_dt=0.25, nsteps=5000) == (5000, 1)
    # human-ish: t_end ~44k days, dt 0.25 -> multiplier ~36
    assert aig.compute_timing(44000, target_dt=0.25, nsteps=5000) == (5000, 36)


def test_compute_timing_rejects_nonpositive():
    with pytest.raises(ValueError):
        aig.compute_timing(0)
    with pytest.raises(ValueError):
        aig.compute_timing(100, target_dt=0)


# --- log-log fits + prediction --------------------------------------------

def test_build_loglog_fits_structure():
    fits = aig.build_loglog_fits()
    assert set(fits) == set(aig._RATIO_ROWS)
    for f in fits.values():
        assert math.isfinite(f["slope"]) and math.isfinite(f["intercept"])
        assert f["n"] >= 10  # a reasonable number of reference organisms


def test_predict_initial_theta_positive_and_scaled():
    theta = aig.predict_initial_theta(885.0, xc=20.0)
    assert len(theta) == 4
    assert all(np.isfinite(x) and x > 0 for x in theta)
    assert theta[3] == 20.0
    # eta/beta/epsilon scale with xc and xc^2
    theta1 = aig.predict_initial_theta(885.0, xc=1.0)
    assert theta[0] == pytest.approx(theta1[0] * 20.0, rel=1e-9)
    assert theta[2] == pytest.approx(theta1[2] * 400.0, rel=1e-9)


def test_predict_initial_theta_rejects_bad_ml():
    with pytest.raises(ValueError):
        aig.predict_initial_theta(0)
    with pytest.raises(ValueError):
        aig.predict_initial_theta(float("inf"))


# --- external-hazard detection --------------------------------------------

def _gompertz_like(rng, n=3000, loc=200, scale=15):
    """Concentrated deaths -> near-zero early hazard, then a sharp rise."""
    t = loc + rng.gumbel(0, scale, n)
    return np.clip(t, 0, None)


def test_detect_external_hazard_disabled():
    ds = Dataset(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert aig.detect_external_hazard(ds, enabled=False) == (False, None)


def test_detect_external_hazard_pure_aging_is_false():
    rng = np.random.default_rng(0)
    ds = Dataset(_gompertz_like(rng), np.ones(3000))
    has, val = aig.detect_external_hazard(ds)
    assert has is False and val is None


class _HazStub:
    """Minimal dataset stub exposing a controlled ``.hazard`` and median."""

    def __init__(self, t, hz, ml):
        self.hazard = (np.asarray(t, float), np.asarray(hz, float))
        self._ml = ml

    def getMedianLifetime(self):
        return self._ml


def test_detect_external_hazard_constant_floor_recovered():
    # Flat log-hazard floor (rate 0.02) for t<40, then an aging rise. Deterministic.
    t = np.linspace(0, 100, 201)
    floor_rate = 0.02
    hz = np.where(t < 40, floor_rate, floor_rate * np.exp(0.08 * (t - 40)))
    ds = _HazStub(t, hz, ml=70.0)
    has, val = aig.detect_external_hazard(ds)
    assert has is True
    # config value should be -log(rate) = -log(0.02) ~ 3.91
    assert val == pytest.approx(-math.log(floor_rate), abs=0.3)


def test_detect_external_hazard_monotonic_rise_is_false():
    # log-hazard rising from the very start (pure aging) -> no floor.
    t = np.linspace(1, 100, 201)
    hz = 1e-4 * np.exp(0.09 * t)
    ds = _HazStub(t, hz, ml=70.0)
    assert aig.detect_external_hazard(ds) == (False, None)


# --- evaluate_fit: sign/zero handling -------------------------------------

class _FakeSim:
    def __init__(self, median, steep):
        self._m, self._s = median, steep

    def getMedianLifetime(self):
        return self._m

    def getSteepness(self, method="IQR"):
        return self._s


def test_evaluate_fit_positive_devs():
    ds = _FakeSim(100.0, -2.0)        # data
    sim = _FakeSim(110.0, -2.2)       # 10% off both
    md, sd = aig.evaluate_fit(sim, ds)
    assert md == pytest.approx(0.10, abs=1e-9)
    assert sd == pytest.approx(0.10, abs=1e-9)   # abs() handles the negative sign


def test_evaluate_fit_flags_infinite_median():
    ds = _FakeSim(100.0, 1.0)
    sim = _FakeSim(float("inf"), 1.0)
    md, sd = aig.evaluate_fit(sim, ds)
    assert not math.isfinite(md)


# --- trend-start detection -------------------------------------------------

class _TrendStub:
    """Dataset stub exposing the pieces detect_trend_start needs: a NA fitter
    whose smoothed_hazard_ returns a controlled (t, hazard) curve, plus
    getSurvival / getMaxLifetime / getMedianLifetime."""

    class _NAF:
        def __init__(self, t, hz):
            self._t, self._hz = t, hz

        def smoothed_hazard_(self, bandwidth=None):
            return pd.DataFrame({"hz": self._hz}, index=self._t)

    def __init__(self, t, hz, surv_t, surv_s, maxlt, median):
        self.naf = self._NAF(np.asarray(t, float), np.asarray(hz, float))
        self._st = np.asarray(surv_t, float); self._ss = np.asarray(surv_s, float)
        self._maxlt = maxlt; self._median = median

    def getMaxLifetime(self):
        return self._maxlt

    def getMedianLifetime(self):
        return self._median

    def getSurvival(self, interpolate_time=None, time_range=None):
        return self._st, self._ss


def _linspace_survival(maxlt, n=200):
    t = np.linspace(0, maxlt, n)
    s = np.clip(1 - t / (1.05 * maxlt), 1e-3, 1.0)
    return t, s


def test_detect_trend_start_bathtub_finds_dip():
    # Bathtub log-hazard: high juvenile, dip ~300, then aging rise.
    maxlt = 1000.0
    t = np.linspace(1, maxlt, 200)
    logh = np.where(t < 300, -6 - 0.004 * (t - 300), -6 + 0.004 * (t - 300))  # V at t=300
    st, ss = _linspace_survival(maxlt)
    ds = _TrendStub(t, np.exp(logh), st, ss, maxlt, median=520.0)
    t_min = aig.detect_trend_start(ds)
    assert t_min is not None
    assert 200 <= t_min <= 400          # near the dip at 300
    assert t_min <= 0.5 * 520.0         # capped at half-median


def test_detect_trend_start_monotonic_returns_floor():
    # Monotonic-rising log-hazard from the start -> no dip -> floor.
    maxlt = 1000.0
    t = np.linspace(1, maxlt, 200)
    ds = _TrendStub(t, np.exp(-9 + 0.005 * t), *_linspace_survival(maxlt), maxlt, median=500.0)
    t_min = aig.detect_trend_start(ds, floor_frac=0.08)
    assert t_min is not None
    assert abs(t_min - 0.08 * maxlt) <= 5    # ~floor (80)


def test_detect_trend_start_too_few_points():
    maxlt = 1000.0
    t = np.linspace(1, maxlt, 5)
    ds = _TrendStub(t, np.exp(-6 + 0.001 * t), *_linspace_survival(maxlt), maxlt, median=500.0)
    assert aig.detect_trend_start(ds) is None


def test_detect_trend_start_capped_at_half_median():
    # Dip placed late (~700) but median small (400) -> capped at 200.
    maxlt = 1000.0
    t = np.linspace(1, maxlt, 200)
    logh = np.where(t < 700, -6 - 0.003 * (t - 700), -6 + 0.003 * (t - 700))
    ds = _TrendStub(t, np.exp(logh), *_linspace_survival(maxlt), maxlt, median=400.0)
    t_min = aig.detect_trend_start(ds)
    assert t_min is not None and t_min <= 0.5 * 400.0 + 1


# --- conditional median / steepness ---------------------------------------

class _SurvStub:
    def __init__(self, t, s):
        self._t, self._s = np.asarray(t, float), np.asarray(s, float)

    def getSurvival(self, interpolate_time=None, time_range=None):
        if time_range is None:
            return self._t, self._s
        a, b = time_range
        m = (self._t >= a) & (self._t <= b)
        t, s = self._t[m], self._s[m]
        s = s / s[0] if s.size and s[0] > 0 else s  # renormalise to 1 at window start
        return t, s


def test_conditional_median_steepness_inverse_cv():
    # Steepness is now the inverse CV (mean/std) of the death times inside the window;
    # the median still comes from the conditional survival curve.
    rng = np.random.default_rng(0)
    dt = rng.uniform(200, 1000, size=3000)
    ds = Dataset(dt, np.ones_like(dt))
    tr = [200, 1000]
    med, steep = aig.conditional_median_steepness(ds, tr)
    died = dt[(dt >= tr[0]) & (dt <= tr[1])]
    assert steep == pytest.approx(float(np.mean(died) / np.std(died)), rel=1e-6)
    assert med == pytest.approx(float(np.median(dt)), rel=0.1)


def test_conditional_median_steepness_too_few_points():
    med, steep = aig.conditional_median_steepness(_SurvStub([0, 1], [1, 0.9]), [0, 1])
    assert math.isnan(med) and math.isnan(steep)


# --- getCV / getSteepness: inverse CV + time_range (backward compatible) ----

def test_getcv_and_inverse_cv_windowed():
    dt = np.array([10., 20., 30., 40., 50., 500., 600.])   # last two outside the window
    ds = Dataset(dt, np.ones_like(dt))
    win = [0, 100]
    inwin = dt[(dt >= win[0]) & (dt <= win[1])]
    assert ds.getCV(time_range=win) == pytest.approx(float(np.std(inwin) / np.mean(inwin)), rel=1e-9)
    # inverseCV == 1/CV, and it is higher over the tighter window than full range
    assert ds.getSteepness("inverseCV", time_range=win) == pytest.approx(
        float(np.mean(inwin) / np.std(inwin)), rel=1e-9)
    assert ds.getSteepness("inverseCV", time_range=win) > ds.getSteepness("inverseCV")


def test_getcv_windowed_too_few_deaths_is_nan():
    ds = Dataset(np.array([10., 500., 600.]), np.ones(3))
    assert math.isnan(ds.getCV(time_range=[0, 100]))          # only one death in window
    assert math.isnan(ds.getSteepness("inverseCV", time_range=[0, 100]))


def test_getsteepness_iqr_default_unchanged():
    # Default method stays 'IQR' and ignores time_range when not passed (backward compat).
    dt = np.linspace(10, 1000, 400)
    ds = Dataset(dt, np.ones_like(dt))
    assert np.isfinite(ds.getSteepness())        # default IQR still works, no kwargs
    assert ds.getSteepness("IQR") == ds.getSteepness()
