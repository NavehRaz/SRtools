"""Automated initial-parameter guessing for the SR model.

Given a mortality :class:`~SRtools.deathTimesDataSet.Dataset`, this module produces a
reasonable, *validated* initial ``theta = [eta, beta, epsilon, xc]`` for the
Saturating-Removal model and the timing parameters (``nsteps``,
``time_step_multiplier``) needed to run it.

The flow (see :func:`auto_fit`):

1. Predict ``theta`` from the data's median lifetime using log-log relationships
   fitted over well-characterised reference organisms (:func:`build_loglog_fits`,
   :func:`predict_initial_theta`).
2. Optionally detect a constant-then-rising external-hazard floor
   (:func:`detect_external_hazard`).
3. Refine ``theta`` cheaply with :class:`SRtools.initialParamsFinder.Guess`.
4. Re-simulate at full resolution and accept only if the simulated median lifespan
   *and* steepness are within ``dev_threshold`` of the data (:func:`evaluate_fit`).

This is the programmatic core behind ``app/run_auto_calibration.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference log-log fits: median lifetime -> parameter ratios
# ---------------------------------------------------------------------------

#: Organisms excluded from the reference fit (outliers / different time units).
DEFAULT_EXCLUDE = ("cats_BPH", "ecoli")

#: Row of ``summery_mode.csv`` holding each organism's model median lifetime.
_ML_ROW = "best fit no ext hazard_MedianLifetime"

#: Ratio rows (in ``summery_mode_overall.csv``) predicted from median lifetime.
_RATIO_ROWS = ("eta/xc", "beta/xc", "epsilon/xc^2")


def _preset_dir():
    """Return the packaged ``Preset_values`` directory."""
    return os.path.join(os.path.dirname(__file__), "Preset_values")


def build_loglog_fits(exclude=DEFAULT_EXCLUDE, preset_dir=None):
    """Build log-log linear fits of parameter ratios vs. median lifetime.

    Reconstructs the relationships used by the Species360 calibration notebook:
    for each ratio in :data:`_RATIO_ROWS`, fit ``log(ratio) = slope*log(ML) +
    intercept`` across the reference organisms (those present in both preset
    tables with finite values, minus ``exclude``).

    Returns
    -------
    dict
        ``{ratio_name: {"slope": float, "intercept": float, "n": int}}``.
    """
    pv = preset_dir or _preset_dir()
    raw_mode = pd.read_csv(os.path.join(pv, "summery_mode.csv"), index_col=0)
    value_cols = [c for c in raw_mode.columns if "95% CI" not in c]
    df_mode = raw_mode[value_cols].apply(pd.to_numeric, errors="coerce")
    df_overall = pd.read_csv(
        os.path.join(pv, "summery_mode_overall.csv"), index_col=0
    ).apply(pd.to_numeric, errors="coerce")

    exclude = set(exclude)
    organisms = sorted((set(df_mode.columns) | set(df_overall.columns)) - exclude)

    rows = {}
    for org in organisms:
        if _ML_ROW not in df_mode.index or org not in df_mode.columns:
            continue
        ml = df_mode.loc[_ML_ROW, org]
        vals = {
            r: (df_overall.loc[r, org] if (r in df_overall.index and org in df_overall.columns) else np.nan)
            for r in _RATIO_ROWS
        }
        if pd.notna(ml) and ml > 0 and all(pd.notna(v) and v > 0 for v in vals.values()):
            rows[org] = {"ML": float(ml), **{r: float(vals[r]) for r in _RATIO_ROWS}}

    param_df = pd.DataFrame(rows).T
    if param_df.empty:
        raise ValueError("No reference organisms with complete preset data found.")

    log_ml = np.log(param_df["ML"].to_numpy(dtype=float))
    fits = {}
    for r in _RATIO_ROWS:
        slope, intercept = np.polyfit(log_ml, np.log(param_df[r].to_numpy(dtype=float)), 1)
        fits[r] = {"slope": float(slope), "intercept": float(intercept), "n": int(len(param_df))}
    return fits


_FITS_CACHE = None


def _get_fits(fits=None):
    global _FITS_CACHE
    if fits is not None:
        return fits
    if _FITS_CACHE is None:
        _FITS_CACHE = build_loglog_fits()
    return _FITS_CACHE


def predict_initial_theta(median_lifetime, xc=20.0, fits=None):
    """Predict ``[eta, beta, epsilon, xc]`` from a median lifetime.

    Uses the log-log fits to predict ``eta/xc``, ``beta/xc`` and ``epsilon/xc^2``
    at the given ``median_lifetime`` (same time unit as the reference organisms,
    i.e. days), then scales them by the chosen ``xc``. ``xc`` is only a starting
    scale; :func:`auto_fit` refines all four parameters afterwards.

    Raises
    ------
    ValueError
        If ``median_lifetime`` is not a finite positive number.
    """
    if not np.isfinite(median_lifetime) or median_lifetime <= 0:
        raise ValueError(f"median_lifetime must be finite and > 0, got {median_lifetime!r}")
    fits = _get_fits(fits)
    log_ml = math.log(median_lifetime)
    ratios = {r: math.exp(f["slope"] * log_ml + f["intercept"]) for r, f in fits.items()}
    eta = ratios["eta/xc"] * xc
    beta = ratios["beta/xc"] * xc
    epsilon = ratios["epsilon/xc^2"] * xc ** 2
    return [eta, beta, epsilon, float(xc)]


# ---------------------------------------------------------------------------
# Timing: keep the effective dt at a constant order of magnitude
# ---------------------------------------------------------------------------

#: Target effective time step (in data units, i.e. days). Calibrated so the
#: effective dt = t_end / (nsteps * multiplier) stays the same order of
#: magnitude across organisms (mice -> multiplier ~1-2, humans -> ~tens),
#: matching ``initialParamsFinder.defaultGuess`` (Mice=3, Humans=30).
DEFAULT_TARGET_DT = 0.25


def compute_timing(t_end, target_dt=DEFAULT_TARGET_DT, nsteps=5000):
    """Return ``(nsteps, time_step_multiplier)`` for a constant effective dt.

    ``time_step_multiplier = max(1, ceil(t_end / (target_dt * nsteps)))`` so that
    the effective step ``t_end / (nsteps * multiplier)`` is approximately
    ``target_dt`` regardless of organism lifespan, while ``nsteps`` stays fixed
    (the user's usual >=5000). Do NOT route this through ``getSr(step_size=...)``,
    which would override ``nsteps``.
    """
    if t_end <= 0 or target_dt <= 0 or nsteps <= 0:
        raise ValueError("t_end, target_dt and nsteps must all be positive")
    multiplier = max(1, math.ceil(t_end / (target_dt * nsteps)))
    return int(nsteps), int(multiplier)


# ---------------------------------------------------------------------------
# External-hazard detection: constant log-hazard plateau, then a rise
# ---------------------------------------------------------------------------


def detect_external_hazard(
    ds,
    enabled=True,
    min_plateau_frac=0.10,
    flat_slope_ratio=0.5,
):
    """Detect a constant external-hazard floor and return its config value.

    A genuine age-independent (external) hazard makes the *log* hazard **flat early
    on**, before age-related mortality takes over. Pure aging (Gompertz/Weibull) has
    a log-hazard that rises monotonically from the very start — no flat early stretch.
    So the discriminator is the *early* log-hazard slope vs the later (aging) rise:

    * ``s_early`` = slope of ``log(hazard)`` over the first ``min_plateau_frac`` of
      the median lifespan;
    * ``s_late``  = slope from there up to the median.

    A floor is flagged only when the curve rises (``s_late > 0``) and the early region
    is much flatter than that rise (``s_early <= flat_slope_ratio * s_late``). The
    stored value is ``-h`` with ``h`` = early-window mean ``log(hazard)`` (the
    simulator uses ``external rate = exp(-external_hazard)``).

    .. note::
       This is a heuristic on the *smoothed* empirical hazard. It reliably rejects
       pure-aging curves (its key safety property — it won't add external hazard to a
       normal survival curve), but borderline low floors are genuinely ambiguous.
       Detection is therefore **opt-in** in the CLI; the candidate value is reported
       in the QC sheet regardless, for you to judge with domain knowledge.

    Parameters
    ----------
    enabled : bool
        If ``False``, returns ``(False, None)`` immediately.
    flat_slope_ratio : float
        How flat the early region must be relative to the aging rise. Lower =
        stricter (closer to a perfect plateau).

    Returns
    -------
    (bool, float | None)
        ``(has_external_hazard, exth_config_value)``.
    """
    if not enabled:
        return False, None

    t, hz = ds.hazard
    t = np.asarray(t, dtype=float)
    hz = np.asarray(hz, dtype=float)

    ml = ds.getMedianLifetime()
    if not np.isfinite(ml) or ml <= 0:
        return False, None

    # Work on log-hazard where the (smoothed) hazard is strictly positive.
    valid = np.isfinite(hz) & (hz > 0) & np.isfinite(t)
    t, logh = t[valid], np.log(hz[valid])
    if t.size < 8:
        return False, None

    early_end = t[0] + min_plateau_frac * ml
    early = t <= early_end
    late = (t > early_end) & (t <= ml)
    if early.sum() < 4 or late.sum() < 4:
        return False, None

    s_early = np.polyfit(t[early] - t[early][0], logh[early], 1)[0]
    s_late = np.polyfit(t[late] - t[late][0], logh[late], 1)[0]
    if s_late <= 0:                            # hazard doesn't rise -> no aging signal
        return False, None
    if s_early > flat_slope_ratio * s_late:    # early region not flat enough
        return False, None

    return True, -float(np.mean(logh[early]))


# ---------------------------------------------------------------------------
# Aging-trend-start detection -> fit time_range [t_min, t_end]
# ---------------------------------------------------------------------------


def detect_trend_start(ds, bw_frac=0.06, surv_cut=0.10, floor_frac=0.08):
    """Find where the sustained aging rise begins (the fit's ``t_min``).

    Encodes the manual heuristic of reading a *smoothed* log-hazard plot and
    picking the change-point after which the hazard rises monotonically (the
    bottom of the dip that follows early/juvenile mortality):

    1. Smooth the hazard with a **lifespan-scaled** bandwidth
       (``bw_frac * maxLifetime``) via the dataset's Nelson-Aalen fitter — NOT
       the cached ``ds.hazard`` (default ``bandwidth=3`` is essentially noise on
       multi-thousand-unit lifespans).
    2. Restrict to the "bulk" (ages up to where KM survival first hits
       ``surv_cut``) to avoid the noisy sparse tail.
    3. ``t_peak`` = age of the maximum smoothed log-hazard in the bulk;
       ``t_min`` = age of the **minimum** smoothed log-hazard over ``[0, t_peak]``
       (the dip preceding the aging rise).
    4. Floor at ``floor_frac * maxLifetime`` (monotonic-rising hazards have no
       dip, so this just trims the youngest sliver, like a "min age") and cap at
       ``0.5 * median`` (never trim more than half — keeps enough deaths in the
       window for a stable fit).

    Returns
    -------
    int | None
        The detected ``t_min`` (rounded), or ``None`` if it cannot be determined
        (too few hazard points / no usable survival cut).
    """
    maxlt = ds.getMaxLifetime()
    if not np.isfinite(maxlt) or maxlt <= 0:
        return None

    bw = bw_frac * maxlt
    try:
        sh = ds.naf.smoothed_hazard_(bandwidth=bw)
    except Exception:
        return None
    t = np.asarray(sh.index, dtype=float)
    hz = np.asarray(sh.iloc[:, 0], dtype=float)
    valid = np.isfinite(hz) & (hz > 0) & np.isfinite(t)
    t, logh = t[valid], np.log(hz[valid])
    if t.size < 8:
        return None

    # Bulk cutoff: age where KM survival first reaches surv_cut (fallbacks below).
    ts, ss = ds.getSurvival()
    ts = np.asarray(ts, dtype=float); ss = np.asarray(ss, dtype=float)
    if np.min(ss) <= surv_cut:
        age_cut = float(ts[np.argmin(np.abs(ss - surv_cut))])
    else:
        med = ds.getMedianLifetime()
        age_cut = float(med) if np.isfinite(med) else float(maxlt)

    bulk = t <= age_cut
    if bulk.sum() < 4:
        return None
    tb, lb = t[bulk], logh[bulk]
    t_peak = tb[np.argmax(lb)]
    pre = tb <= t_peak
    if pre.sum() < 1:
        return None
    t_min = float(tb[pre][np.argmin(lb[pre])])

    t_min = max(t_min, floor_frac * maxlt)
    med = ds.getMedianLifetime()
    if np.isfinite(med) and med > 0:
        t_min = min(t_min, 0.5 * med)
    if t_min <= 0:
        return None
    return int(round(t_min))


# ---------------------------------------------------------------------------
# Fit quality: deviation of simulated median lifespan & steepness from data
# ---------------------------------------------------------------------------


def evaluate_fit(sim, ds):
    """Return ``(median_pct_dev, steepness_pct_dev)`` of ``sim`` vs. ``ds``.

    Both are absolute relative deviations (sign-safe; ``getSteepness`` can be
    either sign). A non-finite or undefined metric yields ``inf`` for that
    deviation so callers can flag it.
    """
    def _rel(sim_val, data_val):
        if not np.isfinite(sim_val) or not np.isfinite(data_val) or data_val == 0:
            return float("inf")
        return abs((sim_val - data_val) / data_val)

    median_dev = _rel(sim.getMedianLifetime(), ds.getMedianLifetime())
    try:
        steep_dev = _rel(sim.getSteepness("IQR"), ds.getSteepness("IQR"))
    except (ZeroDivisionError, ValueError, FloatingPointError):
        steep_dev = float("inf")
    return median_dev, steep_dev


def conditional_median_steepness(obj, time_range):
    """Median & steepness of ``obj`` over a window, from the conditional survival.

    ``getMedianLifetime``/``getSteepness`` ignore ``time_range``, so compute both
    from ``obj.getSurvival(time_range=...)`` — which returns the survival curve
    sliced to the window and renormalised to 1 at ``time_range[0]`` (the same
    conditional renormalisation for a data Dataset and a simulation object).
    Steepness uses the same ``-median/(q3-q1)`` convention as
    ``Dataset.getSteepness('IQR')`` so values stay comparable.

    Returns ``(median, steepness)``; either is ``nan`` if undefined (too few
    points, survival never crosses 0.5, or zero IQR).
    """
    try:
        t, s = obj.getSurvival(time_range=list(time_range))  # keyword: sigs differ across types
    except Exception:
        return float("nan"), float("nan")
    t = np.asarray(t, dtype=float); s = np.asarray(s, dtype=float)
    if t.size < 4:
        return float("nan"), float("nan")
    median = float("nan") if np.min(s) > 0.5 else float(t[np.argmin(np.abs(s - 0.5))])
    q1 = t[np.argmin(np.abs(s - 0.25))]
    q3 = t[np.argmin(np.abs(s - 0.75))]
    if q3 == q1 or not np.isfinite(median):
        return median, float("nan")
    return median, -median / (q3 - q1)


def evaluate_fit_conditional(sim, ds, time_range):
    """Like :func:`evaluate_fit` but on the CONDITIONAL median & steepness over
    ``time_range`` (used when a fit window is active)."""
    def _rel(sim_val, data_val):
        if not np.isfinite(sim_val) or not np.isfinite(data_val) or data_val == 0:
            return float("inf")
        return abs((sim_val - data_val) / data_val)

    sim_med, sim_steep = conditional_median_steepness(sim, time_range)
    data_med, data_steep = conditional_median_steepness(ds, time_range)
    return _rel(sim_med, data_med), _rel(sim_steep, data_steep)


# ---------------------------------------------------------------------------
# auto_fit: predict -> refine (cheap) -> validate (full) -> accept/flag
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Outcome of :func:`auto_fit` for one dataset."""

    species: str = ""
    theta: list = None                       # [eta, beta, epsilon, xc]
    has_ext: bool = False                     # external hazard applied?
    exth: float = None                        # applied external-hazard value (or None)
    ext_candidate: float = None               # detected candidate value (always computed)
    ndims: int = 4
    t_end: float = 0.0
    nsteps: int = 5000
    time_step_multiplier: int = 1
    median_dev: float = float("inf")
    steep_dev: float = float("inf")
    status: str = "ok"                        # ok | best_effort | insufficient_data | steepness_undefined
    n_restarts: int = 0
    seed: int = 0
    data_ml: float = float("nan")
    data_steepness: float = float("nan")
    data_survival: tuple = None               # (t, s) for the overview PDF
    guess_survival: tuple = None              # (t, s) for the overview PDF
    # Trend-start / fit-window fields (when time_range_mode is on)
    t_min: float = float("nan")               # detected aging-trend start
    time_range: list = None                   # [t_min, t_end] written to the config
    cond_data_ml: float = float("nan")        # data median over the window (conditional)
    cond_data_steepness: float = float("nan") # data steepness over the window (conditional)
    data_survival_window: tuple = None        # conditional (renormalised) data survival
    guess_survival_window: tuple = None       # conditional (renormalised) guess survival

    @property
    def passed(self):
        return self.median_dev <= 0.15 and self.steep_dev <= 0.15


def auto_fit(
    ds,
    species="",
    *,
    target_dt=DEFAULT_TARGET_DT,
    xc=20.0,
    method="scipy",
    search_npeople=5000,
    search_nsteps=2500,
    search_max_n=3000,
    valid_npeople=20000,
    valid_nsteps=5000,
    normal_iters=120,
    retry_iters=200,
    max_restarts=2,
    dev_threshold=0.15,
    detect_ext=False,
    ext_kwargs=None,
    time_range_mode=True,
    min_events=30,
    seed=0,
    parallel=True,
    fits=None,
):
    """Produce a validated initial ``theta`` (+ timing) for one dataset.

    Predicts ``theta`` from the data median, refines it cheaply
    (``search_*`` resolution) via :class:`SRtools.initialParamsFinder.Guess`, then
    re-simulates at full resolution (``valid_*``) and accepts only if both the
    median-lifespan and steepness deviations are within ``dev_threshold``. On
    failure it escalates (up to ``max_restarts`` perturbed restarts with the
    larger ``retry_iters`` budget) and returns the best attempt, flagged.

    External hazard: the candidate floor is always computed and stored in
    ``FitResult.ext_candidate``; it is only *applied* (``ndims=5``) when
    ``detect_ext=True`` and a candidate is found.

    Time range (``time_range_mode=True``): detect the aging-trend start
    (:func:`detect_trend_start`) and fit only the window ``[t_min, t_end]`` — the
    fit's ``baysianDistance`` and the 15% gate then use the **conditional** median &
    steepness over that window, and ``[t_min, t_end]`` is written to the config.
    Excluding early/juvenile mortality usually improves the aging-shape fit.

    Parameters
    ----------
    method : {"scipy", "random"}
        Refinement engine: scipy Nelder-Mead (default) or the random-search
        :meth:`make_guess` schedule.
    """
    from .initialParamsFinder import Guess
    from . import SR_hetro as srh

    res = FitResult(species=species, seed=seed)

    # ---- preconditions -----------------------------------------------------
    ml = ds.getML()
    res.data_ml = float(ml) if np.isfinite(ml) else float("nan")
    try:
        res.data_steepness = float(ds.getSteepness("IQR"))
    except Exception:
        res.data_steepness = float("nan")
    n_events = int(np.sum(np.asarray(ds.events) == 1))
    try:
        res.data_survival = ds.getSurvival()
    except Exception:
        res.data_survival = None
    if not np.isfinite(ml) or ml <= 0 or n_events < min_events:
        res.status = "insufficient_data"
        return res

    # ---- timing ------------------------------------------------------------
    t_end = float(round(1.1 * ds.getMaxLifetime()))
    res.t_end = t_end
    nsteps_cfg, mult = compute_timing(t_end, target_dt=target_dt, nsteps=valid_nsteps)
    res.nsteps, res.time_step_multiplier = nsteps_cfg, mult

    # ---- aging-trend start -> fit window [t_min, t_end] --------------------
    tr = None
    if time_range_mode:
        t_min = detect_trend_start(ds)
        if t_min is not None and 0 < t_min < t_end:
            tr = [int(t_min), int(t_end)]
            res.t_min = float(t_min)
            res.time_range = tr
            res.cond_data_ml, res.cond_data_steepness = conditional_median_steepness(ds, tr)
            try:
                res.data_survival_window = ds.getSurvival(time_range=tr)
            except Exception:
                res.data_survival_window = None

    # ---- external hazard (always detect; apply only if requested) ----------
    cand_has, cand_val = detect_external_hazard(ds, enabled=True, **(ext_kwargs or {}))
    res.ext_candidate = cand_val if cand_has else None
    apply_ext = bool(detect_ext and cand_has)
    res.has_ext = apply_ext
    res.exth = cand_val if apply_ext else None
    res.ndims = 5 if apply_ext else 4
    ext_arg = res.exth if apply_ext else np.inf

    # ---- initial theta -----------------------------------------------------
    theta0 = predict_initial_theta(ml, xc=xc, fits=fits)

    # Subsample the data for the (cheap) search objective only. baysianDistance's
    # KDE cost scales with the number of data death times, so large datasets make
    # the optimizer very slow; the survival curve is well-approximated by a few
    # thousand points. Validation and the 15% gate always use the FULL dataset.
    search_ds = ds
    n_dt = len(ds.death_times)
    if search_max_n and n_dt > search_max_n:
        from .deathTimesDataSet import Dataset
        idx = np.sort(np.random.default_rng(seed).choice(n_dt, size=search_max_n, replace=False))
        props = ({p: ds.properties[p][idx] for p in ds.properties} if ds.properties else None)
        search_ds = Dataset(ds.death_times[idx], ds.events[idx], properties=props, bandwidth=ds.bandwidth)

    def _eval(sim):
        return evaluate_fit_conditional(sim, ds, tr) if tr is not None else evaluate_fit(sim, ds)

    def _validate(theta):
        sim = srh.getSrHetro(
            theta, n=valid_npeople, nsteps=valid_nsteps, t_end=t_end,
            external_hazard=ext_arg, time_step_multiplier=mult, hetro=False, parallel=parallel,
        )
        md, sd = _eval(sim)
        return sim, md, sd

    def _early_stop(sim):
        md, sd = _eval(sim)
        return md <= dev_threshold and sd <= dev_threshold

    def _calibrate_eta(g):
        """Robustly scale eta so the guess median approaches the data median.

        Higher eta -> faster damage -> shorter median. Handles the "no deaths yet"
        case (guess median = inf for long-lived species) by bumping eta up with a
        bounded factor instead of the raw (possibly inf) ratio, which would set
        eta = inf and produce a degenerate simulation. When a fit window is active,
        target the CONDITIONAL median over the window (warm-starting on the full
        median pushes the windowed optimizer into a bad basin).
        """
        if tr is not None:
            target = conditional_median_steepness(g.ds, tr)[0]
            if not np.isfinite(target) or target <= 0:
                target = g.ds.getMedianLifetime()
            measure = lambda: conditional_median_steepness(g.guess, tr)[0]
        else:
            target = g.ds.getMedianLifetime()
            measure = lambda: g.guess.getMedianLifetime()
        if not np.isfinite(target) or target <= 0:
            return
        for _ in range(8):
            try:
                gm = measure()
            except Exception:
                break
            if not np.isfinite(gm) or gm <= 0:      # too few deaths -> increase eta
                factor = 4.0
            else:
                ratio = gm / target
                if 0.9 <= ratio <= 1.1:
                    break
                factor = min(max(ratio, 0.1), 10.0)  # bounded step toward target
            g.theta[0] *= factor
            try:
                g.guess = g._sim(g.theta, g.ds.t_end)
            except Exception:
                break

    def _search(theta_start, iters, sd_seed):
        g = Guess(
            list(theta_start), search_ds, t_end=t_end, nsteps=search_nsteps, npeople=search_npeople,
            time_step_multiplier=mult, external_hazard=ext_arg, time_range=tr, parallel=parallel,
        )
        _calibrate_eta(g)  # fast, robust median match before refinement
        if method == "scipy":
            g.optimize(maxiter=iters, seed=sd_seed)
        elif method == "random":
            g.make_guess(niter=iters, step_size=5, method="all_at_once",
                         plot=False, print_thetas=False, seed=sd_seed, early_stop=_early_stop)
            g.make_guess(niter=max(5, iters // 4), step_size=2, method="one_at_a_time",
                         plot=False, print_thetas=False, early_stop=_early_stop)
        else:
            raise ValueError("method must be 'scipy' or 'random'")
        return g.theta

    # ---- attempt 0 + escalation -------------------------------------------
    attempts = []  # (theta, sim, md, sd)
    theta = _search(theta0, normal_iters, seed)
    sim, md, sd = _validate(theta)
    attempts.append((theta, sim, md, sd))

    rng = np.random.default_rng(seed)
    while not (md <= dev_threshold and sd <= dev_threshold) and res.n_restarts < max_restarts:
        res.n_restarts += 1
        pert = np.asarray(theta0, dtype=float) * rng.uniform(0.3, 3.0, size=len(theta0))
        th = _search(list(pert), retry_iters, seed + res.n_restarts)
        sim_r, md_r, sd_r = _validate(th)
        attempts.append((th, sim_r, md_r, sd_r))
        _, _, md, sd = min(attempts, key=lambda a: max(a[2], a[3]))

    theta_b, sim_b, md_b, sd_b = min(attempts, key=lambda a: max(a[2], a[3]))
    res.theta = [float(x) for x in theta_b]
    res.median_dev, res.steep_dev = float(md_b), float(sd_b)
    try:
        res.guess_survival = sim_b.getSurvival()
    except Exception:
        res.guess_survival = None
    if tr is not None:
        try:
            res.guess_survival_window = sim_b.getSurvival(time_range=tr)
        except Exception:
            res.guess_survival_window = None

    if md_b <= dev_threshold and sd_b <= dev_threshold:
        res.status = "ok"
    elif md_b <= dev_threshold and not np.isfinite(sd_b):
        res.status = "steepness_undefined"
    else:
        res.status = "best_effort"
    return res
