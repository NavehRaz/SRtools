"""Single-file PDF overview of data-vs-initial-guess survival curves.

:func:`plot_survival_overview` writes ONE paginated PDF of small-multiple panels
(data survival vs. the fitted initial-guess survival), so you can flip through many
organisms quickly and spot bad fits at a glance — green panels pass the deviation
gate, red panels do not. Curves are taken from the cached
``FitResult.data_survival`` / ``guess_survival`` (no re-simulation).
"""

from __future__ import annotations

import math

import numpy as np


def _coerce_xy(surv):
    """Return ``(t, s)`` arrays from a getSurvival() result, or ``None``."""
    if surv is None:
        return None
    try:
        t, s = surv[0], surv[1]
        t = np.asarray(t, dtype=float)
        s = np.asarray(s, dtype=float)
        if t.size == 0 or s.size == 0:
            return None
        n = min(t.size, s.size)
        return t[:n], s[:n]
    except Exception:
        return None


def plot_survival_overview(results, out_pdf, rows=4, cols=3, dev_threshold=0.15,
                           title="Initial-guess survival fits"):
    """Write one PDF of data-vs-guess survival panels, paginated ``rows x cols``.

    Parameters
    ----------
    results : list[FitResult]
        Fit results (any order). Each panel shows the data and guess survival
        curves with the species name and the median/steepness % deviations.
    out_pdf : str
        Output PDF path (exactly one file is written).
    rows, cols : int
        Panels per page. Default 4x3 = 12 readable panels per landscape page.

    Returns
    -------
    int
        Number of pages written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    per_page = rows * cols
    n = len(results)
    pages = max(1, math.ceil(n / per_page))

    def _fmt(d):
        return "n/a" if (d is None or not np.isfinite(d)) else f"{100 * d:.0f}%"

    with PdfPages(out_pdf) as pdf:
        for p in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5))
            axes = np.atleast_1d(axes).ravel()
            fig.suptitle(f"{title}  (page {p + 1}/{pages})", fontsize=11)
            for k in range(per_page):
                ax = axes[k]
                idx = p * per_page + k
                if idx >= n:
                    ax.axis("off")
                    continue
                r = results[idx]
                ok = (r.status == "ok")
                color = "#1a7f37" if ok else "#cf222e"

                data_xy = _coerce_xy(r.data_survival)
                guess_xy = _coerce_xy(r.guess_survival)
                if data_xy is not None:
                    ax.plot(data_xy[0], data_xy[1], color="#333333", lw=1.4, label="data")
                if guess_xy is not None:
                    ax.plot(guess_xy[0], guess_xy[1], color="#0969da", lw=1.2,
                            ls="--", label="guess")
                if data_xy is None and guess_xy is None:
                    ax.text(0.5, 0.5, r.status, ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color=color)

                t_min = getattr(r, "t_min", float("nan"))
                if np.isfinite(t_min):
                    ax.axvline(t_min, color="#999999", ls=":", lw=0.8)

                ax.set_ylim(-0.02, 1.02)
                ax.set_title(r.species, fontsize=8, color=color)
                ax.set_xlabel(
                    f"med {_fmt(r.median_dev)} | steep {_fmt(r.steep_dev)}"
                    + ("" if ok else f" | {r.status}"),
                    fontsize=7, color=color,
                )
                ax.tick_params(labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.3 if not ok else 0.8)
                if k == 0 and ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6, loc="upper right")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    return pages


def _died(obj):
    """Return the ages at observed death (events == 1) for a Dataset/sim, or None."""
    try:
        dt = np.asarray(obj.getDeathTimes(), dtype=float)
        ev = np.asarray(obj.events)
        return dt[ev == 1]
    except Exception:
        try:
            dt = np.asarray(obj.death_times, dtype=float)
            ev = np.asarray(obj.events)
            return dt[ev == 1]
        except Exception:
            return None


def plot_diagnostics_overview(results, datasets, out_pdf, per_page=4, sim_npeople=8000,
                              sim_nsteps=5000, bw_frac=0.04, dev_threshold=0.15, show=False):
    """Three-panel diagnostics (survival, hazard, death-time distribution) per organism.

    For each :class:`~SRtools.auto_initial_guess.FitResult` (with a usable ``theta``),
    the fitted guess is re-simulated and overlaid on the data — black = data, blue
    dashed = initial guess — in a row of three panels. Paginated into ONE PDF; a dotted
    line marks the detected ``t_min`` (fit window start) when present.

    Parameters
    ----------
    results : list[FitResult]
    datasets : dict[str, Dataset]
        Maps ``FitResult.species`` -> the (trimmed) data Dataset it was fit on.
    out_pdf : str
    per_page : int
        Organisms (rows) per page.
    bw_frac : float
        Hazard smoothing bandwidth as a fraction of maxLifetime (data & guess use the
        same bandwidth so the hazard panels are comparable).
    show : bool
        Also display each page inline (for notebooks).

    Returns
    -------
    int
        Number of pages written.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .SR_hetro import getSrHetro
    from .deathTimesDataSet import Dataset

    usable = [r for r in results if getattr(r, "theta", None) is not None and r.species in datasets]
    pages = max(1, math.ceil(len(usable) / per_page)) if usable else 1

    with PdfPages(out_pdf) as pdf:
        for p in range(pages):
            chunk = usable[p * per_page:(p + 1) * per_page]
            nrow = max(1, len(chunk))
            fig, axes = plt.subplots(nrow, 3, figsize=(13, 3.0 * nrow), squeeze=False)
            for row in range(nrow):
                if row >= len(chunk):
                    for c in range(3):
                        axes[row][c].axis("off")
                    continue
                r = chunk[row]
                ds0 = datasets[r.species]
                maxlt = ds0.getMaxLifetime()
                bw = max(3.0, bw_frac * maxlt) if np.isfinite(maxlt) and maxlt > 0 else 3.0
                color = "#1a7f37" if r.status == "ok" else "#cf222e"
                t_min = getattr(r, "t_min", float("nan"))

                # Data with a smoothing bandwidth matched to the guess; re-sim the guess.
                ds = Dataset(np.asarray(ds0.death_times, float), np.asarray(ds0.events),
                             bandwidth=bw)
                ext = r.exth if getattr(r, "has_ext", False) and r.exth is not None else np.inf
                sim = None
                try:
                    sim = getSrHetro(np.asarray(r.theta, float), n=sim_npeople, nsteps=sim_nsteps,
                                     t_end=r.t_end, external_hazard=ext,
                                     time_step_multiplier=int(r.time_step_multiplier),
                                     hetro=False, bandwidth=bw, parallel=True)
                except Exception:
                    sim = None

                ax_s, ax_h, ax_d = axes[row]
                # survival
                try:
                    ds.plotSurvival(ax_s, CI=False, color="black", lw=1.3, label="data")
                    if sim is not None:
                        sim.plotSurvival(ax_s, CI=False, color="tab:blue", ls="--", lw=1.2, label="guess")
                except Exception:
                    pass
                ax_s.set_ylim(-0.02, 1.02); ax_s.set_ylabel("survival", fontsize=7)
                ax_s.legend(fontsize=6, loc="upper right")
                # hazard (log y)
                try:
                    ds.plotHazard(ax_h, CI=False, color="black", lw=1.2)
                    if sim is not None:
                        sim.plotHazard(ax_h, CI=False, color="tab:blue", ls="--", lw=1.1)
                    ax_h.set_yscale("log")
                except Exception:
                    pass
                ax_h.set_ylabel("hazard (log)", fontsize=7)
                # death-time distribution (normalised histograms of died ages)
                dd, gd = _died(ds), (_died(sim) if sim is not None else None)
                if dd is not None and dd.size:
                    hi = np.nanmax(dd) if gd is None or not gd.size else max(np.nanmax(dd), np.nanmax(gd))
                    bins = np.linspace(0, hi, 41)
                    ax_d.hist(dd, bins=bins, density=True, histtype="step", color="black", label="data")
                    if gd is not None and gd.size:
                        ax_d.hist(gd, bins=bins, density=True, histtype="step", color="tab:blue",
                                  ls="--", label="guess")
                ax_d.set_ylabel("death-time density", fontsize=7)

                for ax in (ax_s, ax_h, ax_d):
                    if np.isfinite(t_min):
                        ax.axvline(t_min, color="#999999", ls=":", lw=0.8)
                    ax.tick_params(labelsize=6)
                    ax.set_xlabel("age", fontsize=6)
                ax_s.set_title(
                    f"{r.species}  [{r.status}]  med {100*r.median_dev:.0f}% | steep {100*r.steep_dev:.0f}%",
                    fontsize=8, color=color, loc="left",
                )
            fig.tight_layout()
            pdf.savefig(fig)
            if show:
                plt.show()
            plt.close(fig)
    return pages
