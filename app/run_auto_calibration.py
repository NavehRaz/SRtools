#!/usr/bin/env python
"""Batch-generate Baysian03 initial-guess configs from Species360 datasets.

For each dataset it predicts an initial ``theta``, refines it, validates the fit
(median lifespan & steepness within a threshold), and writes:

  * ``<out>``                 — multi-column configurations.xlsx (Sheet1) + QC sheet
  * ``<out_dir>/survival_overview.pdf`` — one PDF of data-vs-guess survival curves

Run from the SRtools checkout with the SR environment, e.g.::

    ~/SRvenv/bin/python app/run_auto_calibration.py \
        --datasets-root ../../code_4/Species360/selected_datasets \
        --out ../../code_4/Species360/configurations_auto.xlsx
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

from SRtools.deathTimesDataSet import dsFromFile, Dataset
from SRtools import auto_initial_guess as aig
from SRtools import species_config_builder as scb
from SRtools.calibration_report import plot_survival_overview


def _resolve_targets(args):
    """Return a list of (path, class_name, species) to process."""
    root = args.datasets_root
    targets = []
    if args.list_file:
        with open(args.list_file) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if os.path.isfile(line):
                    path = line
                else:
                    # "Class,Species_Sex" or "Class/Species_Sex"
                    parts = [p for p in line.replace(",", "/").split("/") if p]
                    if len(parts) < 2:
                        print(f"  skipping unparseable list entry: {line!r}")
                        continue
                    cls, sp = parts[-2], parts[-1]
                    sp = sp[:-4] if sp.endswith(".csv") else sp
                    path = os.path.join(root, cls, sp + ".csv")
                cls = os.path.basename(os.path.dirname(path))
                sp = os.path.splitext(os.path.basename(path))[0]
                targets.append((path, cls, sp))
    else:
        for path in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
            cls = os.path.basename(os.path.dirname(path))
            sp = os.path.splitext(os.path.basename(path))[0]
            targets.append((path, cls, sp))
    return targets


def _load_trimmed(path, n_remove):
    """Load a dataset and drop the ``n_remove`` longest-lived individuals.

    Mirrors the Species360 notebook (cell 0): trimming outliers stabilises
    ``getMaxLifetime`` (which drives ``t_end``).
    """
    ds = dsFromFile(path, properties=["death_cause"])
    if n_remove > 0 and ds.n > n_remove:
        order = np.argsort(ds.death_times)
        keep = order[:-n_remove]
        props = ({p: ds.properties[p][keep] for p in ds.properties}
                 if ds.properties else None)
        ds = Dataset(ds.death_times[keep], ds.events[keep],
                     properties=props, bandwidth=ds.bandwidth)
    return ds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", required=True,
                        help="root with <Class>/<Species_Sex>.csv files")
    parser.add_argument("--out", required=True, help="output configurations .xlsx path")
    parser.add_argument("--list-file", default=None,
                        help="optional file of species (path, or 'Class,Species_Sex' per line); "
                             "default = all CSVs under --datasets-root")
    parser.add_argument("--base-prefix", default="Species360_Calibration",
                        help="shared path prefix for folder/data_file/run_file_mcmc")
    # model / fit knobs
    parser.add_argument("--method", choices=["scipy", "random"], default="scipy")
    parser.add_argument("--target-dt", type=float, default=aig.DEFAULT_TARGET_DT)
    parser.add_argument("--xc", type=float, default=20.0)
    parser.add_argument("--dev-threshold", type=float, default=0.15)
    parser.add_argument("--n-remove", type=int, default=10)
    parser.add_argument("--min-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    # resolution / budgets
    parser.add_argument("--search-npeople", type=int, default=5000)
    parser.add_argument("--search-nsteps", type=int, default=2500)
    parser.add_argument("--search-max-n", type=int, default=3000,
                        help="cap on data death-times used for the search objective "
                             "(speeds up large datasets; validation uses the full data)")
    parser.add_argument("--valid-npeople", type=int, default=20000)
    parser.add_argument("--valid-nsteps", type=int, default=5000)
    parser.add_argument("--normal-iters", type=int, default=120)
    parser.add_argument("--retry-iters", type=int, default=200)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--no-parallel", action="store_true",
                        help="disable in-sim parallelism (use for debugging)")
    # external hazard
    parser.add_argument("--detect-external-hazard", action="store_true",
                        help="apply detected external hazard (ndims=5). Off by default; the "
                             "candidate value is reported in QC regardless.")
    parser.add_argument("--ext-flat-slope-ratio", type=float, default=0.5,
                        help="how flat the early log-hazard must be vs the aging rise to "
                             "count as an external-hazard floor (lower = stricter)")
    parser.add_argument("--no-time-range", action="store_true",
                        help="disable aging-trend-start detection (fit the full [0, t_end] "
                             "instead of the detected [t_min, t_end] window)")
    # report
    parser.add_argument("--no-report", action="store_true", help="skip the survival-overview PDF")
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=3)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    targets = _resolve_targets(args)
    if not targets:
        print("No datasets found.", file=sys.stderr)
        return 1
    print(f"Processing {len(targets)} dataset(s) with method={args.method} ...")

    columns, qc_rows, results = [], [], []
    ext_kwargs = {"flat_slope_ratio": args.ext_flat_slope_ratio}
    for i, (path, cls, species) in enumerate(targets, 1):
        try:
            ds = _load_trimmed(path, args.n_remove)
            res = aig.auto_fit(
                ds, species=species, method=args.method, target_dt=args.target_dt, xc=args.xc,
                search_npeople=args.search_npeople, search_nsteps=args.search_nsteps,
                search_max_n=args.search_max_n,
                valid_npeople=args.valid_npeople, valid_nsteps=args.valid_nsteps,
                normal_iters=args.normal_iters, retry_iters=args.retry_iters,
                max_restarts=args.max_restarts, dev_threshold=args.dev_threshold,
                detect_ext=args.detect_external_hazard, ext_kwargs=ext_kwargs,
                time_range_mode=not args.no_time_range,
                min_events=args.min_events, seed=args.seed, parallel=not args.no_parallel,
            )
        except Exception as e:  # never abort the batch
            print(f"[{i}/{len(targets)}] {species}: ERROR {e}")
            res = aig.FitResult(species=species, status=f"error:{type(e).__name__}")
        results.append(res)
        qc_rows.append(scb.build_qc_row(res))
        col = scb.build_config_column(res, cls, base_prefix=args.base_prefix)
        if col is not None:
            columns.append(col)
        tmin_s = f"t_min={res.t_min:.0f}" if np.isfinite(res.t_min) else "t_min=-"
        print(f"[{i}/{len(targets)}] {species:35s} {res.status:18s} "
              f"med={res.median_dev:.3f} steep={res.steep_dev:.3f} ndims={res.ndims} {tmin_s}")

    scb.write_configurations_excel(columns, args.out, qc_rows=qc_rows)
    print(f"\nWrote {len(columns)} config column(s) -> {args.out}  (+ QC sheet)")

    if not args.no_report:
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "survival_overview.pdf")
        n_pages = plot_survival_overview(results, pdf_path, rows=args.grid_rows,
                                         cols=args.grid_cols, dev_threshold=args.dev_threshold)
        print(f"Wrote survival overview ({n_pages} page(s)) -> {pdf_path}")

    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"\nSummary: {n_ok}/{len(results)} within {args.dev_threshold:.0%} on median & steepness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
