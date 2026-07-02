#!/usr/bin/env python
"""Small experiment to choose good default calibration settings.

Sweeps a handful of method / resolution combinations on a few reference datasets
and reports, for each combination, how many fits land within the deviation gate and
how long they take. Run once at setup time; pick the cheapest combo that reliably
passes and bake it into ``auto_initial_guess`` / the CLI defaults.

Writes ``tune_defaults_results.csv`` and prints a ranked table.

Example::

    ~/SRvenv/bin/python app/tune_defaults.py \
        --datasets-root ../../code_4/Species360/selected_datasets \
        --species Amphibia/Anaxyrus_baxteri_F Reptilia/Pogona_vitticeps_M
"""

import argparse
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from SRtools.deathTimesDataSet import dsFromFile, Dataset
from SRtools import auto_initial_guess as aig


def _load_trimmed(path, n_remove=10):
    ds = dsFromFile(path, properties=["death_cause"])
    if n_remove > 0 and ds.n > n_remove:
        order = np.argsort(ds.death_times)
        keep = order[:-n_remove]
        props = ({p: ds.properties[p][keep] for p in ds.properties} if ds.properties else None)
        ds = Dataset(ds.death_times[keep], ds.events[keep], properties=props, bandwidth=ds.bandwidth)
    return ds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--species", nargs="+", required=True,
                        help="relative '<Class>/<Species_Sex>' entries to tune on")
    parser.add_argument("--out", default="tune_defaults_results.csv")
    parser.add_argument("--dev-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["scipy", "random"])
    parser.add_argument("--search-npeople", nargs="+", type=int, default=[2500, 5000])
    parser.add_argument("--search-nsteps", nargs="+", type=int, default=[1500, 2500])
    parser.add_argument("--target-dt", nargs="+", type=float, default=[0.25])
    parser.add_argument("--normal-iters", type=int, default=60)
    parser.add_argument("--valid-npeople", type=int, default=8000)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    datasets = []
    for sp in args.species:
        path = os.path.join(args.datasets_root, sp + (".csv" if not sp.endswith(".csv") else ""))
        datasets.append((sp, _load_trimmed(path)))

    rows = []
    combos = list(itertools.product(args.methods, args.search_npeople,
                                    args.search_nsteps, args.target_dt))
    print(f"Sweeping {len(combos)} combos x {len(datasets)} datasets ...")
    for method, snp, sns, tdt in combos:
        passed, devs, secs = 0, [], 0.0
        for sp, ds in datasets:
            t0 = time.time()
            r = aig.auto_fit(ds, species=sp, method=method, target_dt=tdt,
                             search_npeople=snp, search_nsteps=sns,
                             valid_npeople=args.valid_npeople, valid_nsteps=5000,
                             normal_iters=args.normal_iters, max_restarts=0,
                             dev_threshold=args.dev_threshold, seed=args.seed)
            secs += time.time() - t0
            passed += int(r.status == "ok")
            devs.append(max(r.median_dev, r.steep_dev))
        rows.append({
            "method": method, "search_npeople": snp, "search_nsteps": sns, "target_dt": tdt,
            "passed": passed, "n": len(datasets),
            "worst_dev_mean": float(np.mean([d for d in devs if np.isfinite(d)] or [np.inf])),
            "seconds": round(secs, 1),
        })
        print(f"  {method:6s} np={snp} ns={sns} dt={tdt}: "
              f"{passed}/{len(datasets)} pass, {secs:.0f}s")

    df = pd.DataFrame(rows).sort_values(
        ["passed", "worst_dev_mean", "seconds"], ascending=[False, True, True])
    df.to_csv(args.out, index=False)
    print("\nRanked (best first):")
    print(df.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
