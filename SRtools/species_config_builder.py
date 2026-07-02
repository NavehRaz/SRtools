"""Turn :class:`~SRtools.auto_initial_guess.FitResult` objects into a Baysian03
configuration Excel.

The output matches ``Species360/configurations.csv`` exactly: one column per
species, keys in column A (blank header), in :data:`ROW_ORDER`. It is read back by
:class:`SRtools.config_lib.ExcelConfigParser`. A second ``QC`` sheet carries the
per-species fit-quality summary (it is never the first sheet, so it does not
interfere with config parsing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Row order of a configuration column (verbatim from the Species360 notebook /
#: ``single_configuration_*.csv``).
ROW_ORDER = [
    "eta", "beta", "epsilon", "xc", "ExtH", "h5_file_name", "folder", "nsteps",
    "time_step_multiplier", "npeople", "t_end", "n_jobs", "nwalkers", "n_mcmc_steps",
    "job_name", "initial_memory", "max_memory", "name", "run_file_mcmc", "metric",
    "data_file", "variations", "prior", "transform", "time_range", "external_hazard",
    "queue", "results_csv_file_name", "index", "nbins", "hetro", "data_dt", "ndims",
    "test", "TIME_UNIT", "POST_CREATED",
]

#: Static defaults (everything not derived from the fit). Matches the notebook.
DEFAULTS = {
    "npeople": 20000, "n_jobs": 250, "nwalkers": 20, "n_mcmc_steps": 1000,
    "initial_memory": 1, "max_memory": 1, "metric": "baysian", "variations": "[0.1, 10]",
    "prior": 10, "transform": 1, "time_range": "", "queue": "short", "index": 1,
    "nbins": 100, "hetro": 0, "data_dt": 1, "test": 0, "TIME_UNIT": "days",
    "POST_CREATED": 0,
}

#: Species360 class -> datasets sub-folder name.
CLASS_TO_FOLDER = {
    "Mammalia": "Mammalia", "Aves": "Aves", "Amphibia": "Amphibia",
    "Reptilia": "Reptilia", "Chondrichthyes": "Chondrichthyes",
}

#: QC columns written to the second sheet.
QC_COLUMNS = [
    "species", "status", "median_dev", "steep_dev", "ndims", "has_ext",
    "ext_candidate", "n_restarts", "seed", "data_ml", "data_steepness", "t_end",
    "nsteps", "time_step_multiplier", "t_min", "time_range", "cond_data_ml",
    "cond_data_steepness",
]


def build_config_column(fit_result, class_name, base_prefix="Species360_Calibration",
                        class_to_folder=None):
    """Return an ordered ``pd.Series`` (a configuration column) for one fit.

    Returns ``None`` if the fit produced no usable theta (e.g. ``insufficient_data``).
    Paths follow the ``single_configuration_*.csv`` example exactly: ``data_file``
    has no extension; ``t_end`` is written as an integer (``run_manager.py`` casts it
    with ``int()`` and would crash on a float string).
    """
    if fit_result.theta is None:
        return None

    class_to_folder = class_to_folder or CLASS_TO_FOLDER
    species = fit_result.species
    folder = class_to_folder.get(class_name, class_name)
    eta, beta, epsilon, xc = fit_result.theta

    has_ext = fit_result.has_ext
    exth_val = fit_result.exth if has_ext else ""

    values = dict(DEFAULTS)
    values.update({
        "eta": eta, "beta": beta, "epsilon": epsilon, "xc": xc,
        "ExtH": exth_val,
        "h5_file_name": species,
        "folder": f"{base_prefix}/simulation_results/{folder}/{species}",
        "nsteps": int(fit_result.nsteps),
        "time_step_multiplier": int(fit_result.time_step_multiplier),
        "t_end": int(round(fit_result.t_end)),
        "job_name": species,
        "name": species,
        "run_file_mcmc": f"{base_prefix}/run_mcmc_excel.csh",
        "data_file": f"{base_prefix}/datasets/{folder}/{species}",
        "external_hazard": exth_val,
        "results_csv_file_name": f"{species}.csv",
        "ndims": int(fit_result.ndims),
    })
    # Fit window (aging-trend start). Written as a "[t_min, t_end]" string that
    # config_to_dict parses back to a [int, int] list via ast.literal_eval.
    tr = getattr(fit_result, "time_range", None)
    if tr is not None:
        values["time_range"] = f"[{int(round(tr[0]))}, {int(round(tr[1]))}]"
    return pd.Series({k: values.get(k, "") for k in ROW_ORDER}, name=species)


def build_qc_row(fit_result):
    """Return a dict of QC fields for one fit (always available, even on failure)."""
    return {
        "species": fit_result.species,
        "status": fit_result.status,
        "median_dev": fit_result.median_dev,
        "steep_dev": fit_result.steep_dev,
        "ndims": fit_result.ndims,
        "has_ext": fit_result.has_ext,
        "ext_candidate": fit_result.ext_candidate,
        "n_restarts": fit_result.n_restarts,
        "seed": fit_result.seed,
        "data_ml": fit_result.data_ml,
        "data_steepness": fit_result.data_steepness,
        "t_end": fit_result.t_end,
        "nsteps": fit_result.nsteps,
        "time_step_multiplier": fit_result.time_step_multiplier,
        "t_min": getattr(fit_result, "t_min", float("nan")),
        "time_range": getattr(fit_result, "time_range", None),
        "cond_data_ml": getattr(fit_result, "cond_data_ml", float("nan")),
        "cond_data_steepness": getattr(fit_result, "cond_data_steepness", float("nan")),
    }


def write_configurations_excel(columns, out_path, qc_rows=None):
    """Write a multi-column configurations Excel + a QC sheet.

    Parameters
    ----------
    columns : list[pd.Series]
        Config columns from :func:`build_config_column` (``None`` entries skipped).
    out_path : str
        Destination ``.xlsx`` path.
    qc_rows : list[dict], optional
        Rows from :func:`build_qc_row` for the ``QC`` sheet.
    """
    columns = [c for c in columns if c is not None]
    if not columns:
        raise ValueError("No configuration columns to write (all fits unusable).")

    # Sheet1: index = ROW_ORDER (blank header), one column per species.
    df = pd.concat(columns, axis=1).reindex(ROW_ORDER)
    df.index.name = None

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=True, header=True)
        if qc_rows:
            qc_df = pd.DataFrame(qc_rows).reindex(columns=QC_COLUMNS)
            qc_df.to_excel(writer, sheet_name="QC", index=False, header=True)
    return out_path
