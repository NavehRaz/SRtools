from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SRtools import Dataset, Life_table
from SRtools import presets
from SRtools.SR_hetro import getSrHetro


MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE = 35_000_000
PREVIEW_SIMULATION_WORK_UNITS = 250_000
PREVIEW_SIMULATION_MAX_SAMPLE_SIZE = 1_000
MAX_TIME_STEP_MULTIPLIER = 20
DEFAULT_BANDWIDTH = 3
TIME_UNITS = ["hours", "days", "weeks", "years", "generations"]
TIME_UNIT_TO_DAYS = {
    "hours": 1.0 / 24.0,
    "days": 1.0,
    "weeks": 7.0,
    "years": 365.0,
    "generations": 3.0 / 24.0,
}
CORE_PARAMETER_SPECS = {
    "eta": {"label": "eta", "min": 1e-10, "max": 1e2},
    "beta": {"label": "beta", "min": 1e-8, "max": 1e4},
    "kappa": {"label": "kappa", "min": 1e-6, "max": 1e3},
    "epsilon": {"label": "epsilon", "min": 1e-12, "max": 1e4},
    "xc": {"label": "xc", "min": 1e-6, "max": 1e5},
}
SWEEP_PARAMETER_LABELS = {
    "eta": "eta",
    "beta": "beta",
    "kappa": "kappa",
    "epsilon": "epsilon",
    "xc": "xc",
    "external_hazard_exponent": "external hazard exponent",
}
FRIENDLY_HUMAN_PRESETS = {
    "core:Denmark_M_1900_hetro": "Human M",
    "core:Denmark_F_1900_hetro": "Human F",
}
FRIENDLY_HIDDEN_PRESETS = {"core:combined_human_M", "core:combined_human_F"}
STYLE_CONFIGS = {
    "Friendly": {
        "template": "plotly_white",
        "line_width": 3.2,
        "reference_width": 2,
        "font_size": 15,
        "title_size": 22,
        "legend_title": "Series",
        "colors": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#332288", "#88CCEE"],
        "gridcolor": "rgba(120, 130, 145, 0.18)",
    },
    "Scientific": {
        "template": "simple_white",
        "line_width": 2,
        "reference_width": 1.5,
        "font_size": 12,
        "title_size": 16,
        "legend_title": "Dataset",
        "colors": ["#1B1B1B", "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#9C755F", "#76B7B2", "#E15759"],
        "gridcolor": "rgba(0, 0, 0, 0.12)",
    },
}

PRESET_CATALOG_CSV = """source,preset_name,suggested_alias,shortlist,time_units
core,combined_human_M,Human male combined,yes,years
core,combined_human_F,Human female combined,yes,years
core,mice_M,Mouse male,yes,weeks
core,mice_F,Mouse female,yes,weeks
core,cats_BPH,Cat BPH,no,years
core,Labradors_vetCompass,Dog Labrador retriever,yes,years
core,Staffy_vetCompass,Dog Staffordshire terrier,yes,years
core,Jack_Russell_vetCompass,Dog Jack Russell terrier,yes,years
core,German_Shepherd_vetCompass,Dog German shepherd,yes,years
core,Guiniea_pig_VC,Guinea pig,yes,years
core,celegans,C. elegans,yes,days
core,ecoli,E. coli,yes,hours
core,yeast,Budding yeast,yes,generations
core,drosophila_441,Drosophila DGRP 441,yes,days
core,drosophila_M22_25deg,Drosophila M22 25 C,no,days
core,Denmark_M_1900_hetro,Denmark male 1900,yes,years
core,Denmark_F_1900_hetro,Denmark female 1900,yes,years
core,Denmark_M_1890_hetro,Denmark male 1890,no,years
core,Denmark_F_1890_hetro,Denmark female 1890,no,years
core,Sweden_M_1910_hetro,Sweden male 1910,yes,years
core,Sweden_F_1910_hetro,Sweden female 1910,yes,years
core,Sweden_F_1910_homo,Sweden female 1910 homogeneous,no,years
core,Denmark_M_1900_homo,Denmark male 1900 homogeneous,no,years
core,Denmark_M_1890_homo,Denmark male 1890 homogeneous,no,years
core,cats_vp_M,Cat male,yes,years
core,cats_vp_F,Cat female,yes,years
core,drosophila_853,Drosophila DGRP 853,yes,days
core,drosophila_707,Drosophila DGRP 707,yes,days
core,drosophila_217,Drosophila DGRP 217,yes,days
core,drosophila_136,Drosophila DGRP 136,yes,days
core,drosophila_195,Drosophila DGRP 195,yes,days
core,drosophila_105,Drosophila DGRP 105,yes,days
core,Wdah_control,Drosophila Wdah control,no,days
core,Wdah_chronic,Drosophila Wdah rapamycin,no,days
core,Wdah_chronic2,Drosophila Wdah rapamycin 2,no,days
smurfs,Killifish,Killifish,yes,days
smurfs,drosophila_M22_25deg,Drosophila M22 25 C,no,days
smurfs,Wdah_control,Wdah control,no,days
smurfs,Wdah_chronic,Wdah chronic,no,days
smurfs,Wdah_chronic2,Wdah chronic 2,no,days
smurfs,drosophila_136,Drosophila 136,no,days
smurfs,drosophila_195,Drosophila 195,no,days
smurfs,drosophila_105,Drosophila 105,no,days
smurfs,smurfs_195,SMURFS 195,no,days
smurfs,smurfs_136,SMURFS 136,no,days
smurfs,smurfs_105,SMURFS 105,no,days
smurfs,smurfs_91,SMURFS 91,no,days
smurfs,smurfs_88,SMURFS 88,no,days
smurfs,smurfs_83,SMURFS 83,no,days
"""


@dataclass
class PlotObject:
    name: str
    kind: str
    source: object
    time_unit: str
    params: dict | None = None


def _finite_float(value, default=np.inf):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(value):
        return default
    return value


def normalize_yes_no(value) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def normalize_time_unit(value, default="days") -> str:
    unit = str(value or "").strip().lower()
    return unit if unit in TIME_UNIT_TO_DAYS else default


def preset_folder(source: str):
    return "smurfs" if source == "smurfs" else None


def default_heterogeneity(preset_name: str, config_value: bool) -> bool:
    name = preset_name.lower()
    if "homo" in name:
        return False
    human_markers = ("human", "denmark", "sweden", "combined_human")
    if any(marker in name for marker in human_markers):
        return True
    return False


def convert_duration(value, from_unit: str, to_unit: str):
    return np.asarray(value, dtype=float) * TIME_UNIT_TO_DAYS[from_unit] / TIME_UNIT_TO_DAYS[to_unit]


def display_external_hazard(exponent: float) -> str:
    if not np.isfinite(exponent):
        return "0"
    return f"{np.exp(-exponent):.4g}"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(float(value), min_value), max_value)


def object_time_unit(obj: PlotObject) -> str:
    return normalize_time_unit(getattr(obj, "time_unit", "days"), default="days")


def friendly_shortlist_active(visualization_style: str, use_shortlist: bool) -> bool:
    return visualization_style == "Friendly" and use_shortlist


def visible_presets_for_mode(preset_catalog: list[dict], visualization_style: str, use_shortlist: bool) -> list[dict]:
    friendly_shortlist = friendly_shortlist_active(visualization_style, use_shortlist)
    visible = []
    for index, preset in enumerate(preset_catalog):
        if use_shortlist and not preset["shortlist"]:
            continue
        if friendly_shortlist and preset["id"] in FRIENDLY_HIDDEN_PRESETS:
            continue
        visible.append({**preset, "_catalog_index": index})
    if friendly_shortlist:
        visible.sort(key=lambda preset: ({"core:Denmark_M_1900_hetro": 0, "core:Denmark_F_1900_hetro": 1}.get(preset["id"], 2), preset["_catalog_index"]))
    return visible


def preset_display_name(preset: dict, visualization_style: str, use_shortlist: bool) -> str:
    if friendly_shortlist_active(visualization_style, use_shortlist) and preset["id"] in FRIENDLY_HUMAN_PRESETS:
        return FRIENDLY_HUMAN_PRESETS[preset["id"]]
    if use_shortlist:
        return preset["alias"]
    return f"{preset['alias']} ({preset['source']})"


@st.cache_data(show_spinner=False)
def load_preset_catalog() -> list[dict]:
    rows = []
    for row in csv.DictReader(StringIO(PRESET_CATALOG_CSV.strip())):
        source = row["source"].strip()
        preset_name = row["preset_name"].strip()
        alias = row["suggested_alias"].strip()
        time_unit = normalize_time_unit(row.get("time_units"), default="days")
        rows.append(
            {
                "id": f"{source}:{preset_name}",
                "source": source,
                "preset_name": preset_name,
                "alias": alias,
                "shortlist": normalize_yes_no(row.get("shortlist")),
                "time_unit": time_unit,
                "folder": preset_folder(source),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def load_preset_defaults(preset_name: str, source: str, time_unit: str) -> dict:
    folder = preset_folder(source)
    external_hazard_exponent = 30.0
    try:
        theta = presets.getTheta(preset_name=preset_name, type="mode_overall", time_unit=time_unit, ExtH=True, folder=folder)
        external_hazard_exponent = _finite_float(theta[4], default=30.0)
    except KeyError:
        theta = presets.getTheta(preset_name=preset_name, type="mode_overall", time_unit=time_unit, ExtH=False, folder=folder)
    try:
        config = presets.get_config_params(
            preset_name=preset_name,
            config_params=["nsteps", "time_step_multiplier", "npeople", "t_end", "hetro"],
            types=[int, int, int, int, bool],
            time_unit=time_unit,
            folder=folder,
        )
    except Exception:
        config = {"nsteps": 5000, "time_step_multiplier": 1, "npeople": 5000, "t_end": 100, "hetro": False}
    return {
        "eta": float(theta[0]),
        "beta": float(theta[1]),
        "kappa": 0.5,
        "epsilon": float(theta[2]),
        "xc": float(theta[3]),
        "external_hazard_exponent": external_hazard_exponent,
        "nsteps": int(config["nsteps"]),
        "time_step_multiplier": int(config["time_step_multiplier"]),
        "npeople": int(config["npeople"]),
        "t_end": int(config["t_end"]),
        "hetro": default_heterogeneity(preset_name, bool(config.get("hetro", False))),
    }


def params_from_defaults(defaults: dict, time_unit: str) -> dict:
    return {
        "eta": float(defaults["eta"]),
        "beta": float(defaults["beta"]),
        "kappa": float(defaults.get("kappa", 0.5)),
        "epsilon": float(defaults["epsilon"]),
        "xc": float(defaults["xc"]),
        "external_hazard": float(defaults["external_hazard_exponent"]),
        "external_hazard_exponent": float(defaults["external_hazard_exponent"]),
        "time_unit": normalize_time_unit(time_unit),
        "hetro": bool(defaults["hetro"]),
        "eta_var": 0.0,
        "beta_var": 0.0,
        "kappa_var": 0.0,
        "epsilon_var": 0.0,
        "xc_var": 0.2 if defaults["hetro"] else 0.0,
        "npeople": int(min(defaults["npeople"], 5_000)),
        "nsteps": int(max(defaults["nsteps"], 1)),
        "t_end": int(min(defaults["t_end"], 250)),
        "time_step_multiplier": int(min(defaults["time_step_multiplier"], MAX_TIME_STEP_MULTIPLIER)),
        "method": "brownian_bridge",
    }


def preview_params(params: dict) -> dict:
    preview = params.copy()
    preview["npeople"] = int(min(max(100, preview["npeople"]), PREVIEW_SIMULATION_MAX_SAMPLE_SIZE))
    max_preview_steps = max(50, PREVIEW_SIMULATION_WORK_UNITS // preview["npeople"])
    preview["nsteps"] = int(min(max(50, preview["nsteps"]), max_preview_steps))
    preview["time_step_multiplier"] = int(min(max(1, preview["time_step_multiplier"]), MAX_TIME_STEP_MULTIPLIER))
    return preview


def params_cache_key(params: dict) -> tuple:
    keys = [
        "eta",
        "beta",
        "kappa",
        "epsilon",
        "xc",
        "external_hazard",
        "time_unit",
        "hetro",
        "eta_var",
        "beta_var",
        "kappa_var",
        "epsilon_var",
        "xc_var",
        "npeople",
        "nsteps",
        "t_end",
        "time_step_multiplier",
        "method",
    ]
    frozen = []
    for key in keys:
        value = params[key]
        if isinstance(value, float):
            value = round(value, 12)
        frozen.append((key, value))
    return tuple(frozen)


def read_uploaded_table(uploaded_file, sheet_name=None) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if suffix == "csv":
        return pd.read_csv(BytesIO(data))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(data), sheet_name=sheet_name)
    raise ValueError("Supported upload formats are CSV and XLSX.")


def excel_sheet_names(uploaded_file) -> list[str]:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if suffix not in {"xlsx", "xls"}:
        return []
    excel = pd.ExcelFile(BytesIO(uploaded_file.getvalue()))
    return list(excel.sheet_names)


def make_dataset(df: pd.DataFrame, time_col: str, event_col: str | None, event_is_censored: bool) -> Dataset:
    clean = df.copy()
    clean = clean.dropna(subset=[time_col])
    death_times = clean[time_col].astype(float).to_numpy()
    if event_col:
        events = clean[event_col].fillna(0).astype(int).to_numpy()
    else:
        events = np.ones(len(clean), dtype=int)
    return Dataset(
        death_times,
        events,
        bandwidth=DEFAULT_BANDWIDTH,
        event_is_censored=event_is_censored,
    )


def make_life_table(df: pd.DataFrame, age_col: str, alive_col: str, hazard_col: str | None) -> Life_table:
    clean = df.dropna(subset=[age_col, alive_col]).sort_values(age_col)
    hazard = clean[hazard_col].astype(float).to_numpy() if hazard_col else None
    return Life_table(
        clean[age_col].astype(float).to_numpy(),
        clean[alive_col].astype(float).to_numpy(),
        hazard_values=hazard,
        bandwidth=DEFAULT_BANDWIDTH,
    )


def trim_xy(x: Iterable[float], y: Iterable[float], time_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= time_range[0]) & (x_arr <= time_range[1])
    return x_arr[mask], y_arr[mask]


def object_median_lifetime(obj: PlotObject, plot_unit: str) -> float:
    from_unit = object_time_unit(obj)
    median = getattr(obj.source, "median_lifetime", None)
    if median is None and hasattr(obj.source, "getMedianLifetime"):
        try:
            median = obj.source.getMedianLifetime()
        except Exception:
            median = None
    if median is None or not np.isfinite(float(median)) or float(median) <= 0:
        return np.nan
    return float(convert_duration(float(median), from_unit, plot_unit))


def survival_frame(
    obj: PlotObject,
    time_range: tuple[float, float],
    plot_unit: str,
    renormalize: bool = False,
    scale_by_median: bool = False,
) -> pd.DataFrame:
    from_unit = object_time_unit(obj)
    t, s = obj.source.getSurvival()
    t = convert_duration(t, from_unit, plot_unit)
    t, s = trim_xy(t, s, time_range)
    if renormalize and len(s) > 0 and s[0] > 0:
        s = s / s[0]
    if scale_by_median:
        median = object_median_lifetime(obj, plot_unit)
        if np.isfinite(median) and median > 0:
            t = t / median
    return pd.DataFrame({"time": t, "value": s, "name": obj.name, "kind": obj.kind, "plot": "survival", "time_unit": plot_unit})


def hazard_frame(obj: PlotObject, time_range: tuple[float, float], plot_unit: str) -> pd.DataFrame:
    from_unit = object_time_unit(obj)
    t, h = obj.source.getHazard()
    t = convert_duration(t, from_unit, plot_unit)
    h = np.asarray(h, dtype=float) * TIME_UNIT_TO_DAYS[plot_unit] / TIME_UNIT_TO_DAYS[from_unit]
    t, h = trim_xy(t, h, time_range)
    return pd.DataFrame({"time": t, "value": h, "name": obj.name, "kind": obj.kind, "plot": "hazard", "time_unit": plot_unit})


def death_distribution_frame(obj: PlotObject, time_range: tuple[float, float], bin_width: float, plot_unit: str) -> pd.DataFrame:
    from_unit = object_time_unit(obj)
    native_range = tuple(convert_duration(np.array(time_range), plot_unit, from_unit))
    native_bin_width = float(convert_duration(bin_width, plot_unit, from_unit))
    bins = np.arange(native_range[0], native_range[1] + native_bin_width, native_bin_width)
    if len(bins) < 2:
        bins = np.array([native_range[0], native_range[1]])
    try:
        probs, edges = obj.source.get_death_times_distribution(bins=bins, dt=native_bin_width, time_range=native_range)
    except TypeError:
        probs, edges = obj.source.get_death_times_distribution(bins=bins, dt=native_bin_width)
    left = np.asarray(edges[:-1], dtype=float)
    right = np.asarray(edges[1:], dtype=float)
    left = convert_duration(left, from_unit, plot_unit)
    right = convert_duration(right, from_unit, plot_unit)
    probs = np.asarray(probs[: len(left)], dtype=float)
    mask = np.isfinite(left) & np.isfinite(right) & np.isfinite(probs)
    return pd.DataFrame(
        {
            "time": left[mask],
            "bin_right": right[mask],
            "value": probs[mask],
            "name": obj.name,
            "kind": obj.kind,
            "plot": "death_distribution",
            "time_unit": plot_unit,
        }
    )


def apply_plot_style(fig: go.Figure, style: dict) -> go.Figure:
    fig.update_layout(
        template=style["template"],
        font={"size": style["font_size"]},
        title={"font": {"size": style["title_size"]}},
        legend_title_text=style["legend_title"],
        margin={"l": 30, "r": 20, "t": 60, "b": 45},
    )
    fig.update_xaxes(gridcolor=style["gridcolor"], zeroline=False)
    fig.update_yaxes(gridcolor=style["gridcolor"], zeroline=False)
    return fig


def plot_lines(
    frames: list[pd.DataFrame],
    title: str,
    y_title: str,
    x_title: str = "Time",
    log_y: bool = False,
    style: dict | None = None,
) -> go.Figure:
    style = style or STYLE_CONFIGS["Friendly"]
    fig = go.Figure()
    colors = style["colors"]
    for index, frame in enumerate(frames):
        if frame.empty:
            continue
        label = frame["name"].iloc[0]
        is_reference = frame["kind"].iloc[0] in {"data", "life table"}
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["value"],
                mode="lines",
                line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                name=label,
                line={
                    "color": "rgba(95, 99, 104, 0.75)" if is_reference else colors[index % len(colors)],
                    "width": style["reference_width"] if is_reference else style["line_width"],
                    "dash": "dot" if frame["kind"].iloc[0] == "preview" else None,
                },
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return apply_plot_style(fig, style)


def frame_for_plot(
    obj: PlotObject,
    plot_kind: str,
    time_range: tuple[float, float],
    plot_unit: str,
    bin_width: float,
    renormalize: bool = False,
    scale_by_median: bool = False,
) -> pd.DataFrame:
    if plot_kind == "Survival":
        return survival_frame(obj, time_range, plot_unit, renormalize=renormalize, scale_by_median=scale_by_median)
    if plot_kind == "Hazard":
        return hazard_frame(obj, time_range, plot_unit)
    return death_distribution_frame(obj, time_range, bin_width, plot_unit)


def plot_sweep_overlay(
    frames: list[pd.DataFrame],
    reference_frames: list[pd.DataFrame],
    parameter_label: str,
    title: str,
    y_title: str,
    x_title: str,
    log_y: bool,
    style: dict,
) -> go.Figure:
    fig = go.Figure()
    colors = sample_colorscale("Viridis", np.linspace(0.08, 0.92, max(len(frames), 1)))
    for index, frame in enumerate(frames):
        if frame.empty:
            continue
        sweep_value = frame["sweep_value"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["value"],
                mode="lines",
                line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                name=f"{parameter_label}={sweep_value:.3g}",
                line={"color": colors[index], "width": style["line_width"]},
            )
        )
    for frame in reference_frames:
        if frame.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["value"],
                mode="lines",
                line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                name=frame["name"].iloc[0],
                line={"color": "rgba(95, 99, 104, 0.7)", "width": style["reference_width"], "dash": "dot"},
            )
        )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    if log_y:
        fig.update_yaxes(type="log")
    return apply_plot_style(fig, style)


def plot_sweep_animation(
    frames: list[pd.DataFrame],
    reference_frames: list[pd.DataFrame],
    parameter_label: str,
    title: str,
    y_title: str,
    x_title: str,
    log_y: bool,
    style: dict,
) -> go.Figure:
    fig = go.Figure()
    if not frames:
        return fig

    initial = frames[0]
    fig.add_trace(
        go.Scatter(
            x=initial["time"],
            y=initial["value"],
            mode="lines",
            line_shape="hv" if initial["plot"].iloc[0] == "death_distribution" else "linear",
            name=f"{parameter_label}={initial['sweep_value'].iloc[0]:.3g}",
            line={"color": style["colors"][0], "width": style["line_width"]},
        )
    )
    for frame in reference_frames:
        if frame.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["value"],
                mode="lines",
                line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                name=frame["name"].iloc[0],
                line={"color": "rgba(95, 99, 104, 0.7)", "width": style["reference_width"], "dash": "dot"},
            )
        )

    animation_frames = []
    for frame in frames:
        sweep_value = frame["sweep_value"].iloc[0]
        animation_frames.append(
            go.Frame(
                name=f"{sweep_value:.6g}",
                data=[
                    go.Scatter(
                        x=frame["time"],
                        y=frame["value"],
                        mode="lines",
                        line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                        name=f"{parameter_label}={sweep_value:.3g}",
                        line={"color": style["colors"][0], "width": style["line_width"]},
                    )
                ],
            )
        )
    fig.frames = animation_frames
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": f"{parameter_label}="},
                "steps": [
                    {
                        "label": frame.name,
                        "method": "animate",
                        "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for frame in animation_frames
                ],
            }
        ],
    )
    if log_y:
        fig.update_yaxes(type="log")
    return apply_plot_style(fig, style)


def all_objects(uploaded_objects: list[PlotObject]) -> list[PlotObject]:
    return st.session_state.get("simulations", []) + uploaded_objects


def run_simulation(name: str, params: dict) -> PlotObject:
    theta = np.array([params["eta"], params["beta"], params["epsilon"], params["xc"]], dtype=float)
    sim = getSrHetro(
        theta,
        kappa=params["kappa"],
        n=params["npeople"],
        nsteps=params["nsteps"],
        t_end=params["t_end"],
        external_hazard=params["external_hazard"],
        time_step_multiplier=params["time_step_multiplier"],
        eta_var=params["eta_var"],
        beta_var=params["beta_var"],
        epsilon_var=params["epsilon_var"],
        xc_var=params["xc_var"],
        kappa_var=params["kappa_var"],
        hetro=params["hetro"],
        method=params["method"],
        parallel=False,
    )
    return PlotObject(name=name, kind="simulation", source=sim, time_unit=params["time_unit"], params=params.copy())


@st.cache_data(show_spinner=False)
def run_cached_preview(name: str, frozen_params: tuple) -> PlotObject:
    params = dict(frozen_params)
    return run_simulation(name, params)


def editable_params_from_simulation(sim: PlotObject) -> dict:
    params = getattr(sim, "params", None)
    if params:
        return params.copy()

    source = sim.source
    external_hazard = _finite_float(getattr(source, "external_hazard", 30.0), default=30.0)
    return {
        "eta": float(getattr(source, "eta", 0.0)),
        "beta": float(getattr(source, "beta", 1.0)),
        "kappa": float(getattr(source, "kappa", 0.5)),
        "epsilon": float(getattr(source, "epsilon", 0.0)),
        "xc": float(getattr(source, "xc", 1.0)),
        "external_hazard": external_hazard,
        "external_hazard_exponent": external_hazard,
        "time_unit": object_time_unit(sim),
        "hetro": any(
            float(getattr(source, attr, 0.0)) > 0
            for attr in ("eta_var", "beta_var", "kappa_var", "epsilon_var", "xc_var")
        ),
        "eta_var": float(getattr(source, "eta_var", 0.0)),
        "beta_var": float(getattr(source, "beta_var", 0.0)),
        "kappa_var": float(getattr(source, "kappa_var", 0.0)),
        "epsilon_var": float(getattr(source, "epsilon_var", 0.0)),
        "xc_var": float(getattr(source, "xc_var", 0.0)),
        "npeople": int(getattr(source, "npeople", 5000)),
        "nsteps": int(getattr(source, "nsteps", 5000)),
        "t_end": int(getattr(source, "t_end", 100)),
        "time_step_multiplier": int(getattr(source, "time_step_multiplier", 1)),
        "method": getattr(source, "method", "brownian_bridge"),
    }


def set_draft_state(params: dict, name: str | None = None, selected_index: int | None = None) -> None:
    params = params.copy()
    params["external_hazard"] = float(params.get("external_hazard_exponent", params.get("external_hazard", 30.0)))
    params["external_hazard_exponent"] = float(params["external_hazard"])
    st.session_state["draft_params"] = params
    st.session_state["selected_sim_index"] = selected_index
    if name is not None:
        st.session_state["draft_name"] = name
        st.session_state["draft_name_input"] = name
    for key, spec in CORE_PARAMETER_SPECS.items():
        value = clamp(params[key], spec["min"], spec["max"])
        st.session_state[f"{key}_exact"] = value
        st.session_state[f"{key}_log"] = float(np.log10(value))
    st.session_state["external_hazard_exact"] = float(params["external_hazard_exponent"])
    st.session_state["external_hazard_slider"] = float(params["external_hazard_exponent"])
    for key in ["hetro", "eta_var", "beta_var", "kappa_var", "epsilon_var", "xc_var", "npeople", "nsteps", "t_end", "time_step_multiplier", "method"]:
        st.session_state[f"draft_{key}"] = params[key]


def queue_draft_state(params: dict, name: str | None = None, selected_index: int | None = None) -> None:
    st.session_state["pending_draft_state"] = (params.copy(), name, selected_index)


def sync_exact_from_log(parameter: str) -> None:
    spec = CORE_PARAMETER_SPECS[parameter]
    st.session_state[f"{parameter}_exact"] = float(10 ** st.session_state[f"{parameter}_log"])
    st.session_state[f"{parameter}_exact"] = clamp(st.session_state[f"{parameter}_exact"], spec["min"], spec["max"])


def sync_log_from_exact(parameter: str) -> None:
    spec = CORE_PARAMETER_SPECS[parameter]
    value = clamp(st.session_state[f"{parameter}_exact"], spec["min"], spec["max"])
    st.session_state[f"{parameter}_exact"] = value
    st.session_state[f"{parameter}_log"] = float(np.log10(value))


def sync_external_exact_from_slider() -> None:
    st.session_state["external_hazard_exact"] = float(st.session_state["external_hazard_slider"])


def sync_external_slider_from_exact() -> None:
    value = clamp(st.session_state["external_hazard_exact"], 0.0, 60.0)
    st.session_state["external_hazard_exact"] = value
    st.session_state["external_hazard_slider"] = value


def render_log_parameter_control(parameter: str) -> float:
    spec = CORE_PARAMETER_SPECS[parameter]
    left, right = st.columns([0.62, 0.38])
    left.slider(
        spec["label"],
        min_value=float(np.log10(spec["min"])),
        max_value=float(np.log10(spec["max"])),
        step=0.01,
        key=f"{parameter}_log",
        format="10^%.2f",
        on_change=sync_exact_from_log,
        args=(parameter,),
    )
    right.number_input(
        f"{spec['label']} exact",
        min_value=spec["min"],
        max_value=spec["max"],
        format="%.8g",
        key=f"{parameter}_exact",
        on_change=sync_log_from_exact,
        args=(parameter,),
    )
    return float(st.session_state[f"{parameter}_exact"])


def draft_params_from_controls() -> dict:
    params = st.session_state["draft_params"].copy()
    for key in CORE_PARAMETER_SPECS:
        params[key] = float(st.session_state[f"{key}_exact"])
    params["external_hazard"] = float(st.session_state["external_hazard_exact"])
    params["external_hazard_exponent"] = float(st.session_state["external_hazard_exact"])
    params["hetro"] = bool(st.session_state["draft_hetro"])
    for key in ["eta_var", "beta_var", "kappa_var", "epsilon_var", "xc_var"]:
        params[key] = float(st.session_state[f"draft_{key}"])
    params["npeople"] = int(st.session_state["draft_npeople"])
    params["nsteps"] = int(st.session_state["draft_nsteps"])
    params["t_end"] = int(st.session_state["draft_t_end"])
    params["time_step_multiplier"] = int(st.session_state["draft_time_step_multiplier"])
    params["method"] = st.session_state["draft_method"]
    params["time_unit"] = normalize_time_unit(params.get("time_unit"), default="days")
    st.session_state["draft_params"] = params
    st.session_state["draft_name"] = st.session_state["draft_name_input"]
    return params


def render_model_parameter_controls() -> None:
    for parameter in CORE_PARAMETER_SPECS:
        render_log_parameter_control(parameter)


def render_external_hazard_controls() -> None:
    eh_left, eh_right = st.columns([0.62, 0.38])
    eh_left.slider(
        "external hazard exponent",
        min_value=0.0,
        max_value=60.0,
        step=0.1,
        key="external_hazard_slider",
        on_change=sync_external_exact_from_slider,
        help="The model uses an external hazard rate of exp(-x). Enter 0 for the maximum rate; larger values weaken it.",
    )
    eh_right.number_input(
        "external exact",
        min_value=0.0,
        max_value=60.0,
        format="%.8g",
        key="external_hazard_exact",
        on_change=sync_external_slider_from_exact,
    )
    st.caption(f"External hazard rate = exp(-x) = {display_external_hazard(st.session_state['external_hazard_exact'])}")


def render_heterogeneity_controls() -> None:
    st.checkbox("Enable heterogeneity", key="draft_hetro")
    h1, h2 = st.columns(2)
    h1.number_input("eta_var", min_value=0.0, format="%.4g", key="draft_eta_var")
    h2.number_input("beta_var", min_value=0.0, format="%.4g", key="draft_beta_var")
    h1.number_input("kappa_var", min_value=0.0, format="%.4g", key="draft_kappa_var")
    h2.number_input("epsilon_var", min_value=0.0, format="%.4g", key="draft_epsilon_var")
    h1.number_input("xc_var", min_value=0.0, format="%.4g", key="draft_xc_var")


def render_runtime_controls(show_sample_size: bool = True) -> None:
    r1, r2 = st.columns(2)
    if show_sample_size:
        r1.number_input("sample size", min_value=100, max_value=25_000, step=100, key="draft_npeople")
    max_nsteps = max(100, MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE // int(st.session_state["draft_npeople"]))
    if st.session_state["draft_nsteps"] > max_nsteps:
        st.session_state["draft_nsteps"] = max_nsteps
    r2.number_input("steps", min_value=1, max_value=max_nsteps, step=1, key="draft_nsteps")
    r1.number_input("end time", min_value=1, max_value=250, step=1, key="draft_t_end")
    r2.number_input(
        "step multiplier",
        min_value=1,
        max_value=MAX_TIME_STEP_MULTIPLIER,
        step=1,
        key="draft_time_step_multiplier",
    )
    st.selectbox("method", ["brownian_bridge", "euler"], key="draft_method")


st.set_page_config(page_title="SRtools Explorer", layout="wide")
st.title("SRtools Explorer")
st.markdown(
    """
    Simulate survival from organism presets, upload observed death-time or life-table data, and compare several simulations or datasets on the same survival, hazard, and death-time distribution plots.

    The Saturating Removal model represents aging as accumulated damage. Damage production grows with time through `eta`, repair/removal is controlled by `beta` and saturates with `kappa`, stochastic variation is controlled by `epsilon`, and death occurs when damage crosses the threshold `xc` or through the optional external hazard. Heterogeneity terms let these parameters vary across individuals in a simulated cohort.
    """
)

st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        min-height: 2.6rem;
        white-space: normal;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "simulations" not in st.session_state:
    st.session_state["simulations"] = []

preset_catalog = load_preset_catalog()
preset_by_id = {preset["id"]: preset for preset in preset_catalog}

with st.sidebar:
    st.header("Controls")
    visualization_style = st.radio("Visualization style", ["Friendly", "Scientific"], horizontal=True)
    style_config = STYLE_CONFIGS[visualization_style]

    st.subheader("Preset")
    use_shortlist = st.checkbox("Shortlist", value=True)
    visible_presets = visible_presets_for_mode(preset_catalog, visualization_style, use_shortlist)
    preset_id = st.selectbox(
        "Organism preset",
        [preset["id"] for preset in visible_presets],
        index=0,
        format_func=lambda value: preset_display_name(preset_by_id[value], visualization_style, use_shortlist),
    )
    selected_preset = preset_by_id[preset_id]
    selected_preset_display = preset_display_name(selected_preset, visualization_style, use_shortlist)
    defaults = load_preset_defaults(selected_preset["preset_name"], selected_preset["source"], selected_preset["time_unit"])
    preset_draft = params_from_defaults(defaults, selected_preset["time_unit"])

    pending_draft_state = st.session_state.pop("pending_draft_state", None)
    if pending_draft_state is not None:
        set_draft_state(*pending_draft_state)
    elif "draft_params" not in st.session_state:
        set_draft_state(preset_draft, selected_preset_display, None)

    if st.button("Load preset values"):
        set_draft_state(preset_draft, selected_preset_display, None)
        st.rerun()

    st.divider()
    st.subheader("Draft simulation")
    selected_sim_index = st.session_state.get("selected_sim_index")
    selected_sim = (
        st.session_state["simulations"][selected_sim_index]
        if selected_sim_index is not None and 0 <= selected_sim_index < len(st.session_state["simulations"])
        else None
    )
    if selected_sim is not None:
        st.caption(f"Editing saved simulation: {selected_sim.name}")
    else:
        st.caption(f"Preset: {selected_preset_display}; native time unit: {st.session_state['draft_params']['time_unit']}")

    st.text_input("Simulation name", key="draft_name_input")

    if visualization_style == "Friendly":
        st.number_input("Sample size", min_value=100, max_value=25_000, step=100, key="draft_npeople")
        st.caption("The preview updates automatically. Save a full simulation when you like the settings.")
        with st.expander("Model parameters", expanded=False):
            render_model_parameter_controls()
        with st.expander("Advanced simulation settings", expanded=False):
            render_external_hazard_controls()
            st.divider()
            render_heterogeneity_controls()
            st.divider()
            render_runtime_controls(show_sample_size=False)
    else:
        render_model_parameter_controls()
        st.divider()
        render_external_hazard_controls()
        st.divider()
        render_heterogeneity_controls()
        st.divider()
        render_runtime_controls(show_sample_size=True)

    draft_params = draft_params_from_controls()
    work_units = int(draft_params["npeople"]) * int(draft_params["nsteps"])
    preview_work_units = min(draft_params["npeople"], PREVIEW_SIMULATION_MAX_SAMPLE_SIZE) * preview_params(draft_params)["nsteps"]
    st.caption(f"Full work units: {work_units:,} / {MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE:,}")
    st.caption(f"Preview work units: {preview_work_units:,}")

    replace_existing = st.checkbox("Replace selected simulation", value=True, disabled=selected_sim is None)
    if st.button("Run full simulation", type="primary"):
        if work_units > MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE:
            st.error("Reduce sample size or steps before running.")
        else:
            sim_name = st.session_state["draft_name_input"]
            with st.spinner("Running full simulation..."):
                result = run_simulation(sim_name, draft_params)
            if selected_sim is not None and replace_existing:
                st.session_state["simulations"][selected_sim_index] = result
                queue_draft_state(result.params, result.name, selected_sim_index)
                st.success(f"Updated {sim_name}")
            else:
                st.session_state["simulations"].append(result)
                queue_draft_state(result.params, result.name, len(st.session_state["simulations"]) - 1)
                st.success(f"Added {sim_name}")
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Add as copy"):
        if work_units > MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE:
            st.error("Reduce sample size or steps before running.")
        else:
            copy_name = f"{st.session_state['draft_name_input']} copy"
            copy_params = draft_params.copy()
            with st.spinner("Running full simulation..."):
                st.session_state["simulations"].append(run_simulation(copy_name, copy_params))
            queue_draft_state(copy_params, copy_name, len(st.session_state["simulations"]) - 1)
            st.rerun()
    if st.session_state["simulations"] and c2.button("Clear saved"):
        st.session_state["simulations"] = []
        st.session_state["selected_sim_index"] = None
        st.rerun()

draft_params = st.session_state["draft_params"]
preview_object = run_cached_preview("Live preview", params_cache_key(preview_params(draft_params)))
preview_object.kind = "preview"

st.header("Uploaded Data")
uploads = st.file_uploader("Upload CSV or XLSX files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
uploaded_objects: list[PlotObject] = []

for uploaded in uploads:
    with st.expander(uploaded.name, expanded=False):
        sheet_names = excel_sheet_names(uploaded)
        sheet = st.selectbox("Sheet", sheet_names, key=f"{uploaded.name}-sheet") if sheet_names else None
        df = read_uploaded_table(uploaded, sheet_name=sheet)
        st.dataframe(df.head(20), width="stretch")
        mode = st.radio("Data type", ["Death times", "Life table"], horizontal=True, key=f"{uploaded.name}-mode")
        columns = list(df.columns)
        display_name = st.text_input("Display name", value=uploaded.name, key=f"{uploaded.name}-name")
        upload_time_unit = st.selectbox("Time unit", TIME_UNITS, index=TIME_UNITS.index("days"), key=f"{uploaded.name}-unit")

        if mode == "Death times":
            default_time_index = columns.index("death_times") if "death_times" in columns else 0
            time_col = st.selectbox("Death time column", columns, index=default_time_index, key=f"{uploaded.name}-time")
            event_options = ["All observed deaths"] + columns
            event_col_choice = st.selectbox("Event/censor column", event_options, key=f"{uploaded.name}-event")
            event_is_censored = st.checkbox("Selected column is censor indicator", value=False, key=f"{uploaded.name}-censor")
            event_col = None if event_col_choice == "All observed deaths" else event_col_choice
            uploaded_objects.append(PlotObject(display_name, "data", make_dataset(df, time_col, event_col, event_is_censored), upload_time_unit))
        else:
            age_col = st.selectbox("Age column", columns, index=0, key=f"{uploaded.name}-age")
            alive_col = st.selectbox("Number alive column", columns, index=min(1, len(columns) - 1), key=f"{uploaded.name}-alive")
            hazard_options = ["No hazard column"] + columns
            hazard_choice = st.selectbox("Hazard column", hazard_options, key=f"{uploaded.name}-hazard")
            hazard_col = None if hazard_choice == "No hazard column" else hazard_choice
            uploaded_objects.append(PlotObject(display_name, "life table", make_life_table(df, age_col, alive_col, hazard_col), upload_time_unit))

saved_objects = all_objects(uploaded_objects)
objects = [preview_object] + saved_objects

st.header("Plots")
if st.session_state["simulations"]:
    st.subheader("Saved simulations")
    card_columns = st.columns(min(3, len(st.session_state["simulations"])))
    for index, sim in enumerate(st.session_state["simulations"]):
        params = editable_params_from_simulation(sim)
        label = (
            f"{index + 1}. {sim.name}\n"
            f"eta={params['eta']:.3g}, beta={params['beta']:.3g}, kappa={params['kappa']:.3g}, xc={params['xc']:.3g}"
        )
        if index == st.session_state.get("selected_sim_index"):
            label = "Editing: " + label
        if card_columns[index % len(card_columns)].button(label, key=f"sim-card-{index}"):
            queue_draft_state(params, sim.name, index)
            st.rerun()

names = [obj.name for obj in objects]
active_names = st.multiselect("Series", names, default=names)
active_objects = [obj for obj in objects if obj.name in active_names]

default_plot_unit = object_time_unit(active_objects[0]) if active_objects else "days"
plot_time_unit = st.selectbox("Plot time unit", TIME_UNITS, index=TIME_UNITS.index(default_plot_unit))
max_t = (
    max(float(convert_duration(float(getattr(obj.source, "t_end", 1.0)), object_time_unit(obj), plot_time_unit)) for obj in active_objects)
    if active_objects
    else 1.0
)
plot_controls = st.columns(6)
range_start = plot_controls[0].number_input("Range start", min_value=0.0, value=0.0)
range_end = plot_controls[1].number_input("Range end", min_value=max(range_start + 1.0, 1.0), value=max(max_t, range_start + 1.0))
bin_width = plot_controls[2].number_input("Death-time bin width", min_value=0.001, value=1.0)
log_y = plot_controls[3].checkbox("Log y-axis", value=False)
renormalize_survival = plot_controls[4].checkbox("Renormalize survival", value=False)
scale_survival_by_median = plot_controls[5].checkbox("Scale survival by median", value=False)
time_range = (float(range_start), float(range_end))

plot_types = st.multiselect(
    "Plot types",
    ["Survival", "Hazard", "Death-time distribution"],
    default=["Survival", "Hazard", "Death-time distribution"],
)

download_frames: list[pd.DataFrame] = []
if "Survival" in plot_types:
    frames = [
        survival_frame(
            obj,
            time_range,
            plot_time_unit,
            renormalize=renormalize_survival,
            scale_by_median=scale_survival_by_median,
        )
        for obj in active_objects
    ]
    download_frames.extend(frames)
    y_title = "Conditional survival probability" if renormalize_survival else "Survival probability"
    x_title = "Time / median lifetime" if scale_survival_by_median else f"Time ({plot_time_unit})"
    fig = plot_lines(frames, "Survival", y_title, x_title=x_title, log_y=False, style=style_config)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download survival plot HTML", fig.to_html(), file_name="survival.html", mime="text/html")

if "Hazard" in plot_types:
    frames = [hazard_frame(obj, time_range, plot_time_unit) for obj in active_objects]
    download_frames.extend(frames)
    fig = plot_lines(frames, "Hazard", f"Hazard (per {plot_time_unit})", x_title=f"Time ({plot_time_unit})", log_y=log_y, style=style_config)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download hazard plot HTML", fig.to_html(), file_name="hazard.html", mime="text/html")

if "Death-time distribution" in plot_types:
    frames = [death_distribution_frame(obj, time_range, float(bin_width), plot_time_unit) for obj in active_objects]
    download_frames.extend(frames)
    fig = plot_lines(frames, "Death-time Distribution", "Probability of death", x_title=f"Time ({plot_time_unit})", log_y=log_y, style=style_config)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download death distribution plot HTML", fig.to_html(), file_name="death_distribution.html", mime="text/html")

effect_title = "Explore one parameter" if visualization_style == "Friendly" else "Parameter effect"
with st.expander(effect_title, expanded=False):
    if visualization_style == "Friendly":
        st.caption("Pick one setting and see how the curves change across a range.")
    else:
        st.caption("Generate preview-resolution simulations across one parameter range.")
    base_options = ["Current draft"] + [f"Saved {index + 1}: {sim.name}" for index, sim in enumerate(st.session_state["simulations"])]
    base_choice = st.selectbox("Base", base_options)
    if base_choice == "Current draft":
        base_params = draft_params.copy()
        base_name = st.session_state["draft_name"]
    else:
        saved_index = base_options.index(base_choice) - 1
        base_sim = st.session_state["simulations"][saved_index]
        base_params = editable_params_from_simulation(base_sim)
        base_name = base_sim.name

    sweep_cols = st.columns(4)
    sweep_parameter = sweep_cols[0].selectbox("Parameter", list(SWEEP_PARAMETER_LABELS.keys()), format_func=lambda key: SWEEP_PARAMETER_LABELS[key])
    current_sweep_value = float(base_params[sweep_parameter])
    if sweep_parameter == "external_hazard_exponent":
        default_min = max(0.0, current_sweep_value - 5.0)
        default_max = min(60.0, current_sweep_value + 5.0)
        sweep_min = sweep_cols[1].number_input("Minimum", min_value=0.0, max_value=60.0, value=float(default_min), format="%.8g")
        sweep_max = sweep_cols[2].number_input("Maximum", min_value=0.0, max_value=60.0, value=float(max(default_max, default_min + 0.1)), format="%.8g")
    else:
        spec = CORE_PARAMETER_SPECS[sweep_parameter]
        default_min = clamp(current_sweep_value / 2.0, spec["min"], spec["max"])
        default_max = clamp(current_sweep_value * 2.0, spec["min"], spec["max"])
        if default_max <= default_min:
            default_max = clamp(default_min * 10.0, spec["min"], spec["max"])
        sweep_min = sweep_cols[1].number_input("Minimum", min_value=spec["min"], max_value=spec["max"], value=float(default_min), format="%.8g")
        sweep_max = sweep_cols[2].number_input("Maximum", min_value=spec["min"], max_value=spec["max"], value=float(default_max), format="%.8g")
    sweep_count = sweep_cols[3].number_input("Steps", min_value=3, max_value=20, value=7, step=1)

    sweep_view_cols = st.columns(3)
    sweep_plot_kind = sweep_view_cols[0].selectbox("Plot", ["Survival", "Hazard", "Death-time distribution"])
    show_references = sweep_view_cols[1].checkbox("Show selected series as reference", value=True)
    generate_sweep = sweep_view_cols[2].button("Generate effect", type="primary")

    if generate_sweep:
        if sweep_max <= sweep_min:
            st.error("Maximum must be larger than minimum.")
        else:
            if sweep_parameter == "external_hazard_exponent":
                sweep_values = np.linspace(float(sweep_min), float(sweep_max), int(sweep_count))
            else:
                sweep_values = np.geomspace(float(sweep_min), float(sweep_max), int(sweep_count))
            sweep_frames = []
            with st.spinner("Generating parameter effect..."):
                for value in sweep_values:
                    params = base_params.copy()
                    params[sweep_parameter] = float(value)
                    if sweep_parameter == "external_hazard_exponent":
                        params["external_hazard"] = float(value)
                    sweep_obj = run_cached_preview(
                        f"{base_name}: {SWEEP_PARAMETER_LABELS[sweep_parameter]}={value:.3g}",
                        params_cache_key(preview_params(params)),
                    )
                    sweep_obj.kind = "sweep"
                    frame = frame_for_plot(
                        sweep_obj,
                        sweep_plot_kind,
                        time_range,
                        plot_time_unit,
                        float(bin_width),
                        renormalize=renormalize_survival,
                        scale_by_median=scale_survival_by_median,
                    )
                    frame["sweep_value"] = float(value)
                    sweep_frames.append(frame)

            reference_frames = []
            if show_references:
                for obj in active_objects:
                    if obj.kind == "preview":
                        continue
                    reference_frames.append(
                        frame_for_plot(
                            obj,
                            sweep_plot_kind,
                            time_range,
                            plot_time_unit,
                            float(bin_width),
                            renormalize=renormalize_survival,
                            scale_by_median=scale_survival_by_median,
                        )
                    )

            if sweep_plot_kind == "Survival":
                sweep_y_title = "Conditional survival probability" if renormalize_survival else "Survival probability"
                sweep_x_title = "Time / median lifetime" if scale_survival_by_median else f"Time ({plot_time_unit})"
            elif sweep_plot_kind == "Hazard":
                sweep_y_title = f"Hazard (per {plot_time_unit})"
                sweep_x_title = f"Time ({plot_time_unit})"
            else:
                sweep_y_title = "Probability of death"
                sweep_x_title = f"Time ({plot_time_unit})"
            sweep_title = f"{sweep_plot_kind}: changing {SWEEP_PARAMETER_LABELS[sweep_parameter]}"
            overlay_fig = plot_sweep_overlay(
                sweep_frames,
                reference_frames,
                SWEEP_PARAMETER_LABELS[sweep_parameter],
                sweep_title,
                sweep_y_title,
                sweep_x_title,
                log_y if sweep_plot_kind != "Survival" else False,
                style_config,
            )
            st.plotly_chart(overlay_fig, width="stretch")
            st.download_button(
                "Download parameter overlay HTML",
                overlay_fig.to_html(),
                file_name="parameter_effect_overlay.html",
                mime="text/html",
            )

            animation_fig = plot_sweep_animation(
                sweep_frames,
                reference_frames,
                SWEEP_PARAMETER_LABELS[sweep_parameter],
                f"Animated {sweep_title}",
                sweep_y_title,
                sweep_x_title,
                log_y if sweep_plot_kind != "Survival" else False,
                style_config,
            )
            st.plotly_chart(animation_fig, width="stretch")
            st.download_button(
                "Download parameter animation HTML",
                animation_fig.to_html(),
                file_name="parameter_effect_animation.html",
                mime="text/html",
            )

if download_frames:
    csv = pd.concat(download_frames, ignore_index=True).to_csv(index=False)
    st.download_button("Download plotted data CSV", csv, file_name="srtools_plot_data.csv", mime="text/csv")
