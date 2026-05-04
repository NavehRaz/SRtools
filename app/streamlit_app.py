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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SRtools import Dataset, Life_table
from SRtools import presets
from SRtools.SR_hetro import getSrHetro


MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE = 35_000_000
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


def object_time_unit(obj: PlotObject) -> str:
    return normalize_time_unit(getattr(obj, "time_unit", "days"), default="days")


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


def plot_lines(frames: list[pd.DataFrame], title: str, y_title: str, x_title: str = "Time", log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    for frame in frames:
        if frame.empty:
            continue
        label = frame["name"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["value"],
                mode="lines",
                line_shape="hv" if frame["plot"].iloc[0] == "death_distribution" else "linear",
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        legend_title_text="Series",
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


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


st.set_page_config(page_title="SRtools Explorer", layout="wide")
st.title("SRtools Explorer")
st.markdown(
    """
    Simulate survival from organism presets, upload observed death-time or life-table data, and compare several simulations or datasets on the same survival, hazard, and death-time distribution plots.

    The Saturating Removal model represents aging as accumulated damage. Damage production grows with time through `eta`, repair/removal is controlled by `beta` and saturates with `kappa`, stochastic variation is controlled by `epsilon`, and death occurs when damage crosses the threshold `xc` or through the optional external hazard. Heterogeneity terms let these parameters vary across individuals in a simulated cohort.
    """
)

if "simulations" not in st.session_state:
    st.session_state["simulations"] = []

preset_catalog = load_preset_catalog()
preset_by_id = {preset["id"]: preset for preset in preset_catalog}

with st.sidebar:
    st.header("Simulation")
    use_shortlist = st.checkbox("Shortlist", value=True)
    visible_presets = [preset for preset in preset_catalog if preset["shortlist"] or not use_shortlist]
    preset_id = st.selectbox(
        "Organism preset",
        [preset["id"] for preset in visible_presets],
        index=0,
        format_func=lambda value: preset_by_id[value]["alias"]
        if use_shortlist
        else f"{preset_by_id[value]['alias']} ({preset_by_id[value]['source']})",
    )
    selected_preset = preset_by_id[preset_id]
    preset_name = selected_preset["preset_name"]
    preset_time_unit = selected_preset["time_unit"]
    defaults = load_preset_defaults(preset_name, selected_preset["source"], preset_time_unit)
    edit_index = st.session_state.get("editing_sim_index")
    editing_sim = None
    if edit_index is not None and 0 <= edit_index < len(st.session_state["simulations"]):
        editing_sim = st.session_state["simulations"][edit_index]
    elif edit_index is not None:
        st.session_state.pop("editing_sim_index", None)
        st.session_state.pop("editing_sim_params", None)
        st.session_state.pop("editing_sim_name", None)

    if st.session_state["simulations"]:
        edit_options = ["New simulation"] + [
            f"{idx + 1}. {sim.name}" for idx, sim in enumerate(st.session_state["simulations"])
        ]
        selected_edit = st.selectbox("Edit previous simulation", edit_options)
        e1, e2 = st.columns(2)
        if e1.button("Load"):
            if selected_edit == "New simulation":
                st.session_state.pop("editing_sim_index", None)
                st.session_state.pop("editing_sim_params", None)
                st.session_state.pop("editing_sim_name", None)
            else:
                selected_index = edit_options.index(selected_edit) - 1
                selected_sim = st.session_state["simulations"][selected_index]
                st.session_state["editing_sim_index"] = selected_index
                st.session_state["editing_sim_params"] = editable_params_from_simulation(selected_sim)
                st.session_state["editing_sim_name"] = selected_sim.name
            st.rerun()
        if e2.button("Preset"):
            st.session_state.pop("editing_sim_index", None)
            st.session_state.pop("editing_sim_params", None)
            st.session_state.pop("editing_sim_name", None)
            st.rerun()

    form_defaults = defaults.copy()
    form_defaults.update(st.session_state.get("editing_sim_params", {}))
    form_time_unit = normalize_time_unit(form_defaults.get("time_unit"), default=preset_time_unit)
    form_name = st.session_state.get("editing_sim_name", selected_preset["alias"])

    with st.form("simulation_form"):
        sim_name = st.text_input("Simulation name", value=form_name)
        if editing_sim is not None:
            st.caption(f"Editing: {editing_sim.name}; native time unit: {form_time_unit}")
        else:
            st.caption(f"Preset: {preset_name}; native time unit: {preset_time_unit}")
        c1, c2 = st.columns(2)
        eta = c1.number_input("eta", min_value=0.0, value=float(form_defaults["eta"]), format="%.8g")
        beta = c2.number_input("beta", min_value=0.000001, value=float(form_defaults["beta"]), format="%.8g")
        kappa = c1.number_input("kappa", min_value=0.000001, value=float(form_defaults["kappa"]), format="%.8g")
        epsilon = c2.number_input("epsilon", min_value=0.0, value=float(form_defaults["epsilon"]), format="%.8g")
        xc = c1.number_input("xc", min_value=0.000001, value=float(form_defaults["xc"]), format="%.8g")
        external_hazard_exponent = st.number_input(
            "external hazard exponent x",
            min_value=0.0,
            value=float(form_defaults["external_hazard_exponent"]),
            format="%.8g",
            help="The model uses an external hazard rate of exp(-x). Enter 0 for the maximum rate; larger values weaken it.",
        )
        st.caption(f"External hazard rate = exp(-x) = {display_external_hazard(external_hazard_exponent)}")

        st.divider()
        hetro = st.checkbox("Enable heterogeneity", value=bool(form_defaults["hetro"]))
        h1, h2 = st.columns(2)
        eta_var = h1.number_input("eta_var", min_value=0.0, value=float(form_defaults.get("eta_var", 0.0)), format="%.4g")
        beta_var = h2.number_input("beta_var", min_value=0.0, value=float(form_defaults.get("beta_var", 0.0)), format="%.4g")
        kappa_var = h1.number_input("kappa_var", min_value=0.0, value=float(form_defaults.get("kappa_var", 0.0)), format="%.4g")
        epsilon_var = h2.number_input("epsilon_var", min_value=0.0, value=float(form_defaults.get("epsilon_var", 0.0)), format="%.4g")
        xc_var = h1.number_input("xc_var", min_value=0.0, value=float(form_defaults.get("xc_var", 0.2 if form_defaults["hetro"] else 0.0)), format="%.4g")

        st.divider()
        r1, r2 = st.columns(2)
        npeople = r1.number_input("sample size", min_value=100, max_value=25_000, value=min(int(form_defaults["npeople"]), 5_000), step=100)
        max_nsteps = max(100, MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE // int(npeople))
        nsteps = r2.number_input(
            "steps",
            min_value=1,
            max_value=max_nsteps,
            value=min(max(int(form_defaults["nsteps"]), 1), max_nsteps),
            step=1,
        )
        t_end = r1.number_input("end time", min_value=1, max_value=250, value=min(int(form_defaults["t_end"]), 250), step=1)
        time_step_multiplier = r2.number_input(
            "step multiplier",
            min_value=1,
            max_value=MAX_TIME_STEP_MULTIPLIER,
            value=min(int(form_defaults["time_step_multiplier"]), MAX_TIME_STEP_MULTIPLIER),
            step=1,
        )
        methods = ["brownian_bridge", "euler"]
        default_method = form_defaults.get("method", "brownian_bridge")
        if default_method not in methods:
            default_method = "brownian_bridge"
        method = st.selectbox("method", methods, index=methods.index(default_method))
        replace_existing = st.checkbox("Replace loaded simulation", value=True, disabled=editing_sim is None)
        submitted = st.form_submit_button("Run simulation" if editing_sim is None else "Run edited simulation")

    work_units = int(npeople) * int(nsteps)
    st.caption(
        f"Simulation steps x sample size: {work_units:,} / {MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE:,}"
    )
    if submitted:
        if work_units > MAX_SIMULATION_STEPS_TIMES_SAMPLE_SIZE:
            st.error("Reduce sample size or steps before running.")
        else:
            params = {
                "eta": eta,
                "beta": beta,
                "kappa": kappa,
                "epsilon": epsilon,
                "xc": xc,
                "external_hazard": external_hazard_exponent,
                "external_hazard_exponent": external_hazard_exponent,
                "time_unit": form_time_unit,
                "hetro": hetro,
                "eta_var": eta_var,
                "beta_var": beta_var,
                "epsilon_var": epsilon_var,
                "xc_var": xc_var,
                "kappa_var": kappa_var,
                "npeople": int(npeople),
                "nsteps": int(nsteps),
                "t_end": int(t_end),
                "time_step_multiplier": int(time_step_multiplier),
                "method": method,
            }
            with st.spinner("Running simulation..."):
                result = run_simulation(sim_name, params)
                if editing_sim is not None and replace_existing:
                    st.session_state["simulations"][edit_index] = result
                    st.session_state.pop("editing_sim_index", None)
                    st.session_state.pop("editing_sim_params", None)
                    st.session_state.pop("editing_sim_name", None)
                    st.success(f"Updated {sim_name}")
                else:
                    st.session_state["simulations"].append(result)
                    st.success(f"Added {sim_name}")

    if st.session_state["simulations"] and st.button("Clear simulations"):
        st.session_state["simulations"] = []

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

objects = all_objects(uploaded_objects)

st.header("Plots")
if not objects:
    st.info("Run a simulation or upload data to create plots.")
    st.stop()

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
    fig = plot_lines(frames, "Survival", y_title, x_title=x_title, log_y=False)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download survival plot HTML", fig.to_html(), file_name="survival.html", mime="text/html")

if "Hazard" in plot_types:
    frames = [hazard_frame(obj, time_range, plot_time_unit) for obj in active_objects]
    download_frames.extend(frames)
    fig = plot_lines(frames, "Hazard", f"Hazard (per {plot_time_unit})", x_title=f"Time ({plot_time_unit})", log_y=log_y)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download hazard plot HTML", fig.to_html(), file_name="hazard.html", mime="text/html")

if "Death-time distribution" in plot_types:
    frames = [death_distribution_frame(obj, time_range, float(bin_width), plot_time_unit) for obj in active_objects]
    download_frames.extend(frames)
    fig = plot_lines(frames, "Death-time Distribution", "Probability of death", x_title=f"Time ({plot_time_unit})", log_y=log_y)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download death distribution plot HTML", fig.to_html(), file_name="death_distribution.html", mime="text/html")

if download_frames:
    csv = pd.concat(download_frames, ignore_index=True).to_csv(index=False)
    st.download_button("Download plotted data CSV", csv, file_name="srtools_plot_data.csv", mime="text/csv")
