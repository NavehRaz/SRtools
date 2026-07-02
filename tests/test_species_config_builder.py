"""Tests for species_config_builder: column format + Excel round-trip."""

import numpy as np
import pytest

from SRtools import auto_initial_guess as aig
from SRtools import species_config_builder as scb
from SRtools.config_lib import ExcelConfigParser, config_to_dict


def _make_result(species, theta, has_ext=False, exth=None, ndims=4):
    return aig.FitResult(
        species=species, theta=theta, has_ext=has_ext, exth=exth, ndims=ndims,
        t_end=2970.4, nsteps=5000, time_step_multiplier=3,
        median_dev=0.06, steep_dev=0.09, status="ok",
        data_ml=1273.0, data_steepness=1.38,
    )


def test_build_config_column_format():
    r = _make_result("Test_sp", [0.0002, 0.76, 1.5, 16.7])
    col = scb.build_config_column(r, "Mammalia", base_prefix="Species360_Calibration")
    assert list(col.index) == scb.ROW_ORDER
    assert col["data_file"] == "Species360_Calibration/datasets/Mammalia/Test_sp"  # no extension
    assert col["folder"] == "Species360_Calibration/simulation_results/Mammalia/Test_sp"
    assert col["results_csv_file_name"] == "Test_sp.csv"  # csv only here
    assert col["t_end"] == 2970  # rounded int
    assert col["ndims"] == 4
    assert col["ExtH"] == "" and col["external_hazard"] == ""


def test_build_config_column_external_hazard():
    r = _make_result("Ext_sp", [0.0002, 0.76, 1.5, 16.7], has_ext=True, exth=4.2, ndims=5)
    col = scb.build_config_column(r, "Aves")
    assert col["ndims"] == 5
    assert col["ExtH"] == 4.2 and col["external_hazard"] == 4.2


def test_build_config_column_none_when_no_theta():
    r = aig.FitResult(species="bad", theta=None, status="insufficient_data")
    assert scb.build_config_column(r, "Mammalia") is None


def test_excel_roundtrip_via_configparser(tmp_path):
    rA = _make_result("Species_A", [0.0002, 0.76, 1.5, 16.7])
    rB = _make_result("Species_B", [0.0003, 0.50, 2.0, 20.0])
    cols = [scb.build_config_column(rA, "Mammalia"), scb.build_config_column(rB, "Aves")]
    qc = [scb.build_qc_row(rA), scb.build_qc_row(rB)]
    out = tmp_path / "configurations.xlsx"
    scb.write_configurations_excel(cols, str(out), qc_rows=qc)

    # Round-trip Species_A
    cfg = ExcelConfigParser(str(out), "Species_A")
    d = config_to_dict(cfg, mcmc_convert=True)
    assert d["nsteps"] == 5000 and isinstance(d["nsteps"], int)
    assert d["ndims"] == 4 and isinstance(d["ndims"], int)
    assert isinstance(d["t_end"], int) and d["t_end"] == 2970
    assert d["hetro"] == 0
    assert d["time_step_multiplier"] == 3
    # blank external_hazard -> absent or None (never a crash)
    assert d.get("external_hazard") in (None, "", float("nan")) or "external_hazard" not in d
    assert d["data_file"] == "Species360_Calibration/datasets/Mammalia/Species_A"

    # The QC sheet must not disturb parsing of either column.
    cfgB = ExcelConfigParser(str(out), "Species_B")
    dB = config_to_dict(cfgB, mcmc_convert=True)
    assert dB["name"] == "Species_B" and dB["nsteps"] == 5000


def test_write_excel_skips_none_columns(tmp_path):
    r = _make_result("Only_sp", [0.0002, 0.76, 1.5, 16.7])
    out = tmp_path / "c.xlsx"
    scb.write_configurations_excel([scb.build_config_column(r, "Mammalia"), None], str(out))
    cfg = ExcelConfigParser(str(out), "Only_sp")
    assert config_to_dict(cfg)["name"] == "Only_sp"


def test_write_excel_raises_when_all_none(tmp_path):
    with pytest.raises(ValueError):
        scb.write_configurations_excel([None], str(tmp_path / "x.xlsx"))


def test_build_config_column_writes_time_range():
    r = _make_result("Win_sp", [0.0002, 0.76, 1.5, 16.7])
    r.t_min = 500.0
    r.time_range = [500, 7000]
    col = scb.build_config_column(r, "Mammalia")
    assert col["time_range"] == "[500, 7000]"


def test_time_range_roundtrips_to_int_list(tmp_path):
    r = _make_result("Win_sp", [0.0002, 0.76, 1.5, 16.7])
    r.time_range = [500, 7000]
    out = tmp_path / "c.xlsx"
    scb.write_configurations_excel([scb.build_config_column(r, "Mammalia")], str(out),
                                   qc_rows=[scb.build_qc_row(r)])
    d = config_to_dict(ExcelConfigParser(str(out), "Win_sp"))
    assert d["time_range"] == [500, 7000]
    assert all(isinstance(x, int) for x in d["time_range"])


def test_blank_time_range_roundtrips_none(tmp_path):
    r = _make_result("No_win", [0.0002, 0.76, 1.5, 16.7])  # time_range stays None
    out = tmp_path / "c.xlsx"
    scb.write_configurations_excel([scb.build_config_column(r, "Mammalia")], str(out))
    d = config_to_dict(ExcelConfigParser(str(out), "No_win"))
    assert d.get("time_range") in (None, "") or "time_range" not in d
