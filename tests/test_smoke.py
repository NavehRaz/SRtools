import importlib.resources as resources

import numpy as np

import SRtools
from SRtools import Dataset, Life_table
from SRtools.SR_hetro import getSrHetro
from SRtools import presets


def test_public_imports_work():
    assert "Dataset" in SRtools.__all__
    assert Dataset is not None
    assert Life_table is not None


def test_preset_csvs_are_packaged():
    preset_file = resources.files("SRtools").joinpath("Preset_values", "summery_mode_overall.csv")
    config_file = resources.files("SRtools").joinpath("Preset_values", "All_config.csv")
    assert preset_file.is_file()
    assert config_file.is_file()
    assert len(presets.get_preset_names()) > 0


def test_dataset_and_life_table_smoke():
    ds = Dataset(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, dtype=int))
    t, s = ds.getSurvival()
    ht, h = ds.getHazard()
    assert len(t) > 0
    assert len(s) == len(t)
    assert len(h) == len(ht)

    lt = Life_table(np.array([0.0, 1.0, 2.0, 3.0]), np.array([10, 8, 5, 1]))
    lt_t, lt_s = lt.getSurvival()
    assert len(lt_t) == len(lt_s)
    assert lt_s[0] == 1.0


def test_tiny_heterogeneous_simulation_smoke():
    sim = getSrHetro(
        np.array([5.0, 1.0, 0.5, 2.0]),
        n=20,
        nsteps=200,
        t_end=5,
        time_step_multiplier=1,
        hetro=False,
        method="euler",
    )
    t, s = sim.getSurvival()
    ht, h = sim.getHazard()
    death_times = sim.getDeathTimes()
    assert len(t) > 0
    assert len(s) == len(t)
    assert len(h) == len(ht)
    assert len(death_times) > 0
