"""Lean tests for the initialParamsFinder.Guess improvements (tiny sims)."""

import numpy as np

from SRtools.SR_hetro import getSrHetro
from SRtools.deathTimesDataSet import Dataset
from SRtools.initialParamsFinder import Guess


def _tiny_dataset():
    sim = getSrHetro(np.array([5.0, 1.0, 0.5, 2.0]), n=400, nsteps=300, t_end=5,
                     time_step_multiplier=1, hetro=False, method="euler")
    dt = sim.getDeathTimes()
    return Dataset(np.asarray(dt, float), np.ones(len(dt), dtype=int))


def _guess(ds):
    return Guess([5.0, 1.0, 0.5, 2.0], ds, t_end=5, nsteps=300, npeople=400,
                 time_step_multiplier=1, parallel=False)


def test_make_guess_accepts_seed_and_early_stop():
    ds = _tiny_dataset()
    g = _guess(ds)
    calls = {"n": 0}

    def stop(_sim):
        calls["n"] += 1
        return True  # stop immediately

    out = g.make_guess(niter=5, seed=1, early_stop=stop, plot=False, print_thetas=False)
    assert out is not None
    assert calls["n"] >= 1            # early_stop was consulted
    assert len(g.theta) == 4


def test_optimize_runs_and_returns_guess():
    ds = _tiny_dataset()
    g = _guess(ds)
    out = g.optimize(maxiter=8, seed=0)
    assert out is not None
    assert np.isfinite(out.getMedianLifetime())
    assert len(g.theta) == 4 and all(t > 0 for t in g.theta)
