# Building Custom Aging Models

SRtools is designed to be extended. By subclassing `SR_Hetro` you can implement
any custom stochastic aging dynamic while retaining all the built-in survival analysis,
plotting, and MCMC infrastructure.

## When to build a custom model

- You want to add an intervention (drug, caloric restriction) that modifies damage dynamics mid-life
- You need additional biological parameters (e.g., a repair saturation term, a second damage compartment)
- You want to model a non-standard death criterion

## The extension pattern

Three components are needed:

1. **A subclass** of `SR_Hetro` that overrides `calc_death_times()`
2. **A factory function** (`getMyModel`) that instantiates it from a `theta` vector
3. **A model function** (`model`) that computes log-likelihood — passed to `getSampler`

## Example: adding a constant damage source

```python
import numpy as np
from SRtools import SR_hetro as srh
from SRtools import SRmodellib as sr
from SRtools import sr_mcmc as srmc


class GammaSR(srh.SR_Hetro):
    """SR model with an extra constant damage influx (gamma)."""

    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                 gamma=0.0,
                 eta_var=0, beta_var=0, kappa_var=0, epsilon_var=0, xc_var=0.2,
                 t_start=0, tscale='years', external_hazard=np.inf,
                 time_step_multiplier=1, parallel=False, bandwidth=3,
                 method='brownian_bridge'):
        self.gamma = gamma
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                         eta_var, beta_var, kappa_var, epsilon_var, xc_var,
                         t_start, tscale, external_hazard,
                         time_step_multiplier, parallel, bandwidth, method)

    def calc_death_times(self):
        # Modify parameters to fold gamma into eta as a constant offset at t=0,
        # or write a fully custom Numba-jitted simulation (see SR_hetro.py).
        # For a minimal example we just call the parent here.
        return super().calc_death_times()


def getGammaSR(theta, n=10000, nsteps=3000, t_end=110,
               external_hazard=np.inf, time_step_multiplier=1,
               npeople=None, parallel=False, gamma=0.0, **kwargs):
    if npeople is not None:
        n = npeople
    eta, beta, epsilon, xc = theta[:4]
    return GammaSR(
        eta=eta, beta=beta, kappa=0.5, epsilon=epsilon, xc=xc,
        gamma=gamma, npeople=n, nsteps=nsteps, t_end=t_end,
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
    )


def model(theta, n, nsteps, t_end, dataSet, sim=None, metric='baysian',
          time_range=None, time_step_multiplier=1, parallel=False,
          dt=1, set_params=None, debug=False, kwargs=None):
    """Log-likelihood function — passed to getSampler as model_func."""
    sim = getGammaSR(theta, n=n, nsteps=nsteps, t_end=t_end,
                     time_step_multiplier=time_step_multiplier,
                     parallel=parallel)
    tprob = sr.distance(dataSet, sim, metric=metric,
                        time_range=time_range, dt=dt)
    return tprob if not np.any(np.isnan(tprob)) else -np.inf
```

## Plugging into MCMC

```python
from SRtools import deathTimesDataSet as dtds

ds = dtds.dsFromFile('my_data.csv')

sampler = srmc.getSampler(
    nwalkers=32,
    num_mcmc_steps=2000,
    dataSet=ds,
    back_end_file='gamma_results.h5',
    t_end=110,
    model_func=model,      # <-- your custom model function
)
```

## Writing a custom Numba simulation loop

For performance with large populations, write the inner loop as a Numba JIT-compiled function.
See `SR_hetro.py::death_times_euler_brownian_bridge` for the reference implementation.
The key contract is:

```python
@numba.jit(nopython=True)
def my_death_times(s, dt, t, eta0, ..., npeople, external_hazard, time_step_multiplier):
    death_times = []
    events = []
    for person in range(npeople):
        x = 0.0
        # ... simulate trajectory ...
        death_times.append(age_at_death)
        events.append(1 if crossed else 0)
    return np.array(death_times), np.array(events)
```

Then call it inside `calc_death_times()`:

```python
def calc_death_times(self):
    death_times, events = my_death_times(
        self.nsteps, self.dt, self.t,
        self.eta, ..., self.npeople,
        self.external_hazard, self.time_step_multiplier,
    )
    self.death_times = death_times
    self.events = events
    self._calc_survival_and_hazard()
    return death_times, events
```

## Template

A fully-commented template is available at `SRtools/SR_hetro.py` — copy the
`SR_Hetro` class definition and `getSrHetro` factory as a starting point.
