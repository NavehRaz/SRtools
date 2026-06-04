# SRtools

A Python library for analyzing mortality and aging data using the **Saturating Removal (SR) model** — a mechanistic, stochastic model that describes aging as the result of damage accumulation and cellular repair. SRtools lets you simulate survival curves, fit parameters to your own mortality data, and compare aging strategies across species.

**Published as**: `srtools-aging` on PyPI (imports as `SRtools`).  
**Paper**: Naveh Raz et al., *A damage accumulation model reveals strategies of aging across species*, Research Square 2025. [https://doi.org/10.21203/rs.3.rs-6946440/v1](https://doi.org/10.21203/rs.3.rs-6946440/v1)

---

## The SR Model

The SR model tracks a single internal damage variable *X(t)* for each individual. Death occurs when damage crosses a critical threshold *xc*. The dynamics follow the stochastic differential equation:

```
dX/dt = η·t  −  β·X/(X + κ)  +  √(2ε)·ξ
```

| Parameter | Symbol | Biological meaning |
|-----------|--------|--------------------|
| `eta` | η | Rate at which new damage accumulates (increases with age) |
| `beta` | β | Maximum damage removal capacity |
| `kappa` | κ | Half-saturation constant for damage removal |
| `epsilon` | ε | Stochasticity / noise in damage dynamics |
| `xc` | x_c | Critical damage threshold — individual dies when X exceeds this |

The key competition between accumulation (η·t) and removal (β·X/(X+κ)) — which saturates at high damage — is what produces realistic, Gompertz-like survival curves.

---

## Installation

```bash
pip install srtools-aging
```

> **Note**: The PyPI package is called `srtools-aging` (the name `srtools` was already taken). You import it as `SRtools`.

**Install from source** (for development):

```bash
git clone https://github.com/NavehRaz/SRtools
cd SRtools
pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.10, NumPy, SciPy, Matplotlib, lifelines, emcee, numba, and others (see `requirements.txt`).

---

## Quick Start

### Tier 1 — Load your data and plot a survival curve

Your data should be a CSV file with a column named `death dt` containing the age at death (or last observation) for each individual.

```python
from SRtools import deathTimesDataSet as dtds

ds = dtds.dsFromFile('my_data.csv')

# Basic survival statistics
print(ds.getMedianLifetime())
survival_times, survival_values = ds.getSurvival()
hazard_times, hazard_values = ds.getHazard()

# Plot
ds.plotSurvival()
ds.plotHazard()
```

Or construct a dataset directly from arrays:

```python
import numpy as np
from SRtools import Dataset

death_times = np.array([45, 62, 71, 80, 55, 90, ...])
events = np.ones_like(death_times)        # 1 = died, 0 = censored

ds = Dataset(death_times, events)
ds.plotSurvival()
```

For **aggregate life-table data** (age bins and counts of survivors), use `Life_table`:

```python
from SRtools import Life_table
import numpy as np

ages   = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
n_alive = np.array([1000, 950, 900, 800, 600, 400, 200, 80, 20, 2])

lt = Life_table(ages, n_alive)
lt.plotSurvival()
```

---

### Tier 2 — Simulate with built-in organism presets

SRtools ships with fitted SR parameters for a range of organisms. Use them to generate a simulation without any fitting step.

```python
from SRtools import presets

# See all available presets
print(presets.get_preset_names())
# e.g.: combined_human_M, combined_human_F, mice_M, mice_F,
#       celegans, yeast, ecoli, drosophila_853, cats_vp_M,
#       Labradors_vetCompass, Sweden_M_1910_hetro, ...

# Simulate with human (male combined) parameters
sim = presets.getSim('combined_human_M')
sim.plotSurvival()

# Retrieve only the parameter vector [eta, beta, epsilon, xc]
theta = presets.getTheta('combined_human_M')
print(theta)   # [eta, beta, epsilon, xc]

# Compare two species on one plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
presets.getSim('combined_human_M').plotSurvival(ax=ax, label='Human (M)')
presets.getSim('mice_M').plotSurvival(ax=ax, label='Mouse (M)')
ax.legend()
plt.show()
```

The `type` argument controls which estimate is used (`"mode_overall"` default, `"mode"`, or `"max_likelihood"`):

```python
theta_mle = presets.getTheta('combined_human_M', type='max_likelihood')
```

---

### Tier 3 — Fit parameters to your data with MCMC

This is the full Bayesian inference pipeline. It uses `emcee` ensemble sampling and can take minutes to hours depending on the dataset size and number of steps.

```python
from SRtools import sr_mcmc as srmc
from SRtools import deathTimesDataSet as dtds
from SRtools import samples_utils as su
from SRtools import SR_hetro as srh

# 1. Load data
ds = dtds.dsFromFile('my_data.csv')

# 2. Run MCMC — results are saved to an HDF5 backend file
sampler = srmc.getSampler(
    nwalkers=32,
    num_mcmc_steps=2000,
    dataSet=ds,
    back_end_file='results.h5',
    t_end=110,           # maximum age in your data
    npeople=10000,       # simulated individuals per likelihood evaluation
    nsteps=3000,         # simulation time steps
)

# 3. Load completed samples (discard burn-in, thin for autocorrelation)
samples_trans, lnprobs = srmc.loadSamplesFromDir(
    'results/',          # folder containing the .h5 file(s)
    best=False,
    thin=5,
    discard=200,
)

# 4. Build a Posterior object and save it
# save_to_file writes two files:
#   posterior.csv        — discretized posterior metadata (human-readable)
#   posterior_data.npz   — raw samples and log-probabilities (compressed binary)
post = su.Posterior(samples_trans, lnprobs, bins=100, log=True)
post.save_to_file('posterior.csv')

# 5. Extract best-fit parameters and simulate
best_theta_trans = post.best_raw_sample()
best_theta = srmc.inv_transform(best_theta_trans)   # back-transform from log space

best_sim = srh.getSrHetro(best_theta, npeople=10000, nsteps=3000, t_end=110)

# 6. Compare simulation to data
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ds.plotSurvival(ax=ax, label='data', linestyle='--')
best_sim.plotSurvival(ax=ax, label='best fit')
ax.legend()
plt.show()
```

To reload a saved posterior in a later session:

```python
post = su.Posterior.load_from_file('posterior.csv')
```

---

### Tier 4 — Extend the model with custom dynamics

Subclass `SR_Hetro` and override `calc_death_times()` to implement any custom stochastic aging model. The subclass inherits all dataset and plotting methods automatically.

```python
import numpy as np
from SRtools import SR_hetro as srh
from SRtools import SRmodellib as sr
from SRtools import sr_mcmc as srmc


class MySR(srh.SR_Hetro):
    """SR model with an additional constant damage source (gamma)."""

    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                 gamma=0.0,                      # custom parameter
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
        # Modify the simulation here.
        # Must return (death_times_array, events_array).
        # Call the parent to get the base simulation, then post-process,
        # or write a fully custom Numba-jitted function (see SR_hetro.py).
        return super().calc_death_times()


# Factory function — matches the signature expected by getSampler
def getMySR(theta, n=10000, nsteps=3000, t_end=110,
            external_hazard=np.inf, time_step_multiplier=1,
            npeople=None, parallel=False, gamma=0.0, **kwargs):
    if npeople is not None:
        n = npeople
    eta, beta, epsilon, xc = theta[:4]
    return MySR(eta=eta, beta=beta, kappa=0.5, epsilon=epsilon, xc=xc,
                gamma=gamma, npeople=n, nsteps=nsteps, t_end=t_end,
                external_hazard=external_hazard,
                time_step_multiplier=time_step_multiplier, parallel=parallel)


# Model function for MCMC (returns log-likelihood)
def model(theta, n, nsteps, t_end, dataSet, sim=None, metric='baysian',
          time_range=None, time_step_multiplier=1, parallel=False,
          dt=1, set_params=None, debug=False, kwargs=None):
    sim = getMySR(theta, n=n, nsteps=nsteps, t_end=t_end,
                  time_step_multiplier=time_step_multiplier, parallel=parallel)
    tprob = sr.distance(dataSet, sim, metric=metric, time_range=time_range, dt=dt)
    return tprob if not np.any(np.isnan(tprob)) else -np.inf


# Pass the custom model function to getSampler
sampler = srmc.getSampler(
    nwalkers=32, num_mcmc_steps=1000, dataSet=ds,
    back_end_file='custom_results.h5', t_end=110,
    model_func=model,
)
```

---

## Core API Reference

### Dataset classes

| Class | Use case | Key methods |
|-------|----------|-------------|
| `Dataset` | Individual-level death times | `getSurvival()`, `getHazard()`, `plotSurvival()`, `plotHazard()`, `getMedianLifetime()`, `toCsv()` |
| `DatasetCollection` | Manage and compare multiple datasets | — |
| `Life_table` | Aggregate survival data (age bins + counts) | Same interface as `Dataset` |

Load from file: `dtds.dsFromFile(path)` — reads a CSV/Excel with a `death dt` column.

### SR model variants

| Class / function | When to use |
|-----------------|-------------|
| `getSrHetro(theta, ...)` | **Recommended** — heterogeneous population, Brownian-bridge simulation |
| `SR_Hetro` | Base class for custom model extensions |
| `SR_lf` | Simple homogeneous model with lifelines backend |
| `SR` | Low-level parent class; prefer `getSrHetro` |

### Parametric fitters

All fitters wrap lifelines and share the `Dataset` interface.

| Class | Distribution |
|-------|-------------|
| `WeibullFitter` | Weibull |
| `ExtendedWeibullFitter` | Extended Weibull |
| `GompertzFitter` | Gompertz |
| `GompertzMakehamFitter` | Gompertz–Makeham |
| `MakehamGompertzFitter` | Makeham–Gompertz |

### Bayesian analysis

| Function / class | Purpose |
|-----------------|---------|
| `srmc.getSampler(...)` | Run MCMC with emcee; saves to HDF5 |
| `srmc.loadSamplesFromDir(...)` | Load and flatten samples from result folder |
| `srmc.inv_transform(theta_trans)` | Back-transform log-space samples |
| `su.Posterior(samples, lnprobs, bins)` | Analyze and store posterior |
| `post.best_raw_sample()` | Best-fit (highest log-prob) sample |
| `post.get_mode()` / `get_mean()` / `get_ci()` | Summary statistics |
| `post.plot_corner()` | Corner plot of marginal distributions |
| `JointPosterior` | Combine posteriors across multiple MCMC runs |

### Presets

| Function | Purpose |
|----------|---------|
| `presets.get_preset_names()` | List all available organism/population presets |
| `presets.getTheta(name)` | Return `[eta, beta, epsilon, xc]` for a preset |
| `presets.getSim(name)` | Return a ready-to-use `SR_Hetro` simulation |
| `presets.get_config_params(name)` | Return the simulation config (nsteps, t_end, …) |

---

## Built-in Organism Presets

The following preset names are available. Pass them to `getTheta()` or `getSim()`.

**Humans**
- `combined_human_M`, `combined_human_F`
- `Sweden_M_1910_hetro`, `Sweden_F_1910_hetro`, `Sweden_F_1910_homo`
- `Denmark_M_1900_hetro`, `Denmark_M_1890_hetro`, `Denmark_F_1900_hetro`, `Denmark_F_1890_hetro`
- `Denmark_M_1900_homo`, `Denmark_M_1890_homo`

**Rodents & small mammals**
- `mice_M`, `mice_F`
- `Guiniea_pig_VC`

**Companion animals**
- `cats_vp_M`, `cats_vp_F`, `cats_BPH`
- `Labradors_vetCompass`, `Staffy_vetCompass`, `Jack_Russell_vetCompass`, `German_Shepherd_vetCompass`

**Invertebrates / model organisms**
- `celegans`
- `drosophila_853`, `drosophila_707`, `drosophila_441`, `drosophila_217`, `drosophila_136`, `drosophila_195`, `drosophila_105`, `drosophila_M22_25deg`
- `Wdah_chronic`, `Wdah_chronic2`, `Wdah_control`
- `yeast`, `ecoli`

---

## Data Format

### Individual-level CSV

```
death dt,event
45.3,1
62.1,1
80.0,0
```

- `death dt`: age at death or last observation (required)
- `event`: 1 = death observed, 0 = censored (optional; defaults to all 1)

Load with `dtds.dsFromFile('file.csv')` or `dtds.dsFromFile('file.csv', events_column='event')`.

### Life-table data

Use `Life_table(ages, n_alive)` when you have aggregate survival counts per age bin rather than individual records.

---

## Running Large MCMC Campaigns (HPC / Cluster)

For large-scale studies across many species or conditions, SRtools supports an Excel-based configuration workflow designed for HPC job schedulers (e.g., LSF).

The pattern:
1. Define all runs as columns in `configurations.xlsx` (parameters, data files, MCMC settings)
2. A `run_manager.py` script reads the Excel file and submits one cluster job per column
3. Each job calls `run_file_mcmc_excel.py`, which reads its configuration via `config_lib`
4. Results are saved as HDF5 files, then analysed in templated Jupyter notebooks

```python
from SRtools import config_lib as cl

config = cl.read_excel_config('configurations.xlsx', 'my_run')
cfg = cl.config_to_dict(config, mcmc_convert=True)

eta   = float(cfg['eta'])
nwalkers = int(cfg['nwalkers'])
# ... etc.
```

See [`Extra_calibrations/README.md`](../Extra_calibrations/README.md) for a full worked example of this workflow.

---

## Citation

If you use SRtools in your research, please cite:

1. Naveh Raz, Yifan Yang, Glen Pridham et al. *A damage accumulation model reveals strategies of aging across species*, Research Square 2025. [https://doi.org/10.21203/rs.3.rs-6946440/v1](https://doi.org/10.21203/rs.3.rs-6946440/v1)
2. Karin O., Agrawal A., Porat Z. et al. *Senescent cell turnover slows with age providing an explanation for the Gompertz law.* Nat Commun 10, 5495 (2019). [https://doi.org/10.1038/s41467-019-13192-4](https://doi.org/10.1038/s41467-019-13192-4)

---

## Deprecated Modules

The following modules are kept for backward compatibility but should not be used in new code:

- `probability.py` — use `sr_mcmc.py` and `samples_utils.py` instead
- `prior_gen.py` — use `sr_mcmc.py` instead
- `life_table_old.py` — use `life_table.py` instead

---

## Web App

An interactive Streamlit app for exploring the SR model lives in `app/`. It is **not** a dependency of the core package.

```bash
pip install srtools-aging[app]
streamlit run app/streamlit_app.py
```

---

## License

MIT. See `LICENSE`.

## Contributing

Bug reports and pull requests are welcome at [https://github.com/NavehRaz/SRtools](https://github.com/NavehRaz/SRtools). Please open an issue first to discuss significant changes.

## Contact

Naveh.Raz@weizmann.ac.il
