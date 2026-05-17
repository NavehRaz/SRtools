# Bayesian Parameter Fitting with MCMC

This tutorial walks through the full pipeline for fitting SR model parameters
to your own mortality data using MCMC (Markov Chain Monte Carlo) sampling.

## Overview

The pipeline has four stages:

1. **Load data** — create a `Dataset` from your death-times file
2. **Run MCMC** — use `getSampler` to sample the posterior; results saved to HDF5
3. **Analyse posterior** — build a `Posterior` object from the samples
4. **Simulate best fit** — run `getSrHetro` with the best-fit parameters and compare to data

## 1. Load data

```python
from SRtools import deathTimesDataSet as dtds

ds = dtds.dsFromFile('my_cohort.csv')
```

## 2. Run MCMC

```python
from SRtools import sr_mcmc as srmc

# Rough initial guess: [eta, beta, epsilon, xc]
seed = [0.5, 50, 50, 15]

sampler = srmc.getSampler(
    nwalkers=32,
    num_mcmc_steps=3000,
    dataSet=ds,
    back_end_file='results.h5',   # checkpoint to HDF5
    t_end=110,
    npeople=10000,
    nsteps=3000,
    seed=seed,
    ndim=4,
)
```

:::{note}
`back_end_file` saves every step to disk. If the run is interrupted, set
`restartFromBackEnd=True` to resume from where it stopped.
:::

### Choosing `nwalkers` and `num_mcmc_steps`

- **`nwalkers`**: Use at least `2 * ndim` (8 for 4 parameters). `32` is a safe default.
- **`num_mcmc_steps`**: Plan to discard the first 30–50% as burn-in. `3000` steps
  with 50% burn-in leaves 1500 effective steps per walker.

### Using a prior

By default the prior is derived from the initial seed with a ×10 expansion.
To set explicit bounds:

```python
prior = [
    [0.01, 5.0],    # eta bounds
    [1.0, 500.0],   # beta bounds
    [1.0, 500.0],   # epsilon bounds
    [5.0, 30.0],    # xc bounds
]
sampler = srmc.getSampler(..., prior=prior)
```

## 3. Analyse the posterior

```python
from SRtools import samples_utils as su

# Load samples, discarding 50% burn-in, thinning by 5 for autocorrelation
samples_trans, lnprobs = srmc.loadSamplesFromDir(
    'results/',
    best=False,
    thin=5,
    discard=1500,
)

# Build and save the posterior
post = su.Posterior(samples_trans, lnprobs, bins=100, log=True)
post.save_to_file('posterior.csv')
# Saves: posterior.csv (metadata) + posterior_data.npz (raw samples)
```

To reload in a later session:

```python
post = su.Posterior.load_from_file('posterior.csv')
```

### Summary statistics

```python
mode   = post.get_mode()             # most probable parameter set (binned)
mean   = post.get_mean()
ci     = post.get_ci(percentile=95)  # 95% credible interval
best   = post.best_raw_sample()      # highest log-prob raw sample
```

### Corner plot

```python
import matplotlib.pyplot as plt

labels = [r'$\eta$', r'$\beta$', r'$\varepsilon$', r'$x_c$']
fig = post.plot_corner(labels=labels)
plt.savefig('corner.pdf')
```

## 4. Simulate best fit

```python
from SRtools import SR_hetro as srh
import matplotlib.pyplot as plt

# Back-transform from log-space
best_theta = srmc.inv_transform(post.best_raw_sample())

best_sim = srh.getSrHetro(
    best_theta,
    npeople=10000,
    nsteps=3000,
    t_end=110,
)

fig, ax = plt.subplots()
ds.plotSurvival(ax=ax, label='data', linestyle='--')
best_sim.plotSurvival(ax=ax, label='best fit')
ax.legend()
plt.show()
```

## Running on a compute cluster

For large datasets or many walkers, running on HPC is recommended.
See the [cluster workflow](../api/mcmc.md#cluster-workflow) section for an overview
of the Excel-config + `config_lib` pattern used in the included templates.
