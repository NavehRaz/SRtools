# SRtools

A comprehensive Python library for analyzing mortality and aging data using the **Saturating Removal (SR) model**. The SR model is a stochastic differential equation model that describes aging as a process of damage accumulation and removal, providing a mechanistic framework for understanding survival and mortality patterns.

## Overview

SRtools provides tools for:
- **SR Model Simulation**: Simulate aging trajectories using various SR model variants
- **Survival Analysis**: Analyze death times, survival curves, and hazard functions
- **Parameter Estimation**: Fit SR model parameters to data using maximum likelihood and Bayesian methods
- **Life Table Analysis**: Work with aggregate survival data from life tables
- **Parametric Fitting**: Fit Weibull, Gompertz, and Makeham-Gompertz distributions to survival data
- **Bayesian Inference**: Perform MCMC sampling for parameter estimation and uncertainty quantification

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- And other dependencies (see `requirements.txt`)

### Install from source

```bash
git clone <repository-url>
cd SRtools
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic SR Model Simulation

```python
from SRtools import SR_lf

# Create an SR model simulation
# Parameters: eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end
model = SR_lf(
    eta=0.5,      # Damage production rate
    beta=50,     # Damage removal parameter
    kappa=0.5,    # Removal saturation parameter
    epsilon=50, # Noise parameter
    xc=17.0,      # Critical damage threshold
    npeople=10000,
    nsteps=5000,
    t_end=110
)

# Access survival and hazard functions
survival_times, survival_values = model.getSurvival()
hazard_times, hazard_values = model.getHazard()

# Plot survival curve
model.plotSurvival()
```

### Working with Death Times Data

```python
from SRtools import Dataset
import numpy as np

# Create dataset from death times
death_times = np.array([...])  # Your death times data
events = np.ones_like(death_times)  # Event indicators (1 = death, 0 = censored)

dataset = Dataset(death_times, events)

# Calculate survival and hazard
survival_times, survival_values = dataset.getSurvival()
hazard_times, hazard_values = dataset.getHazard()

# Plot results
dataset.plotSurvival()
dataset.plotHazard()
```

### Life Table Analysis

```python
from SRtools import Life_table
import numpy as np

# Create life table from age bins and number alive
ages = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
n_alive = np.array([1000, 950, 900, 800, 600, 400, 200, 80, 20, 2])

life_table = Life_table(ages, n_alive)

# Access survival and hazard
survival_times, survival_values = life_table.getSurvival()
hazard_times, hazard_values = life_table.getHazard()
```

### Bayesian Parameter Estimation

```python
from SRtools import Dataset
from SRtools.sr_mcmc import run_mcmc

# Load your data
dataset = Dataset(death_times, events)

# Run MCMC to estimate SR model parameters
# See sr_mcmc.py for detailed usage
```

## Core Components

### Dataset Classes

- **`Dataset`**: Base class for working with individual death times data
- **`DatasetCollection`**: Manage multiple datasets
- **`Life_table`**: Work with aggregate survival data from life tables

### SR Model Variants

- **`SR`**: Standard SR model with full parameter set and parent class for advanced models.

It is recommended to work with these:

- **`SR_Hetro`**: SR model with population heterogeneity (most advanced)
- **`SR_lf`**: SR model integrated with lifelines library (recommended for general use)

### Parametric Fitters

- **`WeibullFitter`**: Fit Weibull distribution to survival data
- **`ExtendedWeibullFitter`**: Extended Weibull fitting
- **`GompertzFitter`**: Fit Gompertz distribution
- **`MakehamGompertzFitter`**: Fit Makeham-Gompertz distribution
- **`GompertzMakehamFitter`**: Alternative Makeham-Gompertz fitting

### Bayesian Analysis

- **`Posterior`**: Analyze samples from MCMC to calculate and analyze posterior probabilities
- **`JointPosterior`**: Joint posterior analysis on several MCMC runs
- **`sr_mcmc`**: MCMC sampling functionality using emcee and some supporting utilities and analysis

### Utilities

- **`Guess`**: Simple tool for inital parameter estimation 
- **`plotting_utils`**: Visualization utilities for the comparison of parameters and CI's estimated from different runs
- **`distance_metrics`**: Distance metrics for model comparison (most of them are likelihoods and not distances)
- **`readResults`**: Read and analyze saved results 
- **`readResultsBaysian`**: Read Bayesian analysis results

## SR Model Description

The SR model describes the evolution of damage (X) over time:

dX/dt = eta * t - (beta * X) / (X + kappa) + sqrt(2 * epsilon) * xi

Where:
- **eta**: Damage production rate (linear growth)
- **beta**: Damage removal parameter
- **kappa**: Removal saturation parameter
- **epsilon**: Stochastic noise parameter
- **xc**: Critical damage threshold (death occurs when X > xc)
- **xi**: White noise (Wiener process)

Death occurs when the damage X exceeds the critical threshold xc.

## Key Features

- **Flexible Simulation**: Multiple SR model variants and simulation methods
- **Efficient Computation**: Uses Numba JIT compilation for performance
- **Comprehensive Analysis**: Survival curves, hazard functions, cumulative hazards
- **Statistical Fitting**: Maximum likelihood and Bayesian parameter estimation
- **Visualization**: Built-in plotting functions for survival and hazard analysis
- **Life Table Support**: Work with aggregate survival data
- **MCMC Integration**: Full Bayesian inference using emcee

## Documentation

For detailed API documentation, see the docstrings in individual modules. Key modules include:

- `SRmodellib.py`: Core SR model implementation
- `deathTimesDataSet.py`: Dataset handling and survival analysis
- `life_table.py`: Life table analysis
- `sr_mcmc.py`: MCMC sampling and Bayesian inference
- `weibullFitter.py`: Weibull distribution fitting
- `makhamGompertzFitter.py`: Gompertz-Makeham fitting

## Deprecated Modules

The following modules are deprecated and should not be used in new code:
- `probability.py`
- `prior_gen.py`
- `life_table_old.py`

Use `life_table.py` instead of `life_table_old.py`. For prior generation and probability calculations, use the functionality in `sr_mcmc.py` and `samples_utils.py`.

## Examples

See the ... notebooks for example usage.

## Citation

If you use SRtools in your research, please cite the relevant papers on the SR model of aging:
- 1. Naveh Raz, Yifan Yang, Glen Pridham et al. A damage accumulation model reveals strategies of aging across species, 08 July 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6946440/v1]
- 2. Karin, O., Agrawal, A., Porat, Z. et al. Senescent cell turnover slows with age providing an explanation for the Gompertz law. Nat Commun 10, 5495 (2019). https://doi.org/10.1038/s41467-019-13192-4



## License

[Add license information here]

## Contributing

[Add contributing guidelines here]

## Contact

Naveh.Raz@weizmann.ac.il

