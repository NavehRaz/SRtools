# Tutorials

The tutorials below follow a progression from basic data loading to advanced model customization.
Each builds on the previous one, but they can also be read independently.

## Overview

```{toctree}
:maxdepth: 1

loading_data
presets
mcmc_fitting
custom_models
```

| Tutorial | What you'll learn |
|----------|------------------|
| [Loading data](loading_data) | Import death-times data from CSV/Excel, compute survival curves and hazard functions, and create basic plots |
| [Using presets](presets) | Simulate survival curves for humans, mice, flies, and other organisms using built-in calibrated parameters |
| [MCMC fitting](mcmc_fitting) | Run Bayesian inference with emcee to fit SR model parameters to your own mortality data |
| [Custom models](custom_models) | Subclass `SR_Hetro` to implement custom aging dynamics and plug them into the MCMC pipeline |
