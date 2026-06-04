# SRtools

**A Python library for mortality and aging analysis using the Saturating Removal (SR) model.**

SRtools provides tools for simulating aging dynamics, fitting parameters to mortality data, and comparing aging strategies across species — all grounded in a mechanistic, stochastic model of damage accumulation and repair.

::::{grid} 2
:::{grid-item-card} Getting started
:link: installation
:link-type: doc
Install SRtools and run your first simulation in minutes.
:::
:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Step-by-step guides from loading data to full Bayesian inference.
:::
:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
Complete documentation for all classes and functions.
:::
:::{grid-item-card} GitHub
:link: https://github.com/NavehRaz/SRtools
:link-type: url
Source code, issues, and contributions.
:::
::::

## The SR Model

The SR model describes how internal cellular damage *X(t)* evolves over time. Death occurs when damage crosses a critical threshold *x_c*:

```
dX/dt = η·t  −  β·X/(X + κ)  +  √(2ε)·ξ
```

| Parameter | Biological meaning |
|-----------|-------------------|
| `eta` (η) | Rate of damage accumulation (increases with age) |
| `beta` (β) | Maximum repair/removal capacity |
| `kappa` (κ) | Half-saturation constant for repair |
| `epsilon` (ε) | Stochasticity in damage dynamics |
| `xc` | Death threshold |

## Citation

If you use SRtools in your research, please cite:

> Naveh Raz, Yifan Yang, Glen Pridham et al. *A damage accumulation model reveals strategies of aging across species*, Research Square 2025.  
> [https://doi.org/10.21203/rs.3.rs-6946440/v1](https://doi.org/10.21203/rs.3.rs-6946440/v1)

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

installation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/index
tutorials/loading_data
tutorials/presets
tutorials/mcmc_fitting
tutorials/custom_models
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API Reference

api/index
api/dataset
api/models
api/mcmc
api/posterior
api/presets
```
