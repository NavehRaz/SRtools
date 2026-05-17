# MCMC Sampling

Functions for Bayesian parameter estimation using the `emcee` ensemble sampler.

## Running MCMC

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.getSampler
```

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.getSamplerAutoCorrMon
```

## Loading results

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.loadSamplesFromDir
```

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.loadSamples
```

## Parameter transformations

The MCMC sampler works in a transformed (decorrelated) parameter space for efficiency.
Use these functions to convert between the raw SR parameters and the transformed space.

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.transform
```

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.inv_transform
```

## Utilities

```{eval-rst}
.. autofunction:: SRtools.sr_mcmc.parse_theta

.. autofunction:: SRtools.sr_mcmc.get_bins_from_seed

.. autofunction:: SRtools.sr_mcmc.getThreshold

.. autofunction:: SRtools.sr_mcmc.applyThreshold

.. autofunction:: SRtools.sr_mcmc.getSr
```

## Cluster workflow

For large-scale studies across many organisms, SRtools supports an Excel-based
configuration workflow designed for HPC job schedulers (LSF, SLURM).

The key components (found in the `Extra_calibrations/` reference project):

| File | Purpose |
|------|---------|
| `configurations.xlsx` | One column per MCMC run; rows are parameter/config fields |
| `run_manager.py` | Reads Excel, submits one cluster job per column |
| `run_file_mcmc_excel.py` | Single-run MCMC script, called by each cluster job |
| `run_multiple_configs.py` | Batch-generates analysis notebooks from results |

```python
from SRtools import config_lib as cl

# Read one column from the Excel file
config = cl.read_excel_config('configurations.xlsx', 'my_run')
cfg    = cl.config_to_dict(config, mcmc_convert=True)

eta      = float(cfg['eta'])
nwalkers = int(cfg['nwalkers'])
# ... pass to getSampler
```
