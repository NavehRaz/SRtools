# Installation

## From PyPI (recommended)

```bash
pip install srtools-aging
```

> **Note**: The PyPI package is named `srtools-aging` (the name `srtools` was taken).
> You import it as `SRtools` in Python.

**Requirements**: Python ≥ 3.10.

## From source

```bash
git clone https://github.com/NavehRaz/SRtools
cd SRtools
pip install -e ".[dev]"
```

## Dependencies

SRtools requires the following packages (installed automatically via pip):

| Package | Purpose |
|---------|---------|
| NumPy, SciPy | Numerical computation |
| Matplotlib, Seaborn | Plotting |
| Pandas | Data handling |
| Numba | JIT-compiled simulation loops |
| lifelines | Kaplan-Meier and parametric survival fitting |
| emcee | MCMC ensemble sampling |
| h5py | HDF5 result storage |
| tqdm | Progress bars |

## Verifying the installation

```python
import SRtools
from SRtools import presets

# Simulate a human survival curve using built-in parameters
sim = presets.getSim('combined_human_M')
sim.plotSurvival()
print("SRtools is working correctly!")
```

## Optional: Web app

An interactive Streamlit app lives in `app/`. It is not installed by default:

```bash
pip install srtools-aging[app]
streamlit run app/streamlit_app.py
```

## Building the documentation locally

```bash
pip install srtools-aging[docs]
cd docs
make html
open _build/html/index.html
```
