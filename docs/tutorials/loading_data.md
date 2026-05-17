# Loading and Exploring Mortality Data

This tutorial covers importing individual-level death-times data, computing survival and hazard
functions, and producing publication-ready plots.

## Data format

SRtools expects a CSV or Excel file with at minimum one column of death/observation times.

```
death dt,event
45.3,1
62.1,1
80.0,0
55.7,1
```

- `death dt` — age at death or last observation (any time unit: years, days, weeks, etc.)
- `event` — `1` = death observed, `0` = censored (lost to follow-up or study end). Optional; defaults to all 1.

## Loading from a file

```python
from SRtools import deathTimesDataSet as dtds

ds = dtds.dsFromFile('my_cohort.csv')
```

If your columns have different names:

```python
ds = dtds.dsFromFile(
    'raw_data.xlsx',
    sheet='Experiment_1',
    death_times_column='age_at_death',
    events_column='died',
)
```

For censored data where the events column uses `1 = censored` (rather than `1 = dead`):

```python
ds = dtds.dsFromFile('study.csv', event_is_censored=True)
```

## Constructing from arrays

```python
import numpy as np
from SRtools import Dataset

death_times = np.array([45, 62, 71, 80, 55, 90])
events      = np.ones_like(death_times)   # all observed deaths

ds = Dataset(death_times, events)
```

## Survival and hazard functions

```python
# Kaplan-Meier survival curve  S(t) = P(lifespan > t)
t_surv, s = ds.getSurvival()

# Nelson-Aalen hazard  h(t) = instantaneous mortality rate
t_haz, h = ds.getHazard()

# Summary statistics
print(ds.getMedianLifetime())
```

## Plotting

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ds.plotSurvival(ax=ax1, label='cohort', CI=True)
ds.plotHazard(ax=ax2, label='cohort', trim_by_n=5)

ax1.set_title('Survival')
ax2.set_title('Hazard')
plt.tight_layout()
plt.show()
```

## Aggregate life-table data

If you have counts per age bin rather than individual records:

```python
from SRtools import Life_table
import numpy as np

ages    = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
n_alive = np.array([1000, 970, 940, 900, 820, 700, 500, 250, 80, 10])

lt = Life_table(ages, n_alive)
lt.plotSurvival()
lt.plotHazard()
```

## Subsetting by metadata

Add properties to tag individuals, then filter:

```python
ds = dtds.dsFromFile('study.csv', properties=['genotype', 'diet'])

wt   = ds.subSetByProperty('genotype', 'WT')
mut  = ds.subSetByProperty('genotype', 'mutant')

ax = wt.plotSurvival(label='WT')
mut.plotSurvival(ax=ax, label='mutant')
```

## Bootstrap confidence intervals

```python
n_boot = 200
medians = [ds.sample(len(ds.death_times)).getMedianLifetime() for _ in range(n_boot)]
ci_low, ci_high = np.percentile(medians, [2.5, 97.5])
print(f"Median lifespan: {ds.getMedianLifetime():.1f}  95% CI [{ci_low:.1f}, {ci_high:.1f}]")
```
