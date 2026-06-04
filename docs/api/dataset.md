# Dataset Classes

Classes for loading, storing, and analysing mortality data.
All three classes share the same survival-analysis interface.

## Loading from a file

```{eval-rst}
.. autofunction:: SRtools.deathTimesDataSet.dsFromFile
```

## Dataset

The primary class for individual-level death-times data.

```{eval-rst}
.. autoclass:: SRtools.deathTimesDataSet.Dataset
   :members:
   :member-order: bysource
```

## Life_table

For aggregate survival data (age bins + counts alive).

```{eval-rst}
.. autoclass:: SRtools.life_table.Life_table
   :members:
   :member-order: bysource
```

## DatasetCollection

Manage and compare multiple datasets.

```{eval-rst}
.. autoclass:: SRtools.deathTimesDataSet.DatasetCollection
   :members:
```
