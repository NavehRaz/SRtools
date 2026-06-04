# SR Model Classes

## Recommended: getSrHetro (factory function)

The primary way to create a simulation. Returns a ready-to-use `SR_Hetro` object.

```{eval-rst}
.. autofunction:: SRtools.SR_hetro.getSrHetro
```

## SR_Hetro

Heterogeneous SR model — individuals vary in their parameter values.
Subclass this to implement custom aging dynamics (see [Custom Models](../tutorials/custom_models.md)).

```{eval-rst}
.. autoclass:: SRtools.SR_hetro.SR_Hetro
   :members:
   :member-order: bysource
```

## Parametric fitters

All fitters wrap lifelines and share the `Dataset` interface for plotting and survival analysis.

```{eval-rst}
.. autoclass:: SRtools.weibullFitter.WeibullFitter
   :members:

.. autoclass:: SRtools.weibullFitter.ExtendedWeibullFitter
   :members:

.. autoclass:: SRtools.makhamGompertzFitter.GompertzFitter
   :members:

.. autoclass:: SRtools.makhamGompertzFitter.GompertzMakehamFitter
   :members:

.. autoclass:: SRtools.makhamGompertzFitter.MakehamGompertzFitter
   :members:
```

## Lower-level SR variants

These are available for advanced use but `getSrHetro` / `SR_Hetro` is recommended for most work.

```{eval-rst}
.. autoclass:: SRtools.SRmodellib_lifelines.SR_lf
   :members:

.. autoclass:: SRtools.SRmodellib.SR
   :members:
```
