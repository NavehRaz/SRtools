# Simulating with Built-in Organism Presets

SRtools ships with fitted SR parameters for a wide range of organisms and populations,
derived from published mortality data. This tutorial shows how to use them.

## Listing available presets

```python
from SRtools import presets

names = presets.get_preset_names()
print(names)
# ['combined_human_M', 'combined_human_F', 'mice_M', 'mice_F',
#  'celegans', 'yeast', 'ecoli', 'drosophila_853', ...]
```

## Simulating a preset organism

```python
# Simulate human male survival (combined dataset)
sim = presets.getSim('combined_human_M')
sim.plotSurvival()

print(f"Median lifespan: {sim.getMedianLifetime():.1f} years")
```

## Comparing species

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

presets.getSim('combined_human_M').plotSurvival(ax=ax, label='Human (M)')
presets.getSim('combined_human_F').plotSurvival(ax=ax, label='Human (F)')
presets.getSim('mice_M').plotSurvival(ax=ax, label='Mouse (M)')
presets.getSim('celegans').plotSurvival(ax=ax, label='C. elegans')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Survival')
ax.legend()
plt.tight_layout()
plt.show()
```

## Getting parameter vectors

```python
# Returns [eta, beta, epsilon, xc]
theta = presets.getTheta('combined_human_M')
print(theta)

# Include external hazard as 5th element
theta_ext = presets.getTheta('combined_human_M', ExtH=True)
```

## Choosing the parameter estimate type

```python
# Mode of joint posterior (default, recommended)
theta_mode = presets.getTheta('mice_F', type='mode_overall')

# Maximum-likelihood estimate
theta_mle  = presets.getTheta('mice_F', type='max_likelihood')

# Marginal mode of each parameter independently
theta_marg = presets.getTheta('mice_F', type='mode')
```

## Getting the simulation config

Each preset also stores the recommended simulation settings (population size, time steps, etc.):

```python
cfg = presets.get_config_params('combined_human_M')
# {'nsteps': 5000, 'time_step_multiplier': 1, 'npeople': 25000,
#  't_end': 110, 'time_range': [20, 100], 'hetro': True}
```

`getSim` loads these automatically, but you can override any value:

```python
sim = presets.getSim('combined_human_M', npeople=5000, t_end=90)
```

## Available presets

**Humans**

| Preset name | Description |
|-------------|-------------|
| `combined_human_M` | Combined male human dataset |
| `combined_human_F` | Combined female human dataset |
| `Sweden_M_1910_hetro` | Swedish males, 1910 (heterogeneous model) |
| `Sweden_F_1910_hetro` | Swedish females, 1910 |
| `Denmark_M_1900_hetro` | Danish males, 1900 |
| `Denmark_F_1900_hetro` | Danish females, 1900 |

**Rodents & small mammals**

| Preset name | Description |
|-------------|-------------|
| `mice_M` | Male mice |
| `mice_F` | Female mice |
| `Guiniea_pig_VC` | Guinea pig (VetCompass) |

**Companion animals**

| Preset name | Description |
|-------------|-------------|
| `cats_vp_M` / `cats_vp_F` | Male/female cats (VetCompass) |
| `Labradors_vetCompass` | Labrador retrievers |
| `Staffy_vetCompass` | Staffordshire bull terriers |
| `Jack_Russell_vetCompass` | Jack Russell terriers |
| `German_Shepherd_vetCompass` | German shepherd dogs |

**Model organisms**

| Preset name | Description |
|-------------|-------------|
| `celegans` | *C. elegans* |
| `drosophila_853` | *Drosophila melanogaster* (N=853) |
| `yeast` | Budding yeast |
| `ecoli` | *E. coli* |
