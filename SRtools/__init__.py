# Core dataset classes
from .deathTimesDataSet import Dataset, DatasetCollection

# Life table
from .life_table import Life_table

# SR model classes
from .SRmodellib import SR

# Legacy SR classes
from .legacy_classes import (
    SR_karin_human, 
    SRNDK, 
    SRND_Peckle, 
    SR_GWTW, 
    SR_karin_human_GWTW, 
    SRNDK_GWTW, 
    SRND_Peckle_GWTW,
    SR_envelope
)

# Distance metrics
from .distance_metrics import (
    baysian_dirichlet_distance,
    distance,
    ks_test,
    trim_to_range
)

# Import the deprecated baysianDistance from SRmodellib
from .SRmodellib import baysianDistance

# Utilities
from .utils import (
    gompetz_hazard,
    get_survival_from_hazard,
    get_dimless_groups,
    get_hazard_from_survival,
    karin_params,
    karin_mice_params
)

# SR with lifelines
from .SRmodellib_lifelines import SR_lf, SR_lf_karin_human

# SR with heterogeneity
from .SR_hetro import SR_Hetro

# Parametric fitters
from .weibullFitter import WeibullFitter, ExtendedWeibullFitter
from .makhamGompertzFitter import (
    GompertzMakehamFitter, 
    MakehamGompertzFitter, 
    GompertzFitter
)

# Bayesian analysis classes
from .prior_gen import PriorGen, PriorGenExtended
from .samples_utils import Posterior, JointPosterior

# Parameter search
from .initialParamsFinder import Guess

__all__ = [
    # Core dataset classes
    'Dataset',
    'DatasetCollection',
    'Life_table',
    # SR model classes
    'SR',
    # Legacy SR classes
    'SR_karin_human',
    'SRNDK',
    'SRND_Peckle',
    'SR_GWTW',
    'SR_karin_human_GWTW',
    'SRNDK_GWTW',
    'SRND_Peckle_GWTW',
    'SR_envelope',
    # Distance metrics
    'baysian_dirichlet_distance',
    'baysianDistance',  # deprecated
    'distance',
    'ks_test',
    'trim_to_range',
    # Utilities
    'gompetz_hazard',
    'get_survival_from_hazard',
    'get_dimless_groups',
    'get_hazard_from_survival',
    'karin_params',
    'karin_mice_params',
    # SR with lifelines
    'SR_lf',
    'SR_lf_karin_human',
    # SR with heterogeneity
    'SR_Hetro',
    # Parametric fitters
    'WeibullFitter',
    'ExtendedWeibullFitter',
    'GompertzMakehamFitter',
    'MakehamGompertzFitter',
    'GompertzFitter',
    # Bayesian analysis classes
    'PriorGen',
    'PriorGenExtended',
    'Posterior',
    'JointPosterior',
    # Parameter search
    'Guess',
]
