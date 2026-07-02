# Core dataset classes
from .deathTimesDataSet import Dataset, DatasetCollection

# Life table
from .life_table import Life_table

# SR model classes
from .SRmodellib import (
    SR, 
    SR_karin_human, 
    SRNDK, 
    SRND_Peckle, 
    SR_GWTW, 
    SR_karin_human_GWTW, 
    SRNDK_GWTW, 
    SRND_Peckle_GWTW,
    SR_envelope
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

# Automated initial-guess + Species360 config generation
from .auto_initial_guess import (
    auto_fit,
    FitResult,
    build_loglog_fits,
    predict_initial_theta,
    compute_timing,
    detect_external_hazard,
    detect_trend_start,
    conditional_median_steepness,
    evaluate_fit,
)
from .species_config_builder import (
    build_config_column,
    build_qc_row,
    write_configurations_excel,
    ROW_ORDER,
)
from .calibration_report import plot_survival_overview, plot_diagnostics_overview

__all__ = [
    # Core dataset classes
    'Dataset',
    'DatasetCollection',
    'Life_table',
    # SR model classes
    'SR',
    'SR_karin_human',
    'SRNDK',
    'SRND_Peckle',
    'SR_GWTW',
    'SR_karin_human_GWTW',
    'SRNDK_GWTW',
    'SRND_Peckle_GWTW',
    'SR_envelope',
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
    # Automated initial-guess + config generation
    'auto_fit',
    'FitResult',
    'build_loglog_fits',
    'predict_initial_theta',
    'compute_timing',
    'detect_external_hazard',
    'detect_trend_start',
    'conditional_median_steepness',
    'evaluate_fit',
    'build_config_column',
    'build_qc_row',
    'write_configurations_excel',
    'ROW_ORDER',
    'plot_survival_overview',
    'plot_diagnostics_overview',
]
