"""
Trajectory analysis methods for SR models.

This module contains trajectory-dependent analysis methods that have been moved from
SRmodellib.py. These methods are currently disabled pending refactoring for efficiency,
as the trajectory saving and analysis system needs to be redesigned.

All methods in this module raise NotImplementedError with guidance messages.

Functions moved here:
- get_damage_PDF: Damage probability density functions
- create_distributions: Distribution creation for multiple time points
- age_distribution_by_damage: Age distribution analysis
- damage_transition_probabilities: Transition matrix calculations
- get_sickspan, mean_sickspan, mean_hle_over_le: Health span analysis
- kramer_damage_distribution: Theoretical damage distributions
- plotTrajectoriies, meanDamageTrajectories: Trajectory visualization
- agingStats: Comprehensive aging statistics
- get_median_damage, median_accelerator: Damage statistics
- has_died_in_interval: Death time analysis
- trajectories_accelerator: Numba-accelerated trajectory generation

Note: These methods require trajectory storage which is being refactored.
"""

import numpy as np
from scipy.stats import gaussian_kde
from numba import jit


def _not_implemented_error():
    """Raise NotImplementedError with consistent message."""
    raise NotImplementedError(
        "Trajectory analysis methods are being refactored for efficiency. "
        "This functionality is temporarily disabled. "
        "The save_dist system and trajectory storage needs complete reimplementation. "
        "See SRtools.trajectory_analysis for future implementation."
    )


def get_damage_PDF(self, t, nvalues=40, pdf_method='kde'):
    """
    Returns the damage distribution for a given time t.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    t : float
        The time for which the damage distribution is calculated.
    nvalues : int
        The number of values in the damage distribution.
    pdf_method : str
        The method used to calculate the probability density function (PDF).
        Options are 'kde' (kernel density estimation) or 'hist' (histogram). 
        Default is 'kde'.

    Returns:
    --------
    tuple
        (damage_values, pdf) arrays
    """
    _not_implemented_error()


def create_distributions(self, dist_years, method='kde', nvalues=40):
    """
    Create distributions of damage values for each year in dist_years.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    dist_years : array-like
        An array of years for which to calculate the damage distributions.

    Returns:
    --------
    tuple
        (damage_values, distributions) where distributions is a dict
    """
    _not_implemented_error()


def age_distribution_by_damage(self, x_bin, nvalues=40, pdf_method='kde', age_limits=[50, 90]):
    """
    Calculate the age distribution of individuals with damage values in the range x_bin.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    x_bin : list
        A list of two values representing the range of damage values.
    nvalues : int
        The number of values in the damage distribution.
    pdf_method : str
        The method used to calculate the probability density function (PDF).
        Options are 'kde' (kernel density estimation) or 'hist' (histogram). 
        Default is 'kde'.
    age_limits : list
        Age range limits for analysis.

    Returns:
    --------
    tuple
        (age_values, pdf) arrays
    """
    _not_implemented_error()


def damage_transition_probabilities(self, t_interval, xbins, t_averaging_interval, 
                                   min_age=0, max_age='Auto'):
    """
    Calculates the transition probabilities between damage bins.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    t_interval : float
        The time interval for the transition. Should be an integer multiplication 
        of t_averaging_interval.
    xbins : list
        The damage bins to calculate the transition probabilities to.
    t_averaging_interval : float
        The time averaging interval to calculate the current damage.
    min_age : int
        The minimum age to consider in the transition probabilities calculation.
    max_age : str or float
        The maximum age to consider. 'Auto' uses self.t_end.

    Returns:
    --------
    array-like or None
        Matrix of transition probabilities between damage bins, or None if 
        insufficient data.
    """
    _not_implemented_error()


def get_sickspan(self, xd, min_age=0):
    """
    Calculate sick span (time spent in high damage state before death).
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    xd : float
        Damage threshold defining "sick" state.
    min_age : float
        Minimum age for analysis.

    Returns:
    --------
    tuple
        (sick_spans, hle_over_le) arrays
    """
    _not_implemented_error()


def mean_sickspan(self, xd):
    """
    Calculate mean sick span.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    xd : float
        Damage threshold defining "sick" state.

    Returns:
    --------
    float
        Mean sick span.
    """
    _not_implemented_error()


def mean_hle_over_le(self, xd, min_age=65):
    """
    Calculate mean healthy life expectancy over total life expectancy.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    xd : float
        Damage threshold defining "sick" state.
    min_age : float
        Minimum age for analysis.

    Returns:
    --------
    float
        Mean ratio of healthy to total life expectancy.
    """
    _not_implemented_error()


def kramer_damage_distribution(self, t, nvalues=40, stats=False):
    """
    Returns the theoretical damage distribution for a given time t using Kramer's theory.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    t : float
        The time for which the damage distribution is calculated (must be integer).
    nvalues : int
        The number of values in the damage distribution.
    stats : bool
        Whether to return distribution statistics.

    Returns:
    --------
    tuple
        (damage_values, pdf) or (damage_values, pdf, mean, std) if stats=True
    """
    _not_implemented_error()


def plotTrajectoriies(self, n_trajectories, randomize_index=False, resample_traj=10, 
                     mark_deaths=True, ax=None, fix_looks=True, kwargs={}, y=False):
    """
    Plot damage trajectories for visualization.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    n_trajectories : int
        Number of trajectories to plot.
    randomize_index : bool
        Whether to randomize which trajectories to plot.
    resample_traj : int
        Resampling factor for trajectory plotting.
    mark_deaths : bool
        Whether to mark death points.
    ax : matplotlib.axes
        Axes to plot on.
    fix_looks : bool
        Whether to fix plot appearance.
    kwargs : dict
        Additional plotting arguments.
    y : bool
        Whether to plot y trajectories instead of x.
    """
    _not_implemented_error()


def meanDamageTrajectories(self, time):
    """
    Calculate mean damage trajectories with moving average.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    time : float
        Time window for moving average.

    Returns:
    --------
    array-like
        Mean damage trajectories.
    """
    _not_implemented_error()


def agingStats(t, vals, val_critical=np.inf, nvalues=40, corr_threshold=0.5,
               calc_error_bars=False, confidence_level=0.95, n_resample=150, 
               pdf_method='hist'):
    """
    Calculate aging statistics for a given time point.
    
    DEPRECATED: This function is temporarily disabled pending refactoring.

    Parameters:
    -----------
    t : array-like
        Time points.
    vals : array-like
        Values at each time point for each person indexed as [time,person].
    val_critical : float
        Critical value. Defaults to np.inf.
    nvalues : int
        Number of value bins for distribution. Defaults to 40.
    corr_threshold : float
        Correlation threshold for autocorrelation analysis.
    calc_error_bars : bool
        Whether to calculate error bars.
    confidence_level : float
        Confidence level for error bars.
    n_resample : int
        Number of bootstrap resamples.
    pdf_method : str
        PDF calculation method ('hist' or 'kde').

    Returns:
    --------
    tuple
        Multiple statistics arrays and distributions.
    """
    _not_implemented_error()


def get_median_damage(self):
    """
    Returns the median damage of the trajectories.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Returns:
    --------
    array-like
        Median damage values over time.
    """
    _not_implemented_error()


def has_died_in_interval(self, t_interval):
    """
    Returns a boolean array indicating if an individual has died in the interval t_interval.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    t_interval : list
        A list of two values indicating the time interval for which to check 
        if an individual has died.

    Returns:
    --------
    array-like
        Boolean array indicating if an individual has died in the interval.
    """
    _not_implemented_error()


@jit(nopython=True, parallel=True)
def median_accelerator(medians, trajectories, xc):
    """
    Numba-accelerated median damage calculation.
    
    DEPRECATED: This function is temporarily disabled pending refactoring.

    Parameters:
    -----------
    medians : array-like
        Output array for median values.
    trajectories : array-like
        Damage trajectories.
    xc : float
        Critical damage threshold.
    """
    _not_implemented_error()


def trajectories_accelerator(noise, xt, s, dt, eta, t, beta, k, sdt, boundary='sticking'):
    """
    Numba-accelerated trajectory generation.
    
    DEPRECATED: This function is temporarily disabled pending refactoring.

    Parameters:
    -----------
    noise : array-like
        Random noise array.
    xt : array-like
        Trajectory array.
    s : int
        Number of time steps.
    dt : array-like
        Time step sizes.
    eta : float
        Damage production parameter.
    t : array-like
        Time array.
    beta : float
        Damage removal parameter.
    k : float
        Saturation parameter.
    sdt : array-like
        Square root of time steps.
    boundary : str
        Boundary condition type.

    Returns:
    --------
    array-like
        Generated trajectories.
    """
    _not_implemented_error()


def getTrajectories(self, boundary='sticking'):
    """
    Generate damage trajectories for the SR model.
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    boundary : str
        Boundary condition for trajectory generation.

    Returns:
    --------
    array-like
        Damage trajectories over time.
    """
    _not_implemented_error()


def y_traj_calc(self, y_gamma):
    """
    Calculate y trajectories (if this method exists).
    
    DEPRECATED: This method is temporarily disabled pending refactoring.

    Parameters:
    -----------
    y_gamma : float
        Gamma parameter for y calculation.

    Returns:
    --------
    array-like
        Y trajectories.
    """
    _not_implemented_error()
