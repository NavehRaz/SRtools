"""
SRtools MCMC Module

This module provides comprehensive MCMC (Markov Chain Monte Carlo) sampling functionality
for the Saturating Removal (SR) model of aging and mortality. It includes parameter estimation,
posterior sampling, and analysis tools for survival data.

Key Features:
- MCMC sampling using emcee ensemble sampler
- Parameter transformation and prior handling
- Automatic boundary detection for filtering
- Sample analysis and visualization
- Backend storage and restart capabilities
- Autocorrelation monitoring and convergence detection

The module supports both standard and transformed parameter spaces, with flexible
prior specification and likelihood functions for different metrics (Bayesian, survival).
"""

from . import SRmodellib as sr
import numpy as np
from . import SRmodellib_lifelines as srl
import emcee # a MCMC sampler
import time
import pandas as pd






def parse_theta(theta, set_params):
    """
    Parse parameter array according to fixed parameter dictionary.

    This function handles mixed parameter specification where some parameters
    are fixed (in set_params) and others are free (in theta). Parameters
    are parsed in the standard SR model order.

    Parameters:
    -----------
    theta : array-like
        Array of free parameters to be parsed
    set_params : dict
        Dictionary of fixed parameters. Keys can be: 'eta', 'beta', 'epsilon', 
        'xc', 'external_hazard'

    Returns:
    --------
    dict
        Dictionary containing all parameters with keys: 'eta', 'beta', 'epsilon', 
        'xc', 'external_hazard', 'extra'. Any additional parameters in theta
        beyond the standard 5 are stored in 'extra'. If 'external_hazard' is not
        a parameter (i.e., not present in set_params and not in theta), the returned
        dictionary will simply not include it, and theta will be of length 4.

    Notes:
    ------
    Parameter order: [eta, beta, epsilon, xc, external_hazard]
    If a parameter is in set_params, it's taken from there and skipped in theta.
    If 'external_hazard' is not a parameter (not in set_params and not in theta), 
    this function will still work and return a dictionary with only the four main 
    parameters ('eta', 'beta', 'epsilon', 'xc'), and theta should be of length 4.
    # Example: If set_params = {'beta': 0.5, 'external_hazard': 0.01} and theta = [1.0, 0.2, 0.05],
    # then the mapping will be:
    #   eta = 1.0         (from theta[0])
    #   beta = 0.5        (from set_params)
    #   epsilon = 0.2     (from theta[1])
    #   xc = 0.05         (from theta[2])
    #   external_hazard = 0.01 (from set_params)
    # Example: If set_params = {'beta': 0.5} and theta = [1.0, 0.2, 0.05, 0.01],
    # then the mapping will be:
    #   eta = 1.0         (from theta[0])
    #   beta = 0.5        (from set_params)
    #   epsilon = 0.2     (from theta[1])
    #   xc = 0.05         (from theta[2])
    #   external_hazard = 0.01 (from theta[3])
    # Example: If set_params = {'beta': 0.5} and theta = [1.0, 0.2, 0.05],
    # and 'external_hazard' is not a parameter, the mapping will be:
    #   eta = 1.0         (from theta[0])
    #   beta = 0.5        (from set_params)
    #   epsilon = 0.2     (from theta[1])
    #   xc = 0.05         (from theta[2])
    #   (no external_hazard key in the output)
    """
    param_names = ['eta', 'beta', 'epsilon', 'xc', 'external_hazard']
    param_values = {}
    idx = 0
    for name in param_names:
        if name in set_params:
            # parameter given in set_params, skip reading it from theta
            param_values[name] = set_params[name]
        else:
            # read next position of theta array if available
            if idx < len(theta):
                param_values[name] = theta[idx]
                idx += 1
            else:
                param_values[name] = None
    # store any extra entries in 'extra'
    if idx < len(theta):
        param_values['extra'] = theta[idx:]
    return param_values

def parse_theta_trans(theta_trans, set_params):
    """
    Parse transformed parameter array according to fixed parameter dictionary.
    
    This function handles mixed parameter specification for the transformed
    parameter space. Parameters are parsed in the transformed SR model order.
    
    Parameters:
    -----------
    theta_trans : array-like
        Array of transformed free parameters to be parsed
    set_params : dict
        Dictionary of fixed transformed parameters. Keys can be: 'xc_eta', 
        'eta_beta', 'xc2_epsilon', 'xc', 'external_hazard'
    
    Returns:
    --------
    dict
        Dictionary containing all transformed parameters with keys: 'xc_eta', 
        'eta_beta', 'xc2_epsilon', 'xc', 'external_hazard', 'extra'. Any 
        additional parameters in theta_trans beyond the standard 5 are stored 
        in 'extra'. If 'external_hazard' is not a parameter (not in set_params and not in theta_trans), 
        this function will still work and return a dictionary with only the four main 
        transformed parameters ('xc_eta', 'eta_beta', 'xc2_epsilon', 'xc'), and theta_trans should be of length 4.
    
    Notes:
    ------
    Transformed parameter order: [xc_eta, eta_beta, xc2_epsilon, xc, external_hazard]
    These correspond to: xc/eta, beta/eta, xc²/epsilon, xc, external_hazard
    # Example:
    # Suppose theta_trans = [0.5, 2.0, 0.1, 0.05, 0.01] and set_params = {'xc': 0.07}
    # The function will assign:
    #   'xc_eta'         = 0.5      (from theta_trans[0])
    #   'eta_beta'       = 2.0      (from theta_trans[1])
    #   'xc2_epsilon'    = 0.1      (from theta_trans[2])
    #   'xc'             = 0.07     (from set_params, not from theta_trans[3])
    #   'external_hazard'= 0.01     (from theta_trans[4])
    # If theta_trans has more than 5 elements, the extras are stored in 'extra'.
    # If set_params = {'xc': 0.07} and theta_trans = [0.5, 2.0, 0.1, 0.01] (no external_hazard),
    # the mapping will be:
    #   'xc_eta'         = 0.5      (from theta_trans[0])
    #   'eta_beta'       = 2.0      (from theta_trans[1])
    #   'xc2_epsilon'    = 0.1      (from theta_trans[2])
    #   'xc'             = 0.07     (from set_params)
    #   (no external_hazard key in the output)
    """
    param_names = ['xc_eta', 'eta_beta', 'xc2_epsilon', 'xc', 'external_hazard']
    param_values = {}
    idx = 0
    for name in param_names:
        if name in set_params:
            # parameter given in set_params, skip reading it from theta
            param_values[name] = set_params[name]
        else:
            # read next position of theta array if available
            if idx < len(theta_trans):
                param_values[name] = theta_trans[idx]
                idx += 1
            else:
                param_values[name] = None
    # store any extra entries in 'extra'
    param_values['extra'] = theta_trans[idx:]
    return param_values

def model(theta, n, nsteps, t_end, dataSet, sim=None, metric='baysian', time_range=None, 
          time_step_multiplier=1, parallel=False, dt=1, set_params=None, kwargs=None):
    """
    Evaluate SR model with given parameters and return score according to metric.
    
    This function simulates the Saturating Removal model with the provided parameters
    and compares the simulation results to the observed dataset using the specified
    metric. It handles parameter parsing and simulation setup.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters [eta, beta, epsilon, xc, external_hazard, ...]
    n : int
        Number of individuals to simulate
    nsteps : int
        Number of time steps for simulation
    t_end : float
        End time for simulation
    dataSet : object
        Dataset object containing observed data for comparison
    sim : object, optional
        Pre-computed simulation object. If None, simulation is performed
    metric : str, default='baysian'
        Metric for comparison: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    parallel : bool, default=False
        Whether to use parallel simulation
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    kwargs : dict, optional
        Additional keyword arguments
    
    Returns:
    --------
    float
        Score according to the specified metric. Returns -inf for invalid parameters.
    
    Notes:
    ------
    The function returns -inf if 1/beta < time_step_size or if any NaN values
    are encountered in the probability calculation.
    """
    if set_params is None:
        set_params = {}
    # parse parameters
    pv = parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    theta_sr = np.array([eta, beta, epsilon, xc])
    time_step_size = t_end/(nsteps*time_step_multiplier)
    if 1/beta < time_step_size:
        return -np.inf
    sim = getSr(theta_sr, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel) if sim is None else sim
    
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.any(np.isnan(tprob)):
        return -np.inf

    return tprob



def lnlike(theta, n, nsteps, t_end, dataSet, metric='baysian', time_range=None, 
           time_step_multiplier=1, sim=None, dt=1, set_params=None, model_func=model, parallel=False, kwargs=None):
    """
    Calculate log-likelihood for MCMC sampling.
    
    This function computes the log-likelihood of the model parameters given the
    observed data. It wraps the model function and handles metric-specific
    transformations.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters [eta, beta, epsilon, xc, external_hazard, ...]
    n : int
        Number of individuals to simulate
    nsteps : int
        Number of time steps for simulation
    t_end : float
        End time for simulation
    dataSet : object
        Dataset object containing observed data
    metric : str, default='baysian'
        Metric for comparison: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    sim : object, optional
        Pre-computed simulation object
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    model_func : callable, default=model
        Function to evaluate model
    kwargs : dict, optional
        Additional keyword arguments
    
    Returns:
    --------
    float
        Log-likelihood value. For 'survival' metric, returns 1/score.
    
    Notes:
    ------
    For 'survival' metric, the likelihood is inverted (1/score) to convert
    from distance to likelihood.
    """
    if set_params is None:
        set_params = {}
    
    LnLike = model_func(theta, n, nsteps, t_end, dataSet,sim=sim, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params, parallel=parallel, kwargs=kwargs)
    if metric =='survival':
        LnLike =1/LnLike
    return LnLike

def lnlikeTransformed(theta_trans, n, nsteps, t_end, dataSet, metric='baysian', time_range=None, 
                      time_step_multiplier=1, sim=None, dt=1, set_params=None, model_func=model, parallel=False, kwargs=None):
    """
    Calculate log-likelihood for transformed parameters in MCMC sampling.
    
    This function computes the log-likelihood for the transformed parameter space.
    It first transforms the parameters back to the original space, then computes
    the likelihood using the standard model function.
    
    Parameters:
    -----------
    theta_trans : array-like
        Transformed model parameters [xc_eta, eta_beta, xc2_epsilon, xc, external_hazard, ...]
    n : int
        Number of individuals to simulate
    nsteps : int
        Number of time steps for simulation
    t_end : float
        End time for simulation
    dataSet : object
        Dataset object containing observed data
    metric : str, default='baysian'
        Metric for comparison: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    sim : object, optional
        Pre-computed simulation object
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed transformed parameters
    model_func : callable, default=model
        Function to evaluate model
    kwargs : dict, optional
        Additional keyword arguments
    
    Returns:
    --------
    float
        Log-likelihood value for the transformed parameters
    
    Notes:
    ------
    The function first calls inv_transform() to convert back to original parameter
    space, then computes the likelihood using the standard model function.
    """
    if set_params is None:
        set_params = {}
    theta = inv_transform(theta_trans, set_params)
    
    LnLike = model_func(theta, n, nsteps, t_end, dataSet,sim=sim, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params, parallel=parallel, kwargs=kwargs)
    if metric =='survival':
        LnLike =1/LnLike
    return LnLike

def transform(theta, set_params={}):
    """
    Transform parameters from original to transformed space.
    
    This function converts the standard SR model parameters to a transformed
    parameter space that may be more suitable for MCMC sampling. The transformation
    helps with parameter correlations and sampling efficiency.
    
    Parameters:
    -----------
    theta : array-like
        Original parameters [eta, beta, epsilon, xc, external_hazard, ...]
    set_params : dict, default={}
        Dictionary of fixed parameters
    
    Returns:
    --------
    ndarray
        Transformed parameters [xc_eta, eta_beta, xc2_epsilon, xc, external_hazard, ...]
    
    Notes:
    ------
    Transformation:
    - xc_eta = xc/eta
    - eta_beta = beta/eta  
    - xc2_epsilon = xc²/epsilon
    - xc = xc (unchanged)
    - external_hazard = external_hazard (unchanged)
    - extra parameters are preserved unchanged
    """
    pv = parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    if 'external_hazard' in pv:
        external_hazard = pv['external_hazard']
    else:
        external_hazard = None
    xc_eta = xc/eta
    beta_eta = beta/eta
    xc2_epsilon = xc**2/epsilon
    values = [xc_eta, beta_eta, xc2_epsilon,xc]
    if external_hazard is not None:
        values.append(external_hazard)
    if 'extra' in pv:
        return np.array(values + pv['extra'].tolist())
    else:
        return np.array(values)

def inv_transform(theta_trans, set_params={}):
    """
    Inverse transform parameters from transformed to original space.
    
    This function converts the transformed parameters back to the original
    SR model parameter space. It is the inverse of the transform() function.
    
    Parameters:
    -----------
    theta_trans : array-like
        Transformed parameters [xc_eta, eta_beta, xc2_epsilon, xc, external_hazard, ...]
    set_params : dict, default={}
        Dictionary of fixed transformed parameters
    
    Returns:
    --------
    ndarray
        Original parameters [eta, beta, epsilon, xc, external_hazard, ...]
    
    Notes:
    ------
    Inverse transformation:
    - eta = xc / xc_eta
    - beta = eta_beta * eta
    - epsilon = xc² / xc2_epsilon
    - xc = xc (unchanged)
    - external_hazard = external_hazard (unchanged)
    - extra parameters are preserved unchanged
    """
    pv = parse_theta_trans(theta_trans, set_params)
    for key in ['xc', 'xc_eta', 'eta_beta', 'xc2_epsilon']:
        if pv.get(key) is None:
            print(f"Warning: {key} is NoneType in inv_transform.")
    eta = pv['xc'] / pv['xc_eta']
    beta = pv['eta_beta'] * eta
    epsilon = pv['xc']**2 / pv['xc2_epsilon']
    xc = pv['xc']
    if 'external_hazard' in pv:
        external_hazard = pv['external_hazard']
    else:
        external_hazard = None
    values = [eta, beta, epsilon, xc]
    if external_hazard is not None:
        values.append(external_hazard)
    if 'extra' in pv:
        return np.array(values + pv['extra'].tolist())
    else:
        return np.array(values)

def lnprior(theta, prior):
    """
    Calculate log-prior for MCMC sampling.
    
    This function evaluates the log-prior probability for the given parameters.
    It implements uniform priors within the specified bounds.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters to evaluate
    prior : list
        List of prior bounds for each parameter. Each element should be
        [min_value, max_value] for the corresponding parameter in theta
    
    Returns:
    --------
    float
        Log-prior value. Returns 0.0 if all parameters are within bounds,
        -inf if any parameter is outside bounds.
    
    Raises:
    -------
    ValueError
        If the length of theta doesn't match the length of prior
    
    Notes:
    ------
    The function implements uniform priors, so the log-prior is constant
    (0.0) when all parameters are within bounds, and -inf otherwise.
    """
    #check theta is same length as prior
    if len(theta) != len(prior):
        raise ValueError("The length of theta should be the same as the length of the prior.")

    
    for i, (param, bounds) in enumerate(zip(theta, prior)):
        if not (bounds[0] < param < bounds[1]):
            return -np.inf
    return 0.0
    


def lnprob(theta, n, nsteps, t_end, dataSet, metric='survival', time_range=None, 
           time_step_multiplier=1, prior=None, dt=1, set_params=None, model_func=model, log_samples=False, parallel=False, kwargs=None):
    """
    Calculate log-posterior for MCMC sampling.
    
    This function computes the log-posterior probability by combining the
    log-prior and log-likelihood: log(posterior) = log(prior) + log(likelihood).
    
    Parameters:
    -----------
    theta : array-like
        Model parameters [eta, beta, epsilon, xc, external_hazard, ...]
    n : int
        Number of individuals to simulate
    nsteps : int
        Number of time steps for simulation
    t_end : float
        End time for simulation
    dataSet : object
        Dataset object containing observed data
    metric : str, default='survival'
        Metric for comparison: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    prior : list
        List of prior bounds for each parameter
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    model_func : callable, default=model
        Function to evaluate model
    kwargs : dict, optional
        Additional keyword arguments
    
    Returns:
    --------
    float
        Log-posterior value. Returns -inf if prior is invalid or likelihood
        evaluation fails.
    
    Notes:
    ------
    The function first evaluates the prior. If the prior is -inf (invalid
    parameters), it returns -inf immediately without computing the likelihood.
    """
    if set_params is None:
        set_params = {}
    
    # If log_samples is True, exponentiate theta before processing
    if log_samples:
        theta = np.exp(theta)
    
    lp = lnprior(theta, prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta , n, nsteps, t_end, dataSet=dataSet, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt,model_func=model_func, set_params=set_params, parallel=parallel, kwargs=kwargs)

def lnprobTransformed(theta_trans , n, nsteps, t_end, dataSet, metric = 'survival', time_range=None, time_step_multiplier = 1,prior = None, dt=1, set_params=None,model_func=model, log_samples=False, parallel=False, kwargs=None):
    """
    The posterior function for the MCMC sampler.
    """
    if set_params is None:
        set_params = {}
    
    # If log_samples is True, exponentiate theta_trans before processing
    if log_samples:
        theta_trans = np.exp(theta_trans)
    
    lp = lnprior(theta_trans, prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikeTransformed(theta_trans , n, nsteps, t_end, dataSet=dataSet, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params,model_func=model_func, parallel=parallel, kwargs=kwargs)

def draw_param(bins, log_space=True, log_samples=False):
    """
    Draw random parameters from specified bins.
    
    This function randomly samples parameters from the provided bins. It first
    selects a random bin index, then draws parameters from that bin for all
    dimensions using either uniform or log-uniform sampling.
    
    Parameters:
    -----------
    bins : list
        List of parameter bins. Each element should be a list of [min, max] pairs
        for each parameter dimension. Can also be a list of lists of bins for
        multiple bin options per parameter.
    log_space : bool, default=True
        If True, sample in log space (log-uniform distribution)
        If False, sample in linear space (uniform distribution)
    
    Returns:
    --------
    ndarray
        Array of randomly drawn parameters
    
    Raises:
    -------
    Exception
        If there's an error during parameter drawing
    
    Notes:
    ------
    The function randomly selects a bin index and uses the same index for all
    parameters. This ensures that all parameters are drawn from the same
    "bin configuration".
    """
    bin_index = np.random.randint(0, len(bins[0]))
    try:
        if log_space:
            theta = np.array([np.exp(np.random.uniform(np.log(bins[i][bin_index][0]), np.log(bins[i][bin_index][1]))) for i in range(len(bins))])
        else:
            theta = np.array([np.random.uniform(bins[i][bin_index][0], bins[i][bin_index][1]) for i in range(len(bins))])
        
        # If log_samples is True, log the drawn parameters
        if log_samples:
            theta = np.log(theta)
            
    except Exception as e:
        print("Exception occurred in draw_param. bins:", bins)
        raise
    return theta

#fix this function so that it take the variations as a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements. the usage of specific var_eta, varbeta etc should be dropped and the variations should be used instead, the function should return a list of ndims lists of n_bins lists of 2 elements.
def get_bins_from_seed(seed, ndims=4, variations=[0.7, 1.3]):
    """
    Get the bins from the seed theta. The bins are the seed multiplied by the variations.
    If variations is a list of 2 elements, the same variations are applied to all parameters.
    If variations is a list of ndims lists of 2 elements, each parameter has its own variations.
    If variations is a list of ndims lists of n_bins lists of 2 elements, each parameter has its own bins.

    Returns:
    - bins (list): A list of ndims lists of n_bins lists of 2 elements.
    """
    if len(variations) == 2:
        variations = [variations] * ndims
    elif len(variations) != ndims:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")

    bins = []
    lengths  = []
    for i in range(ndims):
        if len(variations[i]) == 2 and isinstance(variations[i][0], (int, float)):
            bins.append([[seed[i] * variations[i][0], seed[i] * variations[i][1]]])
        else:
            bins.append([[seed[i] * var[0], seed[i] * var[1]] for var in variations[i]])
            lengths.append(len(variations[i]))
    if len(set(lengths)) > 1 and len(lengths) > 1:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")
    if len(lengths) > 0 and len(lengths)< ndims:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")

    return bins

   

def getSampler(nwalkers, num_mcmc_steps, dataSet, seed=None, npeople=10000, nsteps=5000, t_end=None, ndim=4,
               bins=None, variations=[0.7, 1.3], draw_params_in_log_space=True, prior_generator=None,
               back_end_file=None, metric='baysian', time_range=None, time_step_multiplier=1, prior=None, 
               restartFromBackEnd=False, progress=False, transformed=False, dt=1, set_params=None, model_func=model, 
               log_samples=False, parallel=False, **kwargs):
    """
    Create and run MCMC sampler for SR model parameter estimation.
    
    This function sets up and runs an MCMC sampling procedure using the emcee
    ensemble sampler to estimate the posterior distribution of SR model parameters.
    
    Parameters:
    -----------
    nwalkers : int
        Number of walkers in the ensemble sampler
    num_mcmc_steps : int
        Number of MCMC steps to run
    dataSet : object
        Dataset object containing observed data
    seed : array-like, optional
        Seed parameters for generating initial bins. If None, bins must be provided
    npeople : int, default=10000
        Number of individuals to simulate
    nsteps : int, default=5000
        Number of time steps for simulation
    t_end : float, optional
        End time for simulation. If None, uses dataSet.t_end
    ndim : int, default=4
        Number of dimensions (parameters) to sample
    bins : list, optional
        Parameter bins for initial walker positions. If None, generated from seed
    variations : list, default=[0.7, 1.3]
        Variations for generating bins from seed. Can be:
        - List of 2 elements: applied to all parameters
        - List of ndim lists of 2 elements: specific for each parameter
        - List of ndim lists of n_bins lists of 2 elements: multiple bins per parameter
    draw_params_in_log_space : bool, default=True
        Whether to draw initial parameters in log space
    prior_generator : object, optional
        Prior generator object for sampling initial positions
    back_end_file : str, optional
        Path to HDF5 file for storing sampler backend
    metric : str, default='baysian'
        Metric for likelihood calculation: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    prior : list or float, optional
        Prior bounds. Can be:
        - List of [min, max] pairs for each parameter
        - Float: expansion factor for generating bounds from bins
        - None: uses default expansion factor of 10
    restartFromBackEnd : bool, default=False
        Whether to restart from existing backend file
    progress : bool, default=False
        Whether to show progress bar
    transformed : bool, default=False
        Whether to use transformed parameter space
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    model_func : callable, default=model
        Function to evaluate model
    log_samples : bool, default=False
        If True, log the initial sample values and exponentiate theta in lnprob/lnprobTransformed
    **kwargs : dict
        Additional keyword arguments (for backward compatibility)
    
    Returns:
    --------
    emcee.EnsembleSampler
        Configured and run MCMC sampler object
    
    Notes:
    ------
    The function handles backward compatibility with individual bin parameters
    (eta_bins, beta_bins, etc.) and automatically sets external_hazard from
    dataSet if not provided and ndim=4.
    
    The sampler can be configured to use either the original or transformed
    parameter space, with appropriate likelihood and prior functions.
    """
    ####THIS IS FOR BACKWARDS COMPATIBILITY.##### 
    eta_bins = kwargs.get('eta_bins', None)
    beta_bins = kwargs.get('beta_bins', None)
    epsilon_bins = kwargs.get('epsilon_bins', None)
    xc_bins = kwargs.get('xc_bins', None)
    
    if prior_generator is not None:
        pass
    elif bins is None and all(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        bins = [eta_bins, beta_bins, epsilon_bins, xc_bins]
    elif bins is None and any(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        raise ValueError("Either all bins should be None or all of them specified.")
    ###########################################

    #CHANGE THIS AFTER YOU SORT VARIABLE LENGTH THETA
    if set_params is None and ndim == 4:
        set_params = {'external_hazard': dataSet.external_hazard}
        
    if prior_generator is None:
        #check if only some of the bins are specified and raise an exception if so.
        if bins is not None and len(bins) != ndim:
            raise ValueError("Either all bins should be specified or None of them.")

        if bins is None and seed is None:
            raise ValueError("Either the seed or the bins should be specified.")
        
        if bins is None:
            bins = get_bins_from_seed(seed, variations = variations)

    if t_end is None:
        t_end = dataSet.t_end

    #check if the prior is specified and generate a default prior if not.
    #  If the prior is a scalar a default prior is generated using it, otherwise 10 is used. 
    if prior is None:
        prior = 10
    if prior_generator is not None:
        if type(prior) is float or type(prior) is int:
            prior = prior_generator.getBounds(expansion_factor=prior)
        else:
            prior = prior_generator.getBounds()
    elif  type(prior) is float or type(prior) is int:
        v= prior
        prior = [[np.min(bins[i])/v,np.max(bins[i])*v] for i in range(ndim)]
    elif len(prior) != ndim:
        raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
    else:
        for i,p in enumerate(prior):
            if type(p) is float or type(p) is int:
                prior[i] = [np.min(bins[i])/p,np.max(bins[i])*p]
            elif len(p) <=2:
                prior[i] = [np.min(bins[i])/p[0],np.max(bins[i])*p[-1]]
            elif len(p) >2:
                raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")

    

    args = [ npeople, nsteps, t_end, dataSet, metric, time_range, time_step_multiplier, prior, dt, set_params,model_func,log_samples,parallel,kwargs]
    # Set the initial positions of the walkers
    if prior_generator is None:
        pos = [draw_param(bins=bins,log_space=draw_params_in_log_space, log_samples=log_samples) for i in range(nwalkers)]
    else:
        pos = prior_generator.sample(n_samples=nwalkers,temperature=0.5)
    
    
    if transformed:
        lp = lnprobTransformed
    else:
        lp = lnprob

    start=time.time()
    if back_end_file is not None and not restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args,backend=backend)
        sampler.run_mcmc(pos, num_mcmc_steps, progress=progress)
    elif back_end_file is not None and restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp,backend=backend, args=args)
        pos, prob, state = sampler.run_mcmc(None, num_mcmc_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args)
        sampler.run_mcmc(pos, num_mcmc_steps,progress='notebook')
    end = time.time()
    print("Time elapsed: ", end-start)
    return sampler


def getSamplerAutoCorrMon(nwalkers, num_mcmc_steps, dataSet, seed=None, npeople=10000, nsteps=5000, t_end=None, ndim=4,
                          bins=None, variations=[0.7, 1.3], draw_params_in_log_space=True,
                          back_end_file=None, metric='baysian', time_range=None, time_step_multiplier=1, prior=None,
                          restartFromBackEnd=False, progress=False, plot_correlations=False, transformed=False, dt=1,
                          set_params=None, model_func=model, log_samples=False, parallel=False, **kwargs):
    """
    Create and run MCMC sampler with autocorrelation monitoring.
    
    This function sets up and runs an MCMC sampling procedure with automatic
    convergence detection based on autocorrelation analysis. The sampler stops
    when the autocorrelation time has converged.
    
    Parameters:
    -----------
    nwalkers : int
        Number of walkers in the ensemble sampler
    num_mcmc_steps : int
        Maximum number of MCMC steps to run
    dataSet : object
        Dataset object containing observed data
    seed : array-like, optional
        Seed parameters for generating initial bins. If None, bins must be provided
    npeople : int, default=10000
        Number of individuals to simulate
    nsteps : int, default=5000
        Number of time steps for simulation
    t_end : float, optional
        End time for simulation. If None, uses dataSet.t_end
    ndim : int, default=4
        Number of dimensions (parameters) to sample
    bins : list, optional
        Parameter bins for initial walker positions. If None, generated from seed
    variations : list, default=[0.7, 1.3]
        Variations for generating bins from seed
    draw_params_in_log_space : bool, default=True
        Whether to draw initial parameters in log space
    back_end_file : str, optional
        Path to HDF5 file for storing sampler backend
    metric : str, default='baysian'
        Metric for likelihood calculation: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    prior : list or float, optional
        Prior bounds. Can be list of [min, max] pairs or expansion factor
    restartFromBackEnd : bool, default=False
        Whether to restart from existing backend file
    progress : bool, default=False
        Whether to show progress bar
    plot_correlations : bool, default=False
        Whether to plot autocorrelation convergence
    transformed : bool, default=False
        Whether to use transformed parameter space
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    model_func : callable, default=model
        Function to evaluate model
    log_samples : bool, default=False
        If True, log the initial sample values and exponentiate theta in lnprob/lnprobTransformed
    **kwargs : dict
        Additional keyword arguments (for backward compatibility)
    
    Returns:
    --------
    tuple
        (sampler, autocorr, niters, index) where:
        - sampler: emcee.EnsembleSampler object
        - autocorr: array of autocorrelation time estimates
        - niters: array of iteration numbers
        - index: number of convergence checks performed
    
    Notes:
    ------
    The sampler automatically stops when:
    1. All autocorrelation times are less than iteration/100
    2. Relative change in autocorrelation time is less than 1%
    
    Convergence is checked every 100 iterations.
    """
    import matplotlib.pyplot as plt

    ####THIS IS FOR BACKWARDS COMPATIBILITY.##### 
    eta_bins = kwargs.get('eta_bins', None)
    beta_bins = kwargs.get('beta_bins', None)
    epsilon_bins = kwargs.get('epsilon_bins', None)
    xc_bins = kwargs.get('xc_bins', None)

    if bins is None and all(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        bins = [eta_bins, beta_bins, epsilon_bins, xc_bins]
    elif bins is None and any(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        raise ValueError("Either all bins should be None or all of them specified.")
    ###########################################

    if set_params is None and ndim == 4:
        set_params = {'external_hazard': dataSet.external_hazard}

    if bins is None and seed is None:
        raise ValueError("Either the seed or the bins should be specified.")

    if bins is None:
        bins = get_bins_from_seed(seed, variations=variations)

    if t_end is None:
        t_end = dataSet.t_end

    if prior is None:
        prior = 10
    if type(prior) is float or type(prior) is int:
        v = prior
        prior = [[np.min(bins[i]) / v, np.max(bins[i]) * v] for i in range(ndim)]
    elif len(prior) != ndim:
        raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
    else:
        for i, p in enumerate(prior):
            if len(p) > 2:
                raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
            elif type(p) is float or type(p) is int:
                prior[i] = [np.min(bins[i]) / p, np.max(bins[i]) * p]

    args = [npeople, nsteps, t_end, dataSet, metric, time_range, time_step_multiplier, prior, dt, set_params, model_func, log_samples, parallel, kwargs]
    pos = [draw_param(bins=bins, log_space=draw_params_in_log_space, log_samples=log_samples) for i in range(nwalkers)]

    if transformed:
        lp = lnprobTransformed
    else:
        lp = lnprob

    start = time.time()
    if back_end_file is not None and not restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args, backend=backend)
    elif back_end_file is not None and restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, backend=backend, args=args)
        pos, prob, state = sampler.run_mcmc(None, num_mcmc_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args)

    index = 0
    niters = np.empty(num_mcmc_steps // 100)
    autocorr = np.empty(num_mcmc_steps // 100)
    old_tau = np.inf
    if plot_correlations:
        fig, ax = plt.subplots()

    for sample in sampler.sample(pos, iterations=num_mcmc_steps, progress=progress):
        if sampler.iteration % 100:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        niters[index] = sampler.iteration
        index += 1

        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if plot_correlations:
            ax.cla()
            ax.plot(niters[:index], autocorr[:index])
            ax.plot(niters[:index], niters[:index] / 100, linestyle='dashed', color='gray')
            ax.set_xlabel('iteration')
            ax.set_ylabel(r'autocorrelation time estimate')

        if converged:
            break
        old_tau = tau

    end = time.time()
    print("Time elapsed: ", end - start)
    return sampler, autocorr, niters, index



def loadSamples(back_end_file, flat=True, thin=1, discard=0):
    """
    Load MCMC samples from HDF5 backend file.
    
    This function loads the MCMC samples and log probabilities from an emcee
    HDF5 backend file with options for thinning and discarding initial samples.
    
    Parameters:
    -----------
    back_end_file : str
        Path to the HDF5 backend file
    flat : bool, default=True
        Whether to flatten the samples (combine all walkers)
    thin : int, default=1
        Thinning factor (take every nth sample)
    discard : int, default=0
        Number of initial samples to discard
    
    Returns:
    --------
    tuple
        (samples, lnprobs) where:
        - samples: array of parameter samples
        - lnprobs: array of log probability values
    """
    backend = emcee.backends.HDFBackend(back_end_file)
    samples = backend.get_chain(flat = flat, thin =thin, discard = discard)
    lnprobs = backend.get_log_prob(flat = flat,thin =thin, discard = discard)
    return samples,lnprobs


def loadSamplesFromDir(dirs, best=True, flat=True, n_per_file=800, thin=1, discard=0, debug=False):
    """
    Load MCMC samples from multiple HDF5 files in directories.
    
    This function loads samples from all HDF5 files in the specified directories
    and concatenates them into a single array. Optionally selects only the best
    samples from each file.
    
    Parameters:
    -----------
    dirs : str or list
        Directory path(s) containing HDF5 files
    best : bool, default=True
        Whether to select only the best samples from each file
    flat : bool, default=True
        Whether to flatten the samples (combine all walkers)
    n_per_file : int, default=800
        Number of best samples to take from each file (if best=True)
    thin : int, default=1
        Thinning factor (take every nth sample)
    discard : int, default=0
        Number of initial samples to discard
    debug : bool, default=False
        Whether to print detailed error information
    
    Returns:
    --------
    tuple
        (samples, lnprobs) where:
        - samples: concatenated array of parameter samples
        - lnprobs: concatenated array of log probability values
    
    Notes:
    ------
    If best=True, the function selects the n_per_file samples with the highest
    log probabilities from each file. This is useful for combining results from
    multiple MCMC runs while keeping only the most likely parameter sets.
    """
    import os
    samples = []
    lnprobs = []
    if type(dirs) is not list:
        dirs = [dirs]
    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".h5"):
                try:
                    samples_,lnprobs_ = loadSamples(os.path.join(dir,file),flat = flat)
                except:
                    print("Error loading file: ",file)
                    if debug:
                        import traceback
                        print("Error details:")
                        traceback.print_exc()
                        continue
                # If we want the best thetas, we only want the 100 thetas with the highest log probability.
                if best:
                    idx = np.argsort(lnprobs_)[-n_per_file:]
                    samples_ = samples_[idx]
                    lnprobs_ = lnprobs_[idx]
                samples.append(samples_)
                lnprobs.append(lnprobs_)
    if flat:
        samples = np.concatenate(samples)
        lnprobs = np.concatenate(lnprobs)
    return samples,lnprobs


def getSr(theta, n=25000, nsteps=6000, t_end=110, external_hazard=np.inf, time_step_multiplier=1, npeople=None, parallel=False, step_size=None, method='brownian_bridge'):
    """
    Generate SR model simulation with given parameters.

    This function creates a Saturating Removal model simulation using the provided
    parameters. It's a convenience wrapper around the SR_lf simulation function.

    Parameters:
    -----------
    theta : array-like
        Model parameters [eta, beta, epsilon, xc]
    n : int, default=25000
        Number of individuals to simulate
    nsteps : int, default=6000
        Number of time steps for simulation
    t_end : float, default=110
        End time for simulation
    external_hazard : float, default=np.inf
        External hazard rate
    time_step_multiplier : int, default=1
        Multiplier for time step size
    npeople : int, optional
        Alternative parameter name for n (for backward compatibility)
    parallel : bool, default=False
        Whether to use parallel simulation
    step_size : float, optional
        If provided, overrides nsteps and time_step_multiplier to achieve the desired step size.
    method : str, default='brownian_bridge'
        Method to use for death times calculation. Options:
        - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
        - 'euler': Standard Euler method

    Returns:
    --------
    object
        SR model simulation object

    Notes:
    ------
    The function uses kappa=0.5 as a fixed parameter. If npeople is provided,
    it overrides the n parameter.

    If step_size is given, nsteps and time_step_multiplier are ignored and recalculated so that
    t_end/(nsteps*time_step_multiplier) = step_size. If nsteps*time_step_multiplier <= 6000, time_step_multiplier=1, else
    increase time_step_multiplier until nsteps <= 6000. Both nsteps and time_step_multiplier are integers.
    """
    if npeople is not None:
        n = npeople

    if step_size is not None:
        # Calculate total number of steps needed
        total_steps = int(np.ceil(t_end / step_size))
        # Try to keep nsteps <= 6000 by increasing time_step_multiplier
        nsteps = total_steps
        time_step_multiplier = 1
        while nsteps > 6000:
            time_step_multiplier += 1
            nsteps = int(np.ceil(total_steps / time_step_multiplier))
        # Ensure nsteps and time_step_multiplier are at least 1
        nsteps = max(1, nsteps)
        time_step_multiplier = max(1, time_step_multiplier)

    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    sim = srl.SR_lf(
        eta=eta,
        beta=beta,
        epsilon=epsilon,
        xc=xc,
        kappa=0.5,
        npeople=n,
        nsteps=nsteps,
        t_end=t_end,
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
        method=method
    )
    return sim

def save_best_thetas(samples,lnprobs,save_path,best_threshold=0.9):
    """
    Save the best thetas from the samples.
    """
    best_thetas = samples[np.where(lnprobs>best_threshold)]
    np.save(save_path,best_thetas)
    return best_thetas

def load_thetas(path):
    return np.load(path)

def karin_theta():
    return [0.49275,54.75,51.83,17]

def get_params_from_thetas(thetas):
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])
    # INSERT_YOUR_CODE
    # Handle extra params if theta is longer
    extras = []
    if len(thetas[0]) > 4:
        for i in range(len(thetas[0])-4):
            extras.append(np.array([theta[4+i] for theta in thetas]))
    return [etas,betas,epsilons,xcs]+extras


def plot_thetas_and_probs(thetas, probs, marked_thetas = None,marked_probs =None, annotations = None, xscale = 'linear',yscale = 'log',threshold = None):
    import matplotlib.pyplot as plt
    #annotations should be the same length as marked_thetas
    
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    if threshold is not None:
        thetas = thetas[probs>threshold]
        probs = probs[probs>threshold]

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas/etas
    bxe = betas*xcs/epsilons
    Fx = betas**2/(etas*xcs)
    Dx = betas*epsilons/(etas*(xcs**2))

    td = xcs**2/epsilons
    bke = betas*kappa/epsilons
    Fk = betas**2/(etas*kappa)
    Dk = betas*epsilons/(etas*(kappa**2))

    s = (xcs**1.5)*(etas**0.5)/epsilons
    slope = etas*xcs/epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk

    marked_etas = np.array([theta[ETA] for theta in marked_thetas])
    marked_betas = np.array([theta[BETA] for theta in marked_thetas])
    marked_epsilons = np.array([theta[EPSILON] for theta in marked_thetas] )
    marked_xcs = np.array([theta[XC] for theta in marked_thetas])

    marked_t3 = marked_betas/marked_etas
    marked_bxe = marked_betas*marked_xcs/marked_epsilons
    marked_Fx = marked_betas**2/(marked_etas*marked_xcs)
    marked_Dx = marked_betas*marked_epsilons/(marked_etas*(marked_xcs**2))

    marked_td = marked_xcs**2/marked_epsilons
    marked_bke = marked_betas*kappa/marked_epsilons
    marked_Fk = marked_betas**2/(marked_etas*kappa)
    marked_Dk = marked_betas*marked_epsilons/(marked_etas*(kappa**2))

    marked_s = (marked_xcs**1.5)*(marked_etas**0.5)/marked_epsilons
    marked_slope = marked_etas*marked_xcs/marked_epsilons
    marked_Xceps = marked_xcs/marked_epsilons
    marked_Pk = marked_Fk/marked_Dk

    fig, axs = plt.subplots(5, 4,figsize=(35,30))
    arrow_color='red'
    headlength = 0.5
    headwidth = 0.25
    annot_fontsize = 16
    xytext_shift = 5
    for i in range(4):
        axs[0, i].hist([etas, betas, epsilons, xcs][i], bins=50)
        axs[0, i].set_xlabel(['ETA', 'BETA', 'EPSILON', 'XC'][i])
        axs[0, i].set_ylabel('Frequency')

        axs[1, i].scatter(probs, [etas, betas, epsilons, xcs][i], c=probs, cmap='viridis', s=1)
        axs[1, i].set_ylabel(['ETA', 'BETA', 'EPSILON', 'XC'][i])
        axs[1, i].set_xlabel('lnprobs')
        axs[1, i].set_yscale(yscale)
        axs[1, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[1, i].scatter(marked_probs, [marked_etas, marked_betas, marked_epsilons, marked_xcs][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[1, i].annotate(txt, (marked_probs[j], [marked_etas, marked_betas, marked_epsilons, marked_xcs][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                   fontsize=annot_fontsize)

        axs[2, i].scatter(probs, [t3, bxe, Fx, Dx][i], c=probs, cmap='viridis', s=1)
        axs[2, i].set_ylabel(['beta/eta', 'intercept=beta*xc/eps', 'Fx=beta^2/(eta*xc)', 'Dx=beta*eps/(eta*xc^2)'][i])
        axs[2, i].set_xlabel('lnprobs')
        axs[2, i].set_yscale(yscale)
        axs[2, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[2, i].scatter(marked_probs, [marked_t3, marked_bxe, marked_Fx, marked_Dx][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[2, i].annotate(txt, (marked_probs[j], [marked_t3, marked_bxe, marked_Fx, marked_Dx][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.01, headlength=headlength, headwidth=headwidth),
                                   fontsize=annot_fontsize)

        axs[3, i].scatter(probs, [td, bke, Fk, Dk][i], c=probs, cmap='viridis', s=1)
        axs[3, i].set_ylabel(['xc^2/eps', 'beta*kappa/eps', 'Fk=beta^2/(eta*kappa)', 'Dk=beta*eps/(eta*kappa^2)'][i])
        axs[3, i].set_xlabel('lnprobs')
        axs[3, i].set_yscale(yscale)
        axs[3, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[3, i].scatter(marked_probs, [marked_td, marked_bke, marked_Fk, marked_Dk][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[3, i].annotate(txt, (marked_probs[j], [marked_td, marked_bke, marked_Fk, marked_Dk][i][j]),
                                    xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                    fontsize=annot_fontsize)

        axs[4, i].scatter(probs, [s, slope, Xceps, Pk][i], c=probs, cmap='viridis', s=1)
        axs[4, i].set_ylabel(['s=(xc^1.5*eta^0.5)/eps', 'slope=eta*xc/eps', 'xc/eps', 'Fk/Dk'][i])
        axs[4, i].set_xlabel('lnprobs')
        axs[4, i].set_yscale(yscale)
        axs[4, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[4, i].scatter(marked_probs, [marked_s, marked_slope, marked_Xceps, marked_Pk][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[4, i].annotate(txt, (marked_probs[j], [marked_s, marked_slope, marked_Xceps, marked_Pk][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                    fontsize=annot_fontsize)

def getStats(thetas, probs, threshold = None):
    """
    Prints the stats for the thetas with log probability greater than the threshold.
    gives the valuse of best fit, median, means, std and median absolute deviation.
    The valuse are printed for each parameter, and for 
    beta/eta, beta*xc/eps, Fx, Dx, xc^2/eps, beta*kappa/eps, Fk, Dk, s, slope, xc/eps, Fk/Dk.
    """
    from scipy.stats import median_abs_deviation
    import pandas as pd
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas/etas
    bxe = betas*xcs/epsilons
    Fx = betas**2/(etas*xcs)
    Dx = betas*epsilons/(etas*(xcs**2))

    td = xcs**2/epsilons
    bke = betas*kappa/epsilons
    Fk = betas**2/(etas*kappa)
    Dk = betas*epsilons/(etas*(kappa**2))

    s = (xcs**1.5)*(etas**0.5)/epsilons
    slope = etas*xcs/epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk

    if threshold is not None:
        etas = etas[probs>threshold]
        betas = betas[probs>threshold]
        epsilons = epsilons[probs>threshold]
        xcs = xcs[probs>threshold]

        t3 = t3[probs>threshold]
        bxe = bxe[probs>threshold]
        Fx = Fx[probs>threshold]
        Dx = Dx[probs>threshold]

        td = td[probs>threshold]
        bke = bke[probs>threshold]
        Fk = Fk[probs>threshold]
        Dk = Dk[probs>threshold]

        s = s[probs>threshold]
        slope = slope[probs>threshold]
        Xceps = Xceps[probs>threshold]
        Pk = Pk[probs>threshold]

    #all valuse should be rounded to 2 decimal places or to 3 non-zero decimal places. (1987.3424 -> 1987.34, 0.0006675 -> 0.000667)
    def round_value(value):
        if value == 0:
            return 0
        elif abs(value) < 1:
            r = int(np.abs(np.log10(abs(value))))
            return round(value, r+2)
        else:
            return round(value, 2)

    data = {
        'Best fit': [round_value(etas[np.argmax(probs)]), round_value(betas[np.argmax(probs)]), round_value(epsilons[np.argmax(probs)]), round_value(xcs[np.argmax(probs)]), round_value(t3[np.argmax(probs)]), round_value(bxe[np.argmax(probs)]), round_value(Fx[np.argmax(probs)]), round_value(Dx[np.argmax(probs)]), round_value(td[np.argmax(probs)]), round_value(bke[np.argmax(probs)]), round_value(Fk[np.argmax(probs)]), round_value(Dk[np.argmax(probs)]), round_value(s[np.argmax(probs)]), round_value(slope[np.argmax(probs)]), round_value(Xceps[np.argmax(probs)]), round_value(Pk[np.argmax(probs)])],
        'Median': [round_value(np.median(etas)), round_value(np.median(betas)), round_value(np.median(epsilons)), round_value(np.median(xcs)), round_value(np.median(t3)), round_value(np.median(bxe)), round_value(np.median(Fx)), round_value(np.median(Dx)), round_value(np.median(td)), round_value(np.median(bke)), round_value(np.median(Fk)), round_value(np.median(Dk)), round_value(np.median(s)), round_value(np.median(slope)), round_value(np.median(Xceps)), round_value(np.median(Pk))],
        'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk))],
        'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk))],
        'Median absolute deviation': [round_value(median_abs_deviation(etas)), round_value(median_abs_deviation(betas)), round_value(median_abs_deviation(epsilons)), round_value(median_abs_deviation(xcs)), round_value(median_abs_deviation(t3)), round_value(median_abs_deviation(bxe)), round_value(median_abs_deviation(Fx)), round_value(median_abs_deviation(Dx)), round_value(median_abs_deviation(td)), round_value(median_abs_deviation(bke)), round_value(median_abs_deviation(Fk)), round_value(median_abs_deviation(Dk)), round_value(median_abs_deviation(s)), round_value(median_abs_deviation(slope)), round_value(median_abs_deviation(Xceps)), round_value(median_abs_deviation(Pk))]
    }

    columns = ['Eta', 'Beta', 'Epsilon', 'Xc', 'Beta/Eta', 'Beta*Xc/Eps', 'Fx=Beta^2/(Eta*Xc)', 'Dx=Beta*Eps/(Eta*Xc^2)', 'Xc^2/Eps', 'Beta*Kappa/Eps', 'Fk=Beta^2/(Eta*Kappa)', 'Dk=Beta*Eps/(Eta*Kappa^2)', 's=(Xc^1.5*Eta^0.5)/Eps', 'Slope=Eta*Xc/Eps', 'Xc/Eps', 'Fk/Dk']

    df = pd.DataFrame(data, index=columns)
    dfr = df.transpose()

    print(dfr.to_string())
    return dfr


def getThreshold(ds,metric = 'survival',time_range = None):
    """
    Get the threshold for the log probability. The threshold is the log probability CI.
    The supplied ds should implement a getConfidenceInterval() method.
    """
    CI = ds.getConfidenceInterval()
    t,s = ds.getSurvival()
    #trim to the time range
    if time_range is not None:
        idx = (t>=time_range[0]) & (t<=time_range[1])
        t = t[idx]
        s = s[idx]
        CI[0] = CI[0][idx]
        CI[1] = CI[1][idx]
    if metric == 'survival':
        d0 =np.mean((s-CI[0])**2)
        d1 = np.mean((s-CI[1])**2)
    else:
        #throw an exception if the metric is not implemented
        raise Exception(f"{metric} Not implemented")
    thresh0 = 1/(0.1*d0)
    thresh1 = 1/(0.1*d1)
    #return the maximum of the two thresholds
    return max(thresh0,thresh1)

def applyThreshold(thetas,lnprobs,ds,metric = 'survival',time_range = None):
    """
    Apply the threshold to the thetas and the log probabilities.
    """
    threshold = getThreshold(ds,metric = metric,time_range = time_range)
    print(f"Threshold: {threshold}")
    #print the number of thetas that are above the threshold
    print(f"Number of thetas above the threshold: {len(thetas[lnprobs>threshold])}")
    print(f"Number of thetas below the threshold: {len(thetas[lnprobs<threshold])}")
    return thetas[lnprobs>threshold],lnprobs[lnprobs>threshold]
    

def custom_corner(
    samples,
    lnprobs,
    labels=['eta', 'beta', 'epsilon', 'xc', 'ext h'],
    truths=None,
    scale='log',
    grid=True,
    figsize=(15, 15),
    quantiles=[0.16, 0.5, 0.84],
    show_color_bar=True,
    alpha=0.5,
    axes=None,
    cmap='viridis',
    truth_color='r'
):
    """
    A custom corner plot for the samples.
    Plots the samples and colors them according to the log probabilities.
    params:
    - samples (ndarray): The samples.
    - lnprobs (ndarray): The log probabilities.
    - labels (list): The labels for the parameters.
    - truths (list): The true values for the parameters. If None, the median of the samples is used.
    - scale (str): The scale for the axes.
    - grid (bool): Whether to plot the grid.
    - figsize (tuple): The size of the figure.
    - quantiles (list): The quantiles to plot.
    - show_color_bar (bool): Whether to show the color bar for the log probabilities.
    - alpha (float): Alpha value for scatter points (default 0.5).
    - axes (np.ndarray or None): Optionally provide axes to plot on. If None, a new figure and axes are created.
    - cmap (str or Colormap): Colormap for scatter points (default 'viridis').
    - truth_color (str or color): Color for the truth lines (default 'r').
    returns:
    - fig: The figure.
    - axes: The axes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    ndim = samples.shape[1]

    # If axes are not provided, create them
    if axes is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=figsize)
        fig.subplots_adjust(hspace=0.15, wspace=0.15)
        own_fig = True
    else:
        fig = axes[0, 0].figure
        own_fig = False

    scatter = None
    for i in range(ndim):
        for j in range(i + 1):
            ax = axes[i, j]
            if i == j:
                if scale == 'log':
                    ax.hist(
                        samples[:, i],
                        bins=np.logspace(np.log10(np.min(samples[:, i])), np.log10(np.max(samples[:, i])), 50),
                        color="k",
                        histtype="step"
                    )
                    ax.set_xscale('log')
                else:
                    ax.hist(samples[:, i], bins=100, color="k", histtype="step")
                if truths is not None:
                    ax.axvline(truths[i], color=truth_color)
                ax.set_yticks([])
            else:
                scatter = ax.scatter(
                    samples[:, j],
                    samples[:, i],
                    c=lnprobs,
                    cmap=cmap,
                    s=1,
                    alpha=alpha
                )
                if truths is not None:
                    ax.axvline(truths[j], color=truth_color)
                    ax.axhline(truths[i], color=truth_color)
                if scale == 'log':
                    ax.set_xscale('log')
                    ax.set_yscale('log')
            if i < ndim - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(labels[j])
            if j > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(labels[i])
            if grid:
                ax.grid(True)
        for j in range(i + 1, ndim):
            axes[i, j].set_visible(False)

    for quantile in quantiles:
        for i in range(ndim):
            axes[i, i].axvline(np.percentile(samples[:, i], quantile * 100), color="k", linestyle="dashed")

    # Add color bar if requested
    if show_color_bar and scatter is not None:
        if own_fig:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(scatter, cax=cbar_ax, label='Log Probability')
        else:
            fig.colorbar(scatter, ax=axes, label='Log Probability', fraction=0.046, pad=0.04)

    return fig, axes


def grid_search(theta, npeople, nsteps, t_end, dataSet, metric='baysian', time_range=None, 
                time_step_multiplier=1, prior=None, dt=1, set_params=None, model_func=model, 
                log_samples=False, kwargs=None, max_magnitude=1.1, max_tries=5, n_iterations=10, transformed=False, verbose=True, progress=False, parallel=False):
    """
    Perform grid search optimization around a given theta point.
    
    This method explores parameter space by taking steps in all directions around
    the initial theta point. It supports all the same options as getSampler for
    consistent likelihood evaluation.
    
    Parameters:
    -----------
    theta : array-like
        Initial parameter values to search around
    npeople : int
        Number of people to simulate
    nsteps : int
        Number of simulation steps
    t_end : float
        End time for simulation
    dataSet : array-like
        Observed data for likelihood calculation
    metric : str, optional
        Likelihood metric ('survival' or 'bayesian'), default 'survival'
    time_range : array-like, optional
        Time range for likelihood calculation
    time_step_multiplier : int, optional
        Multiplier for time steps, default 1
    prior : array-like, optional
        Prior bounds for parameters
    dt : float, optional
        Time step size, default 1
    set_params : dict, optional
        Fixed parameters dictionary
    model_func : callable, optional
        Model function to use, default is model
    log_samples : bool, optional
        Whether parameters are in log space, default False
    kwargs : dict, optional
        Additional keyword arguments for model function
    max_magnitude : float, optional
        Maximum fold difference in parameter values, default 1.1
    max_tries : int, optional
        Maximum number of random attempts when no improvement found, default 5
    n_iterations : int, optional
        Number of grid search iterations, default 10
    transformed : bool, optional
        Whether theta is in transformed space (log space), default False
    verbose : bool, optional
        Whether to print progress information, default True
    progress : bool, optional
        Whether to show progress bar, default False
    parallel : bool, optional
        Whether to use parallel processing in model evaluation, default False
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'theta_best': Best parameter values found
        - 'likelihood_best': Best likelihood value
        - 'convergence_history': List of likelihood values during search
        - 'iterations_completed': Number of iterations completed
        - 'improvements_found': Number of improvements found
        - 'all_thetas_tested': Array of all parameter combinations tested
        - 'all_likelihoods': Array of corresponding likelihood values
    """
    if set_params is None:
        set_params = {}
    if kwargs is None:
        kwargs = {}
    
    # Import tqdm for progress bar if needed
    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            if verbose:
                print("Warning: tqdm not available. Progress bar disabled.")
            progress = False
        
    # Convert theta to numpy array
    theta = np.array(theta)
    n_params = len(theta)
    
    # Handle prior parameter similar to getSampler
    if prior is None:
        prior = 10
    if isinstance(prior, (int, float)):
        # Create bins based on theta values and prior expansion factor
        prior = [[theta[i] / prior, theta[i] * prior] for i in range(n_params)]
    
    # Initialize best values
    theta_best = theta.copy()
    
    # Calculate initial likelihood
    if transformed:
        # Use lnprobTransformed for transformed parameters
        likelihood_best = lnprobTransformed(theta, npeople, nsteps, t_end, dataSet=dataSet,
                                          metric=metric, time_range=time_range,
                                          time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                          set_params=set_params, model_func=model_func, 
                                          log_samples=log_samples, parallel=parallel, kwargs=kwargs)
    else:
        # Use lnprob for regular parameters
        likelihood_best = lnprob(theta, npeople, nsteps, t_end, dataSet=dataSet,
                                metric=metric, time_range=time_range,
                                time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                set_params=set_params, model_func=model_func, 
                                log_samples=log_samples, parallel=parallel, kwargs=kwargs)
    
    convergence_history = [likelihood_best]
    improvements_found = 0
    
    # Store all tested thetas and their likelihoods
    all_thetas_tested = [theta_best.copy()]
    all_likelihoods = [likelihood_best]
    
    if verbose:
        print(f"Starting grid search with initial likelihood: {likelihood_best:.6f}")
    
    # Create progress bar if requested
    if progress:
        iteration_range = tqdm(range(n_iterations), desc="Grid Search", unit="iter")
    else:
        iteration_range = range(n_iterations)
    
    for iteration in iteration_range:
        if verbose and not progress:
            print(f"Iteration {iteration + 1}/{n_iterations}")
        
        # Generate all possible directions (3^n_params - 1, excluding the zero vector)
        # Each dimension can be -1, 0, or 1; skip the all-zeros direction
        from itertools import product
        directions = []
        for direction_tuple in product([-1, 0, 1], repeat=n_params):
            if all(x == 0 for x in direction_tuple):
                continue  # skip the zero vector
            directions.append(np.array(direction_tuple))
        
        # Try each direction with max_magnitude step
        improved = False
        best_direction_likelihood = likelihood_best
        best_direction_theta = theta_best.copy()
        
        for direction in directions:
            # Calculate new theta by applying fold change to each parameter
            theta_new = theta_best.copy()
            
            for i in range(n_params):
                if direction[i] > 0:
                    # Increase parameter by max_magnitude fold
                    if log_samples:
                        # In log space, add the log of the fold change
                        theta_new[i] = theta_best[i] + np.log(max_magnitude)
                    else:
                        # In linear or transformed space, multiply by the fold change
                        theta_new[i] = theta_best[i] * max_magnitude
                else:
                    # Decrease parameter by max_magnitude fold
                    if log_samples:
                        # In log space, subtract the log of the fold change
                        theta_new[i] = theta_best[i] - np.log(max_magnitude)
                    else:
                        # In linear or transformed space, divide by the fold change
                        theta_new[i] = theta_best[i] / max_magnitude
            
            # Calculate likelihood for new theta
            try:
                if transformed:
                    # Use lnprobTransformed for transformed parameters
                    likelihood_new = lnprobTransformed(theta_new, npeople, nsteps, t_end, dataSet=dataSet,
                                                    metric=metric, time_range=time_range,
                                                    time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                                    set_params=set_params, model_func=model_func, 
                                                    log_samples=log_samples, parallel=parallel, kwargs=kwargs)
                else:
                    # Use lnprob for regular parameters
                    likelihood_new = lnprob(theta_new, npeople, nsteps, t_end, dataSet=dataSet,
                                          metric=metric, time_range=time_range,
                                          time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                          set_params=set_params, model_func=model_func, 
                                          log_samples=log_samples, parallel=parallel, kwargs=kwargs)
                
                # Store all tested thetas and likelihoods
                all_thetas_tested.append(theta_new.copy())
                all_likelihoods.append(likelihood_new)
                
                if likelihood_new > best_direction_likelihood:
                    best_direction_likelihood = likelihood_new
                    best_direction_theta = theta_new.copy()
                    improved = True
                    
            except (ValueError, RuntimeError):
                # Skip invalid parameter combinations
                continue
        
        # If we found an improvement, use it
        if improved:
            theta_best = best_direction_theta.copy()
            likelihood_best = best_direction_likelihood
            convergence_history.append(likelihood_best)
            improvements_found += 1
            if verbose:
                print(f"  Improvement found! New likelihood: {likelihood_best:.6f}")
            if progress:
                iteration_range.set_postfix({
                    'likelihood': f'{likelihood_best:.6f}',
                    'improvements': improvements_found
                })
        else:
            # No improvement found in this iteration
            if verbose:
                print(f"  No improvement found in iteration {iteration + 1}")
            
            # Try random magnitudes in all directions up to max_tries times
            improvement_found = False
            for try_num in range(max_tries):
                # Try all directions with random magnitudes
                for direction in directions:
                    # Random magnitude between 1.0 and max_magnitude
                    random_magnitude = np.random.uniform(1.0, max_magnitude)
                    
                    # Calculate new theta by applying random fold change to each parameter
                    theta_new = theta_best.copy()
                    
                    for i in range(n_params):
                        if direction[i] > 0:
                            # Increase parameter by random_magnitude fold
                            if log_samples:
                                # In log space, add the log of the fold change
                                theta_new[i] = theta_best[i] + np.log(random_magnitude)
                            else:
                                # In linear or transformed space, multiply by the fold change
                                theta_new[i] = theta_best[i] * random_magnitude
                        elif direction[i] < 0:
                            # Decrease parameter by random_magnitude fold
                            if log_samples:
                                # In log space, subtract the log of the fold change
                                theta_new[i] = theta_best[i] - np.log(random_magnitude)
                            else:
                                # In linear or transformed space, divide by the fold change
                                theta_new[i] = theta_best[i] / random_magnitude
                        # If direction[i] == 0, keep theta_new[i] = theta_best[i] (no change)
                    
                    # Calculate likelihood for new theta
                    try:
                        if transformed:
                            # Use lnprobTransformed for transformed parameters
                            likelihood_new = lnprobTransformed(theta_new, npeople, nsteps, t_end, dataSet=dataSet,
                                                            metric=metric, time_range=time_range,
                                                            time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                                            set_params=set_params, model_func=model_func, 
                                                            log_samples=log_samples, parallel=parallel, kwargs=kwargs)
                        else:
                            # Use lnprob for regular parameters
                            likelihood_new = lnprob(theta_new, npeople, nsteps, t_end, dataSet=dataSet,
                                                  metric=metric, time_range=time_range,
                                                  time_step_multiplier=time_step_multiplier, prior=prior, dt=dt,
                                                  set_params=set_params, model_func=model_func, 
                                                  log_samples=log_samples, parallel=parallel, kwargs=kwargs)
                        
                        # Store all tested thetas and likelihoods
                        all_thetas_tested.append(theta_new.copy())
                        all_likelihoods.append(likelihood_new)
                        
                        if likelihood_new > likelihood_best:
                            theta_best = theta_new.copy()
                            likelihood_best = likelihood_new
                            convergence_history.append(likelihood_best)
                            improvements_found += 1
                            improvement_found = True
                            if verbose:
                                print(f"  Random improvement found! New likelihood: {likelihood_best:.6f}")
                            if progress:
                                iteration_range.set_postfix({
                                    'likelihood': f'{likelihood_best:.6f}',
                                    'improvements': improvements_found
                                })
                            # Break out of both direction loop and try_num loop
                            break
                            
                    except (ValueError, RuntimeError):
                        # Skip invalid parameter combinations
                        continue
                
                # If improvement found, break out of try_num loop
                if improvement_found:
                    break
            else:
                # No improvement found in random tries either
                if verbose:
                    print(f"  No improvement found in {max_tries} random attempts")
    
    # Close progress bar if it was used
    if progress:
        iteration_range.close()
    
    if verbose:
        print(f"Grid search completed. Final likelihood: {likelihood_best:.6f}")
        print(f"Total improvements found: {improvements_found}")
    
    return {
        'theta_best': theta_best,
        'likelihood_best': likelihood_best,
        'convergence_history': convergence_history,
        'iterations_completed': n_iterations,
        'improvements_found': improvements_found,
        'all_thetas_tested': np.array(all_thetas_tested),
        'all_likelihoods': np.array(all_likelihoods)
    }
