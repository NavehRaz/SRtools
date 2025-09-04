"""
This class provides utility functions for the project.
"""

import numpy as np


def weibull_hazard(t, shape=1, scale=1, translation=0, npoints =100):
    """
    Weibull hazard function. If t is a numpy array, the function will return an array of the same size with the
    corresponding hazard values. if its a list or tupel of lenght 2 it will generate a numpy array with npoints
    between the two values.

    :param t: time
    :param shape: shape parameter
    :param scale: scale parameter
    :param translation: translation parameter
    :param npoints: number of points to generate
    :return: hazard function value
    """
    if isinstance(t, (list, tuple)) and len(t) == 2:
        t = np.linspace(t[0], t[1], npoints)
    return shape/scale * ((t-translation)/scale)**(shape-1) 

def gompertz_makeham_hazard(t, intercept=1, slope=1, makeham=0, npoints=100):
    """
    Gompertz-Makeham hazard function. If t is a numpy array, the function will return an array of the same size with the
    corresponding hazard values. if its a list or tupel of lenght 2 it will generate a numpy array with npoints
    between the two values.

    :param t: time
    :param alpha: alpha parameter
    :param beta: beta parameter
    :param gamma: gamma parameter
    :param translation: translation parameter
    :param npoints: number of points to generate
    :return: hazard function value
    """
    if isinstance(t, (list, tuple)) and len(t) == 2:
        t = np.linspace(t[0], t[1], npoints)
    return intercept*np.exp(slope*(t)) + makeham

# Function to append or create CSV
def append_or_create_csv(data, filepath, doubles=False, override=False):
    import os
    import pandas as pd
    if os.path.exists(filepath):
        # Read existing data
        existing_data = pd.read_csv(filepath, index_col=0)
        
        # Check for duplicate columns
        duplicate_cols = set(existing_data.columns) & set(data.columns)
        
        if duplicate_cols and not doubles and not override:
            import warnings
            warnings.warn(f"Columns {duplicate_cols} already exist and neither doubles nor override is allowed - skipping these columns")
            # Skip duplicate columns by only keeping non-duplicate columns from new data
            data = data.drop(columns=duplicate_cols)
            
        if duplicate_cols and not doubles and override:
            # Remove duplicate columns from existing data
            existing_data = existing_data.drop(columns=duplicate_cols)
            
        # Combine with new data
        combined_data = pd.concat([existing_data, data], axis=1)
        # Save combined data
        combined_data.to_csv(filepath)
    else:
        # Create new file
        data.to_csv(filepath)

def rescaleTheta(theta, s=None, from_unit=None, to_unit=None):
    """
    Rescales theta parameters: theta[0]*s^2, theta[1]*s, theta[2]*s.
    
    Parameters:
    -----------
    theta : array-like
        Array of theta parameters to rescale
    s : float, optional
        Scaling factor (to_unit/from_unit). If None, calculated from from_unit and to_unit
    from_unit : str, optional
        Original unit. Required if s is None. Options: 'days', 'hours', 'generations', 'weeks', 'years'
    to_unit : str, optional
        Target unit. Required if s is None. Options: 'days', 'hours', 'generations', 'weeks', 'years'
    
    Returns:
    --------
    numpy.ndarray
        Rescaled theta parameters
    """
    import numpy as np
    
    # Unit conversion factors (relative to days)
    unit_factors = {
        'days': 1,
        'hours': 1/24,
        'generations': 3/24,
        'weeks': 7,
        'years': 365
    }
    
    # Calculate scaling factor if not provided
    if s is None:
        if from_unit is None or to_unit is None:
            raise ValueError("If s is None, both from_unit and to_unit must be specified")
        
        if from_unit not in unit_factors or to_unit not in unit_factors:
            raise ValueError(f"Invalid unit. Must be one of: {list(unit_factors.keys())}")
        
        s = unit_factors[to_unit] / unit_factors[from_unit]
    
    # Convert theta to numpy array if it isn't already
    theta = np.array(theta)
    
    # Apply rescaling: theta[0]*s^2, theta[1]*s, theta[2]*s
    rescaled_theta = theta.copy()
    rescaled_theta[0] *= s**2
    rescaled_theta[1] *= s
    rescaled_theta[2] *= s
    
    return rescaled_theta


def thetaToSeed(theta, filename):
    """
    Saves the theta parameters as a seed CSV file.

    The CSV will have columns: Eta, Beta, Epsilon, Xc, and a single row labeled 'Estimate' containing the current theta values.
    """
    import pandas as pd
    df = pd.DataFrame(theta, columns=['Estimate'], index=['Eta', 'Beta', 'Epsilon', 'Xc'])
    df=df.T
    df.to_csv(filename)

def param_to_index(param_name):
    """
    Converts a parameter name to an index.
    """
    if param_name == 'eta':
        return 0
    elif param_name == 'beta':
        return 1
    elif param_name == 'epsilon':
        return 2
    elif param_name == 'xc':
        return 3
    elif param_name == 'external_hazard':
        return 4
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
    
def index_to_param(index):
    """
    Converts an index to a parameter name.
    """
    if index == 0:
        return 'eta'
    elif index == 1:
        return 'beta'
    elif index == 2:
        return 'epsilon'
    elif index == 3:
        return 'xc'
    elif index == 4:
        return 'external_hazard'
    else:
        raise ValueError(f"Unknown index: {index}")