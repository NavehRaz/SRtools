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