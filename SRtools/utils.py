"""
This class provides utility functions for the project.
"""

import ast
import os

import numpy as np
import pandas as pd


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


def read_summary_csv(file=None, columns='all', format='mode'):
    """
    Load preset summary CSVs and optionally subset columns while parsing confidence intervals.

    Parameters
    ----------
    file : str, optional
        Explicit path to a summary CSV. If None, resolved from `format`.
    columns : 'all' or array-like, optional
        Which value columns to read (excluding the parameter column). Defaults to all.
    format : {'mode', 'mode_overall', 'max_likelihood'}, optional
        Determines the default CSV file and whether confidence intervals exist.

    Returns
    -------
    tuple
        (values_by_param, ci_by_param, column_names)
        - values_by_param: dict[str, list] with parameter name keys and ordered column values.
        - ci_by_param: dict[str, list] mirroring values order, containing [low, high] or None when absent.
        - column_names: list[str] of selected non-CI column names.
    """
    allowed_formats = {
        'mode': 'summery_mode.csv',
        'mode_overall': 'summery_mode_overall.csv',
        'max_likelihood': 'summery_max_likelihood.csv',
    }

    if file is None:
        if format not in allowed_formats:
            raise ValueError(f"Unknown format '{format}'. Expected one of {list(allowed_formats.keys())}")
        file = os.path.join(os.path.dirname(__file__), 'Preset_values', allowed_formats[format])

    df = pd.read_csv(file, index_col=0)

    if isinstance(columns, str) and columns != 'all':
        selected_columns = [columns]
    elif columns == 'all':
        selected_columns = 'all'
    else:
        selected_columns = list(columns)

    if format == 'mode':
        value_columns = [col for col in df.columns if '95% CI' not in col]
        ci_columns = {col.replace(' 95% CI', ''): col for col in df.columns if '95% CI' in col}
    else:
        value_columns = list(df.columns)
        ci_columns = {}

    if selected_columns == 'all':
        selected_value_columns = value_columns
    else:
        missing = [col for col in selected_columns if col not in value_columns]
        if missing:
            raise KeyError(f"Columns {missing} not found in summary file {file}")
        selected_value_columns = selected_columns

    def parse_ci(ci_value):
        if pd.isna(ci_value):
            return None
        if isinstance(ci_value, (list, tuple)):
            return [float(ci_value[0]), float(ci_value[1])]
        if isinstance(ci_value, str):
            ci_value = ci_value.strip()
            if not ci_value:
                return None
            try:
                parsed = ast.literal_eval(ci_value)
            except (ValueError, SyntaxError):
                return None
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return [float(parsed[0]), float(parsed[1])]
        return None

    values_by_param = {}
    ci_by_param = {} if ci_columns else {}

    for param_name, row in df.iterrows():
        values = []
        cis = []
        for col in selected_value_columns:
            values.append(row[col])
            if col in ci_columns:
                ci_value = parse_ci(df.loc[param_name, ci_columns[col]])
                cis.append(ci_value)
            elif ci_columns:
                cis.append(None)
        values_by_param[param_name] = values
        if ci_columns:
            ci_by_param[param_name] = cis
            
    return values_by_param, ci_by_param, selected_value_columns

def get_summery_csv_df(file=None, columns='all', format='mode'):
    """
    Load preset summary CSVs and optionally subset columns, returning a pandas DataFrame.
    
    Similar to read_summary_csv but returns a DataFrame instead of dictionaries.

    Parameters
    ----------
    file : str, optional
        Explicit path to a summary CSV. If None, resolved from `format`.
    columns : 'all' or array-like, optional
        Which value columns to read (excluding the parameter column). Defaults to all.
    format : {'mode', 'mode_overall', 'max_likelihood'}, optional
        Determines the default CSV file and whether confidence intervals exist.

    Returns
    -------
    pandas.DataFrame
        DataFrame with parameters as index and selected columns (including CI columns if present).
    """
    allowed_formats = {
        'mode': 'summery_mode.csv',
        'mode_overall': 'summery_mode_overall.csv',
        'max_likelihood': 'summery_max_likelihood.csv',
    }

    if file is None:
        if format not in allowed_formats:
            raise ValueError(f"Unknown format '{format}'. Expected one of {list(allowed_formats.keys())}")
        file = os.path.join(os.path.dirname(__file__), 'Preset_values', allowed_formats[format])

    df = pd.read_csv(file, index_col=0)

    if isinstance(columns, str) and columns != 'all':
        selected_columns = [columns]
    elif columns == 'all':
        selected_columns = 'all'
    else:
        selected_columns = list(columns)

    if format == 'mode':
        value_columns = [col for col in df.columns if '95% CI' not in col]
        ci_columns = {col.replace(' 95% CI', ''): col for col in df.columns if '95% CI' in col}
    else:
        value_columns = list(df.columns)
        ci_columns = {}

    if selected_columns == 'all':
        selected_value_columns = value_columns
    else:
        missing = [col for col in selected_columns if col not in value_columns]
        if missing:
            raise KeyError(f"Columns {missing} not found in summary file {file}")
        selected_value_columns = selected_columns

    # Build the result DataFrame with selected columns
    result_columns = []
    for col in selected_value_columns:
        result_columns.append(col)
        if col in ci_columns:
            result_columns.append(ci_columns[col])

    # Select only the columns we want
    result_df = df[result_columns].copy()

    return result_df


def random_logspace_vector(vec, dist):
    """
    Given a vector and a positive distance, return a new random vector such that
    the norm of the difference in log-space is exactly log(dist).
    The direction is random but the log-space norm constraint is satisfied.

    Args:
        vec (array-like): Original vector (all elements should be > 0).
        dist (float): Multiplicative distance (dist > 0).

    Returns:
        np.ndarray: New vector, positive, at log-space distance log(dist).
    """
    vec = np.asarray(vec)
    if np.any(vec <= 0):
        raise ValueError("All elements of input vector must be positive for log-space operation.")
    log_vec = np.log(vec)
    n = len(vec)
    # Random direction
    direction = np.random.randn(n)
    direction /= np.linalg.norm(direction)
    # Step in log space
    step = direction * (np.log(dist))
    new_log_vec = log_vec + step
    new_vec = np.exp(new_log_vec)
    return new_vec