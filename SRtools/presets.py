"""
This file is used to generate presets for the SR model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
import ast
from . import SR_hetro as srh

# Dictionary mapping aliases to actual preset names
PRESET_ALIASES = {
    # Mice aliases
    'Mice_M': 'mice_M',
    'mice_m': 'mice_M', 
    'Mice M': 'mice_M',
    'mice M': 'mice_M',
    'mice m': 'mice_M',
    'male mice': 'mice_M',
    'Male Mice': 'mice_M',
    
    # Female mice aliases
    'Mice_F': 'mice_F',
    'mice_f': 'mice_F',
    'Mice F': 'mice_F', 
    'mice F': 'mice_F',
    'mice f': 'mice_F',
    'female mice': 'mice_F',
    'Female Mice': 'mice_F',
    
    # Cats aliases
    'cats': 'cats_BPH',
    'Cats': 'cats_BPH',
    
    # Drosophila aliases
    'drosophila': 'drosophila_441',
    'Drosophila': 'drosophila_441',
    'flies': 'drosophila_441',
    'drosophila melanogaster': 'drosophila_441',
    'Drosophila Melanogaster': 'drosophila_441',
    
    # Human male aliases
    'humans_M': 'combined_human_M',
    'Humans M': 'combined_human_M',
    'Humans_M': 'combined_human_M',
    'humans male': 'combined_human_M',
    'Humans Male': 'combined_human_M',
    'humans_M_combined': 'combined_human_M',
    
    # Human female aliases
    'humans_F': 'combined_human_F',
    'Humans F': 'combined_human_F',
    'Humans_F': 'combined_human_F',
    'humans female': 'combined_human_F',
    'Humans Female': 'combined_human_F',
    'humans_F_combined': 'combined_human_F',

    # Denmark male hetro aliases
    'Denmark_M_1900': 'Denmark_M_1900_hetro',
    'Denmark_M_1890': 'Denmark_M_1890_hetro',

    # Denmark female hetro aliases
    'Denmark_F_1900': 'Denmark_F_1900_hetro',
    'Denmark_F_1890': 'Denmark_F_1890_hetro',
    
    # Labrador aliases
    'Labradors': 'Labradors_vetCompass',
    'dogs': 'Labradors_vetCompass',
    'Dogs': 'Labradors_vetCompass',
    'dog labrador': 'Labradors_vetCompass',
    'Dog Labrador': 'Labradors_vetCompass',

    # Staffy aliases
    'Staffy': 'Staffy_vetCompass',
    'staffy': 'Staffy_vetCompass',
    'Staffy_vetCompass': 'Staffy_vetCompass',
    'staffy_vetCompass': 'Staffy_vetCompass',
    'dog staffy': 'Staffy_vetCompass',
    'Dog Staffy': 'Staffy_vetCompass',
    
    # German Shepherd aliases
    'German Shepherd': 'GermanShepherd_vetCompass',
    'German Shepherd_vetCompass': 'GermanShepherd_vetCompass',
    'german shepherd': 'GermanShepherd_vetCompass',
    'German Shepherd_vetCompass': 'GermanShepherd_vetCompass',
    'dog german shepherd': 'GermanShepherd_vetCompass',
    'Dog German Shepherd': 'GermanShepherd_vetCompass',
    
    # Jack Russell aliases
    'Jack Russell': 'Jack_Russell_vetCompass',
    'Jack Russell_vetCompass': 'Jack_Russell_vetCompass',
    'jack russell': 'Jack_Russell_vetCompass',
    'Jack Russell_vetCompass': 'Jack_Russell_vetCompass',
    'dog jack russell': 'Jack_Russell_vetCompass',
    'Dog Jack Russell': 'Jack_Russell_vetCompass',

    # C. elegans aliases
    'C. elegans': 'celegans',
    'c. elegans': 'celegans',
    'Celegans': 'celegans',
    'celegans': 'celegans',
    
    #E. coli aliases
    'E. coli': 'ecoli',
    'e. coli': 'ecoli',
    'Ecoli': 'ecoli',
    'E coli': 'ecoli',

    #Yeast aliases
    'Yeast': 'yeast',
    'Budding yeast': 'yeast',
    'budding yeast': 'yeast',

    #Guinea pigs aliases
    'Guinea pigs': 'Guiniea_pig_VC',
    'guinea pigs': 'Guiniea_pig_VC',
    'Guinea Pig': 'Guiniea_pig_VC',
    'guinea pig': 'Guiniea_pig_VC',
    'gpig': 'Guiniea_pig_VC',
    'Guinea Pig VC': 'Guiniea_pig_VC',
    'guinea pig vc': 'Guiniea_pig_VC',
    'guinea pig vc': 'Guiniea_pig_VC',
}


params_and_time_powers_dict ={
    'xc/eta': 2,
    'beta/eta': 1,
    'xc^2/epsilon': 1,
    'xc': 0,
    'eta': -2,
    'epsilon': -1,
    'beta': -1,
    'sqrt(xc/eta)': 1,
    's= eta^0.5*xc^1.5/epsilon': 0,
    'beta*xc/epsilon': 0,
    'eta*xc/epsilon': -1,
    'Fx=beta^2/eta*xc': 0,
    'Dx =beta*epsilon/eta*xc^2': 0,
    'Pk=beta*k/epsilon': 0,
    'Fk=beta^2/eta*k': 0,
    'Dk =beta*epsilon/eta*k^2': 0,
    'Fk^2/Dk=beta^3/eta*epsilon': 0,
    'epsilon/beta^2':1,
    'k/beta':1,
    'k^2/epsilon':1,
    'best fit no ext hazard_MedianLifetime': 1,
    'best fit no ext hazard_MaxLifetime': 1,
    'best fit_MedianLifetime': 1,
    'best fit_MaxLifetime': 1,
    'data_MedianLifetime': 1,
    'data_MaxLifetime': 1,
    'ML_lnprob': 0,
    'ExtH': 0,
    'eta/xc': -2,
    'beta/xc': -1,
    'epsilon/xc^2': -1,
    'k/xc': 0
}

FOLDER_ALIASES = {
    'smurf': 'extras/Preset_values/smurfs',
    'smurfs': 'extras/Preset_values/smurfs',
}


def _resolve_preset_dir(folder=None):
    """
    Resolve the preset directory path.

    Parameters
    ----------
    folder : str or None
        - None  -> default ``Preset_values/`` next to this module.
        - A key in ``FOLDER_ALIASES`` (e.g. ``'smurf'``) -> mapped path
          relative to this module.
        - Any other string -> treated as a path.  Absolute paths are used
          as-is; relative paths are resolved relative to this module's
          directory.

    Returns
    -------
    str
        Absolute path to the preset directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if folder is None:
        return os.path.join(current_dir, "Preset_values")

    if folder in FOLDER_ALIASES:
        return os.path.join(current_dir, FOLDER_ALIASES[folder])

    if os.path.isabs(folder):
        return folder

    return os.path.join(current_dir, folder)


def getKarinTheta():
    return [0.49275,54.75,51.83,17]

    
def getTheta(preset_name="humans_M_combined",type = "mode_overall",time_unit='auto',ExtH =False, file=None, folder=None):
    """
    This function is used to get the theta values for a given organism (preset_name).
    The preset can originate from different file types: 
     - mode_overall (default): Overall mode of of posterior distribution
     - mode: Mode of marginal posterior distribution of each parameter
     - max_likelihood: parameters from run with highest likelihood
    
    Time unit conversion:
     - 'auto': Automatically determine based on preset (humans/dogs/cats->years, ecoli->hours, yeast->generations)
     - 'days': No conversion (original units)
     - 'years': Convert from days to years (s=365)
     - 'hours': Convert from days to hours (s=1/24)
     - 'generations': Convert from days to generations (s=3/24)
    
    Parameter scaling: eta->eta*s^2, beta->beta*s, epsilon->epsilon*s, xc->xc
    
    If file is provided, it will be used directly without checking aliases or current_dir.

    folder : str or None
        Preset directory. None -> default Preset_values/. 'smurf' -> extras/Preset_values/smurfs.
        Can also be an absolute or relative path.
    """
    
    # If file is provided, use it directly
    if file is not None:
        # Load the CSV file directly
        try:
            df = pd.read_csv(file, index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find preset file: {file}")
        
        # Determine time unit conversion factor
        if time_unit == 'auto':
            # Auto-detect based on preset name
            if any(keyword in preset_name.lower() for keyword in ['human','sweden','denmark', 'dog', 'cat', 'labrador', 'staffy', 'german', 'jack','pig','guinea pigs']):
                time_unit = 'years'
            elif any(keyword in preset_name.lower() for keyword in ['ecoli', 'e. coli']):
                time_unit = 'hours'
            elif any(keyword in preset_name.lower() for keyword in ['mice', 'mouse']):
                time_unit = 'weeks'
            elif any(keyword in preset_name.lower() for keyword in ['yeast']):
                time_unit = 'generations'
            else:
                time_unit = 'days'  # Default for other organisms
        
        # Set conversion factor based on time unit
        if time_unit == 'days':
            s = 1.0
        elif time_unit == 'years':
            s = 365.0
        elif time_unit == 'hours':
            s = 1.0/24.0
        elif time_unit == 'generations':
            s = 3.0/24.0
        elif time_unit == 'weeks':
            s = 7.0
        else:
            raise ValueError(f"Unknown time_unit: {time_unit}. Must be one of 'auto', 'days', 'years', 'hours', 'generations'")
        
        # Check if the preset_name exists in the columns
        if preset_name not in df.columns:
            available_presets = list(df.columns)
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
        
        # Extract the required parameters: xc/eta, beta/eta, xc^2/epsilon, xc
        try:
            xc_eta = df.loc['xc/eta', preset_name]
            beta_eta = df.loc['beta/eta', preset_name]
            xc_2_epsilon = df.loc['xc^2/epsilon', preset_name]
            xc = df.loc['xc', preset_name]
        except KeyError as e:
            raise KeyError(f"Parameter {e} not found in the preset data")
        
        eta = xc/xc_eta
        beta = beta_eta*eta
        epsilon = xc**2/xc_2_epsilon

        # Apply time unit conversion if needed
        if s != 1.0:
            print(f"Converting time units: days -> {time_unit} (s={s})")
            eta = eta * (s**2)  # eta -> eta*s^2
            beta = beta * s     # beta -> beta*s
            epsilon = epsilon * s  # epsilon -> epsilon*s
            xc = xc  # xc -> xc (no change)
        
        if ExtH:
            external_hazard = df.loc['ExtH', preset_name]
            # If the external_hazard is empty string, convert to None. Otherwise, keep as is.
            if external_hazard == '':
                external_hazard = None
            return np.array([eta, beta, epsilon, xc, external_hazard])
        else:
            return np.array([eta, beta, epsilon, xc])

    #load the preset
    # Check if preset_name is an alias and convert to actual name
    original_preset_name = preset_name
    if preset_name in PRESET_ALIASES:
        preset_name = PRESET_ALIASES[preset_name]
        print(f"Using alias '{original_preset_name}' -> '{preset_name}'")

    # Determine time unit conversion factor
    if time_unit == 'auto':
        # Auto-detect based on preset name
        if any(keyword in preset_name.lower() for keyword in ['human','sweden','denmark', 'dog', 'cat', 'labrador', 'staffy', 'german', 'jack','pig','guinea pigs']):
            time_unit = 'years'
        elif any(keyword in preset_name.lower() for keyword in ['ecoli', 'e. coli']):
            time_unit = 'hours'
        elif any(keyword in preset_name.lower() for keyword in ['mice', 'mouse']):
            time_unit = 'weeks'
        elif any(keyword in preset_name.lower() for keyword in ['yeast']):
            time_unit = 'generations'
        else:
            time_unit = 'days'  # Default for other organisms
    
    # Set conversion factor based on time unit
    if time_unit == 'days':
        s = 1.0
    elif time_unit == 'years':
        s = 365.0
    elif time_unit == 'hours':
        s = 1.0/24.0
    elif time_unit == 'generations':
        s = 3.0/24.0
    elif time_unit == 'weeks':
        s = 7.0
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}. Must be one of 'auto', 'days', 'years', 'hours', 'generations'")
    
    # Determine which CSV file to load based on the type parameter
    if type == "mode_overall":
        csv_filename = "summery_mode_overall.csv"
    elif type == "mode":
        csv_filename = "summery_mode.csv"
    elif type == "max_likelihood":
        csv_filename = "summery_max_likelihood.csv"
    else:
        raise ValueError(f"Unknown type: {type}. Must be one of 'mode_overall', 'mode', or 'max_likelihood'")
    
    preset_dir = _resolve_preset_dir(folder)
    csv_file = os.path.join(preset_dir, csv_filename)
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find preset file: {csv_file}")
    
    # Check if the preset_name exists in the columns
    if preset_name not in df.columns:
        available_presets = list(df.columns)
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
    
    # Extract the required parameters: xc/eta, beta/eta, xc^2/epsilon, xc
    try:
        xc_eta = df.loc['xc/eta', preset_name]
        beta_eta = df.loc['beta/eta', preset_name]
        xc_2_epsilon = df.loc['xc^2/epsilon', preset_name]
        xc = df.loc['xc', preset_name]
    except KeyError as e:
        raise KeyError(f"Parameter {e} not found in the preset data")
    
    eta = xc/xc_eta
    beta = beta_eta*eta
    epsilon = xc**2/xc_2_epsilon

    # Apply time unit conversion if needed
    if s != 1.0:
        print(f"Converting time units: days -> {time_unit} (s={s})")
        eta = eta * (s**2)  # eta -> eta*s^2
        beta = beta * s     # beta -> beta*s
        epsilon = epsilon * s  # epsilon -> epsilon*s
        xc = xc  # xc -> xc (no change)
    
    if ExtH:
        external_hazard = df.loc['ExtH', preset_name]
        # If the external_hazard is empty string, convert to None. Otherwise, keep as is.
        if external_hazard == '':
            external_hazard = None
        return np.array([eta, beta, epsilon, xc, external_hazard])
    else:
        return np.array([eta, beta, epsilon, xc])
    

def getParams(
    preset_name="humans_M_combined",
    type="mode_overall",
    time_unit="auto",
    file=None,
    params=["eta/xc", "beta/xc", "epsilon/xc^2", "xc"],
    folder=None,
):
    """
    Get arbitrary preset parameters (rows) with consistent time-unit conversion.

    Parameters
    ----------
    preset_name : str
        Preset column name (aliases supported via PRESET_ALIASES).
    type : str
        Which preset CSV to use: 'mode_overall', 'mode', or 'max_likelihood'.
    time_unit : str
        'auto', 'days', 'weeks', 'years', 'hours', or 'generations'.
        Conversion assumes the preset table is in 'days', and converts via factor s:
          - years: s=365
          - weeks: s=7
          - hours: s=1/24
          - generations: s=3/24
    ExtH : bool
        If True and `params` is None, append 'ExtH' to the default parameter list.
    file : str or None
        Optional path to a preset CSV to load directly (skips preset_dir resolution).
    params : list[str] or None
        List of parameter names (must be keys in `params_and_time_powers_dict` and
        present as rows in the preset CSV). Returned in the same order.
        Default is ['eta/xc', 'beta/xc', 'epsilon/xc^2', 'xc'].
    folder : str or None
        Preset directory. None -> default Preset_values/. 'smurf' -> extras/Preset_values/smurfs.
        Can also be an absolute or relative path.

    Returns
    -------
    list
        List of parameter values, ordered to match `params`.

    Notes
    -----
    Time-unit conversion uses `params_and_time_powers_dict[param] = p` and applies:
        value_converted = value * s**(-p)
    """

    # If file is provided, use it directly; otherwise resolve from type.
    if file is not None:
        try:
            df = pd.read_csv(file, index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find preset file: {file}")
    else:
        # Check if preset_name is an alias and convert to actual name
        original_preset_name = preset_name
        if preset_name in PRESET_ALIASES:
            preset_name = PRESET_ALIASES[preset_name]
            print(f"Using alias '{original_preset_name}' -> '{preset_name}'")

        if type == "mode_overall":
            csv_filename = "summery_mode_overall.csv"
        elif type == "mode":
            csv_filename = "summery_mode.csv"
        elif type == "max_likelihood":
            csv_filename = "summery_max_likelihood.csv"
        else:
            raise ValueError(
                "Unknown type: {t}. Must be one of 'mode_overall', 'mode', or 'max_likelihood'".format(
                    t=type
                )
            )

        preset_dir = _resolve_preset_dir(folder)
        csv_file = os.path.join(preset_dir, csv_filename)
        try:
            df = pd.read_csv(csv_file, index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find preset file: {csv_file}")

    # Auto-detect time_unit when requested
    if time_unit == "auto":
        preset_name_lower = preset_name.lower()
        if any(
            keyword in preset_name_lower
            for keyword in [
                "human",
                "sweden",
                "denmark",
                "dog",
                "cat",
                "labrador",
                "staffy",
                "german",
                "jack",
                "pig",
                "guinea pigs",
            ]
        ):
            time_unit = "years"
        elif any(keyword in preset_name_lower for keyword in ["ecoli", "e. coli"]):
            time_unit = "hours"
        elif any(keyword in preset_name_lower for keyword in ["mice", "mouse"]):
            time_unit = "weeks"
        elif any(keyword in preset_name_lower for keyword in ["yeast"]):
            time_unit = "generations"
        else:
            time_unit = "days"

    # Set conversion factor based on time unit
    if time_unit == "days":
        s = 1.0
    elif time_unit == "years":
        s = 365.0
    elif time_unit == "hours":
        s = 1.0 / 24.0
    elif time_unit == "generations":
        s = 3.0 / 24.0
    elif time_unit == "weeks":
        s = 7.0
    else:
        raise ValueError(
            f"Unknown time_unit: {time_unit}. Must be one of 'auto', 'days', 'weeks', 'years', 'hours', 'generations'"
        )

    # Validate preset column
    if preset_name not in df.columns:
        available_presets = list(df.columns)
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {available_presets}"
        )

    def _coerce_value(v):
        # Normalize missing/empty values
        if v == "" or v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, str):
            # Some tables store lists/tuples as strings; try to parse.
            try:
                v_parsed = ast.literal_eval(v)
                v = v_parsed
            except Exception:
                # Fall back to float parsing if possible
                try:
                    return float(v)
                except Exception:
                    return v
        return v

    out = []
    for par in params:
        if par not in params_and_time_powers_dict:
            raise KeyError(
                f"Unknown parameter '{par}'. Must be a key in params_and_time_powers_dict."
            )
        try:
            raw = df.loc[par, preset_name]
        except KeyError as e:
            raise KeyError(
                f"Parameter {e} not found in the preset data (row '{par}', column '{preset_name}')."
            )

        val = _coerce_value(raw)
        if val is None:
            out.append(None)
            continue

        p = params_and_time_powers_dict[par]

        # Apply conversion: multiply by s**(-p) (supports scalars + array-likes)
        try:
            out.append(np.asarray(val, dtype=float) * (s ** (-p)))
        except Exception:
            # Non-numeric values are returned as-is (no scaling)
            out.append(val)

    return out


def get_preset_names(type = "mode_overall", folder=None):
    """
    This function is used to get the names of all presets.

    folder : str or None
        Preset directory. None -> default Preset_values/. 'smurf' -> extras/Preset_values/smurfs.
    """
    if type == "mode_overall":
        csv_filename = "summery_mode_overall.csv"
    elif type == "mode":
        csv_filename = "summery_mode.csv"
    elif type == "max_likelihood":
        csv_filename = "summery_max_likelihood.csv"
    else:
        raise ValueError(f"Unknown type: {type}. Must be one of 'mode_overall', 'mode', or 'max_likelihood'")

    preset_dir = _resolve_preset_dir(folder)
    csv_file = os.path.join(preset_dir, csv_filename)
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find preset file: {csv_file}")

    #return a list of all presets
    return list(df.columns)

def get_config_params(
    preset_name="humans_M_combined",
    config_params=['nsteps', 'time_step_multiplier', 'npeople', 't_end', 'time_range','hetro'],
    types=[int, int, int, int, list,bool],
    time_unit=None,
    verbose=False,
    folder=None,
    ):
    """
    This function is used to get the configuration parameters for a given preset. They are returned as a dictionary.
    Optionally, you can specify a target time_unit (e.g., 'years', 'days', 'hours', 'generations').
    If time_unit is given and differs from the detected original_time_unit, t_end and time_range will be converted accordingly.

    folder : str or None
        Preset directory. None -> default Preset_values/. 'smurf' -> extras/Preset_values/smurfs.
    """
    
    
    if time_unit == 'auto':
        time_unit = None
    # Helper: get conversion factor from original_time_unit to time_unit
    def get_time_unit_conversion_factor(from_unit, to_unit):
        # Define conversion factors to days
        to_days = {
            'days': 1.0,
            'weeks': 7.0,
            'years': 365,
            'hours': 1.0 / 24.0,
            'generations': 3.0/24.0, 
        }
        # If units are the same, factor is 1
        
        if from_unit == to_unit:
            return 1.0
        # Convert from original to days, then days to target
        if from_unit not in to_days or to_unit not in to_days:
            raise ValueError(f"Unknown time unit(s): {from_unit}, {to_unit}")
        # Convert to days, then to target
        days = to_days[from_unit]
        factor = to_days[to_unit] * days
        return factor
    
    # Auto-detect original time unit from preset_name
    preset_name_lower = preset_name.lower()
    if any(keyword in preset_name_lower for keyword in ['human','sweden','denmark', 'dog', 'cat', 'labrador', 'staffy', 'german', 'jack','pig','guinea pigs']):
        original_time_unit = 'years'
    elif any(keyword in preset_name_lower for keyword in ['ecoli', 'e. coli']):
        original_time_unit = 'hours'
    elif any(keyword in preset_name_lower for keyword in ['mice', 'mouse']):
        original_time_unit = 'weeks'
    elif any(keyword in preset_name_lower for keyword in ['yeast']):
        original_time_unit = 'generations'
    else:
        original_time_unit = 'days'  # Default for other organisms

    # Check if preset_name is an alias and convert to actual name
    original_preset_name = preset_name
    if preset_name in PRESET_ALIASES:
        preset_name = PRESET_ALIASES[preset_name]
        print(f"Using alias '{original_preset_name}' -> '{preset_name}'")
    if preset_name == 'combined_human_M':
        preset_name = 'Denmark_M_1900_hetro'
        print(f"Using specific cohort config -> 'Denmark_M_1900_hetro'")
    if preset_name == 'combined_human_F':
        preset_name = 'Denmark_F_1900_hetro'
        print(f"Using specific cohort config for combined_human_F -> 'Denmark_F_1900_hetro'")

    csv_filename = 'All_config.csv'

    preset_dir = _resolve_preset_dir(folder)
    csv_file = os.path.join(preset_dir, csv_filename)

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find preset file: {csv_file}")

    # Get the config values as a dict
    # Handle possible NaN for time_range gracefully
    config = {}
    for i in range(len(config_params)):
        value = df.loc[config_params[i], preset_name]
        param_type = types[i]
        # If the type is not int, float, str, or bool, parse with ast.literal_eval
        if param_type not in [int, float, str, bool]:
            # If value is NaN or None, keep as None
            if pd.isna(value) or value is None:
                config[config_params[i]] = None
            else:
                config[config_params[i]] = ast.literal_eval(value)
        else:
            config[config_params[i]] = param_type(value)

    # Optionally handle time unit conversion
    if time_unit is not None and original_time_unit != time_unit:
        s = get_time_unit_conversion_factor(original_time_unit, time_unit)
        # Convert t_end if present
        if 't_end' in config.keys():
            config['t_end'] = int(round(int(config['t_end']) * s))
        # Convert time_range if present in config_params or in the CSV
        if 'time_range' in config.keys():
            # Try to get time_range from config or from CSV
            if 'time_range' in config and config['time_range'] is not None:
                if verbose:
                    print(f"type of time_range: {type(config['time_range'])}")
                    print(f"Converting time_range from {config['time_range']} in {original_time_unit} to {time_unit}")
                traw = config['time_range']
                traw = [int(round(x * s)) for x in traw]
                config['time_range'] = traw

    return config


def getSim(
    preset_name="humans_M_combined",
    type="mode_overall",
    time_unit='auto',
    config_params=['nsteps', 'time_step_multiplier', 'npeople', 't_end', 'hetro'],
    theta=None,
    nsteps=None,
    npeople=None,
    t_end=None,
    time_step_multiplier=None,
    method='brownian_bridge',
    parallel=False,
    ExtH = True,
    folder=None,
):
    """
    Returns an srh.SR_Hetro object for the given preset, using the correct theta and configuration parameters.
    Optionally override theta, nsteps, npeople, t_end, or time_step_multiplier.

    folder : str or None
        Preset directory. None -> default Preset_values/. 'smurf' -> extras/Preset_values/smurfs.
    """
    # Get theta for the preset unless overridden
    if theta is None:
        theta_val = getTheta(preset_name=preset_name, type=type, time_unit=time_unit, ExtH=ExtH, folder=folder)
    else:
        theta_val = theta

    # Get config parameters for the preset
    if time_unit == 'auto':
        config_time_unit = None
    else:
        config_time_unit = time_unit
    config = get_config_params(preset_name=preset_name, time_unit=config_time_unit, config_params=config_params, folder=folder)
    
    if config['hetro'] is None:
        config['hetro'] = False
    # Remove 'time_range' from config if present
    config.pop('time_range', None)




    # Override config values if provided
    if nsteps is not None:
        config['nsteps'] = nsteps
    if npeople is not None:
        config['npeople'] = npeople
    if t_end is not None:
        config['t_end'] = t_end
    if time_step_multiplier is not None:
        config['time_step_multiplier'] = time_step_multiplier

    return srh.getSrHetro(theta_val[:-1], external_hazard=theta_val[-1], method=method, parallel=parallel, **config)