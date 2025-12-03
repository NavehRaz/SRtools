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


def getTheta(preset_name="humans_M_combined",type = "mode_overall",time_unit='auto',ExtH =False):
    """
    This function is used to get the theta values for a given organism (preset_name).
    The preset can originate from different fir types: 
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
    """

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
    
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the Preset_values directory
    preset_dir = os.path.join(current_dir, "Preset_values")
    # Construct the full path to the CSV file
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
    

def get_preset_names(type = "mode_overall"):
    """
    This function is used to get the names of all presets.
    """
    if type == "mode_overall":
        csv_filename = "summery_mode_overall.csv"
    elif type == "mode":
        csv_filename = "summery_mode.csv"
    elif type == "max_likelihood":
        csv_filename = "summery_max_likelihood.csv"
    else:
        raise ValueError(f"Unknown type: {type}. Must be one of 'mode_overall', 'mode', or 'max_likelihood'")

    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the Preset_values directory
    preset_dir = os.path.join(current_dir, "Preset_values")
    # Construct the full path to the CSV file
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
    verbose=False
    ):
    """
    This function is used to get the configuration parameters for a given preset. They are returned as a dictionary.
    Optionally, you can specify a target time_unit (e.g., 'years', 'days', 'hours', 'generations').
    If time_unit is given and differs from the detected original_time_unit, t_end and time_range will be converted accordingly.
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

    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the Preset_values directory
    preset_dir = os.path.join(current_dir, "Preset_values")
    # Construct the full path to the CSV file
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
    ExtH = True
):
    """
    Returns an srh.SR_Hetro object for the given preset, using the correct theta and configuration parameters.
    Optionally override theta, nsteps, npeople, t_end, or time_step_multiplier.
    """
    # Get theta for the preset unless overridden
    if theta is None:
        theta_val = getTheta(preset_name=preset_name, type=type, time_unit=time_unit, ExtH=ExtH)
    else:
        theta_val = theta

    # Get config parameters for the preset
    if time_unit == 'auto':
        config_time_unit = None
    else:
        config_time_unit = time_unit
    config = get_config_params(preset_name=preset_name, time_unit=config_time_unit, config_params=config_params)
    
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