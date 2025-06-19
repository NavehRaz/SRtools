"""
This file is used to generate presets for the SR model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings

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
    'humans_M': 'humans_M_combined',
    'Humans M': 'humans_M_combined',
    'Humans_M': 'humans_M_combined',
    'humans male': 'humans_M_combined',
    'Humans Male': 'humans_M_combined',
    
    # Human female aliases
    'humans_F': 'humans_F_combined',
    'Humans F': 'humans_F_combined',
    'Humans_F': 'humans_F_combined',
    'humans female': 'humans_F_combined',
    'Humans Female': 'humans_F_combined',
    
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
    'Jack Russell': 'JackRussell_vetCompass',
    'Jack Russell_vetCompass': 'JackRussell_vetCompass',
    'jack russell': 'JackRussell_vetCompass',
    'Jack Russell_vetCompass': 'JackRussell_vetCompass',
    'dog jack russell': 'JackRussell_vetCompass',
    'Dog Jack Russell': 'JackRussell_vetCompass',

    # C. elegans aliases
    'C. elegans': 'celegance',
    'c. elegans': 'celegance',
    'Celegance': 'celegance',
    
    #E. coli aliases
    'E. coli': 'ecoli',
    'e. coli': 'ecoli',
    'Ecoli': 'ecoli',
    'E coli': 'ecoli',

    #Yeast aliases
    'Yeast': 'yeast',
    'Budding yeast': 'yeast',
    'budding yeast': 'yeast',
    
}


def getTheta(preset_name="humans_M_combined",type = "mode_overall",time_unit='auto'):
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
        if any(keyword in preset_name.lower() for keyword in ['human', 'dog', 'cat', 'labrador', 'staffy', 'german', 'jack']):
            time_unit = 'years'
        elif any(keyword in preset_name.lower() for keyword in ['ecoli', 'e. coli']):
            time_unit = 'hours'
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
    
    # Return the vector [eta, beta, epsilon, xc]
    return np.array([eta, beta, epsilon, xc])
    
    
