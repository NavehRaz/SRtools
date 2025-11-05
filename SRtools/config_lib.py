import numpy as np
import ast
import os
import pandas as pd


class ExcelConfigParser:
    """
    A ConfigParser-compatible class for reading/writing Excel configuration files.
    Each sheet represents a section. Column A contains keys, Row 1 contains config names.
    """
    
    def __init__(self, filename, config_name):
        """
        Initialize ExcelConfigParser.
        
        Args:
            filename: Path to Excel file
            config_name: Name of the configuration column to use
        """
        self.filename = filename
        self.config_name = config_name
        self._data = {}  # {section_name: {key: value}}
        self._sections = []  # List of section names
        self._load_data()
    
    def _load_data(self):
        """Load data from Excel file."""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(self.filename, engine='openpyxl')
            sheet_names = excel_file.sheet_names
            
            # If no DEFAULT sheet exists, use first sheet as DEFAULT
            if 'DEFAULT' not in sheet_names and len(sheet_names) > 0:
                # Rename first sheet to DEFAULT in our internal representation
                self._sections = ['DEFAULT'] + [s for s in sheet_names if s != sheet_names[0]]
            else:
                self._sections = sheet_names.copy()
            
            # Load data from each sheet
            for sheet_name in sheet_names:
                # Read with header in row 1 (index 0 in pandas)
                df = pd.read_excel(self.filename, sheet_name=sheet_name, engine='openpyxl', header=0)
                
                # Use first sheet as DEFAULT if DEFAULT doesn't exist
                section_name = 'DEFAULT' if (sheet_name == sheet_names[0] and 'DEFAULT' not in sheet_names) else sheet_name
                
                if df.empty:
                    self._data[section_name] = {}
                    continue
                
                # Check if config_name column exists (excluding the first column which is keys)
                config_cols = [c for c in df.columns if c != df.columns[0]]
                if self.config_name not in df.columns:
                    # Config column not found in this sheet - create empty section instead of error
                    # This allows sheets like SUBMISSION_* to exist without all config columns
                    self._data[section_name] = {}
                    continue
                
                # Get keys from first column (column A)
                keys_col = df.iloc[:, 0].astype(str)
                # Get values from config_name column
                values_col = df[self.config_name].astype(str)
                
                # Create dictionary for this section
                section_data = {}
                for idx in range(len(keys_col)):
                    key = str(keys_col.iloc[idx]).strip()
                    if key and key != 'nan':
                        value = str(values_col.iloc[idx])
                        # Skip blank cells - don't add to dictionary so get(key, default) returns default
                        if pd.isna(values_col.iloc[idx]) or value == 'nan' or value.strip() == '':
                            continue  # Skip blank cells - don't add to dictionary
                        section_data[key] = value
                
                self._data[section_name] = section_data
        except Exception as e:
            raise ValueError(f"Error loading Excel config file '{self.filename}': {str(e)}")
    
    def sections(self):
        """Return list of section names."""
        return self._sections.copy()
    
    def has_section(self, section):
        """Check if section exists."""
        return section in self._sections
    
    def add_section(self, section):
        """Add a new section."""
        if section in self._sections:
            return
        self._sections.append(section)
        self._data[section] = {}
    
    def items(self, section):
        """Return list of (key, value) tuples for a section."""
        if section not in self._data:
            return []
        return list(self._data[section].items())
    
    def get(self, *args, **kwargs):
        """
        Get value from section or DEFAULT section.
        
        Usage:
            - config.get('DEFAULT', 'key') or config.get('DEFAULT', 'key', fallback='default')
            - config.get('key', 'fallback_value') or config.get('key', fallback='default')  # Searches DEFAULT section
        
        Args:
            *args: Either (section, key) or (key,) or (key, fallback_value) for DEFAULT section
            **kwargs: fallback keyword argument
        """
        fallback = kwargs.get('fallback', None)
        
        if len(args) == 2:
            # Check if first arg is a section name or a key (if it's a known section, use ConfigParser style)
            if args[0] in self._sections:
                # Two-argument form: get(section, key)
                section, key = args
                if section not in self._data:
                    if fallback is not None:
                        return fallback
                    raise KeyError(f"Section '{section}' not found")
                if key not in self._data[section]:
                    if fallback is not None:
                        return fallback
                    raise KeyError(f"Key '{key}' not found in section '{section}'")
                return self._data[section][key]
            else:
                # Two-argument form: get(key, fallback_value) - searches DEFAULT section
                key, fallback_val = args
                fallback = fallback_val if fallback is None else fallback
                if 'DEFAULT' in self._data and key in self._data['DEFAULT']:
                    return self._data['DEFAULT'][key]
                if fallback is not None:
                    return fallback
                raise KeyError(f"Key '{key}' not found in DEFAULT section")
        elif len(args) == 1:
            # One-argument form: get(key) - searches DEFAULT section
            key = args[0]
            if 'DEFAULT' in self._data and key in self._data['DEFAULT']:
                return self._data['DEFAULT'][key]
            if fallback is not None:
                return fallback
            raise KeyError(f"Key '{key}' not found in DEFAULT section")
        else:
            raise TypeError(f"get() takes 1 or 2 positional arguments but {len(args)} were given")
    
    def write(self, filename=None):
        """Write config to Excel file, preserving existing columns."""
        if filename is None:
            filename = self.filename
        
        # First, read existing file to preserve other columns
        try:
            existing_sheets = {}
            excel_file = pd.ExcelFile(self.filename, engine='openpyxl')
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.filename, sheet_name=sheet_name, engine='openpyxl', header=0)
                existing_sheets[sheet_name] = df
        except:
            existing_sheets = {}
        
        # Use ExcelWriter to write all sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for section in self._sections:
                # Map section name back to sheet name (handle DEFAULT mapping)
                sheet_name = section
                if section == 'DEFAULT' and section not in existing_sheets:
                    # Find original first sheet name if it exists
                    if existing_sheets:
                        sheet_name = list(existing_sheets.keys())[0]
                
                # Prepare data for this section
                if section in self._data and self._data[section]:
                    keys = list(self._data[section].keys())
                    values = [self._data[section][key] for key in keys]
                    
                    # If sheet exists, preserve existing structure
                    if sheet_name in existing_sheets:
                        df = existing_sheets[sheet_name].copy()
                        key_col_name = df.columns[0]  # Assume first column is keys
                        
                        # Update or add config_name column
                        if self.config_name not in df.columns:
                            df[self.config_name] = ''
                        
                        # Create a mapping of key to value for quick lookup
                        key_value_map = dict(zip(keys, values))
                        
                        # Update existing rows that match keys
                        for idx in range(len(df)):
                            existing_key = str(df.iloc[idx, 0]).strip()
                            if existing_key in key_value_map:
                                col_idx = df.columns.get_loc(self.config_name)
                                df.iloc[idx, col_idx] = str(key_value_map[existing_key])
                        
                        # Add new keys that don't exist in the sheet
                        existing_keys_set = set(str(df.iloc[i, 0]).strip() for i in range(len(df)) if not pd.isna(df.iloc[i, 0]))
                        new_keys_to_add = [(k, v) for k, v in zip(keys, values) if k not in existing_keys_set]
                        
                        if new_keys_to_add:
                            new_rows_data = {key_col_name: [k for k, v in new_keys_to_add]}
                            new_rows_data[self.config_name] = [str(v) for k, v in new_keys_to_add]
                            # Fill other columns with empty strings for new rows
                            for col in df.columns:
                                if col not in new_rows_data:
                                    new_rows_data[col] = [''] * len(new_keys_to_add)
                            new_rows_df = pd.DataFrame(new_rows_data)
                            df = pd.concat([df, new_rows_df], ignore_index=True)
                    else:
                        # Create new DataFrame with keys in column A and config_name column
                        df = pd.DataFrame({
                            'Key': keys,
                            self.config_name: values
                        })
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
                else:
                    # Empty section - create minimal structure or preserve existing
                    if sheet_name in existing_sheets:
                        df = existing_sheets[sheet_name].copy()
                        if self.config_name not in df.columns:
                            df[self.config_name] = ''
                    else:
                        df = pd.DataFrame({
                            'Key': [],
                            self.config_name: []
                        })
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
    
    def __setitem__(self, section, items):
        """Set items in a section (dict-like interface)."""
        if section not in self._sections:
            self.add_section(section)
        if isinstance(items, dict):
            self._data[section].update(items)
        else:
            raise TypeError("Items must be a dictionary")
    
    def __getitem__(self, section):
        """Get section data (dict-like interface)."""
        if section not in self._data:
            self.add_section(section)
            self._data[section] = {}
        return self._data[section]


def read_excel_config(filename, config_name):
    """
    Read Excel configuration file and return ExcelConfigParser instance.
    
    Args:
        filename: Path to Excel file
        config_name: Name of the configuration column to use (header in row 1)
    
    Returns:
        ExcelConfigParser instance
    """
    return ExcelConfigParser(filename, config_name)


def read_configs(folder_path):
    """
    Read the all the config files in the config folder and return the latest parameters as a dictionary.
    """
    import configparser
    import os
    #make alist of all the .ini files in the folder
    config_files = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.ini') and not file.startswith('._'):
            config_files.append(os.path.join(folder_path, file))
    config = configparser.ConfigParser()
    config.read(config_files)
    #get the latest parameters
    return config

def config_to_dict(config, mcmc_convert=True, custom_keys=None):
    """
    Convert the config file to a dictionary.
    """
    config_dict = {}
    # If several sections have the same key, 
    # keep the one from the last section as we iterate in order.
    for section in config.sections():
        for key, value in config.items(section):
            config_dict[key] = value

    # Helper function to get the correct converter
    def get_converter(value_type):
        converters = {
            'int': int,
            'float': float,
            'bool': ast.literal_eval,
            'astliteral': ast.literal_eval,
            'str': str
        }
        return converters.get(value_type)

    # Convert the values to the correct type
    if mcmc_convert:
        predefined_keys = {
            'nsteps': 'int',
            'npeople': 'int',
            't_end': 'int',
            'nwalkers': 'int',
            'h5_file_name': 'str',
            'n_mcmc_steps': 'int',
            'metric': 'str',
            'time_range': 'astliteral',
            'time_step_multiplier': 'int',
            'data_file': 'str',
            'seed_file': 'str',
            'variations': 'astliteral',
            'prior': 'astliteral',
            'transform': 'astliteral',
            'external_hazard': 'astliteral',
            'data_dt': 'float',
            'ndims': 'int',
            'hetro': 'astliteral'
        }

        # Merge predefined keys with custom keys if provided
        if custom_keys:
            predefined_keys.update(custom_keys)

        for key, value_type in predefined_keys.items():
            if key in config_dict.keys():
                converter = get_converter(value_type)
                if not converter:
                    continue
                raw_val = config_dict[key]
                # Treat empty strings as None and skip conversion
                if isinstance(raw_val, str) and raw_val.strip() == '':
                    config_dict[key] = None
                    continue
                try:
                    config_dict[key] = converter(raw_val)
                except (ValueError, SyntaxError, TypeError):
                    # Leave original value if conversion fails
                    config_dict[key] = raw_val

    return config_dict

def _is_thumbnail_file(filename):
    """
    Check if a file is a thumbnail file that should be ignored.
    """
    thumbnail_patterns = [
        'Thumbs.db',
        '.DS_Store',
        '._',
        'thumb',
        'thumbnail'
    ]
    filename_lower = filename.lower()
    return any(pattern.lower() in filename_lower for pattern in thumbnail_patterns)

def add_submition_folder(config, folder, path):
    """
    Add the submission folder to the config file in a new section.
    The new section name is called f'SUBMISSION_{i}'.
    where i is the value of the index in the DEFAULT section.
    After adding the submission folder the index in the DEFAULT section is updated to i+1
      and the config file is saved to path. If path is a folder then it should contain a single config file .
    """
    # Check if this is an ExcelConfigParser instance
    is_excel_config = isinstance(config, ExcelConfigParser)
    
    index = int(config.get('DEFAULT', 'index'))
    submission_section = f'SUBMISSION_{index}'
    if not config.has_section(submission_section):
        config.add_section(submission_section)
    config[submission_section] = {'submission_folder': folder}
    config['DEFAULT']['index'] = str(index + 1)
    
    # Handle Excel config files
    if is_excel_config:
        # For Excel configs, use the filename from the config object
        # If path is provided and is an Excel file, use it; otherwise use config's filename
        if path and (path.endswith('.xlsx') or path.endswith('.xls')):
            config.write(path)
        else:
            config.write()  # Write to original file
    else:
        # Handle INI config files (original behavior)
        #If path is a folder then it should contain a single config file, otherwise it should be the path to the config file
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if not _is_thumbnail_file(f)]
            if not files:
                raise ValueError(f"No valid files found in {path}")
            path = os.path.join(path, files[0])
        with open(path, 'w') as configfile:
            config.write(configfile)

def save_config_file(params,save_path):
    """
    Save the parameters to the config file.
    """
    import configparser
    config = configparser.ConfigParser()
    config['DEFAULT'] = params
    #if path does not end with .ini add it
    if not save_path.endswith('.ini'):
        save_path += '.ini'
    #create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as configfile:
        config.write(configfile)
    return


def get_lims(nsteps,kappa=0.5,tend=115,threshold =50):
    eta_lim =[-np.inf,np.inf]
    beta_lim =[-np.inf,threshold*nsteps*kappa/(2*tend) ]
    epsilon_lim =[-np.inf,np.inf]
    xc_lim =[-np.inf,np.inf] 
    lims = [eta_lim,beta_lim,epsilon_lim,xc_lim]
    return lims



def make_new_config_for_JM_test(config_path):
    config = read_configs(config_path)
    index = int(config.get('DEFAULT', 'index'))
    base_name = config.get('DEFAULT', 'base_name')
    new_params ={'name':f"{base_name}_{index+1}",
                 'index':index+1
                 }
    save_config_file(new_params,f'{config_path}/config{index+1}')
    return

def make_new_config_for_JM(config_path, results_path, submit_folder = None):
    config = sr_mcmc.read_configs(config_path)
    name = config.get('DEFAULT', 'name')
    folder = config.get('DEFAULT', 'folder')
    thetas_path = config.get('DEFAULT', 'thetas_path')
    deltas = ast.literal_eval(config.get('DEFAULT', 'deltas'))
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    npeople = int(config.get('DEFAULT', 'npeople'))
    npeople_gwtw = int(config.get('DEFAULT', 'npeople_gwtw'))
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    ngrid_steps = int(config.get('DEFAULT', 'ngrid_steps'))
    index = int(config.get('DEFAULT', 'index'))
    base_name = config.get('DEFAULT', 'base_name')
    dataSet_path = config.get('DEFAULT', 'dataSet_path')
    year = int(config.get('DEFAULT', 'year'))
    n_iterations = int(config.get('DEFAULT', 'n_iterations'))
    stochastic = config.get('DEFAULT', 'stochastic')
    metric = config.get('DEFAULT', 'metric')

    dataSet = sr_mcmc.getData(path=dataSet_path, year =year)
   
    theta_sorted, lnprobs_sorted= sr_mcmc.calc_liklihoods_from_survival_files([results_path],dataSet,
                                                                              npeople =npeople, npeople_gwtw=npeople_gwtw,
                                                                                nsteps=nsteps,tend=115,metric = metric)

    lims = get_lims(nsteps)
    
    ranges = [(1/deltas[0], deltas[0]), (1/deltas[1], deltas[1]),
               (1/deltas[2], deltas[2]), (1/deltas[3], deltas[3])]
    ngrid_steps2 =[ngrid_steps+2,ngrid_steps+2,ngrid_steps+2,ngrid_steps+2]
    if submit_folder is None:
        submit_folder = folder
    save_path = f"{submit_folder}/thetas_{base_name}_{index+1}.npy"
    if stochastic == 'True':
        thetas2, deltas = sr_mcmc.make_thetas_for_grid_search_qmc(theta_sorted[0:20],ranges=ranges,
                                                          nsteps=ngrid_steps2,save_path=save_path,
                                                          deltas=True,lims=lims)
        deltas = np.array(deltas)*1.2
        deltas = deltas.tolist()
    else:
        thetas2, deltas = sr_mcmc.make_thetas_for_grid_search(theta_sorted[0:20],ranges=ranges,
                                                          nsteps=ngrid_steps2,save_path=save_path,grid='log',
                                                          deltas=True,lims=lims)
    len_thetas = len(thetas2[0,:])
    
    n_iterations = n_iterations - 1
    n_per_run = len_thetas//2000
    if len_thetas%2000 != 0:
        n_per_run += 1
    n_per_run = int(n_per_run)
    new_params ={'name':f"{base_name}_{index+1}",
                'index':index+1,
                'len_thetas':len_thetas,
                'thetas_path':save_path,
                'deltas':deltas,
                'n_iterations':n_iterations,
                'submit_folder':submit_folder,
                'n_per_run':n_per_run
                 }
    sr_mcmc.save_config_file(new_params,f'{config_path}/config{index+1}')
    return