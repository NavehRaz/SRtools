import numpy as np
import ast
import os



def read_configs(folder_path):
    """
    Read the all the config files in the config folder and return the latest parameters as a dictionary.
    """
    import configparser
    import os
    #make alist of all the .ini files in the folder
    config_files = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.ini'):
            config_files.append(os.path.join(folder_path,file))
    config = configparser.ConfigParser()
    config.read(config_files)
    #get the latest parameters
    return config

def config_to_dict(config, mcmc_convert=True, custom_keys=None):
    """
    Convert the config file to a dictionary.
    """
    config_dict = {}
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
                if converter:
                    try:
                        conv = converter(config_dict[key])
                    except (ValueError, SyntaxError):
                        conv = converter(config_dict[key]) # Handle conversion errors gracefully
                    config_dict[key] = conv

    return config_dict

def add_submition_folder(config, folder, path):
    """
    Add the submission folder to the config file in a new section.
    The new section name is called f'SUBMISSION_{i}'.
    where i is the value of the index in the DEFAULT section.
    After adding the submission folder the index in the DEFAULT section is updated to i+1
      and the config file is saved to path. If path is a folder then it should contain a single config file .
    """
    index = int(config.get('DEFAULT', 'index'))
    submission_section = f'SUBMISSION_{index}'
    if not config.has_section(submission_section):
        config.add_section(submission_section)
    config[submission_section] = {'submission_folder': folder}
    config['DEFAULT']['index'] = str(index + 1)
    #If path is a folder then it should contain a single config file, otherwise it should be the path to the config file
    if os.path.isdir(path):
        path = os.path.join(path, os.listdir(path)[0])
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