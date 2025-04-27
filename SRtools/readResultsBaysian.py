import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba

"""
This script reads the results from the simulation results csvs and has methods to plots the results.
"""

def readResultsFile(file):
    """
    Reads the results from a csv file and returns a pandas dataframe and a file name.
    """
    file_name = os.path.basename(file)
    #remove the extension
    file_name = os.path.splitext(file_name)[0]
    print(f"Reading {file_name}")
    return file_name, pd.read_csv(file, index_col=0)

def readResultsFolder(folder):
    """
    Reads the results from a folder and returns a dictionary with the file name as the key and the pandas dataframe as the value.
    """
    results = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_name, df = readResultsFile(os.path.join(folder, file))
            results[file_name] = df
    return results

def getPrintDict():
    """
    Returns a dictionary with the latex expresion to print for different columns names    
    """
    pdict = {
        'eta': r'$\eta$',
        'beta': r'$\beta$',
        'epsilon': r'$\epsilon$',
        'xc': r'$x_c$',
        'xc/eta': r'$\frac{xc}{\eta}$',
        'beta/eta': r'$\frac{\beta}{\eta}$',
        'xc^2/epsilon': r'$\frac{xc^2}{\epsilon}$',
        'sqrt(xc/eta)': r'$\sqrt{\frac{xc}{\eta}}$',
        's= eta^0.5*xc^1.5/epsilon': r'$s=\frac{\eta^{0.5} xc^{1.5}}{\epsilon}$',
        'beta*xc/epsilon': r'$\frac{\beta xc}{\epsilon}$',
        'eta*xc/epsilon': r'$\frac{\eta xc}{\epsilon}$',
        'Fx=beta^2/eta*xc': r'$F_x=\frac{\beta^2}{\eta xc}$',
        'Dx =beta*epsilon/eta*xc^2': r'$D_x=\frac{\beta \epsilon}{\eta xc^2}$',
        'Pk=beta*k/epsilon': r'$P_k=\frac{\beta k}{\epsilon}$',
        'Fk=beta^2/eta*k': r'$F_k=\frac{\beta^2}{\eta k}$',
        'Dk =beta*epsilon/eta*k^2': r'$D_k=\frac{\beta \epsilon}{\eta k^2}$',
        'Fk^2/Dk=beta^3/eta*epsilon': r'$F_k^2/D_k=\frac{\beta^3}{\eta \epsilon}$',
        'best fit_MedianLifetime': r'$ML_{best fit}$',
        'best fit_MaxLifetime': r'$MaxL_{best fit}$',
        'data_MedianLifetime': r'$ML_{data}$',
        'data_MaxLifetime': r'$MaxL_{data}$',
    }
    return pdict

def getParamsIndexdict():
    idict = {
        1: 'eta',
        2: 'beta',
        3: 'epsilon',
        4: 'xc',
        5: 'xc/eta',
        6: 'beta/eta',
        7: 'xc^2/epsilon',
        8: 'sqrt(xc/eta)',
        9: 's= eta^0.5*xc^1.5/epsilon',
        10: 'beta*xc/epsilon',
        11: 'eta*xc/epsilon',
        12: 'Fx=beta^2/eta*xc',
        13: 'Dx =beta*epsilon/eta*xc^2',
        14: 'Pk=beta*k/epsilon',
        15: 'Fk=beta^2/eta*k',
        16: 'Dk =beta*epsilon/eta*k^2',
        17: 'Fk^2/Dk=beta^3/eta*epsilon',
        18: 'best fit_MedianLifetime',
        19: 'best fit_MaxLifetime',
        20: 'data_MedianLifetime',
        21: 'data_MaxLifetime',
    }
    return idict

def getPlotProps(file_names,color_map='tab10',mult_species =True):
    """
    Returns a dictionary with the properties to plot for each file name. File names are either of the format 'SPECIES_SEX' or 'SPECIES', where sex may be 'M' or 'F'.
    Same species should have the same color, M should have triangle markers and F should have circle markers. Species with no sex should have square markers.
    """
    plot_props = {}
    colors = {
        'Human': 'blue',
        'Mice': 'green',
        'Drosophila': 'red',
        'Medflies': 'purple',
        'Celegance': 'orange',
        'Yeast': 'brown',
        'Dog': 'magenta',
        'APOE33': 'green',
        'APOE34': 'purple',
        'Middle': 'orange',
        'High': 'green',
        'Low': 'red',
    }
    species_count = {}
    species_colors = {}
    #list of markers:
    markers = [ '^', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']


    for file_name in file_names:
        parts = file_name.split('_')
        if mult_species:
            species = parts[0]
        else:
            species = file_name
        if species not in species_count:
            species_count[species] = 0
        species_count[species] += 1
        if len(parts) > 1:
            sex = parts[1]
            if sex == 'M':
                marker = 's'  # triangle marker
            elif sex == 'F':
                marker = 'o'  # circle marker
            else:
                marker = markers[species_count[species] % len(markers)]  
        else:
            marker = markers[species_count[species] % len(markers)]  

        if color_map is None:
            base_color = to_rgba(colors.get(species, 'black'))  # get the base color
        elif isinstance(color_map, dict):
            base_color = color_map.get(species, 'black')
        elif species not in species_colors:
            cmap = plt.get_cmap(color_map)
            species_colors[species] = cmap(len(species_colors) % cmap.N)
            base_color = species_colors.get(species, 'black')  # default to black if species not found
        
        color = list(base_color)
        color = tuple(color)

        plot_props[file_name] = {'species': species, 'color': color, 'marker': marker}

    return plot_props

def getPlotPropsBreeds(file_names,species_with_breeds={'Dog':'winter','drosophila':'autumn'},color_map='gray',mult_species =True):
    """
    Returns a dictionary with the properties to plot for each file name for the breeeds plot. File names are either of the format 'SPECIES_SEX', 'SPECIES_BREED', SPECIES_SEX_info_info or 'SPECIES', where sex may be 'M' or 'F'.
    Same species should have the same color (unless it is a species with breeds), All specis without breeds should be plotted with the color_map, each species with breeds should have its own color map.
    """
    plot_props = {}
   
    species_count = {}
    species_colors = {}
    
    


    for file_name in file_names:
        parts = file_name.split('_')
        if mult_species:
            species = parts[0]
        else:
            species = file_name
        if species not in species_count:
            species_count[species] = 0
        species_count[species] += 1
        if len(parts) > 1:
            sex = parts[1]
            if sex == 'M':
                marker = 's'  # square marker
            elif sex == 'F':
                marker = 'o'  # circle marker
            else:
                marker = 'v'  
        else:
            marker = 'v'  

        spacing_breeds =55
        spacing =40
        if species not in species_colors or species in species_with_breeds.keys():
            cmap = plt.get_cmap(species_with_breeds.get(species,color_map))
            if species in species_with_breeds.keys():
                species_colors[species] = cmap(species_count[species]*spacing_breeds % cmap.N)
            else:
                species_colors[species] = cmap(len(species_colors)*spacing % cmap.N) 
            base_color = species_colors.get(species, 'black')  # default to black if species not found
        color = list(base_color)
        color = tuple(color)

        plot_props[file_name] = {'species': species, 'color': color, 'marker': marker}

    return plot_props

def getPlotPropsImages(file_image_dict,zooms =0.03,directions=1,offset=1):
    """
    Gets a dictionary of file_names as keys and image_paths as values and zooms.
    Returns a dictionary with file_names as keys and {image: image_instance, zoom: zoom} as values.
    if zoom is an int, use the same zoom for all images, otherwise use the zoom in the list that corresponds to the relvant key.
    """
    plot_props = {}
    for file_name, image_path in file_image_dict.items():
        img = mpimg.imread(image_path)
        if isinstance(zooms, (int, float)):
            zoom = zooms
        elif isinstance(zooms, dict):
            zoom = zooms.get(file_name, 0.03)
        else:
            zoom = 0.03

        if isinstance(directions, dict):
            dir_value = directions.get(file_name, 1)
        else:
            dir_value = directions

        if isinstance(offset, (int, float)):
            offset_value = offset
        elif isinstance(offset, dict):
            offset_value = offset.get(file_name, 1)
        else:
            offset_value = 1

        plot_props[file_name] = {'image': img, 'zoom': zoom, 'direction': dir_value, 'offset': offset_value}
    return plot_props


def plotSingleParam(results, param, divide_by_param=None, multiply_param=None, file_keys='all', type='mode', scale='linear', save_path=None, figsize=(12,5), fact=1, multiple_species=True):
    """
    Plots the values of a single parameter for different files.
    In each df the parameter is the relevant column.
    If type is 'max_likelihood', it plots the max_likelihood value of the parameter with error bars given by the percentile_95 (indexes 'max_likelihood', 'percentile_95').
    If type is 'mean', it plots the mean value of the parameter with error bars given by the Std (indexes 'mean', 'std).
    If type is 'mode', it plots the mode value of the parameter with error bars given by the percentile_95 (indexes 'mode', 'percentile_95').
    In either case the index 'Best fit' is also plotted with alpha = 0.5.
    The title of the plot is the f"{param} {type}".
    The y scale is given by the scale parameter.
    If save_path is not None, the plot is saved in the given path.
    The legend is plotted beside the figure.
    Top and right spines are removed.
    Grid is added.
    If divide_by_param is not None, the parameter is divided by divide_by_param.
    """
    if file_keys == 'all':
        file_keys = results.keys()
    plot_props = getPlotProps(file_keys,mult_species=multiple_species)
    if divide_by_param is not None and multiply_param is not None:
        param_title = f"{getPrintDict().get(param, param)} / {getPrintDict().get(divide_by_param, divide_by_param)} * {getPrintDict().get(multiply_param, multiply_param)}"
    elif divide_by_param is not None:
        param_title = f"{getPrintDict().get(param, param)} / {getPrintDict().get(divide_by_param, divide_by_param)}"
    elif multiply_param is not None:
        param_title = f"{getPrintDict().get(param, param)} * {getPrintDict().get(multiply_param, multiply_param)}"
    else:
        param_title = getPrintDict().get(param, param)
    plt.figure(figsize=figsize)

    for file_key in file_keys:
        df = results[file_key]
        df=df.T
        x = file_key
        if type == 'mean':
            y = df[param]['mean']
            if divide_by_param is not None:
                y /= df[divide_by_param]['mean']
            if multiply_param is not None:
                y *= df[multiply_param]['mean']
            yerr = df[param]['std']
            if isinstance(yerr, str):
                yerr = eval(yerr)
            if divide_by_param is not None:
                yerr /= df[divide_by_param]['mean']
            if multiply_param is not None:
                yerr *= df[multiply_param]['mean']
        elif type == 'mode':
            y = df[param]['mode']
            yerr = eval(df[param]['percentile_95'])
            if divide_by_param is not None:
                y /= df[divide_by_param]['mode']
                yerr = [yerr[0]/df[divide_by_param]['mode'], yerr[1]/df[divide_by_param]['mode']]
            if multiply_param is not None:
                y *= df[multiply_param]['mode']
                yerr = [yerr[0]*df[multiply_param]['mode'], yerr[1]*df[multiply_param]['mode']]
        elif type == 'max_likelihood':
            y = df[param]['max_likelihood']
            yerr = eval(df[param]['percentile_95'])
            if divide_by_param is not None:
                y /= df[divide_by_param]['max_likelihood']
                yerr = [yerr[0]/df[divide_by_param]['max_likelihood'], yerr[1]/df[divide_by_param]['max_likelihood']]
            if multiply_param is not None:
                y *= df[multiply_param]['max_likelihood']
                yerr = [yerr[0]*df[multiply_param]['max_likelihood'], yerr[1]*df[multiply_param]['max_likelihood']]
        else:
            raise ValueError("type must be 'max_likelihood', 'mean', or 'mode'")
        
        # the lower bar of the error bar cannot be lower than 0 if scale is linear and 1e-2*y if scale is log
        # if scale == 'linear' and y - yerr[0] < 0:
        #     yerr = [[y * fact], [yerr[1] * fact]]
        # elif scale == 'log' and y - yerr[0] < 1e-2 * y:
        #     yerr = [[y - 1e-2 * y * fact], [yerr[1] * fact]]

        y = y * fact
        yerr = yerr * fact
        yerr = [max(0,y-yerr[0]), max(0,yerr[1]-y)]
        plt.errorbar(x, y, yerr=[[yerr[0]], [yerr[1]]], label=file_key, 
                 color=plot_props[file_key]['color'], 
                 marker=plot_props[file_key]['marker'], 
                 linestyle='None')
        
        # Plot best fit
        y_best_fit = df[param]['max_likelihood']
        if divide_by_param is not None:
            y_best_fit /= df[divide_by_param]['max_likelihood']
        if multiply_param is not None:
            y_best_fit *= df[multiply_param]['max_likelihood']
        y_best_fit = y_best_fit * fact
        plt.plot(x, y_best_fit, label=f"{file_key} Best fit", 
                 color=plot_props[file_key]['color'], 
                 marker=plot_props[file_key]['marker'], 
                 linestyle='-', alpha=0.3)
    
    plt.yscale(scale)
    plt.title(f"{param_title} {type}")
    plt.xlabel('Index')
    plt.ylabel(param_title)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax = plt.gca()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return ax

def plotParams2D(results, param1, param2, divide_by_param1=None, multiply_param1=None, divide_by_param2=None, multiply_param2=None, file_keys='all', type='mode', xscale='linear', yscale='linear', save_path=None, figsize=(9,6),multiple_species=True, ax=None, full_output=False,full_label=False , legend = True, 
                 plot_props=None, plot_props_images=False,best_fit=True, image_color ='black', drop_images=False, shift_x =1, shift_y =1):
    """
    Plots the values of two parameters for different files, with param2 as a function of param1.
    In each df, the parameters are the relevant columns.
    If type is 'max_likelihood', it plots the max_likelihood value of the parameters with error bars given by the percentile_95 (indexes 'max_likelihood', 'percentile_95').
    If type is 'mean', it plots the mean value of the parameters with error bars given by the Std (indexes 'mean', 'std).
    If type is 'mode', it plots the mode value of the parameters with error bars given by the percentile_95 (indexes 'mode', 'percentile_95').
    In either case, the index 'Best fit' is also plotted with alpha = 0.3.
    The title of the plot is f"{param1} vs {param2} {type}".
    The scale of both axes is given by the scale parameter.
    If save_path is not None, the plot is saved in the given path.
    The legend is plotted beside the figure.
    Top and right spines are removed.
    Grid is added.
    If divide_by_param1 or divide_by_param2 is not None, the parameters are divided by the respective divide_by_param.
    If multiply_param1 or multiply_param2 is not None, the parameters are multiplied by the respective multiply_param.
    """

    if file_keys == 'all':
        file_keys = results.keys()
    if plot_props is None:
        if not plot_props_images:
            plot_props = getPlotProps(file_keys,mult_species=multiple_species)
        else:
            plot_props = {}
            for key in file_keys:
                plot_props[key]=({'color':image_color,'marker':'o'}) 
    inputs_dict = {'param1': param1, 'param2': param2, 'divide_by_param1': divide_by_param1, 'multiply_param1': multiply_param1, 'divide_by_param2': divide_by_param2, 'multiply_param2': multiply_param2}
    for key in ['divide_by_param1', 'multiply_param1', 'divide_by_param2', 'multiply_param2']:
        input_param = inputs_dict[key]
        if input_param is not None and not isinstance(input_param, list):
            inputs_dict[key] = [input_param] 
            
        
    def getTitle(param, inputs_dict,keys):
        divide_by_param = inputs_dict[keys[0]]
        multiply_param = inputs_dict[keys[1]]
        if divide_by_param is not None and multiply_param is not None:
            title = f"{getPrintDict().get(param, param)}  * {' * '.join([getPrintDict().get(p, p) for p in multiply_param])}/ {' / '.join([getPrintDict().get(p, p) for p in divide_by_param])}"
        elif divide_by_param is not None:
            title = f"{getPrintDict().get(param, param)} / {' / '.join([getPrintDict().get(p, p) for p in divide_by_param])}"
        elif multiply_param is not None:
            title = f"{getPrintDict().get(param, param)} * {' * '.join([getPrintDict().get(p, p) for p in multiply_param])}"
        else:
            title = getPrintDict().get(param, param)
        return title
    
    param1_title = getTitle(param1, inputs_dict, ['divide_by_param1', 'multiply_param1'])
    param2_title = getTitle(param2, inputs_dict, ['divide_by_param2', 'multiply_param2'])

    def getValues(df, type, param, multiply_param, divide_by_param):
        #returns x,y,xerr,yerr
        if type not in ['max_likelihood', 'mean', 'mode']:
            raise ValueError("type must be 'max_likelihood', 'mean', or 'mode'")
        errorTypeDict = {'max_likelihood': 'percentile_95', 'mean': 'std', 'mode': 'percentile_95'}
        v = df[param][type]
        v_divide_factor = 1
        verr = df[param][errorTypeDict[type]]
        if isinstance(verr, str):
            verr = ast.literal_eval(verr)
        if not isinstance(verr, list):
            verr = [verr, verr]
        verr = [max(0, v - verr[0]), max(0, verr[1] - v)]
        if divide_by_param is not None:
            for param in divide_by_param:
                v_divide_factor *= df[param][type]
            v /= v_divide_factor
        v_multiply_factor = 1
        if multiply_param is not None:
            for param in multiply_param:
                v_multiply_factor *= df[param][type]
            v *= v_multiply_factor
        verr =[verr[0]*v_multiply_factor, verr[1]*v_multiply_factor]
        verr =[verr[0]/v_divide_factor, verr[1]/v_divide_factor]
        return v, verr


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    x_values, y_values, x_errors, y_errors, keys = [], [], [], [], []
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    for file_key in file_keys:
        
        if  (plot_props_images and file_key in plot_props_images.keys()) or not plot_props_images:
            df = results[file_key]
            df = df.T
            x, xerr = getValues(df, type, param1, inputs_dict['multiply_param1'], inputs_dict['divide_by_param1'])
            y, yerr = getValues(df, type, param2, inputs_dict['multiply_param2'], inputs_dict['divide_by_param2'])
            
            x_values.append(x)
            y_values.append(y)
            x_errors.append(xerr)
            y_errors.append(yerr)
            keys.append(file_key)

            ax.errorbar(x*shift_x, y*shift_y, xerr=[[xerr[0]*shift_x], [xerr[1]*shift_x]], yerr=[[yerr[0]*shift_y], [yerr[1]*shift_y]], label=file_key, 
                        color=plot_props[file_key]['color'], 
                        marker=plot_props[file_key]['marker'], 
                        linestyle='None')
            if best_fit:    
                # Plot best fit
                x_best_fit, _ = getValues(df, 'max_likelihood', param1, inputs_dict['multiply_param1'], inputs_dict['divide_by_param1'])
                y_best_fit, _ = getValues(df, 'max_likelihood', param2, inputs_dict['multiply_param2'], inputs_dict['divide_by_param2'])
                if full_label:
                    blabel=f"{file_key} Best fit"
                else:
                    blabel=None
                ax.plot(x_best_fit*shift_x, y_best_fit*shift_y, label=blabel, 
                        color=plot_props[file_key]['color'], 
                        marker=plot_props[file_key]['marker'], 
                        linestyle='-', alpha=0.3)
        if plot_props_images:
            if not drop_images:
                #this code plots the image at an offset so as not to overlap with the data point.
                #this takes into account that the image is zoomed in, and that the axis might be in log scale.
                #then the picture is bounded by vertical lines on both sides.
                if file_key in plot_props_images.keys():
                    img = plot_props_images[file_key]['image']
                    zoom = plot_props_images[file_key]['zoom']
                    offset_factor_x = plot_props_images[file_key]['offset'][0]
                    offset_factor_y = plot_props_images[file_key]['offset'][1]
                    direction = plot_props_images[file_key]['direction']
                    imwidth = img.shape[1]
                    imheight = img.shape[0]
                    disp_x, disp_y = ax.transData.transform((x, y))
                    disp_offset_x = imwidth * zoom
                    disp_offset_y = imheight * zoom
                    new_disp_x = disp_x + disp_offset_x * offset_factor_x
                    new_disp_y = disp_y + disp_offset_y * offset_factor_y
                    new_data_x, new_data_y = ax.transData.inverted().transform((new_disp_x, new_disp_y))
                    offset_x = new_data_x - x
                    offset_y = new_data_y - y

                    add_image_marker(ax, img, x + direction * offset_x, y + offset_y, zoom,color =image_color)

            # left_disp_x = disp_x + disp_offset_x - imwidth * zoom
            # right_disp_x = disp_x + disp_offset_x + imwidth * zoom
            # left_data_x, _ = ax.transData.inverted().transform((left_disp_x, disp_y))
            # right_data_x, _ = ax.transData.inverted().transform((right_disp_x, disp_y))
            # ax.axvline(left_data_x, color=plot_props[file_key]['color'], linestyle='--', alpha=0.5)
            # ax.axvline(right_data_x, color=plot_props[file_key]['color'], linestyle='--', alpha=0.5)



    ax.set_title(f"{param1_title} vs {param2_title} {type}")
    ax.set_xlabel(param1_title)
    ax.set_ylabel(param2_title)
    ax.grid(True)
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if full_output=='dictioanry': 
        return ax, {'x_values': x_values, 'y_values': y_values, 'x_errors': x_errors, 'y_errors': y_errors, 'keys': keys, 'fig': fig, 'plot_props': plot_props} 
    if full_output:
        return ax, x_values, y_values, x_errors, y_errors, keys, fig , plot_props
    return ax




def make_summery_table(results, type='ML+CI', save_path=None):
    """
    Makes a summary table of the results. The table has the columns 'Species', and all the parameters in the results files.
    The rows are the different files. Each cell has the value of the 'max_likelihood' column for that parameter for that file, and the CI eg:

    | Species | Eta                           | Beta                          | ....
    | species1|{ML of eta} {eta percentile_95}|{ML of eta} {eta percentile_95}| ....
    """
    df = pd.DataFrame()
    for file_key, result_df in results.items():
        # for i, param in enumerate(result_df.index):
        #     print(result_df['max_likelihood'][param]) 
        # print(result_df)
        df = pd.concat([df, pd.Series({**{'Species': file_key}, **{param: f"{result_df['max_likelihood'][param]} {result_df['percentile_95'][param]}" for i, param in enumerate(result_df.index)}}, name=file_key).to_frame().T])
    if save_path:
        df.to_csv(save_path)
    return df


def getTheta(file_path, type='max_likelihood'):
    """
    Returns the theta [eta,beta,epsilon,xc] for the 'max_likelihood'' or'mode' results in the file.
    """
    df = pd.read_csv(file_path, index_col=0)
    if type == 'max_likelihood':
        theta = df['max_likelihood'][['eta', 'beta', 'epsilon', 'xc']].values
    elif type == 'mode':
        theta = df['mode'][['eta', 'beta', 'epsilon', 'xc']].values
    else:
        raise ValueError("type must be 'max_likelihood' or 'mode'")
    return theta
    

def getParams(file_path, params, by_dict=True, type='max_likelihood'):
    """
    Returns the params in params for the 'max_likelihood'' or'mode' results in the file.
    If by dict is True, then if a param is an integer, it is replaced by the corresponding key in getParamsIndexdict().
    Parmeters:
        file_path: str
            The path to the file
        params: list
            The list of parameters to get. If an element is an integer and by_dict=True, it is replaced by the corresponding key in getParamsIndexdict().
        by_dict: bool
            If True, then if a param is an integer, it is replaced by the corresponding key in getParamsIndexdict().
        type: str
            The type of results to get. 'max_likelihood' or 'mode'.
    """ 
    df = pd.read_csv(file_path, index_col=0)
    if by_dict:
        params = [getParamsIndexdict()[param] if isinstance(param, int) else param for param in params]
    if type == 'max_likelihood':
        value = df['max_likelihood'][params].values
    elif type == 'mode':
        value = df['mode'][params].values
    else:
        raise ValueError("type must be 'max_likelihood' or 'mode'")
    return value


def add_image_marker(ax, img, x, y, zoom=1,remove_background=True,transparency=0.6,color='black'):
    if remove_background:
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        # Ensure the image has an alpha channel (RGBA)
        image_rgba = np.dstack((img, np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255))

        # Define white threshold (adjust as needed)
        white_thresh = 240  # Increase if necessary

        # Create mask where white pixels will be removed
        mask = (img[:, :, 0] > white_thresh) & (img[:, :, 1] > white_thresh) & (img[:, :, 2] > white_thresh)

        # Set white pixels to transparent (alpha = 0)
        image_rgba[mask] = [0, 0, 0, 0]

    #set color  everywhere that is not transparent
    image_rgba = np.array(image_rgba)
    image_rgba[image_rgba[:, :, 3] != 0] = list(np.array(to_rgba(color, alpha=1))*255)
    
    
    if transparency < 1:
        image_rgba[:, :, 3] = image_rgba[:, :, 3] * transparency


    im = OffsetImage(image_rgba, zoom=zoom)

    ab = AnnotationBbox(im, (x, y), frameon=False)

    ax.add_artist(ab)
    ab.set_zorder(-1)
