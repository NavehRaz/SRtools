import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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
        'Eta': r'$\eta$',
        'Beta': r'$\beta$',
        'Epsilon': r'$\epsilon$',
        'Xc': r'$x_c$',
        'Beta/Eta': r'$\beta/\eta$',
        'Beta*Xc/Eps': r'$\beta x_c/\epsilon$',
        'Fx=Beta^2/(Eta*Xc)':r"$F_x=\beta^2/(\eta x_c)$",
        'Dx=Beta*Eps/(Eta*Xc^2)':r"$D_x=\beta \epsilon/(\eta x_c^2)$",
        'Xc^2/Eps':r"$x_c^2/\epsilon$",
        'Beta*Kappa/Eps':r"$\beta \kappa/\epsilon$",
        'Fk=Beta^2/(Eta*Kappa)':r"$F_k=\beta^2/(\eta \kappa)$",
        'Dk=Beta*Eps/(Eta*Kappa^2)':r"$Dk=\beta \epsilon/(\eta \kappa^2)$",
        's=(Xc^1.5*Eta^0.5)/Eps':r"$s=(x_c^{1.5}\eta^{0.5})/\epsilon$",
        'Slope=Eta*Xc/Eps':r"$\eta x_c/\epsilon$",
        'Xc/Eps':r"$x_c/\epsilon$",
        'Fk/Dk':r"$F_k/D_k$",
        'Fk^2/Dk':r"$F_k^2/D_k$",
        'Median Lifetime':r"$ML$",
    }
    return pdict

def getParamsIndexdict():
        idict = {
        1:'Eta',
        2:'Beta',
        3:'Epsilon',
        4:'Xc',
        5:'Beta/Eta',
        6:'Beta*Xc/Eps',
        7:'Fx=Beta^2/(Eta*Xc)',
        8:'Dx=Beta*Eps/(Eta*Xc^2)',
        9:'Xc^2/Eps',
        10:'Beta*Kappa/Eps',
        11:'Fk=Beta^2/(Eta*Kappa)',
        12:'Dk=Beta*Eps/(Eta*Kappa^2)',
        13:'s=(Xc^1.5*Eta^0.5)/Eps',
        14:'Slope=Eta*Xc/Eps',
        15:'Xc/Eps',
        16:'Fk/Dk',
        17:'Fk^2/Dk',
        18:'Median Lifetime',
    }
        return idict

def getPlotProps(file_names):
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
        'Yeast': 'brown'
    }
    for file_name in file_names:
        parts = file_name.split('_')
        species = parts[0]
        if len(parts) > 1:
            sex = parts[1]
            if sex == 'M':
                marker = '^'  # triangle marker
            elif sex == 'F':
                marker = 'o'  # circle marker
        else:
            marker = 's'  # square marker
        color = colors.get(species, 'black')  # default to black if species not found
        plot_props[file_name] = {'species':species,'color': color, 'marker': marker}
    return plot_props

def plotSingleParam(results,param,divide_by_param =None,multiply_param =None, file_keys='all', type ='Mean', scale='linear', save_path=None, figsize=(15,7),fact=1):
    """
    Plots the values of a single parameter for different files.
    In each df  the parameter is the relevant column.
    If type is 'Mean', it plots the mean value of the parameter with error bars given by the Std (indexes 'Mean', 'Std). 
    If type is 'Median', it plots the median value of the parameter with error bars given by the Median absolute deviation (indexes 'Median', 'Median absolute deviation').
    in either case the index 'Best fit' is also plotted with alpha = 0.5.
    The title of the plot is the f"{param} {type}" .
    the y scale is given by the scale parameter.
    If save_path is not None, the plot is saved in the given path.
    The legend is plotted beside the figure.
    Top and right spines are removed.
    Grid is added.
    If divide_by_param is not None, the parameter is divided by divide_by_param.
    """
    if file_keys == 'all':
        file_keys = results.keys()
    plot_props = getPlotProps(file_keys)
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
        x = file_key
        if type == 'Mean':
            y = df[param]['Mean']
            if divide_by_param is not None:
                y /= df[divide_by_param]['Mean']
            if multiply_param is not None:
                y *= df[multiply_param]['Mean']
            yerr = df[param]['Std']
            if divide_by_param is not None:
                yerr /= df[divide_by_param]['Mean']
            if multiply_param is not None:
                yerr *= df[multiply_param]['Mean']
        elif type == 'Median':
            y = df[param]['Median']
            if divide_by_param is not None:
                y /= df[divide_by_param]['Median']
            if multiply_param is not None:
                y *= df[multiply_param]['Median']
            yerr = df[param]['Median absolute deviation']
            if divide_by_param is not None:
                yerr /= df[divide_by_param]['Median']
            if multiply_param is not None:
                yerr *= df[multiply_param]['Median']
        else:
            raise ValueError("type must be 'Mean' or 'Median'")
        
        #the lower bar of the error bar cannot be lower than 0 if scale is linear and 1e-2*y if scale is log
        if scale == 'linear' and y-yerr < 0:
            yerr = [[y*fact],[yerr*fact]]
        elif scale == 'log' and y-yerr < 1e-2*y:
            yerr = [[y-1e-2*y*fact],[yerr*fact]]

        y = y*fact

        plt.errorbar(x, y, yerr=yerr, label=file_key, 
                 color=plot_props[file_key]['color'], 
                 marker=plot_props[file_key]['marker'], 
                 linestyle='None')
        
        # Plot best fit
        y_best_fit = df[param]['Best fit']
        if divide_by_param is not None:
            y_best_fit /= df[divide_by_param]['Best fit']
        if multiply_param is not None:
            y_best_fit *= df[multiply_param]['Best fit']
        y_best_fit = y_best_fit*fact
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    

def plotParams2D(results,param1,param2,divide_by_param1 =None,multiply_param1=None, divide_by_param2=None,multiply_param2=None, file_keys='all', type ='Mean', xscale='linear',yscale='linear', save_path=None, figsize=(15,10), ax=None):
    """
    Plots the values of two parameters for different files, with param2 as a function of param1.
    In each df, the parameters are the relevant columns.
    If type is 'Mean', it plots the mean value of the parameters with error bars given by the Std (indexes 'Mean', 'Std). 
    If type is 'Median', it plots the median value of the parameters with error bars given by the Median absolute deviation (indexes 'Median', 'Median absolute deviation').
    In either case, the index 'Best fit' is also plotted with alpha = 0.3.
    The title of the plot is f"{param1} vs {param2} {type}".
    The scale of both axes is given by the scale parameter.
    If save_path is not None, the plot is saved in the given path.
    The legend is plotted beside the figure.
    Top and right spines are removed.
    Grid is added.
    If divide_by_param1 or divide_by_param2 is not None, the parameters are divided by the respective divide_by_param.
    If multiply_param1 or multiply_param2 is not None, the parameters are multiplied by the respective multiply_param.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object used for plotting.
    """

    if file_keys == 'all':
        file_keys = results.keys()
    plot_props = getPlotProps(file_keys)
    if divide_by_param1 is not None and multiply_param1 is not None:
        param1_title = f"{getPrintDict().get(param1, param1)} / {getPrintDict().get(divide_by_param1, divide_by_param1)} * {getPrintDict().get(multiply_param1, multiply_param1)}"
    elif divide_by_param1 is not None:
        param1_title = f"{getPrintDict().get(param1, param1)} / {getPrintDict().get(divide_by_param1, divide_by_param1)}"
    elif multiply_param1 is not None:
        param1_title = f"{getPrintDict().get(param1, param1)} * {getPrintDict().get(multiply_param1, multiply_param1)}"
    else:
        param1_title = getPrintDict().get(param1, param1)
    
    if divide_by_param2 is not None and multiply_param2 is not None:
        param2_title = f"{getPrintDict().get(param2, param2)} / {getPrintDict().get(divide_by_param2, divide_by_param2)} * {getPrintDict().get(multiply_param2, multiply_param2)}"
    elif divide_by_param2 is not None:
        param2_title = f"{getPrintDict().get(param2, param2)} / {getPrintDict().get(divide_by_param2, divide_by_param2)}"
    elif multiply_param2 is not None:
        param2_title = f"{getPrintDict().get(param2, param2)} * {getPrintDict().get(multiply_param2, multiply_param2)}"
    else:
        param2_title = getPrintDict().get(param2, param2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    for file_key in file_keys:
        df = results[file_key]
        if type == 'Mean':
            x = df[param1]['Mean']
            if divide_by_param1 is not None:
                x /= df[divide_by_param1]['Mean']
            if multiply_param1 is not None:
                x *= df[multiply_param1]['Mean']
            y = df[param2]['Mean']
            if divide_by_param2 is not None:
                y /= df[divide_by_param2]['Mean']
            if multiply_param2 is not None:
                y *= df[multiply_param2]['Mean']
            xerr = df[param1]['Std']
            if divide_by_param1 is not None:
                xerr /= df[divide_by_param1]['Mean']
            if multiply_param1 is not None:
                xerr *= df[multiply_param1]['Mean']
            yerr = df[param2]['Std']
            if divide_by_param2 is not None:
                yerr /= df[divide_by_param2]['Mean']
            if multiply_param2 is not None:
                yerr *= df[multiply_param2]['Mean']
        elif type == 'Median':
            x = df[param1]['Median']
            if divide_by_param1 is not None:
                x /= df[divide_by_param1]['Median']
            if multiply_param1 is not None:
                x *= df[multiply_param1]['Median']
            y = df[param2]['Median']
            if divide_by_param2 is not None:
                y /= df[divide_by_param2]['Median']
            if multiply_param2 is not None:
                y *= df[multiply_param2]['Median']
            xerr = df[param1]['Median absolute deviation']
            if divide_by_param1 is not None:
                xerr /= df[divide_by_param1]['Median']
            if multiply_param1 is not None:
                xerr *= df[multiply_param1]['Median']
            yerr = df[param2]['Median absolute deviation']
            if divide_by_param2 is not None:
                yerr /= df[divide_by_param2]['Median']
            if multiply_param2 is not None:
                yerr *= df[multiply_param2]['Median']
        else:
            raise ValueError("type must be 'Mean' or 'Median'")
        
        #the lower bar of the error bar cannot be lower than 0 if scale is linear and 1e-2*y if scale is log
        if yscale == 'linear' and y-yerr < 0:
            yerr = [[y],[yerr]]
        elif yscale == 'log' and y-yerr < 1e-2*y:
            yerr = [[y-1e-2*y],[yerr]]
        
        if xscale == 'linear' and x-xerr < 0:
            xerr = [[x],[xerr]]
        elif xscale == 'log' and x-xerr < 1e-2*x:
            xerr = [[x-1e-2*x],[xerr]]

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, label=file_key, 
                     color=plot_props[file_key]['color'], 
                     marker=plot_props[file_key]['marker'], 
                     linestyle='None')
        
        # Plot best fit
        x_best_fit = df[param1]['Best fit']
        if divide_by_param1 is not None:
            x_best_fit /= df[divide_by_param1]['Best fit']
        if multiply_param1 is not None:
            x_best_fit *= df[multiply_param1]['Best fit']
        y_best_fit = df[param2]['Best fit']
        if divide_by_param2 is not None:
            y_best_fit /= df[divide_by_param2]['Best fit']
        if multiply_param2 is not None:
            y_best_fit *= df[multiply_param2]['Best fit']
        ax.plot(x_best_fit, y_best_fit, label=f"{file_key} Best fit", 
                 color=plot_props[file_key]['color'], 
                 marker=plot_props[file_key]['marker'], 
                 linestyle='-', alpha=0.3)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(f"{param1_title} vs {param2_title} {type}")
    ax.set_xlabel(param1_title)
    ax.set_ylabel(param2_title)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    elif ax is None:
        plt.show()
    
    return ax



def getTheta(result_df, type = 'best', user_input = True, errors = False):
    """
    Returns the theta for the best, mean, or median results in the file.
    Theta =[Eta, Beta, Epsilon, Xc]
    if errors is True, it also returns the errors in the same order (Std for type = 'best'/'mean' or Median absolute deviation for type = 'median')
    """
    if type == 'best':
        if not "Best fit" in result_df.index and user_input:
            theta = result_df.loc['Estimate'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            errs = None
        else:
            theta = result_df.loc['Best fit'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            if errors:
                errs = result_df.loc['Std'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
    elif type == 'mean':
        if not "Mean" in result_df.index and user_input:
            theta = result_df.loc['Estimate'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            errs = None
        else:
            theta = result_df.loc['Mean'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            if errors:
                errs = result_df.loc['Std'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
    elif type == 'median':
        if not "Median" in result_df.index and user_input:
            theta = result_df.loc['Estimate'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            errs = None
        else:
            theta = result_df.loc['Median'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
            if errors:
                errs = result_df.loc['Median absolute deviation'][['Eta', 'Beta', 'Epsilon', 'Xc']].values
    else:
        raise ValueError("type must be 'best', 'mean', or 'median'")
    if errors:
        return theta, errs
    else:
        return theta

