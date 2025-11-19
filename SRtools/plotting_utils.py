"""
Utility functions for plotting from summary CSV files.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast

# Import helper functions from readResultsBaysian
from .readResultsBaysian import getPlotProps, getPrintDict, add_image_marker


def plotParams2D(data, param1, param2, divide_by_param1=None, multiply_param1=None, 
                          divide_by_param2=None, multiply_param2=None, dataset_keys='all', 
                          xscale='linear', yscale='linear', save_path=None, figsize=(9,6), 
                          multiple_species=True, ax=None, full_output=False, full_label=False, 
                          legend=True, plot_props=None, plot_props_images=False, best_fit=True, 
                          image_color='black', drop_images=False, shift_x=1, shift_y=1):
    """
    Plots the values of two parameters from a summary CSV file or DataFrame, with param2 as a function of param1.
    
    The CSV file or DataFrame should have parameters as rows (index) and datasets as columns. Each dataset may have
    a corresponding CI column named "{dataset_name} 95% CI".
    
    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to the summary CSV file (e.g., 'summery_mode.csv') or a pandas DataFrame with the same structure
    param1 : str
        Name of the first parameter (row index in CSV)
    param2 : str
        Name of the second parameter (row index in CSV)
    divide_by_param1 : str or list, optional
        Parameter(s) to divide param1 by
    multiply_param1 : str or list, optional
        Parameter(s) to multiply param1 by
    divide_by_param2 : str or list, optional
        Parameter(s) to divide param2 by
    multiply_param2 : str or list, optional
        Parameter(s) to multiply param2 by
    dataset_keys : 'all' or list, optional
        Which datasets (columns) to plot. Defaults to 'all'.
    xscale : str, optional
        Scale for x-axis ('linear' or 'log'). Defaults to 'linear'.
    yscale : str, optional
        Scale for y-axis ('linear' or 'log'). Defaults to 'linear'.
    save_path : str, optional
        Path to save the figure. If None, figure is shown.
    figsize : tuple, optional
        Figure size. Defaults to (9, 6).
    multiple_species : bool, optional
        Whether to use multiple species coloring. Defaults to True.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    full_output : bool or str, optional
        If True, returns additional data. If 'dictioanry', returns dictionary format.
    full_label : bool, optional
        Whether to include full labels for best fit points.
    legend : bool, optional
        Whether to show legend. Defaults to True.
    plot_props : dict, optional
        Dictionary of plot properties for each dataset. If None, will be generated.
    plot_props_images : bool or dict, optional
        If dict, contains image information for each dataset. If False, images are not plotted.
    best_fit : bool, optional
        Whether to plot best fit points. Note: CSV files may not have best fit data. Defaults to True.
    image_color : str, optional
        Color for images when plot_props_images is used. Defaults to 'black'.
    drop_images : bool, optional
        Whether to drop images even if plot_props_images is provided. Defaults to False.
    shift_x : float, optional
        Multiplier for x values. Defaults to 1.
    shift_y : float, optional
        Multiplier for y values. Defaults to 1.
    
    Returns
    -------
    matplotlib.axes.Axes or tuple
        The axes object used for plotting, or tuple with additional data if full_output is True.
    """
    
    # Handle data - it may be a DataFrame or a file path
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        # It's a file path
        # if not os.path.isabs(data):
        #     # If relative path, try to find it relative to the Preset_values directory
        #     data = os.path.join(os.path.dirname(__file__), 'Preset_values', data)
        df = pd.read_csv(data, index_col=0)
    
    # Get dataset column names (exclude CI columns)
    all_datasets = [col for col in df.columns if ' 95% CI' not in col]
    
    if dataset_keys == 'all':
        dataset_keys = all_datasets
    else:
        # Filter to only include datasets that exist
        dataset_keys = [key for key in dataset_keys if key in all_datasets]
    
    if not dataset_keys:
        raise ValueError("No valid dataset keys found in CSV file")
    
    # Generate plot properties if not provided
    if plot_props is None:
        if not plot_props_images:
            plot_props = getPlotProps(dataset_keys, mult_species=multiple_species)
        else:
            plot_props = {}
            for key in dataset_keys:
                plot_props[key] = {'color': image_color, 'marker': 'o'}
    
    # Handle divide/multiply parameters as lists
    inputs_dict = {
        'param1': param1, 
        'param2': param2, 
        'divide_by_param1': divide_by_param1, 
        'multiply_param1': multiply_param1, 
        'divide_by_param2': divide_by_param2, 
        'multiply_param2': multiply_param2
    }
    
    for key in ['divide_by_param1', 'multiply_param1', 'divide_by_param2', 'multiply_param2']:
        input_param = inputs_dict[key]
        if input_param is not None and not isinstance(input_param, list):
            inputs_dict[key] = [input_param]
    
    def getTitle(param, inputs_dict, keys):
        """Generate title for parameter with divide/multiply operations."""
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
    
    def getValue(df, param, dataset, multiply_params, divide_params):
        """Get parameter value and error from CSV DataFrame."""
        if param not in df.index:
            raise ValueError(f"Parameter '{param}' not found in CSV file")
        
        # Get base value
        v = float(df.loc[param, dataset])
        
        # Apply divide operations
        v_divide_factor = 1
        if divide_params is not None:
            for div_param in divide_params:
                if div_param not in df.index:
                    raise ValueError(f"Divide parameter '{div_param}' not found in CSV file")
                v_divide_factor *= float(df.loc[div_param, dataset])
            v /= v_divide_factor
        
        # Apply multiply operations
        v_multiply_factor = 1
        if multiply_params is not None:
            for mult_param in multiply_params:
                if mult_param not in df.index:
                    raise ValueError(f"Multiply parameter '{mult_param}' not found in CSV file")
                v_multiply_factor *= float(df.loc[mult_param, dataset])
            v *= v_multiply_factor
        
        # Get CI if it exists
        ci_col = f"{dataset} 95% CI"
        verr = None
        if ci_col in df.columns:
            ci_value = df.loc[param, ci_col]
            if pd.notna(ci_value):
                try:
                    if isinstance(ci_value, str):
                        ci_parsed = ast.literal_eval(ci_value)
                    else:
                        ci_parsed = ci_value
                    
                    if isinstance(ci_parsed, (list, tuple)) and len(ci_parsed) == 2:
                        ci_low, ci_high = float(ci_parsed[0]), float(ci_parsed[1])
                        
                        # Apply same transformations to CI bounds
                        ci_low_transformed = ci_low
                        ci_high_transformed = ci_high
                        
                        if divide_params is not None:
                            for div_param in divide_params:
                                div_val = float(df.loc[div_param, dataset])
                                ci_low_transformed /= div_val
                                ci_high_transformed /= div_val
                        
                        if multiply_params is not None:
                            for mult_param in multiply_params:
                                mult_val = float(df.loc[mult_param, dataset])
                                ci_low_transformed *= mult_val
                                ci_high_transformed *= mult_val
                        
                        # Calculate error bars: [lower_error, upper_error]
                        verr = [max(0, v - ci_low_transformed), max(0, ci_high_transformed - v)]
                except (ValueError, SyntaxError):
                    pass
        
        return v, verr
    
    # Create or get axes
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True
    else:
        fig = ax.get_figure()
    
    x_values, y_values, x_errors, y_errors, keys = [], [], [], [], []
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    for dataset_key in dataset_keys:
        # Check if we should plot this dataset (when plot_props_images is used)
        if (plot_props_images and dataset_key in plot_props_images.keys()) or not plot_props_images:
            try:
                # Get values and errors
                x, xerr = getValue(df, param1, dataset_key, 
                                  inputs_dict['multiply_param1'], 
                                  inputs_dict['divide_by_param1'])
                y, yerr = getValue(df, param2, dataset_key, 
                                  inputs_dict['multiply_param2'], 
                                  inputs_dict['divide_by_param2'])
                
                x_values.append(x)
                y_values.append(y)
                x_errors.append(xerr)
                y_errors.append(yerr)
                keys.append(dataset_key)


                # Format error bars for plotting
                if xerr is not None:
                    xerr_plot = [[xerr[0]*shift_x], [xerr[1]*shift_x]]
                else:
                    xerr_plot = None
                
                if yerr is not None:
                    yerr_plot = [[yerr[0]*shift_y], [yerr[1]*shift_y]]
                else:
                    yerr_plot = None
                
                # Handle error bar limits for log scale
                if xscale == 'linear' and xerr is not None and x - xerr[0] < 0:
                    xerr_plot = [[x*shift_x], [xerr[1]*shift_x]]
                elif xscale == 'log' and xerr is not None and x - xerr[0] < 1e-2*x:
                    xerr_plot = [[(x-1e-2*x)*shift_x], [xerr[1]*shift_x]]
                
                if yscale == 'linear' and yerr is not None and y - yerr[0] < 0:
                    yerr_plot = [[y*shift_y], [yerr[1]*shift_y]]
                elif yscale == 'log' and yerr is not None and y - yerr[0] < 1e-2*y:
                    yerr_plot = [[(y-1e-2*y)*shift_y], [yerr[1]*shift_y]]
                
                # Extract plot properties, only including those that exist
                # Valid matplotlib errorbar properties
                valid_errorbar_props = [
                    'color', 'marker', 'label', 'alpha', 'markersize', 
                    'markerfacecolor', 'markeredgewidth', 'markeredgecolor',
                    'linestyle', 'capsize', 'capthick', 'elinewidth', 'elinecolor',
                    'capcolor', 'barsabove', 'lolims', 'uplims', 'xlolims', 'xuplims'
                ]
                
                errorbar_kwargs = {'linestyle': 'None'}
                
                # Use label from plot_props if available, otherwise use dataset_key
                if 'label' in plot_props[dataset_key]:
                    errorbar_kwargs['label'] = plot_props[dataset_key]['label']
                else:
                    errorbar_kwargs['label'] = dataset_key
                
                # Add all other valid properties that exist in plot_props
                for prop in valid_errorbar_props:
                    if prop in plot_props[dataset_key] and prop != 'label':
                        errorbar_kwargs[prop] = plot_props[dataset_key][prop]
                
                # Plot error bars
                ax.errorbar(x*shift_x, y*shift_y, xerr=xerr_plot, yerr=yerr_plot, 
                           **errorbar_kwargs)
                
                # Plot images if plot_props_images is provided
                if plot_props_images:
                    if not drop_images:
                        # This code plots the image at an offset so as not to overlap with the data point.
                        # This takes into account that the image is zoomed in, and that the axis might be in log scale.
                        if dataset_key in plot_props_images.keys():
                            img = plot_props_images[dataset_key]['image']
                            zoom = plot_props_images[dataset_key]['zoom']
                            offset_factor_x = plot_props_images[dataset_key]['offset'][0]
                            offset_factor_y = plot_props_images[dataset_key]['offset'][1]
                            direction = plot_props_images[dataset_key]['direction']
                            imwidth = img.shape[1]
                            imheight = img.shape[0]
                            disp_x, disp_y = ax.transData.transform((x*shift_x, y*shift_y))
                            disp_offset_x = imwidth * zoom
                            disp_offset_y = imheight * zoom
                            new_disp_x = disp_x + disp_offset_x * offset_factor_x
                            new_disp_y = disp_y + disp_offset_y * offset_factor_y
                            new_data_x, new_data_y = ax.transData.inverted().transform((new_disp_x, new_disp_y))
                            offset_x = new_data_x - x*shift_x
                            offset_y = new_data_y - y*shift_y
                            
                            add_image_marker(ax, img, x*shift_x + direction * offset_x, y*shift_y + offset_y, zoom, color=image_color)
            
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping dataset '{dataset_key}': {e}")
                continue
    
    ax.set_title(f"{param1_title} vs {param2_title}")
    ax.set_xlabel(param1_title)
    ax.set_ylabel(param2_title)
    ax.grid(True)
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    elif created_figure:
        plt.show()
    
    if full_output == 'dictioanry':
        return ax, {'x_values': x_values, 'y_values': y_values, 'x_errors': x_errors, 
                    'y_errors': y_errors, 'keys': keys, 'fig': fig, 'plot_props': plot_props}
    if full_output:
        return ax, x_values, y_values, x_errors, y_errors, keys, fig, plot_props
    return ax

