from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import seaborn as sns


class Dataset:
    """
    This class contains the data for the deathTimes dataset and allows access to survival and hazard information.
    Attributes:
        death_times (array-like): Array of death times.
        bandwidth (int, optional): Bandwidth for smoothing the hazard function. Default is 3.
        t_end (float): Maximum death time.
        n (int): Number of times measured.
        kmf (KaplanMeierFitter): Fitted Kaplan-Meier estimator.
        naf (NelsonAalenFitter): Fitted Nelson-Aalen estimator.
        median_lifetime (float): Median lifetime of the dataSet.
        survival (tuple): Survival function timeline and values.
        hazard (tuple): Hazard function timeline and values.
        kmf_confidence_interval (list): 95% confidence interval for the Kaplan-Meier estimator.
        data_dt (float): The time step size for the data.
    Methods:
        calc_survival_and_hazard(events=None):
            Calculates the survival and hazard functions for the dataset.
        plotSurvival(ax=None, time_range=None, **kwargs):
            Plots the survival function.
        plotScaledSurvival(ax=None, time_range=None, **kwargs):
            Plots the scaled survival function.
        get_death_times_distribution(bins=None, dt=1, time_range=None):
            Returns the death times distribution.
        plotDeathTimesDistribution(ax=None, bins=None, use_kde=False, dt=1, time_range=None, **kwargs):
            Plots the death times distribution.
        getSurvival(interpolate_time=None, time_range=None):
            Returns the survival function for the dataset.
        getScaledSurvival(interpolate_time=None):
            Returns the scaled survival function for the dataset.
        getMedianLifetime():
            Returns the median lifetime of the dataset.
        getSteepness(method='IQR'):
            Returns the steepness of the survival function.
        plotHazard(ax=None, time_range=None, **kwargs):
            Plots the hazard function.
        plotScaledHazard(ax=None, time_range=None, **kwargs):
            Plots the scaled hazard function.
        getHazard():
            Returns the hazard function for the dataset.
        sample(n):
            Samples n death times from the dataset and returns a "sampled" cohort.
        plot_survival_bootstrap(ax=None, **kwargs):
            Plots the survival function with bootstrap confidence intervals.
        getConfidenceInterval():
            Returns the 95% confidence interval for the Kaplan-Meier estimator.
        getMaxLifetime():
            Returns the maximum lifetime of the dataset.
        getMedianLifetimeCI():
            Returns the 95% confidence interval of the median lifetime of the dataset.
        getDeathTimes():
            Returns the death times of the dataset.
        survivalAtTimes(times):
            Returns the survival probability at given times.
        subSetByProperty(property, value):
            Returns a subset of the dataset based on a property.
        subsetByProperties(properties, values):
            Returns a subset of the dataset based on a list of properties.
        addProperty(property, value):
            Adds a property to the dataset.
        splitByProperties(properties):
            Splits the dataset by properties and returns a dictionary of datasets.
        removeNans():
            Removes NaNs from the dataset.
        toCsv(file_name, properties=False):
            Saves the dataset to a CSV file.
    """
    
    def __init__(self, death_times,events, external_hazard = np.inf, bandwidth=3, properties = None, data_dt=1, event_is_censored=False):
        """
        This function initializes the Dataset object.
        properties (dict, optional): Dictionary of properties for the dataset.
        data_dt (float, optional): The time step size for the data. Default is 1.
        event_is_censored (bool, optional): If True, the events column is a censor column (1=censored, 0=event).
                                            Events will be flipped when saving. Default is False.
        """
        self.death_times = death_times
        self.t_end = np.max(death_times)
        # If event_is_censored is True, flip the events (1->0, 0->1)
        if event_is_censored:
            self.events = 1 - events
        else:
            self.events = events
        self.event_is_censored = event_is_censored
        self.bandwidth = bandwidth
        self.n = len(death_times)
        self.kmf = None
        self.naf = None
        self.median_lifetime = None
        self.survival = None
        self.hazard = None
        self.external_hazard = external_hazard
        self.properties = properties
        self.data_dt = data_dt
        self.calc_survival_and_hazard(self.events)
    
    @property
    def npeople(self):
        return self.n

    @npeople.setter
    def npeople(self, value):
        self.n = value


    def coarse_grain(self, dt=1):
        """
        Round all death times up to the nearest multiple of ``dt`` and recompute survival statistics.

        Parameters:
            dt (float): Desired temporal resolution. Must be positive.

        Returns:
            Dataset: The dataset instance (for chaining).
        """
        if dt is None:
            return self
        if dt <= 0:
            raise ValueError("dt must be positive.")
        

        death_times = np.asarray(self.death_times, dtype=float)
        eps = np.finfo(float).eps
        coarse_times = np.ceil(death_times / dt - eps) * dt

        self.death_times = coarse_times
        self.t_end = float(np.max(coarse_times)) if coarse_times.size else 0.0
        self.data_dt = dt
        self.calc_survival_and_hazard(self.events)
        return self


    def calc_survival_and_hazard(self, events = None):
        """
        This function calculates the survival and hazard functions for the dataset.
        """
        T = self.death_times
        if events is not None:
            E = events
        else:
            E = np.ones_like(T)
            self.events = E

        
        kmf = KaplanMeierFitter().fit(T, E)
        self.survival = kmf.timeline, np.array(kmf.survival_function_.values)[:,0]
        #95% confidence interval
        self.kmf_confidence_interval = [np.array(kmf.confidence_interval_['KM_estimate_lower_0.95'].values), np.array(kmf.confidence_interval_['KM_estimate_upper_0.95'].values)]
        self.median_lifetime = kmf.median_survival_time_
        self.kmf = kmf

        naf = NelsonAalenFitter().fit(T, event_observed=E)
        self.hazard = naf.timeline, np.array(naf.smoothed_hazard_(bandwidth=self.bandwidth).values)[:,0]
        self.naf = naf

    def plotSurvival(self, ax=None, time_range=None, **kwargs):
        """
        Plots the survival function.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            time_range (tuple, optional): The time range for the plot. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if time_range is not None:
            t,s = self.getSurvival(time_range=time_range)
            ax.plot(t, s, **kwargs)
        else:
            self.kmf.plot_survival_function(ax=ax, **kwargs)

        ax.set_xlabel('Age')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotScaledSurvival(self, ax=None,time_range=None,CI=True, **kwargs):
        """
        Plots the scaled survival function.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            time_range (tuple, optional): The time range for the plot. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        t,s = self.getSurvival(time_range=time_range)
        if time_range is None:
            median_time = self.kmf.median_survival_time_
        else:
            median_time = t[np.argmin(np.abs(s-0.5))]

        ax.plot(t/median_time, s, **kwargs)
        # add confidence interval
        bottom, top = self.getSurvivalCI(time_range)
        if CI:
            if 'color' in kwargs:
                ax.fill_between(t/median_time, bottom, top, alpha=0.3, color=kwargs['color'])
            else:
                ax.fill_between(t/median_time, bottom, top, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def getSurvivalCI(self, time_range = None):
        """
        This function returns the 95% confidence interval for the survival function.
        """
        bottom, top = self.kmf_confidence_interval
        if time_range is not None:
            t,s = self.getSurvival()
            tb,bottom = trim_to_range(t,bottom,time_range, renormalize_survival = True)
            tt,top = trim_to_range(t,top,time_range, renormalize_survival = True)
        return bottom, top
        
    
    def get_death_times_distribution(self, bins = None, dt=1, time_range = None):
        """
        The probability of death in an interval is calculated as the number of deaths in the interval divided by the total number of individuals (npeople).
        The last bin represents the probability of death after the last recorded death time.
        Parameters:
        - bins (array-like, optional): The bins for the distribution. If bins is None, the death times distribution is binned by the time units of the simulation, 
                           with bins = np.linspace(0, self.t_end, self.t_end + 1).
        - time_range (tuple, optional): The time range for the distribution. If time_range is not None, the bins are between time_range[0] and time_range[1], 
                        and death_times is only death_times within the time_range.
        Returns:
        - prob_death (numpy.ndarray): The probability of death in each bin.
        - bin_edges (numpy.ndarray): The edges of the bins used for the distribution.
        """
        if bins is None:
            if time_range is not None:
                bins = np.linspace(time_range[0], time_range[1], int(time_range[1] - time_range[0] + 1))
            else:
                bins = np.linspace(0, self.t_end, int(self.t_end//dt + 1))
        death_times = self.death_times.copy()
        events = self.events
        death_times = death_times[events == 1]
        n =self.n
        if time_range is not None:
            n_death = len(death_times) 
            death_times = death_times[(death_times >= time_range[0]) & (death_times < time_range[1])]
            new_n = len(death_times)
            d= n_death-new_n
            n=n-d
        
        prob_death, bin_edges = np.histogram(death_times, bins=bins)

        if time_range is  None:
            #if the length of the death times < npeople then add -inf to the end of the death times to make death times the same length as npeople
            if len(death_times)<n:
                death_times = np.append(death_times,np.inf*np.ones(n-len(death_times)))
            #append the probability of death after the last death time
            prob_death = np.append(prob_death, n-np.sum(prob_death))
            bin_edges = np.append(bin_edges, np.inf)

        prob_death = prob_death/n
        
        return prob_death, bin_edges
    
    def plotDeathTimesDistribution(self, ax=None, bins=None, use_kde =False, dt=1, time_range=None, **kwargs):
        """
        This function plots the death times distribution.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            bins (array-like, optional): The bins for the distribution. Default is None.
            use_kde (bool, optional): Whether to use Kernel Density Estimation. Default is False.
            dt (float, optional): The time step size for the data. Default is 1.
            time_range (tuple, optional): The time range for the distribution. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        death_times = self.death_times.copy()
        events = self.events
        death_times = death_times[events == 1]
        portion_died = sum(events)/len(events)
        if time_range is not None:
            death_times = death_times[(death_times>=time_range[0]) & (death_times<=time_range[1])]
        if use_kde:
            from scipy.stats import gaussian_kde
            x = np.linspace(0, self.t_end, 1000)
            kde = gaussian_kde(death_times, bw_method='scott')
            ax.plot(x, portion_died*kde(x), **kwargs)
        else:
            prob_death, bin_edges = self.get_death_times_distribution(bins,dt=dt,time_range=time_range)
            ax.step(bin_edges[:-1], prob_death, where='post', **kwargs)
        ax.set_xlabel('Death time (days)')
        ax.set_ylabel('Probability of death')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax, bins
    
    
    
    def getSurvival(self,interpolate_time = None, time_range = None):
        """
        This function returns the survival function for the  dataset.
        
        Parameters:
            interpolate_time (array-like, optional): The time points to interpolate the survival function. Default is None.
            time_range (tuple, optional): The time range for the survival function. Default is None.
        
        Returns:
            t (numpy.ndarray): The time points of the survival function.
            s (numpy.ndarray): The survival probabilities at the time points.
        """
        if interpolate_time is not None:
            t,s = self.survival
            t = t.copy()
            s = s.copy()
            #if interpolate_time is outside the range of t, the survival is assumed to be 0
            if interpolate_time[-1]>t[-1]:
                interpolate_time_trunckated= interpolate_time[interpolate_time<=t[-1]]
            else:
                interpolate_time_trunckated = interpolate_time
            s = np.interp(interpolate_time_trunckated, t, s)
            s_extended = np.zeros(len(interpolate_time))
            s_extended[:len(interpolate_time_trunckated)] = s
            return interpolate_time , s_extended
        elif time_range is not None:
            t,s = self.survival
            t = t.copy()
            s = s.copy()
            t,s = trim_to_range(t,s,time_range, renormalize_survival = True)
        else:
            t,s = self.survival
            t = t.copy()
            s = s.copy()
        return t,s
    

    def getScaledSurvival(self, interpolate_time = None):
        """
        This function returns the survival function for the yeast dataset scaled by the median lifetime.
        
        Parameters:
            interpolate_time (array-like, optional): The time points to interpolate the survival function. Default is None.
        
        Returns:
            t (numpy.ndarray): The time points of the survival function.
            s (numpy.ndarray): The survival probabilities at the time points.
        """
        if interpolate_time is not None:
            t,s = self.survival
            median_time = self.median_lifetime
            t = t/median_time
            #if interpolate_time is outside the range of t, the survival is assumed to be 0
            if interpolate_time[-1]>t[-1]:
                interpolate_time_trunckated= interpolate_time[interpolate_time<=t[-1]]
            else:
                interpolate_time_trunckated = interpolate_time
            s = np.interp(interpolate_time_trunckated, t, s)
            s_extended = np.zeros(len(interpolate_time))
            s_extended[:len(interpolate_time_trunckated)] = s
            return interpolate_time , s_extended
        else:
            return self.survival
    
    
    def getMedianLifetime(self):
        """
        This function returns the median lifetime of the  dataset.
        """
        return self.median_lifetime

    def getSteepness(self, method ='IQR'):
        """
        This function returns the steepness of the survival function as median lifetime/ inter quartile range.
        
        Parameters:
            method (str, optional): The method to calculate steepness. Default is 'IQR'.
        
        Returns:
            steepness (float): The steepness of the survival function.
        """
        t,s = self.survival
        if method == 'IQR':
            q1 = t[np.argmin(np.abs(s-0.25))]
            q3 = t[np.argmin(np.abs(s-0.75))]
            return -self.median_lifetime/(q3-q1)
        elif method == 'AMS10': #adjusted mean over std remove bottom 10% of the data
            #get the death times, remove bottom 10% of the data, return the average over the standard deviation
            death_times = self.death_times.copy()
            death_times = np.sort(death_times)
            n = len(death_times)        
            death_times = death_times[int(n*0.1):n]
            return np.mean(death_times)/np.std(death_times)
        elif method == 'AMS': #adjusted mean over std
            death_times = self.death_times.copy()
            return np.mean(death_times)/np.std(death_times)
        elif method == 'AMS_Sel': #adjusted mean over std selective by external hazard
            if self.external_hazard == np.inf:
                return self.getSteepness(method='AMS')
            else:
                return self.getSteepness(method='AMS10')
        else:
            raise ValueError('method should be either IQR or AMS')
        
    def remaining_lifetime_at_age(self, age, types='median'):
        """
        For individuals who survived to the given age, calculate statistics of their remaining lifespans.

        Parameters:
            age (float): The age at which to calculate remaining lifetimes.
            types (str or list): Which statistics to return. Options:
                - 'median': median of remaining lifespans
                - 'mean': mean of remaining lifespans
                - 'std': standard deviation of remaining lifespans
                - 'distribution': histogram of remaining lifespans (returns (hist, bin_edges))
                - list of any combination, e.g. ['median', 'mean', 'std']

        Returns:
            If types is a string, returns the corresponding value.
            If types is a list, returns a dict with keys from types.
        """
        # Get all death times that survived to at least 'age'
        death_times = np.array(self.death_times)
        remaining = death_times[death_times >= age] - age

        # If no one survived to this age, return None or np.nan
        if len(remaining) == 0:
            if isinstance(types, str):
                return np.nan
            else:
                return {t: np.nan for t in (types if isinstance(types, list) else [types])}

        # Helper to compute each stat
        def compute(stat):
            if stat == 'median':
                return np.median(remaining)
            elif stat == 'mean':
                return np.mean(remaining)
            elif stat == 'std':
                return np.std(remaining)
            elif stat == 'distribution':
                # Return histogram and bin edges
                hist, bin_edges = np.histogram(remaining, bins='auto', density=True)
                return (hist, bin_edges)
            else:
                raise ValueError(f"Unknown type '{stat}' for remainingLifetimeAtAge")

        if isinstance(types, str):
            return compute(types)
        else:
            # Assume iterable/list
            return {stat: compute(stat) for stat in types}

    def plotHazard(self, ax=None, time_range=None, **kwargs):
        """
        Plots the hazard function.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            time_range (array-like, optional): Time range [start, stop] to filter the hazard plot. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        if time_range is not None:
            # Get hazard data and filter by time_range
            t, h = self.hazard
            t = t.copy()
            h = h.copy()
            # Filter to time range
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t_filtered = t[mask]
            h_filtered = h[mask]
            ax.plot(t_filtered, h_filtered, **kwargs)
        else:
            if self.bandwidth is None:
                self.naf.plot_hazard(ax=ax, **kwargs)
            else:
                self.naf.plot_hazard(ax=ax, bandwidth=self.bandwidth, **kwargs)
        ax.set_yscale('log')
        ax.set_xlabel('Age')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotScaledHazard(self, ax=None,scale =None,clean_plot=True,clean_at = 1e-6, CI=True, trim_by_n=10, time_range=None, **kwargs):
        """
        Plots the scaled hazard function.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            scale (float, optional): Scale factor for x-axis. Default is median survival time.
            clean_plot (bool, optional): Whether to clip values below clean_at. Default is True.
            clean_at (float, optional): Minimum value threshold. Default is 1e-6.
            CI (bool, optional): Whether to plot confidence intervals. Default is True.
            trim_by_n (int, optional): Trim plot to start after trim_by_n'th death and stop when 
                                      trim_by_n people are still alive. Default is 0 (no trimming).
            time_range (array-like, optional): Time range [start, stop] to filter the hazard plot. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if scale is None:
            scale = self.kmf.median_survival_time_
        
        # Calculate trim time bounds if trim_by_n > 0
        t_min = None
        t_max = None
        if trim_by_n > 0:
            # Get death times (only actual deaths, not censored)
            death_times = self.death_times[self.events == 1]
            death_times_sorted = np.sort(death_times)
            n_deaths = len(death_times_sorted)
            
            if trim_by_n < n_deaths and (n_deaths - trim_by_n) > trim_by_n:
                # Start after trim_by_n'th death (inclusive of next point)
                # Use the time of the (trim_by_n + 1)'th death as minimum
                t_min = death_times_sorted[trim_by_n] if trim_by_n < n_deaths else None
                
                # Stop when trim_by_n people are still alive
                # This means stop at the time of the (n_deaths - trim_by_n)'th death
                # (exclusive, so we use < t_max)
                t_max = death_times_sorted[n_deaths - trim_by_n - 1] if (n_deaths - trim_by_n) > 0 else None
            else:
                # trim_by_n is too large or would result in invalid range, no trimming
                t_min = None
                t_max = None
        
        # Use lifelines' plot_hazard to get correctly smoothed hazard with CI
        # Create isolated temporary figure for this call (allows multiple hazards on same axes)
        temp_fig = plt.figure(figsize=(1, 1))  # Small figure, won't be displayed
        temp_ax = temp_fig.add_subplot(111)
        
        # Plot with CI if requested - let lifelines handle all smoothing
        plot_kwargs = kwargs.copy()
        # Remove any CI-related parameters that might conflict
        plot_kwargs.pop('CI', None)
        plot_kwargs.pop('ci_show_lines', None)
        
        if CI:
            self.naf.plot_hazard(ax=temp_ax, bandwidth=self.bandwidth, **plot_kwargs)
        else:
            # If no CI, use ci_show=False to disable confidence intervals
            self.naf.plot_hazard(ax=temp_ax, bandwidth=self.bandwidth, ci_show=False, **plot_kwargs)
        
        # Extract data from the temporary plot before closing
        lines = temp_ax.get_lines()
        collections = temp_ax.collections
        
        # Get hazard line data
        hazard_data = None
        if len(lines) > 0:
            hazard_line = lines[0]
            t_orig = np.asarray(hazard_line.get_xdata(), dtype=float)
            h_orig = np.asarray(hazard_line.get_ydata(), dtype=float)
            hazard_data = {
                't': t_orig,
                'h': h_orig,
                'color': hazard_line.get_color(),
                'linestyle': hazard_line.get_linestyle(),
                'linewidth': hazard_line.get_linewidth(),
                'label': hazard_line.get_label() if hazard_line.get_label() else None
            }
        
        # Get CI fill data if present
        ci_data = None
        if CI and len(collections) > 0:
            for collection in collections:
                if hasattr(collection, 'get_paths'):
                    paths = collection.get_paths()
                    if len(paths) > 0:
                        # Get the path vertices - CI is typically a single polygon
                        vertices = paths[0].vertices
                        if len(vertices) > 0:
                            t_ci = vertices[:, 0]
                            h_ci = vertices[:, 1]
                            ci_data = {
                                't': t_ci,
                                'h': h_ci,
                                'alpha': collection.get_alpha() if hasattr(collection, 'get_alpha') else 0.3,
                                'color': collection.get_facecolor()[0] if hasattr(collection, 'get_facecolor') and len(collection.get_facecolor()) > 0 else kwargs.get('color')
                            }
                            break  # Use first collection found
        
        # Close temporary figure immediately after extracting data
        plt.close(temp_fig)
        
        # Now plot to the actual axes with scaling applied
        if hazard_data is not None:
            t_orig = hazard_data['t']
            h_orig = hazard_data['h']
            
            # Apply time_range filtering if requested
            if time_range is not None:
                mask = (t_orig >= time_range[0]) & (t_orig <= time_range[1])
                t_orig = t_orig[mask]
                h_orig = h_orig[mask]
            
            # Apply trim_by_n filtering if requested
            if trim_by_n > 0 and t_min is not None and t_max is not None:
                mask = (t_orig >= t_min) & (t_orig <= t_max)
                t_orig = t_orig[mask]
                h_orig = h_orig[mask]
            
            # Rescale time
            t_scaled = t_orig / scale
            
            # Apply clean_at clipping if requested
            if clean_plot:
                mask = h_orig > clean_at
                t_scaled = t_scaled[mask]
                h_orig = np.maximum(h_orig[mask], clean_at)
            
            # Plot scaled hazard
            ax.plot(t_scaled, h_orig, color=hazard_data['color'], 
                   linestyle=hazard_data['linestyle'], 
                   linewidth=hazard_data['linewidth'],
                   label=hazard_data['label'], **{k: v for k, v in kwargs.items() if k not in ['color', 'linestyle', 'linewidth', 'label']})
        
        # Plot CI if available
        if ci_data is not None:
            t_ci = ci_data['t']
            h_ci = ci_data['h']
            
            # Apply time_range filtering if requested
            if time_range is not None:
                mask = (t_ci >= time_range[0]) & (t_ci <= time_range[1])
                t_ci = t_ci[mask]
                h_ci = h_ci[mask]
            
            # Apply trim_by_n filtering if requested
            if trim_by_n > 0 and t_min is not None and t_max is not None:
                mask = (t_ci >= t_min) & (t_ci <= t_max)
                t_ci = t_ci[mask]
                h_ci = h_ci[mask]
            
            # Rescale time
            t_ci_scaled = t_ci / scale
            
            # Apply clean_at clipping
            if clean_plot:
                mask = h_ci > clean_at
                t_ci_scaled = t_ci_scaled[mask]
                h_ci = np.maximum(h_ci[mask], clean_at)
            
            # Fill the polygon (CI bounds)
            ax.fill(t_ci_scaled, h_ci, 
                   alpha=ci_data['alpha'],
                   color=ci_data['color'],
                   edgecolor='none')
        
        ax.set_yscale('log')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        return ax
    
    def getHazard(self):
        """
        This function returns the hazard function for the yeast dataset.
        """
        return self.hazard
    
    def sample(self, n):
        """
        This function samples n death times from the yeast dataset and returns a "sampled" cohort.
        
        Parameters:
            n (int): The number of samples to draw.
        
        Returns:
            sampled_dataset (Dataset): The sampled dataset.
        """
        indices = np.random.choice(range(self.n), n, replace=True)
        s_detimes = self.death_times[indices]
        s_events = self.events[indices]
        if self.properties is not None:
            s_properties = {prop: self.properties[prop][indices] for prop in self.properties.keys()}
        else:
            s_properties = None
        return Dataset(s_detimes, s_events, external_hazard=self.external_hazard, bandwidth=self.bandwidth, properties=s_properties)
    
    
    def plot_survival_bootstrap(self,ax=None, **kwargs):
        """
        This function plots the survival function with bootstrap confidence intervals.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        def get_survival_bootstrap(death_times, events,n_bootstraps):
            n = len(death_times)
            bootstraps = np.random.choice(n, (n_bootstraps, n), replace=True)
            survival_bootstraps = []
            times =[]
            for bootstrap in bootstraps:
                T = death_times[bootstrap]
                E = events[bootstrap]
                kmf = KaplanMeierFitter().fit(T, E)
                survival_bootstraps.append(np.array(kmf.survival_function_.values))
                times.append(kmf.timeline)
            #create a single timeline for all the bootstraps
            t = np.unique(np.concatenate(times))
            #interplate the survival function for each bootstrap at the common timeline
            for i in range(n_bootstraps):
                # print(survival_bootstraps[i][:,0])
                # print(np.shape(times[i]))
                # print(np.shape(t))
                survival_bootstraps[i] = np.interp(t, times[i], survival_bootstraps[i][:,0])
            # survival_bootstraps = [np.interp(t, times[i], survival_bootstraps[i]) for i in range(n_bootstraps)]
            survival_bootstraps = np.array(survival_bootstraps)
            survival_bootstraps = np.sort(survival_bootstraps, axis=0)
            lower_bound = survival_bootstraps[int(0.025 * n_bootstraps)]
            upper_bound = survival_bootstraps[int(0.975 * n_bootstraps)]
            return lower_bound, upper_bound
        if ax is None:
            fig, ax = plt.subplots()
        t,s = self.survival
        lower_bound, upper_bound = get_survival_bootstrap(self.death_times, self.events, 1000)
        ax.plot(t, s, label='Survival function')
        ax.fill_between(t, lower_bound, upper_bound, alpha=0.3,**kwargs)
        return ax
    
    def getConfidenceInterval(self):
        """
        This function returns the 95% confidence interval for the Kaplan-Meier estimator.
        """
        return self.kmf_confidence_interval
    
    def getMaxLifetime(self):
        """
        This function returns the maximum lifetime of the dataset.
        """
        death_times = self.death_times
        return np.max(death_times)
    
    def getMedianLifetimeCI(self):
        """
        This function returns the 95% confidence interval of the median lifetime of the dataset.
        """
        CI = self.kmf_confidence_interval
        idx = np.argmin(np.abs(self.survival[1]-0.5))
        mCI = [CI[0][idx],CI[1][idx]]
        return mCI

    def getDeathTimes(self):
        """
        This function returns the death times of the dataset.
        """
        return self.death_times
    
    def survivalAtTimes(self, times):
        """
        This function returns the survival probability at a given time using the kmf.
        The functions takes the pd.Series of the kmf and the time at which the survival probability is to be calculated and 
        converts it to a numpy array or a float.
        
        Parameters:
            times (array-like): The time points to get the survival probabilities.
        
        Returns:
            survival_prob (numpy.ndarray or float): The survival probabilities at the given times.
        """
        values = self.kmf.survival_function_at_times(times).values
        if len(values) == 1:
            return values[0]
        else:
            return np.array(values)
        
    def timeAtSurvival(self, survival_prob):
        """
        This function returns the time at which the survival probability is reached.
        """
        # To find the time at which the survival function reaches a given probability, we need to interpolate.
        # We'll use the survival curve (t, s) and interpolate the time at which s == survival_prob.
        t, s = self.survival
        # Ensure s is decreasing, and survival_prob is within the range
        if survival_prob > s[0] or survival_prob < s[-1]:
            raise ValueError("survival_prob is outside the range of the survival function.")
        # Interpolate to find the time at which survival == survival_prob
        return np.interp(survival_prob, s[::-1], t[::-1])
        
    def subSetByProperty(self, property, value):
        """
        This function returns a subset of the dataset based on a property.
        
        Parameters:
            property (str): The property to filter by.
            value (any): The value of the property to filter by.
        
        Returns:
            subset (Dataset): The subset of the dataset.
        """
        if property not in self.properties:
            raise ValueError(f'Property {property} not found in the dataset.')
        idx = self.properties[property] == value
        # if there are no elements with the given property, return None
        if np.sum(idx) == 0:
            return None
        death_times = self.death_times[idx]
        events = self.events[idx]
        properties = {prop: self.properties[prop][idx] for prop in self.properties}
        return Dataset(death_times, events, external_hazard = self.external_hazard, bandwidth=self.bandwidth, properties = properties)
    
    def subsetByProperties(self, properties,values):
        """
        This function returns a subset of the dataset based on a list of properties.
        
        Parameters:
            properties (list): The properties to filter by.
            values (list): The values of the properties to filter by.
        
        Returns:
            subset (Dataset): The subset of the dataset.
        """
        if len(properties) != len(values):
            raise ValueError('The number of properties should match the number of values.')
        idx = np.ones(self.n, dtype=bool)
        for prop, value in zip(properties, values):
            idx = idx & (self.properties[prop] == value)
        if np.sum(idx) == 0:
            return None
        death_times = self.death_times[idx]
        events = self.events[idx]
        properties = {prop: self.properties[prop][idx] for prop in self.properties}
        return Dataset(death_times, events, external_hazard = self.external_hazard, bandwidth=self.bandwidth, properties = properties)
    
    def addProperty(self, property, value):
        """
        This function adds a property to the dataset.
        value (any): Value of the property. should be the same length as the death_times or a single value.
                     If a single value is provided, it is broadcasted to the length of the death_times.
        
        Parameters:
            property (str): The name of the property.
            value (any): The value of the property. Should be the same length as the death_times or a single value.
        """
        if np.size(value) == 1:
            value = np.repeat(value, self.n)
        if self.properties is None:
            self.properties = {property: value}
        else:
            self.properties[property] = value
        return
    
    def splitByProperties(self, properties):
        """
        This function splits the dataset by properties and returns a dictionary of datasets (each with unique values for the properties).
        
        Parameters:
            properties (list): The properties to split by.
        
        Returns:
            datasets (dict): A dictionary of datasets.
        """
        unique_properties = {prop: np.unique(self.properties[prop]) for prop in properties}
        combos = [[]]

        for prop in properties:
            new_combos = []#combos.copy()
            for value in unique_properties[prop]:
                for combo in combos:
                    if len(combos)==1 or ((len(combos) !=1) and combo != []):
                        new_combos.append(combo + [value])
            combos = new_combos
        datasets = {}
        for combo in combos:
            key = ','.join([f'{prop}:_{value}' for prop, value in zip(properties, combo)])
            subset = self.subsetByProperties(properties, combo)
            if subset is not None:
                datasets[key] = subset

        return datasets
    

    def removeNans(self):
        """
        This function removes NaNs from the dataset. If a death time, an event, or a property is NaN, the corresponding agent is removed.
        """
        idx = ~np.isnan(self.death_times) & ~np.isnan(self.events)
        if self.properties is not None:
            for prop in self.properties:
                idx = idx & pd.notnull(self.properties[prop])
        self.death_times = self.death_times[idx]
        self.events = self.events[idx]
        if self.properties is not None:
            for prop in self.properties:
                self.properties[prop] = self.properties[prop][idx]
        self.n = len(self.death_times)
        self.calc_survival_and_hazard(self.events)
        return
    

    def toCsv(self, file_name,properties = False):
        """
        This function saves the dataset to a csv file.
        
        Parameters:
            file_name (str): The name of the CSV file.
            properties (bool, optional): Whether to include properties in the CSV file. Default is False.
        """
        # If event_is_censored is True, flip events back when saving (1->0, 0->1)

        events_to_save = self.events
        data = {'death times': self.death_times, 'events': events_to_save}
        if properties:
            data.update(self.properties)
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        return
        
        
 

class DatasetCollection:
    """
    This class contains a collection of datasets and allows for comparison of survival and hazard functions, 
    and check for batch effects.
    """
    def __init__(self, file_names =None, datasets=None, properties = None, additional_properties = None,warnings = True, death_times_column = None, events_column = None,use_base_file_name = True, event_is_censored=False):
        """
        This function initializes the DatasetCollection object. If file_names is provided, the datasets are loaded from the files.
        If both file_names and datasets are none, an empty dictionary is created.
        Parameters:
            file_names (list, optional): List of file names to load the datasets.
            datasets (list, or dict, optional): List of datasets or dictionary of datasets. If a list is provided a dictionary is created, and index of the dataset is added as a property.
                                                If a dictionary is provided, the keys should adhere to the naming convention given by properties.
            properties (list, optional): List of properties to separate the datasets by. If several files are provided, then the "file_name" property is
                                            added to the properties (separating datasets by files). The naming convention for the keys is f'property1:_{property1_value},property2:_{property2_value}...'.
            additional_properties (list, optional): List of additional properties to load from the files.
            event_is_censored (bool, optional): If True, the events column is a censor column (1=censored, 0=event).
                                                Events will be flipped when loading. Default is False.
        """
        
        #if both file_names and datasets are supplied, raise an error
        if file_names is not None and datasets is not None:
            raise ValueError('Only one of file_names or datasets should be provided.')
        
        #check if additional_properties were provided when file_names is None and print a warning
        if file_names is None and additional_properties is not None and warnings:
            print('WARNING Additional properties were provided without file names. These properties will not be loaded. (This message can be suppressed by setting warnings=False)')
        
        self.datasets = {}
        #concatenate properties and additional_properties
        if properties is not None:
            if additional_properties is not None:
                all_properties = properties + additional_properties
            else:
                all_properties = properties           
        else:
            all_properties = additional_properties

        if file_names is not None:
            #check if file_names is a list
            if not isinstance(file_names, list):
                file_names = [file_names]
            if len(file_names) > 1 and properties is not None:
                properties.append('file_name')
        if file_names is not None:
            for file_name in file_names:
                dataset = dsFromFile(file_name, properties=all_properties, death_times_column = death_times_column, events_column = events_column, event_is_censored=event_is_censored)
                dataset.removeNans()
                if use_base_file_name:
                    file_name = file_name.split('/')[-1]
                if properties is not None:
                    if len(file_names) > 1:
                        dataset.addProperty('file_name', file_name)
                    datasets = dataset.splitByProperties(properties)
                    self.datasets.update(datasets)
                else:
                    key = f"file_name:_{file_name}"
                    self.datasets[key] = dataset
        elif datasets is not None:
            if isinstance(datasets, list):
                if properties is None:
                    properties = []
                properties.append('index')  
                for i, dataset in enumerate(datasets):
                    if properties is not None:
                        dataset.addProperty('index', i)
                    datasets = dataset.splitByProperties(properties)
                    self.datasets.update(datasets)
            elif isinstance(datasets, dict):
                if properties is not None and warnings:
                    print('WARNING Properties are ignored when datasets are provided as a dictionary. (This message can be suppressed by setting warnings=False)')
                self.datasets = datasets
        self.properties = properties
        self.additional_properties = additional_properties

    def nDatasets(self):
        """
        This function returns the number of datasets in the DatasetCollection.
        """
        return len(self.datasets)
    
    def getKeys(self, properties=None, values=None, randomize=False):
        """
        This function returns the keys of the datasets that match the specified properties and values.
        For properties that are not specified, all values are considered.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
        
        Returns:
            list: List of keys of the datasets that match the specified properties and values.
        """
        if properties is None:
            keys = list(self.datasets.keys())
        else:
            if values is None:
                raise ValueError('Values should be provided when properties are specified.')
            if len(properties) != len(values):
                raise ValueError('The number of properties should match the number of values.')
            keys = []
            for key in self.datasets.keys():
                match = [False] * len(properties)
                for prop, value in zip(properties, values):
                    data = key.split(',')
                    for d in data:
                        if prop in d:
                            idx = properties.index(prop)
                            if d.split(':_')[1] in value:
                                match[idx] = True
                if all(match):
                    keys.append(key)
        
        if randomize:
            rng = np.random.default_rng()
            keys = rng.choice(np.array(keys), randomize, replace=False).tolist()
        
        return keys

    def plotSurvival(self, properties=None, values=None, randomize=False, ax=None, **kwargs):
        """
        Plots the survival function of the datasets.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        for key in keys:
            kwargs['label'] = key
            self.datasets[key].plotSurvival(ax=ax, **kwargs)
        ax.set_xlabel('Age')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        return ax
    
    def plotScaledSurvival(self, properties=None, values=None, randomize=False, ax=None, **kwargs):
        """
        Plots the scaled survival function of the datasets.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        for key in keys:
            kwargs['label'] = key
            self.datasets[key].plotScaledSurvival(ax=ax, **kwargs)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def plotHazard(self, properties=None, values=None, randomize=False, ax=None, **kwargs):
        """
        Plots the hazard function of the datasets.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Default is None.
            **kwargs: Additional keyword arguments for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        for key in keys:
            kwargs['label'] = key
            self.datasets[key].plotHazard(ax=ax, **kwargs)
        ax.set_yscale('log')
        ax.set_xlabel('Age')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    
    def get_combined_dataset(self, properties=None, values=None, randomize=False):
        """
        Returns a combined dataset of the specified datasets.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
        
        Returns:
            combined_dataset (Dataset): The combined dataset.
        """
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        if len(keys) == 0:
            import warnings
            warnings.warn("No datasets found matching the given criteria. Returning None.")
            return None
        death_times = np.concatenate([self.datasets[key].death_times for key in keys])
        events = np.concatenate([self.datasets[key].events for key in keys])
        return Dataset(death_times, events, bandwidth=self.datasets[keys[0]].bandwidth)
    
    def get_combined_dataset_averaged(self, properties=None, values=None, randomize=False):
        """
        Returns a combined dataset of the specified datasets, averaged by median lifetime.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
        
        Returns:
            combined_dataset (Dataset): The combined dataset, averaged by median lifetime.
        """
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        death_times = []
        events = []
        medians = []
        for key in keys:
            median_time = self.datasets[key].median_lifetime
            death_times.append(self.datasets[key].death_times / median_time)
            events.append(self.datasets[key].events)
            medians.append(median_time)
        death_times = np.concatenate(death_times)
        median_median = np.median(medians)
        death_times = death_times * median_median
        events = np.concatenate(events)
        return Dataset(death_times, events, bandwidth=self.datasets[keys[0]].bandwidth)
    
    def getSubset(self, properties=None, values=None, randomize=False, size=None):
        """
        Returns a subset of the DatasetCollection.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
            size (str, optional): Condition for the size of the dataset. Default is None.
        
        Returns:
            subset (DatasetCollection): The subset of the DatasetCollection.
        """
        if size is not None and (properties is not None or values is not None):
            raise ValueError('Size and properties/values cannot be provided together.')
        if size is not None:
            # Check if size is an integer
            if isinstance(size, int):
                size = f'<{size}'
            # Check if size is a valid condition: < x, > x, <= x, >= x, == x
            valid = False
            for cond in ['<=', '>=', '==', '<', '>']:
                if size.startswith(cond):
                    # Check if the rest of the string is an integer
                    try:
                        int(size[len(cond):])
                        valid = True
                    except:
                        pass
                    break
            if not valid:
                raise ValueError('Invalid size condition.')
            keys = [key for key in self.datasets.keys() if eval(f'self.datasets[key].n {size}', {'self': self, 'key': key})]
        else:
            keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        new_datasets = {key: self.datasets[key] for key in keys}
        return DatasetCollection(datasets=new_datasets)
    
    def getSubset_averaged(self, properties=None, values=None, randomize=False):
        """
        Returns a subset of the DatasetCollection, averaged by median lifetime.
        
        Parameters:
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): Number of keys to randomly select. Default is False.
        
        Returns:
            subset (DatasetCollection): The subset of the DatasetCollection, averaged by median lifetime.
        """
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        new_datasets = {}
        medians = [self.datasets[key].median_lifetime for key in keys]
        median_median = np.median(medians)
        for key in keys:
            dataset = self.datasets[key]
            death_times = dataset.death_times / dataset.median_lifetime * median_median
            events = dataset.events
            new_datasets[key] = Dataset(death_times, events, bandwidth=dataset.bandwidth)
        return DatasetCollection(datasets=new_datasets)
    
    def removeDatasets(self,size=None, properties=None, values=None, randomize=False):
        """
        This function removes datasets from the DatasetCollection, by size or by properties and values (but not both together).
        Size should by provided as a condition for removal ie size = '<100' or size = '==100'. if size is an integer, it is converted to '<size'.
        Parameters:
            size (str, optional): Condition for the size of the dataset. Default is None.
            properties (list, optional): List of properties to match.
            values (list of lists, optional): List of lists of values to match for each property.
            randomize (int, optional): Number of keys to randomly select. Default is False.
        """
        if size is not None and properties is not None:
            raise ValueError('Size and properties cannot be provided together.')
        if size is not None:
            #check if size is an integer
            if isinstance(size, int):
                size = f'<{size}'
            #check if size is a valid condition: < x, > x, <= x, >= x, == x
            valid = False
            for cond in ['<=', '>=', '==', '<', '>']:
                if size.startswith(cond):
                    #check if the rest of the string is an integer
                    try:
                        int(size[len(cond):])
                        valid = True
                    except:
                        pass
                    break

            keys = [key for key in self.datasets.keys() if eval(f'self.datasets[key].n {size}', {'self': self, 'key': key})]
        else:
            keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        new_datasets = {key: self.datasets[key] for key in self.datasets if key not in keys}
        for key in keys:
            print(f'Removing dataset {key}')
        return DatasetCollection(datasets=new_datasets)

    def drawConfidenceEllipse(self, n_samples=1000, n_per_sample='from distribution', data_percentile=95, sampled_percentile=98, title=None, show_property=None):
        """
        Draws confidence ellipses for the median lifetime and steepness of datasets in a DatasetCollection.
        Parameters:
            n_samples (int, optional): The number of samples to draw from the aggregated dataset. Default is 100.
            n_per_sample (int, optional): The number of samples per dataset. Default is 'from distribution'.
            data_percentile (int, optional): The percentile for the confidence ellipse for the data. Default is 95.
            sampled_percentile (int, optional): The percentile for the confidence ellipse for the sampled data. Default is 95.
            title (str, optional): The title of the plot. Default is None.
            show_property (str, optional): Property to color datasets by. Default is None.
        Returns:
            ax: The axes object of the plot.
        """
        medians = []
        steepness = []
        properties = []

        for key, dataset in self.datasets.items():
            medians.append(dataset.getMedianLifetime())
            steepness.append(dataset.getSteepness())
            if show_property:
                if show_property not in dataset.properties:
                    raise ValueError(f'Property {show_property} not found in the dataset.')
            # Extract the property value from the key name
                # prop_val = key.split(f'{show_property}:_')[1].split(',')[0]
                properties.append(key.split(f'{show_property}:_')[1].split(',')[0])

        sampled_datasets = []
        combined_dataset = self.get_combined_dataset()
        dataset_sizes = [dataset.n for dataset in self.datasets.values()]
        for _ in range(n_samples):
            size = np.random.choice(dataset_sizes) if n_per_sample == 'from distribution' else n_per_sample
            sampled_datasets.append(combined_dataset.sample(size))

        medians_s = []
        steepness_s = []
        for dataset in sampled_datasets:
            medians_s.append(dataset.getMedianLifetime())
            steepness_s.append(dataset.getSteepness())

        def confidence_ellipse(x, y, ax, percentile=95, facecolor='none', **kwargs):
            from matplotlib.patches import Ellipse 
            x = np.array(x)
            y = np.array(y)
            x_median = np.median(x)
            y_median = np.median(y)
            x_std = np.std(x)
            y_std = np.std(y)
            percent_from_edge = (100 - percentile) / 2
            x_95 = np.percentile(x, [percent_from_edge, 100 - percent_from_edge])
            y_95 = np.percentile(y, [percent_from_edge, 100 - percent_from_edge])
            ell_radius_x = max(np.abs(x_95[0] - x_median), np.abs(x_95[1] - x_median))
            ell_radius_y = max(np.abs(y_95[0] - y_median), np.abs(y_95[1] - y_median))
            ellipse = Ellipse((x_median, y_median), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
            ax.add_patch(ellipse)
            return ellipse

        fig, ax = plt.subplots()
        colors = plt.cm.hsv(np.linspace(0, 1, len(np.unique(properties))))
        palette = {prop: colors[i] for i, prop in enumerate(np.unique(properties))}
        confidence_ellipse(medians, steepness, ax, percentile=data_percentile, edgecolor='red', label='confidence interval on data')
        confidence_ellipse(medians_s, steepness_s, ax, percentile=sampled_percentile, edgecolor='blue', label='confidence interval on sampled data')
        ax.scatter(medians_s, steepness_s, alpha=0.5, label=f'sampled data datasets n={n_per_sample}', color='gray')

        if show_property:
            markers = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_']
            for i, prop in enumerate(np.unique(properties)):
                idx = np.array(properties) == prop
                ax.scatter(np.array(medians)[idx], np.array(steepness)[idx], alpha=0.5, label=f'{prop} data datasets', marker=markers[i % len(markers)]) #s=30 - i * 10
        else:
            ax.scatter(medians, steepness, alpha=0.5, label='data datasets', color='blue')

        ax.set_xlabel('Median lifetime')
        ax.set_ylabel('Steepness')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Median lifetime vs Steepness Data and sampled datasets')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=4)
        return ax

    def clean_DatasetCollection(self, n_samples=500, n_per_sample='from distribution', data_percentile=100, sampled_percentile=99):
        """
        Removes from the DatasetCollection datasets that are outside the confidence ellipse of the median lifetime and steepness of the sampled datasets.
        Parameters:
            n_samples (int, optional): The number of samples to draw from the aggregated dataset. Default is 500.
            n_per_sample (int, optional): The number of samples per dataset. Default is 'from distribution'.
            data_percentile (int, optional): The percentile for the confidence ellipse for the data. Default is 100.
            sampled_percentile (int, optional): The percentile for the confidence ellipse for the sampled data. Default is 99.9.
        Returns:
            new_DatasetCollection: The cleaned DatasetCollection.
        """
        medians = []
        steepness = []
        for dataset in self.datasets.values():
            medians.append(dataset.getMedianLifetime())
            steepness.append(dataset.getSteepness())

        sampled_datasets = []
        combined_dataset = self.get_combined_dataset()
        dataset_sizes = [dataset.n for dataset in self.datasets.values()]
        for _ in range(n_samples):
            size = np.random.choice(dataset_sizes) if n_per_sample == 'from distribution' else n_per_sample
            sampled_datasets.append(combined_dataset.sample(size))

        medians_s = []
        steepness_s = []
        for dataset in sampled_datasets:
            medians_s.append(dataset.getMedianLifetime())
            steepness_s.append(dataset.getSteepness())

        def is_within_confidence_ellipse(x, y, x_data, y_data, percentile):
            x = np.array(x)
            y = np.array(y)
            x_median = np.median(x_data)
            y_median = np.median(y_data)
            percent_from_edge = (100 - percentile) / 2
            x_95 = np.percentile(x_data, [percent_from_edge, 100 - percent_from_edge])
            y_95 = np.percentile(y_data, [percent_from_edge, 100 - percent_from_edge])
            ell_radius_x = max(np.abs(x_95[0] - x_median), np.abs(x_95[1] - x_median))
            ell_radius_y = max(np.abs(y_95[0] - y_median), np.abs(y_95[1] - y_median))
            dx = (x - x_median) / ell_radius_x
            dy = (y - y_median) / ell_radius_y
            return dx**2 + dy**2 <= 1

        new_datasets = {}
        for key, dataset in self.datasets.items():
            med = dataset.getMedianLifetime()
            steep = dataset.getSteepness()
            if is_within_confidence_ellipse(med, steep, medians_s, steepness_s, sampled_percentile):
                new_datasets[key] = dataset
            else:
                print(f"Removing dataset {key} with median lifetime {med} and steepness {steep}")

        return DatasetCollection(datasets=new_datasets)

    def plotSurvivalComparison(self, n_datasets=100, properties=None, values=None, randomize=False, ax=None, **kwargs):
        """
        Plots the survival function of datasets in a DatasetCollection for comparison vs sampled datasets.
        Parameters:
            n_datasets (int, optional): The number of datasets to randomly select from the DatasetCollection. Default is 100.
            properties (list, optional): List of properties to match. Default is None.
            values (list of lists, optional): List of lists of values to match for each property. Default is None.
            randomize (int, optional): The number of datasets to randomly select from the DatasetCollection. Default is False.
            ax (axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Default is None.
            kwargs: Additional keyword arguments to pass to the plot function.
        Returns:
            ax (axes.Axes): The axes object of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        combined_dataset = self.get_combined_dataset()
        sizes = [dataset.n for dataset in self.datasets.values()]
        for _ in range(n_datasets):
            size = np.random.choice(sizes)
            sampled_dataset = combined_dataset.sample(size)
            sampled_dataset.plotSurvival(ax=ax[0], **kwargs)
        ax[0].set_xlabel('Age')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].set_title(f'Sampled datasets n datasets={n_datasets}')
        ax[0].get_legend().remove()
        keys = self.getKeys(properties=properties, values=values, randomize=randomize)
        for key in keys:
            kwargs['label'] = key
            self.datasets[key].plotSurvival(ax=ax[1], **kwargs)
        ax[1].set_xlabel('Age')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_title('Data datasets')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

        return ax


def dsFromFile(path, external_hazard = np.inf, properties = None,sheet = None, death_times_column = None, events_column = None,bandwidth = 3, event_is_censored=False,excel_has_header=None,remove_nan_rows = False):
    """
    This function loads the dataset from a file.
    Parameters:
        path (str): Path to the file.
        external_hazard (float, optional): External hazard rate. Default is np.inf.
        properties (list, optional): List of columns to be loaded from the file as properties.
        event_is_censored (bool, optional): If True, the events column is a censor column (1=censored, 0=event).
                                            Events will be flipped when loading. Default is False.

    """
    if sheet is not None and isinstance(path, pd.ExcelFile):
        format = 'xlsx'
    else:
        format = path.split('.')[-1]
    if format == 'csv':
        df = pd.read_csv(path)
    elif format == 'xlsx':
        if sheet is None and len(pd.ExcelFile(path).sheet_names) > 1:
            raise ValueError('Multiple sheets found in the excel file. Please specify the sheet name.')
        df = pd.read_excel(path, sheet_name=sheet, header=excel_has_header)   

    if remove_nan_rows:
        df = df.dropna() 

    if death_times_column is not None:
        death_times = df[death_times_column].values
    elif 'death times' in df.columns:
        death_times = df['death times'].values
    elif 'death_times' in df.columns:
        death_times = df['death_times'].values
    else:
        raise ValueError('Death times column not found in the file.')
    
    if events_column is not None:
        events = df[events_column].values
    elif 'events' in df.columns:
        events = df['events'].values
    elif 'event' in df.columns:
        events = df['event'].values
    elif 'censor' in df.columns:
        events = df['censor'].values
        # If event_is_censored is False, flip 1 and 0 in censor (old behavior)
        if not event_is_censored:
            events = 1 - events
    else:
        raise ValueError('Events or Censor column not found in the file.')
    if properties is not None:
        properties = {property: df[property].values for property in properties if property in df.columns}

    return Dataset(death_times, events, external_hazard = external_hazard, properties = properties, bandwidth=bandwidth, event_is_censored=event_is_censored)
    

def trim_to_range(t,vals,time_range,renormalize_survival = False):
    #trims the values to the time range
    i_start = np.abs(t - time_range[0]).argmin()
    i_end = np.abs(t - time_range[1]).argmin()
    if renormalize_survival:
        vals = vals/vals[i_start]
    return t[i_start:i_end], vals[i_start:i_end]



def dataCollectionFromExcel(
    excel_file_name,
    death_times_column=None,
    events_column=None,
    external_hazard=None,
    properties=None,
    additional_properties=None,
    remove_nan_rows=False,
    bandwidth=3,
    sheet=None,
    excel_has_header=0,
    event_is_censored=False,
    warnings=True,
):
    """
    Loads a dataset collection from an excel file. 
    By default, loads all sheets in the excel file as separate datasets.
    Sheet name is attached as a 'sheet' property for each dataset.
    All arguments accepted by dsFromFile can be passed, except path (replaced by excel_file_name here).

    Parameters:
        excel_file_name (str): Path to the excel file.
        death_times_column (str, optional): Column name for death times.
        events_column (str, optional): Column name for events.
        external_hazard (array-like, optional): External hazard data.
        properties (list, optional): Properties to collect from each dataset.
        additional_properties (list, optional): Additional properties.
        remove_nan_rows (bool, optional): Whether to remove rows with NaN values.
        bandwidth (int, optional): Bandwidth for smoothing hazard.
        sheet (str, list, or None): Optional. Sheet or sheets to load. If None, loads all sheets.
        excel_has_header (int, optional): Which row is the header (default 0).
        event_is_censored (bool): If True, events column is a censor column (1=censored,0=event).
        warnings (bool, optional): Attach warnings to DatasetCollection. Default True.
    """
    xls = pd.ExcelFile(excel_file_name)
    datasets = []

    # Decide which sheets to process
    if sheet is None:
        sheets_to_iterate = xls.sheet_names
    elif isinstance(sheet, (list, tuple)):
        sheets_to_iterate = list(sheet)
    else:
        sheets_to_iterate = [sheet]

    # The logic here ensures behaviour is unchanged for default (all sheets) case
    # Gather all properties like before
    if properties is None and additional_properties is None:
        all_properties = None
    else:
        all_properties = []
        props = [properties, additional_properties]
        for prop in props:
            if prop is not None:
                all_properties += prop

    # For DatasetCollection, ensure 'sheet' is included in the top-level properties list
    if properties is None:
        properties_final = []
    else:
        properties_final = list(properties)
    if 'sheet' not in properties_final:
        properties_final.append('sheet')

    for sh in sheets_to_iterate:
        dataset = dsFromFile(
            xls,
            sheet=sh,
            death_times_column=death_times_column,
            events_column=events_column,
            external_hazard=external_hazard,
            properties=all_properties,
            remove_nan_rows=remove_nan_rows,
            bandwidth=bandwidth,
            excel_has_header=excel_has_header,
            event_is_censored=event_is_censored,
        )
        dataset.addProperty('sheet', sh)
        datasets.append(dataset)

    return DatasetCollection(
        datasets=datasets,
        properties=properties_final,
        additional_properties=additional_properties,
        warnings=warnings
    )



def plotAll(dataSets,labels=None,time_range=None,ax=None):
    """
    plots survival, hazard and death times distribution for all the datasets in dataSets
    """
    if ax is None:
        fig, ax = plt.subplots(1,3,figsize=(15,5))
    for i,dataset in enumerate(dataSets):
        if labels is None:
            label = f'Dataset {i+1}'
        else:
            label = labels[i]
        dataset.plotSurvival(ax=ax[0],label=label,time_range=time_range)
        dataset.plotHazard(ax=ax[1],label=label)
        dataset.plotDeathTimesDistribution(ax=ax[2],label=label,time_range=time_range)
    ax[0].set_title('Survival')
    ax[1].set_title('Hazard')
    ax[2].set_title('Death times distribution')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
   
    return ax