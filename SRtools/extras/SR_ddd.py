"""
SR_DDD: SR model with Death, Disease, and Decline tracking.

This module extends SR_Hetro to track:
- Disease onset times (when damage crosses xd threshold)
- Damage trajectories at configurable intervals
- Provides plotting methods for disease and damage statistics
"""

import numpy as np
from numba import jit
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import os

from .. import SR_hetro as srh
from .. import deathTimesDataSet as dtds

jit_nopython = True


class SR_DDD(srh.SR_Hetro):
    """
    SR model with Death, Disease, and Decline tracking.
    
    Extends SR_Hetro to optionally track:
    - Disease onset (when damage x crosses threshold xd)
    - Damage trajectories at specified time intervals
    
    Parameters:
        xd (float, optional): Disease threshold. When damage x >= xd, individual is "sick".
            If None, no disease tracking is performed.
        x_res (float, optional): Time resolution for saving damage values.
            If 2, save every 2 time units. If None, don't save damage trajectories.
        x_avg_window (float, optional): Window size for averaging damage before saving.
            If x_res=1 and x_avg_window=0.5, average over 0.5 time units, save every 1 time unit.
            If None, no averaging is applied.
    """
    
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                 eta_var=0, beta_var=0, kappa_var=0, epsilon_var=0, xc_var=0,
                 t_start=0, tscale='years', external_hazard=np.inf,
                 time_step_multiplier=1, parallel=False, bandwidth=3,
                 method='brownian_bridge',
                 xd=None, x_res=None, x_avg_window=None, function_ks=None):
        """
        Initialize SR_DDD simulation.
        
        Parameters:
            eta, beta, kappa, epsilon, xc: SR model parameters
            npeople: Number of individuals to simulate
            nsteps: Number of time steps
            t_end: End time of simulation
            eta_var, beta_var, kappa_var, epsilon_var, xc_var: Variance parameters for heterogeneity
            t_start: Start time (default 0)
            tscale: Time scale label (default 'years')
            external_hazard: External hazard rate (default np.inf = no external hazard)
            time_step_multiplier: Multiplier for time step resolution
            parallel: Whether to use parallel computation
            bandwidth: Bandwidth for hazard smoothing
            method: Simulation method ('brownian_bridge' or 'euler')
            xd: Disease threshold (optional)
            x_res: Time resolution for damage recording (optional)
            x_avg_window: Averaging window for damage recording (optional)
            function_ks: List of k values for calculating F = 1 - x/(x + k*kappa) (optional)
        """
        # Store disease and recording params before calling super().__init__
        self.xd = xd
        self.x_res = x_res
        self.x_avg_window = x_avg_window if x_avg_window is not None else 0
        
        # Initialize storage for disease and damage data
        self.disease_onset_times = None
        self.disease_events = None
        self.damage_trajectories = None
        self.damage_times = None
        
        # Individual threshold values (for heterogeneous populations)
        self.individual_xc = None  # Array of each person's xc threshold
        self.individual_xd = None  # Array of each person's xd threshold (if xd is set)
        
        # Disease-related Kaplan-Meier and Nelson-Aalen fitters
        self.disease_kmf = None
        self.disease_naf = None
        self.disease_survival = None
        self.disease_hazard = None
        
        # Functional capacity tracking: F = 1 - x/(x + k*kappa)
        self.function_ks = function_ks if function_ks is not None else []
        self.function_trajectories = {}  # Dict: k -> 2D array [npeople, n_time_points]
        self.function_threshold_times = {}  # Dict: k -> {threshold: times_array}
        self.function_threshold_events = {}  # Dict: k -> {threshold: events_array}
        self.function_threshold_kmf = {}  # Dict: k -> {threshold: KaplanMeierFitter}
        self.function_threshold_naf = {}  # Dict: k -> {threshold: NelsonAalenFitter}
        
        super().__init__(
            eta=eta, beta=beta, kappa=kappa, epsilon=epsilon, xc=xc,
            npeople=npeople, nsteps=nsteps, t_end=t_end,
            eta_var=eta_var, beta_var=beta_var, kappa_var=kappa_var,
            epsilon_var=epsilon_var, xc_var=xc_var,
            t_start=t_start, tscale=tscale, external_hazard=external_hazard,
            time_step_multiplier=time_step_multiplier, parallel=parallel,
            bandwidth=bandwidth, method=method
        )

    def calc_death_times(self):
        """
        Calculate death times, disease onset times, and optionally damage trajectories.
        
        Returns:
            tuple: (death_times, events) arrays
            
        Also populates:
            self.disease_onset_times: Times when damage first crossed xd
            self.disease_events: 1 if disease occurred, 0 if censored
            self.damage_trajectories: 2D array of damage values if x_res is set
            self.damage_times: Time points for damage recordings
        """
        s = len(self.t)
        dt = self.t[1] - self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        # Calculate number of time points to save if x_res is specified
        if self.x_res is not None and self.x_res > 0:
            n_save_points = int(np.ceil(self.t_end / self.x_res)) + 1
            self.damage_times = np.linspace(0, self.t_end, n_save_points)
        else:
            n_save_points = 0
            self.damage_times = None
        
        # Use xd = xc + 1 (unreachable) if no disease tracking
        xd_val = self.xd if self.xd is not None else self.xc + 1
        
        # Select appropriate simulation function
        if self.method == 'brownian_bridge':
            if self.parallel:
                result = death_times_ddd_brownian_bridge_parallel(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, xd_val, sdt, self.npeople,
                    self.external_hazard, self.time_step_multiplier,
                    self.x_res if self.x_res is not None else 0,
                    self.x_avg_window, n_save_points
                )
            else:
                result = death_times_ddd_brownian_bridge(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, xd_val, sdt, self.npeople,
                    self.external_hazard, self.time_step_multiplier,
                    self.x_res if self.x_res is not None else 0,
                    self.x_avg_window, n_save_points
                )
        else:  # euler method
            if self.parallel:
                result = death_times_ddd_euler_parallel(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, xd_val, sdt, self.npeople,
                    self.external_hazard, self.time_step_multiplier,
                    self.x_res if self.x_res is not None else 0,
                    self.x_avg_window, n_save_points
                )
            else:
                result = death_times_ddd_euler(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, xd_val, sdt, self.npeople,
                    self.external_hazard, self.time_step_multiplier,
                    self.x_res if self.x_res is not None else 0,
                    self.x_avg_window, n_save_points
                )
        
        (death_times, events, disease_onset_times, disease_events, 
         damage_trajectories, individual_xc, individual_xd) = result
        
        self.disease_onset_times = np.array(disease_onset_times)
        self.disease_events = np.array(disease_events)
        self.individual_xc = np.array(individual_xc)
        self.individual_xd = np.array(individual_xd) if self.xd is not None else None
        
        if n_save_points > 0:
            self.damage_trajectories = np.array(damage_trajectories)
        else:
            self.damage_trajectories = None
        
        # Calculate functional capacity trajectories if function_ks is set
        if self.damage_trajectories is not None and len(self.function_ks) > 0:
            self._calc_function_trajectories()
        
        # Calculate disease survival and hazard if disease tracking is enabled
        if self.xd is not None:
            self._calc_disease_survival()
        
        return np.array(death_times), np.array(events)

    def _calc_disease_survival(self):
        """
        Calculate disease survival and hazard using Kaplan-Meier and Nelson-Aalen estimators.
        """
        if self.disease_onset_times is None or len(self.disease_onset_times) == 0:
            return
        
        T = self.disease_onset_times
        E = self.disease_events
        
        # Pad arrays if needed
        if len(T) < self.npeople:
            Tn = np.zeros(self.npeople)
            Tn[:len(T)] = T
            Tn[len(T):] = self.t_end
            En = np.zeros(self.npeople)
            En[:len(T)] = E
            T = Tn
            E = En
        
        # Kaplan-Meier for survival
        self.disease_kmf = KaplanMeierFitter().fit(T, E)
        self.disease_survival = (
            self.disease_kmf.timeline,
            np.array(self.disease_kmf.survival_function_.values)[:, 0]
        )
        self.disease_kmf_confidence_interval = [
            np.array(self.disease_kmf.confidence_interval_['KM_estimate_lower_0.95'].values),
            np.array(self.disease_kmf.confidence_interval_['KM_estimate_upper_0.95'].values)
        ]
        self.median_disease_free_time = self.disease_kmf.median_survival_time_
        
        # Nelson-Aalen for hazard
        self.disease_naf = NelsonAalenFitter().fit(T, event_observed=E)
        self.disease_hazard = (
            self.disease_naf.timeline,
            np.array(self.disease_naf.smoothed_hazard_(bandwidth=self.bandwidth).values)[:, 0]
        )

    def _calc_function_trajectories(self):
        """
        Calculate functional capacity trajectories F = 1 - x/(x + k*kappa) for each k in function_ks.
        Must be called after damage_trajectories is populated.
        """
        if self.damage_trajectories is None:
            return
        
        for k in self.function_ks:
            # F = 1 - x/(x + k*kappa)
            # When x = 0, F = 1 (full function)
            # When x -> infinity, F -> 0 (no function)
            denominator = self.damage_trajectories + k * self.kappa
            # Avoid division by zero (shouldn't happen if kappa > 0 and k > 0)
            denominator = np.maximum(denominator, 1e-10)
            F = 1 - self.damage_trajectories / denominator
            self.function_trajectories[k] = F
            
            # Initialize threshold storage for this k
            self.function_threshold_times[k] = {}
            self.function_threshold_events[k] = {}
            self.function_threshold_kmf[k] = {}
            self.function_threshold_naf[k] = {}

    # ==================== Disease Plotting Methods ====================
    
    def getDiseaseSurvival(self, time_range=None):
        """
        Get disease-free survival curve.
        
        Parameters:
            time_range (tuple, optional): (start, end) time range to return
            
        Returns:
            tuple: (times, survival_probabilities)
        """
        if self.disease_survival is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        t, s = self.disease_survival
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            return t[mask], s[mask]
        return t, s

    def getDiseaseSurvivalCI(self, time_range=None):
        """
        Get confidence interval for disease-free survival.
        
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        if self.disease_kmf is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        bottom, top = self.disease_kmf_confidence_interval
        if time_range is not None:
            t, s = self.getDiseaseSurvival()
            mask = (t >= time_range[0]) & (t <= time_range[1])
            bottom = bottom[mask]
            top = top[mask]
        return bottom, top

    def plotDiseaseSurvival(self, ax=None, time_range=None, **kwargs):
        """
        Plot disease-free survival curve (Kaplan-Meier).
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range for the plot
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.disease_kmf is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if time_range is not None:
            t, s = self.getDiseaseSurvival(time_range=time_range)
            ax.plot(t, s, **kwargs)
        else:
            self.disease_kmf.plot_survival_function(ax=ax, **kwargs)
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Disease-free Probability')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotScaledDiseaseSurvival(self, ax=None, time_range=None, CI=True, **kwargs):
        """
        Plot scaled disease-free survival curve.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range for the plot
            CI (bool): Whether to show confidence interval
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.disease_kmf is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        t, s = self.getDiseaseSurvival(time_range=time_range)
        if time_range is None:
            median_time = self.disease_kmf.median_survival_time_
        else:
            median_time = t[np.argmin(np.abs(s - 0.5))]
        
        ax.plot(t / median_time, s, **kwargs)
        
        if CI:
            bottom, top = self.getDiseaseSurvivalCI(time_range)
            if 'color' in kwargs:
                ax.fill_between(t / median_time, bottom, top, alpha=0.3, color=kwargs['color'])
            else:
                ax.fill_between(t / median_time, bottom, top, alpha=0.3)
        
        ax.set_xlabel('Scaled Age (t / median disease-free time)')
        ax.set_ylabel('Disease-free Probability')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotDiseaseHazard(self, ax=None, time_range=None, **kwargs):
        """
        Plot disease hazard function.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range for the plot
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.disease_naf is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if time_range is not None:
            t, h = self.disease_hazard
            mask = (t >= time_range[0]) & (t <= time_range[1])
            ax.plot(t[mask], h[mask], **kwargs)
        else:
            self.disease_naf.plot_hazard(ax=ax, bandwidth=self.bandwidth, **kwargs)
        
        ax.set_yscale('log')
        ax.set_xlabel('Age')
        ax.set_ylabel('Disease Hazard')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def get_disease_times_distribution(self, bins=None, dt=1, time_range=None):
        """
        Get the probability distribution of disease onset times.
        
        Parameters:
            bins (array-like, optional): Bins for histogram
            dt (float): Time step for default bins
            time_range (tuple, optional): Time range to consider
            
        Returns:
            tuple: (probabilities, bin_edges)
        """
        if self.disease_onset_times is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        if bins is None:
            if time_range is not None:
                bins = np.linspace(time_range[0], time_range[1], int((time_range[1] - time_range[0]) / dt + 1))
            else:
                bins = np.linspace(0, self.t_end, int(self.t_end / dt + 1))
        
        disease_times = self.disease_onset_times.copy()
        events = self.disease_events
        disease_times = disease_times[events == 1]
        n = self.npeople
        
        if time_range is not None:
            n_disease = len(disease_times)
            disease_times = disease_times[(disease_times >= time_range[0]) & (disease_times < time_range[1])]
            new_n = len(disease_times)
            d = n_disease - new_n
            n = n - d
        
        prob_disease, bin_edges = np.histogram(disease_times, bins=bins)
        
        if time_range is None:
            if len(disease_times) < n:
                disease_times = np.append(disease_times, np.inf * np.ones(n - len(disease_times)))
            prob_disease = np.append(prob_disease, n - np.sum(prob_disease))
            bin_edges = np.append(bin_edges, np.inf)
        
        prob_disease = prob_disease / n
        return prob_disease, bin_edges

    def plotDiseaseTimesDistribution(self, ax=None, bins=None, use_kde=False, dt=1, time_range=None, **kwargs):
        """
        Plot disease onset times distribution.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            bins (array-like, optional): Bins for histogram
            use_kde (bool): Whether to use kernel density estimation
            dt (float): Time step for default bins
            time_range (tuple, optional): Time range to consider
            **kwargs: Additional arguments passed to plot
            
        Returns:
            tuple: (ax, bins)
        """
        if self.disease_onset_times is None:
            raise ValueError("Disease tracking not enabled (xd is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        disease_times = self.disease_onset_times.copy()
        events = self.disease_events
        disease_times = disease_times[events == 1]
        portion_sick = sum(events) / len(events)
        
        if time_range is not None:
            disease_times = disease_times[(disease_times >= time_range[0]) & (disease_times <= time_range[1])]
        
        if use_kde and len(disease_times) > 1:
            from scipy.stats import gaussian_kde
            x = np.linspace(0, self.t_end, 1000)
            kde = gaussian_kde(disease_times, bw_method='scott')
            ax.plot(x, portion_sick * kde(x), **kwargs)
        else:
            prob_disease, bin_edges = self.get_disease_times_distribution(bins, dt=dt, time_range=time_range)
            ax.step(bin_edges[:-1], prob_disease, where='post', **kwargs)
        
        ax.set_xlabel('Disease onset time')
        ax.set_ylabel('Probability of disease')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax, bins

    # ==================== Damage Trajectory Plotting Methods ====================

    def plotDamageTrajectories(self, ax=None, n_trajectories=10, time_range=None,
                                alpha=0.5, show_thresholds=True, **kwargs):
        """
        Plot individual damage trajectories.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            n_trajectories (int): Number of trajectories to plot
            time_range (tuple, optional): Time range to show
            alpha (float): Transparency of trajectory lines
            show_thresholds (bool): Whether to show xc and xd threshold lines
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.damage_trajectories is None:
            raise ValueError("Damage recording not enabled (x_res is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        t = self.damage_times
        trajectories = self.damage_trajectories
        
        # Apply time range filter
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            trajectories = trajectories[:, mask]
        
        # Select random trajectories
        n_available = trajectories.shape[0]
        n_to_plot = min(n_trajectories, n_available)
        indices = np.random.choice(n_available, n_to_plot, replace=False)
        
        for idx in indices:
            ax.plot(t, trajectories[idx], alpha=alpha, **kwargs)
        
        # Show thresholds with range if heterogeneous
        if show_thresholds:
            if self.individual_xc is not None and len(self.individual_xc) > 1:
                xc_mean = np.mean(self.individual_xc)
                xc_std = np.std(self.individual_xc)
                ax.axhline(y=xc_mean, color='red', linestyle='--', 
                          label=f'xc = {xc_mean:.2f} ± {xc_std:.2f}')
                ax.axhspan(xc_mean - xc_std, xc_mean + xc_std, alpha=0.15, color='red')
            else:
                ax.axhline(y=self.xc, color='red', linestyle='--', label=f'xc = {self.xc:.2f}')
            
            if self.xd is not None:
                if self.individual_xd is not None and len(self.individual_xd) > 1:
                    xd_mean = np.mean(self.individual_xd)
                    xd_std = np.std(self.individual_xd)
                    # Only show range if there's actual variance (xd doesn't vary by default)
                    if xd_std > 0:
                        ax.axhline(y=xd_mean, color='orange', linestyle='--', 
                                  label=f'xd = {xd_mean:.2f} ± {xd_std:.2f}')
                        ax.axhspan(xd_mean - xd_std, xd_mean + xd_std, alpha=0.15, color='orange')
                    else:
                        ax.axhline(y=self.xd, color='orange', linestyle='--', label=f'xd = {self.xd:.2f}')
                else:
                    ax.axhline(y=self.xd, color='orange', linestyle='--', label=f'xd = {self.xd:.2f}')
            ax.legend()
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Damage (x)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotMeanDamage(self, ax=None, time_range=None, CI=True, CI_method='std',
                       only_alive=False, **kwargs):
        """
        Plot mean damage over time with optional confidence interval.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range to show
            CI (bool): Whether to show confidence interval/std
            CI_method (str): 'std' for standard deviation, 'percentile' for 25-75 percentile
            only_alive (bool): If True, only include individuals still alive at each time point
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.damage_trajectories is None:
            raise ValueError("Damage recording not enabled (x_res is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        t = self.damage_times
        trajectories = self.damage_trajectories.copy()
        
        # Apply time range filter
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            trajectories = trajectories[:, mask]
        
        if only_alive:
            # Mask out values after death (where x >= xc)
            alive_mask = trajectories < self.xc
            trajectories = np.where(alive_mask, trajectories, np.nan)
        
        mean_damage = np.nanmean(trajectories, axis=0)
        ax.plot(t, mean_damage, **kwargs)
        
        if CI:
            if CI_method == 'std':
                std_damage = np.nanstd(trajectories, axis=0)
                lower = mean_damage - std_damage
                upper = mean_damage + std_damage
            else:  # percentile
                lower = np.nanpercentile(trajectories, 25, axis=0)
                upper = np.nanpercentile(trajectories, 75, axis=0)
            
            if 'color' in kwargs:
                ax.fill_between(t, lower, upper, alpha=0.3, color=kwargs['color'])
            else:
                ax.fill_between(t, lower, upper, alpha=0.3)
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Mean Damage')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotDamageDistribution(self, ax=None, time_point=None, bins=30, 
                                only_alive=True, **kwargs):
        """
        Plot distribution of damage at a specific time point.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_point (float): Time at which to show damage distribution.
                If None, uses median of time range.
            bins (int): Number of bins for histogram
            only_alive (bool): If True, only include individuals still alive
            **kwargs: Additional arguments passed to hist
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.damage_trajectories is None:
            raise ValueError("Damage recording not enabled (x_res is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if time_point is None:
            time_point = self.damage_times[len(self.damage_times) // 2]
        
        # Find closest time index
        time_idx = np.argmin(np.abs(self.damage_times - time_point))
        actual_time = self.damage_times[time_idx]
        
        damage_values = self.damage_trajectories[:, time_idx]
        
        if only_alive:
            damage_values = damage_values[damage_values < self.xc]
        
        ax.hist(damage_values, bins=bins, density=True, **kwargs)
        
        # Show thresholds with range if heterogeneous
        if self.individual_xc is not None and len(self.individual_xc) > 1:
            xc_mean = np.mean(self.individual_xc)
            xc_std = np.std(self.individual_xc)
            ax.axvline(x=xc_mean, color='red', linestyle='--', 
                      label=f'xc = {xc_mean:.2f} ± {xc_std:.2f}')
            ax.axvspan(xc_mean - xc_std, xc_mean + xc_std, alpha=0.15, color='red')
        else:
            ax.axvline(x=self.xc, color='red', linestyle='--', label=f'xc = {self.xc:.2f}')
        
        if self.xd is not None:
            ax.axvline(x=self.xd, color='orange', linestyle='--', label=f'xd = {self.xd:.2f}')
        
        ax.set_xlabel('Damage (x)')
        ax.set_ylabel('Density')
        ax.set_title(f'Damage Distribution at t = {actual_time:.1f}')
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotDamageQuantiles(self, ax=None, quantiles=None, time_range=None,
                            only_alive=False, show_thresholds=True, **kwargs):
        """
        Plot damage quantile trajectories over time.
        
        Parameters:
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            quantiles (list): List of quantiles to plot (default: [10, 25, 50, 75, 90])
            time_range (tuple, optional): Time range to show
            only_alive (bool): If True, only include individuals still alive
            show_thresholds (bool): Whether to show xc and xd threshold lines
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        if self.damage_trajectories is None:
            raise ValueError("Damage recording not enabled (x_res is None)")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if quantiles is None:
            quantiles = [10, 25, 50, 75, 90]
        
        t = self.damage_times
        trajectories = self.damage_trajectories.copy()
        
        # Apply time range filter
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            trajectories = trajectories[:, mask]
        
        if only_alive:
            alive_mask = trajectories < self.xc
            trajectories = np.where(alive_mask, trajectories, np.nan)
        
        for q in quantiles:
            q_values = np.nanpercentile(trajectories, q, axis=0)
            ax.plot(t, q_values, label=f'{q}th percentile', **kwargs)
        
        if show_thresholds:
            if self.individual_xc is not None and len(self.individual_xc) > 1:
                xc_mean = np.mean(self.individual_xc)
                xc_std = np.std(self.individual_xc)
                ax.axhline(y=xc_mean, color='red', linestyle='--', 
                          label=f'xc = {xc_mean:.2f} ± {xc_std:.2f}')
                ax.axhspan(xc_mean - xc_std, xc_mean + xc_std, alpha=0.15, color='red')
            else:
                ax.axhline(y=self.xc, color='red', linestyle='--', label=f'xc = {self.xc:.2f}')
            
            if self.xd is not None:
                ax.axhline(y=self.xd, color='orange', linestyle='--', label=f'xd = {self.xd:.2f}')
        
        ax.legend()
        ax.set_xlabel('Age')
        ax.set_ylabel('Damage (x)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def getDamageStats(self, stats=None, time_range=None, only_alive=True):
        """
        Get damage statistics over time.

        Parameters:
            stats (list of str): Statistics to compute.
                Default: ['mean', 'std', 'cv'].
                Options: 'mean', 'std', 'cv', 'median', 'min', 'max', 'skewness'.
            time_range (tuple, optional): (start, end) time range
            only_alive (bool): If True, only include individuals still alive at each time point

        Returns:
            dict: Dictionary with 'times' key and one key per requested stat,
                  each mapping to a 1D array over the age axis (as given by x_res).
        """
        if self.damage_trajectories is None:
            raise ValueError("Damage recording not enabled (x_res is None)")

        if stats is None:
            stats = ['mean', 'std', 'cv']

        t = self.damage_times.copy()
        trajectories = self.damage_trajectories.copy()

        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            trajectories = trajectories[:, mask]

        if only_alive:
            alive_mask = trajectories < self.xc
            trajectories = np.where(alive_mask, trajectories, np.nan)

        result = {'times': t}

        mean = None
        std = None
        for stat in stats:
            if stat == 'mean':
                mean = np.nanmean(trajectories, axis=0)
                result['mean'] = mean
            elif stat == 'std':
                std = np.nanstd(trajectories, axis=0)
                result['std'] = std
            elif stat == 'cv':
                if mean is None:
                    mean = np.nanmean(trajectories, axis=0)
                if std is None:
                    std = np.nanstd(trajectories, axis=0)
                result['cv'] = np.where(mean > 0, std / mean, np.nan)
            elif stat == 'median':
                result['median'] = np.nanmedian(trajectories, axis=0)
            elif stat == 'min':
                result['min'] = np.nanmin(trajectories, axis=0)
            elif stat == 'max':
                result['max'] = np.nanmax(trajectories, axis=0)
            elif stat == 'skewness':
                if mean is None:
                    mean = np.nanmean(trajectories, axis=0)
                if std is None:
                    std = np.nanstd(trajectories, axis=0)
                n = np.sum(~np.isnan(trajectories), axis=0)
                m3 = np.nanmean((trajectories - mean[np.newaxis, :]) ** 3, axis=0)
                result['skewness'] = np.where(std > 0, m3 / std ** 3, np.nan)
            else:
                raise ValueError(
                    f"Unknown stat: '{stat}'. "
                    "Options: 'mean', 'std', 'cv', 'median', 'min', 'max', 'skewness'"
                )

        return result

    # ==================== Functional Capacity (F) Methods ====================

    def _validate_function_k(self, k):
        """Validate that k is in function_ks and F trajectories exist."""
        if k not in self.function_ks:
            raise ValueError(f"k={k} not in function_ks={self.function_ks}")
        if k not in self.function_trajectories:
            raise ValueError(f"Function trajectories not calculated for k={k}. "
                           "Ensure x_res is set and simulation has run.")

    def getFunctionStats(self, k, time_range=None):
        """
        Get statistics of functional capacity F for a given k value.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            time_range (tuple, optional): (start, end) time range
            
        Returns:
            dict: Dictionary with keys 'times', 'mean', 'std', 'median', 
                  'percentile_25', 'percentile_75', 'percentile_10', 'percentile_90'
        """
        self._validate_function_k(k)
        
        t = self.damage_times.copy()
        F = self.function_trajectories[k].copy()
        
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            F = F[:, mask]
        
        return {
            'times': t,
            'mean': np.mean(F, axis=0),
            'std': np.std(F, axis=0),
            'median': np.median(F, axis=0),
            'percentile_10': np.percentile(F, 10, axis=0),
            'percentile_25': np.percentile(F, 25, axis=0),
            'percentile_75': np.percentile(F, 75, axis=0),
            'percentile_90': np.percentile(F, 90, axis=0),
        }

    def getFunctionAtTime(self, k, time_point):
        """
        Get F values for all individuals at a specific time point.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            time_point (float): Time at which to get F values
            
        Returns:
            np.ndarray: Array of F values for all individuals
        """
        self._validate_function_k(k)
        
        time_idx = np.argmin(np.abs(self.damage_times - time_point))
        return self.function_trajectories[k][:, time_idx]

    def plotFunctionTrajectories(self, k, ax=None, n_trajectories=10, time_range=None,
                                  alpha=0.5, show_threshold=None, **kwargs):
        """
        Plot individual functional capacity trajectories.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            n_trajectories (int): Number of trajectories to plot
            time_range (tuple, optional): Time range to show
            alpha (float): Transparency of trajectory lines
            show_threshold (float, optional): Show horizontal line at this F threshold
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        t = self.damage_times
        F = self.function_trajectories[k]
        
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            F = F[:, mask]
        
        n_available = F.shape[0]
        n_to_plot = min(n_trajectories, n_available)
        indices = np.random.choice(n_available, n_to_plot, replace=False)
        
        for idx in indices:
            ax.plot(t, F[idx], alpha=alpha, **kwargs)
        
        if show_threshold is not None:
            ax.axhline(y=show_threshold, color='red', linestyle='--', 
                      label=f'F = {show_threshold}')
            ax.legend()
        
        ax.set_xlabel('Age')
        ax.set_ylabel(f'Functional Capacity F (k={k})')
        ax.set_ylim(0, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotMeanFunction(self, k, ax=None, time_range=None, CI=True, CI_method='std', **kwargs):
        """
        Plot mean functional capacity over time with confidence interval.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range to show
            CI (bool): Whether to show confidence interval/std
            CI_method (str): 'std' for standard deviation, 'percentile' for 25-75 percentile
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        stats = self.getFunctionStats(k, time_range=time_range)
        t = stats['times']
        mean_F = stats['mean']
        
        ax.plot(t, mean_F, **kwargs)
        
        if CI:
            if CI_method == 'std':
                lower = mean_F - stats['std']
                upper = mean_F + stats['std']
            else:  # percentile
                lower = stats['percentile_25']
                upper = stats['percentile_75']
            
            if 'color' in kwargs:
                ax.fill_between(t, lower, upper, alpha=0.3, color=kwargs['color'])
            else:
                ax.fill_between(t, lower, upper, alpha=0.3)
        
        ax.set_xlabel('Age')
        ax.set_ylabel(f'Mean Functional Capacity F (k={k})')
        ax.set_ylim(0, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotFunctionDistribution(self, k, ax=None, time_point=None, bins=30, **kwargs):
        """
        Plot distribution of functional capacity at a specific time point.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_point (float): Time at which to show F distribution
            bins (int): Number of bins for histogram
            **kwargs: Additional arguments passed to hist
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if time_point is None:
            time_point = self.damage_times[len(self.damage_times) // 2]
        
        F_values = self.getFunctionAtTime(k, time_point)
        actual_time = self.damage_times[np.argmin(np.abs(self.damage_times - time_point))]
        
        ax.hist(F_values, bins=bins, density=True, **kwargs)
        ax.set_xlabel(f'Functional Capacity F (k={k})')
        ax.set_ylabel('Density')
        ax.set_title(f'F Distribution at t = {actual_time:.1f}')
        ax.set_xlim(0, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotFunctionQuantiles(self, k, ax=None, quantiles=None, time_range=None, 
                               show_threshold=None, **kwargs):
        """
        Plot functional capacity quantile trajectories over time.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            quantiles (list): List of quantiles to plot (default: [10, 25, 50, 75, 90])
            time_range (tuple, optional): Time range to show
            show_threshold (float, optional): Show horizontal line at this F threshold
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if quantiles is None:
            quantiles = [10, 25, 50, 75, 90]
        
        t = self.damage_times
        F = self.function_trajectories[k].copy()
        
        if time_range is not None:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t = t[mask]
            F = F[:, mask]
        
        for q in quantiles:
            q_values = np.percentile(F, q, axis=0)
            ax.plot(t, q_values, label=f'{q}th percentile', **kwargs)
        
        if show_threshold is not None:
            ax.axhline(y=show_threshold, color='red', linestyle='--', 
                      label=f'F = {show_threshold}')
        
        ax.legend()
        ax.set_xlabel('Age')
        ax.set_ylabel(f'Functional Capacity F (k={k})')
        ax.set_ylim(0, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    # ==================== Function Threshold Crossing Methods ====================

    def calcFunctionThresholdCrossing(self, k, threshold):
        """
        Calculate when functional capacity F first drops below a threshold for each individual.
        
        Parameters:
            k (float): The k value for F = 1 - x/(x + k*kappa)
            threshold (float): The F threshold (e.g., 0.5 means 50% functional capacity)
            
        Stores results in:
            self.function_threshold_times[k][threshold]: Array of crossing times
            self.function_threshold_events[k][threshold]: Array of event indicators (1=crossed, 0=censored)
        """
        self._validate_function_k(k)
        
        F = self.function_trajectories[k]
        t = self.damage_times
        npeople = F.shape[0]
        
        crossing_times = np.zeros(npeople)
        crossing_events = np.zeros(npeople, dtype=int)
        
        for i in range(npeople):
            # Find first time F drops below threshold
            below_threshold = F[i] < threshold
            if np.any(below_threshold):
                first_idx = np.argmax(below_threshold)  # First True index
                crossing_times[i] = t[first_idx]
                crossing_events[i] = 1
            else:
                # Censored - didn't cross threshold
                crossing_times[i] = t[-1]
                crossing_events[i] = 0
        
        self.function_threshold_times[k][threshold] = crossing_times
        self.function_threshold_events[k][threshold] = crossing_events
        
        # Calculate KM and NAF for this threshold
        kmf = KaplanMeierFitter().fit(crossing_times, crossing_events)
        self.function_threshold_kmf[k][threshold] = kmf
        
        naf = NelsonAalenFitter().fit(crossing_times, event_observed=crossing_events)
        self.function_threshold_naf[k][threshold] = naf

    def getFunctionThresholdTimes(self, k, threshold):
        """
        Get array of times when F first dropped below threshold.
        
        Parameters:
            k (float): The k value
            threshold (float): The F threshold
            
        Returns:
            np.ndarray: Array of crossing times for each individual
        """
        self._validate_function_k(k)
        
        if threshold not in self.function_threshold_times.get(k, {}):
            raise ValueError(f"Threshold {threshold} not calculated for k={k}. "
                           f"Call calcFunctionThresholdCrossing(k={k}, threshold={threshold}) first.")
        
        return self.function_threshold_times[k][threshold]

    def getFunctionThresholdEvents(self, k, threshold):
        """
        Get array of event indicators for threshold crossing.
        
        Parameters:
            k (float): The k value
            threshold (float): The F threshold
            
        Returns:
            np.ndarray: Array of event indicators (1=crossed, 0=censored)
        """
        self._validate_function_k(k)
        
        if threshold not in self.function_threshold_events.get(k, {}):
            raise ValueError(f"Threshold {threshold} not calculated for k={k}. "
                           f"Call calcFunctionThresholdCrossing(k={k}, threshold={threshold}) first.")
        
        return self.function_threshold_events[k][threshold]

    def plotFunctionThresholdSurvival(self, k, threshold, ax=None, time_range=None, **kwargs):
        """
        Plot Kaplan-Meier survival curve for F staying above threshold.
        
        Parameters:
            k (float): The k value
            threshold (float): The F threshold
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range for the plot
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        # Calculate if not already done
        if threshold not in self.function_threshold_kmf.get(k, {}):
            self.calcFunctionThresholdCrossing(k, threshold)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        kmf = self.function_threshold_kmf[k][threshold]
        
        if time_range is not None:
            times = kmf.timeline
            survival = kmf.survival_function_.values[:, 0]
            mask = (times >= time_range[0]) & (times <= time_range[1])
            ax.plot(times[mask], survival[mask], **kwargs)
        else:
            kmf.plot_survival_function(ax=ax, **kwargs)
        
        ax.set_xlabel('Age')
        ax.set_ylabel(f'P(F > {threshold})')
        ax.set_title(f'Survival above F={threshold} (k={k})')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotFunctionThresholdHazard(self, k, threshold, ax=None, time_range=None, **kwargs):
        """
        Plot hazard of F dropping below threshold.
        
        Parameters:
            k (float): The k value
            threshold (float): The F threshold
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            time_range (tuple, optional): Time range for the plot
            **kwargs: Additional arguments passed to plot
            
        Returns:
            matplotlib.axes.Axes
        """
        self._validate_function_k(k)
        
        # Calculate if not already done
        if threshold not in self.function_threshold_naf.get(k, {}):
            self.calcFunctionThresholdCrossing(k, threshold)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        naf = self.function_threshold_naf[k][threshold]
        
        if time_range is not None:
            hazard = naf.smoothed_hazard_(bandwidth=self.bandwidth)
            times = hazard.index.values
            h_values = hazard.values[:, 0]
            mask = (times >= time_range[0]) & (times <= time_range[1])
            ax.plot(times[mask], h_values[mask], **kwargs)
        else:
            naf.plot_hazard(ax=ax, bandwidth=self.bandwidth, **kwargs)
        
        ax.set_yscale('log')
        ax.set_xlabel('Age')
        ax.set_ylabel(f'Hazard of F < {threshold}')
        ax.set_title(f'Hazard of F dropping below {threshold} (k={k})')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax

    def plotFunctionThresholdDistribution(self, k, threshold, ax=None, bins=None, 
                                           use_kde=False, dt=1, time_range=None, **kwargs):
        """
        Plot distribution of times when F first dropped below threshold.
        
        Parameters:
            k (float): The k value
            threshold (float): The F threshold
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            bins (array-like, optional): Bins for histogram
            use_kde (bool): Whether to use kernel density estimation
            dt (float): Time step for default bins
            time_range (tuple, optional): Time range to consider
            **kwargs: Additional arguments passed to plot/hist
            
        Returns:
            tuple: (ax, bins)
        """
        self._validate_function_k(k)
        
        # Calculate if not already done
        if threshold not in self.function_threshold_times.get(k, {}):
            self.calcFunctionThresholdCrossing(k, threshold)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        crossing_times = self.function_threshold_times[k][threshold].copy()
        crossing_events = self.function_threshold_events[k][threshold]
        
        # Only include actual crossings (not censored)
        crossing_times = crossing_times[crossing_events == 1]
        portion_crossed = np.sum(crossing_events) / len(crossing_events)
        
        if time_range is not None:
            crossing_times = crossing_times[(crossing_times >= time_range[0]) & 
                                           (crossing_times <= time_range[1])]
        
        if len(crossing_times) == 0:
            ax.text(0.5, 0.5, f'No crossings of F < {threshold}', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax, bins
        
        if use_kde and len(crossing_times) > 1:
            from scipy.stats import gaussian_kde
            x = np.linspace(0, self.t_end, 1000)
            kde = gaussian_kde(crossing_times, bw_method='scott')
            ax.plot(x, portion_crossed * kde(x), **kwargs)
        else:
            if bins is None:
                if time_range is not None:
                    bins = np.linspace(time_range[0], time_range[1], 
                                      int((time_range[1] - time_range[0]) / dt + 1))
                else:
                    bins = np.linspace(0, self.t_end, int(self.t_end / dt + 1))
            
            ax.hist(crossing_times, bins=bins, density=True, **kwargs)
        
        ax.set_xlabel(f'Time of F < {threshold}')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of F < {threshold} crossing times (k={k})')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax, bins

    def _get_total_time_above_threshold(self, k, threshold):
        """
        Total time (over lifespan) that F is above threshold for each individual.
        Uses self.death_times as each individual's lifespan; integration is capped at lifespan.
        """
        self._validate_function_k(k)
        t = self.damage_times
        F = self.function_trajectories[k]
        lifespans = self.death_times
        npeople = F.shape[0]
        n_t = len(t)
        result = np.zeros(npeople)
        for i in range(npeople):
            lifespan_i = lifespans[i]
            if lifespan_i <= 0:
                continue
            for j in range(n_t - 1):
                if t[j] >= lifespan_i:
                    break
                if F[i, j] >= threshold:
                    if t[j + 1] <= lifespan_i:
                        result[i] += t[j + 1] - t[j]
                    else:
                        result[i] += lifespan_i - t[j]
        return result

    def getFunctionTimeAboveThresholdStats(self, k, threshold):
        """
        Summary statistics of total time that F is above threshold over each individual's lifespan.
        Uses total time over lifespan (not first crossing). Values are in the same time units as
        the simulation (e.g. years).
        """
        self._validate_function_k(k)
        total_times = self._get_total_time_above_threshold(k, threshold)
        return {
            'mean': np.mean(total_times),
            'median': np.median(total_times),
            'std': np.std(total_times, ddof=1) if len(total_times) > 1 else 0.0,
            'n': len(total_times),
        }

    def getFunctionTimeAboveThresholdDistribution(self, k, threshold, as_percent=False):
        """
        Distribution of total time spent above threshold per individual.
        Returns both total time and percent of lifespan; caller can use either.
        Percent of lifespan is 0 where lifespan is 0.
        """
        self._validate_function_k(k)
        total_time = self._get_total_time_above_threshold(k, threshold)
        lifespans = self.death_times
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_of_lifespan = np.where(
                lifespans > 0,
                100.0 * total_time / lifespans,
                0.0
            )
        return {
            'total_time': total_time,
            'percent_of_lifespan': percent_of_lifespan,
        }

    def plotFunctionTimeAboveThresholdDistribution(self, k, threshold, ax=None, plot_percent=False,
                                                   bins=None,probability =False,density=False, **kwargs):
        """
        Plot histogram of total time above threshold (or percent of lifespan).
        """
        self._validate_function_k(k)
        dist = self.getFunctionTimeAboveThresholdDistribution(k, threshold, as_percent=True)
        if ax is None:
            fig, ax = plt.subplots()
        data = dist['percent_of_lifespan'] if plot_percent else dist['total_time']
        if bins is None:
            if plot_percent:
                bins = np.linspace(0, 100, 26)
            else:
                bins = np.linspace(0, self.t_end, min(51, max(2, int(self.t_end) + 1)))
        # Compute histogram, but don't plot yet
        if probability:
            density = True
        counts, bin_edges = np.histogram(data, bins=bins, density=density)
        if probability:
            # Calculate bin widths and then bin probabilities
            bin_widths = np.diff(bin_edges)
            probabilities = counts * bin_widths  # density * width = probability for each bin
            ax.bar(bin_edges[:-1], probabilities, width=bin_widths, align='edge', **kwargs)
            ax.set_ylabel('Probability')
        else:
            ax.hist(data, bins=bins, density=density, **kwargs)
            ax.set_ylabel('Count')
        ax.set_xlabel('Percent of lifespan above F >= threshold' if plot_percent else
                      'Total time above F >= threshold')
        ax.set_title(f'Time above F >= {threshold} (k={k})')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if probability:
            return ax, bin_edges, probabilities
        else:
            return ax, bins


# ==================== JIT-Compiled Simulation Functions ====================

@jit(nopython=jit_nopython)
def death_times_ddd_brownian_bridge(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                     epsilon0, epsilon_var, xc0, xc_var, xd, sdt, npeople,
                                     external_hazard=np.inf, time_step_multiplier=1,
                                     x_res=0, x_avg_window=0, n_save_points=0):
    """
    Euler method with Brownian bridge crossing detection, extended for disease tracking
    and optional damage trajectory recording.
    
    Parameters:
        xd: Disease threshold (when x >= xd, individual is sick)
        x_res: Time resolution for saving damage values (0 = don't save)
        x_avg_window: Window for averaging damage before saving
        n_save_points: Number of time points to save damage at
    """
    death_times = []
    events = []
    disease_onset_times = []
    disease_events = []
    individual_xc_values = []
    individual_xd_values = []
    
    # Pre-allocate damage trajectories if needed
    if n_save_points > 0:
        damage_trajectories = np.zeros((npeople, n_save_points))
    else:
        damage_trajectories = np.zeros((1, 1))  # Placeholder for numba typing
    
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    else:
        chance_to_die_externally = 0.0
    
    # Calculate save interval in simulation steps
    if x_res > 0:
        save_interval_steps = int(x_res / dt)
        if save_interval_steps < 1:
            save_interval_steps = 1
    else:
        save_interval_steps = 0
    # Calculate averaging window in sub-steps (Brownian steps)
    if x_avg_window > 0:
        avg_window_steps = int(x_avg_window / ndt)
        if avg_window_steps < 1:
            avg_window_steps = 1
    else:
        avg_window_steps = 0
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var) if eta_var > 0 else eta0
        beta = beta0 * np.random.normal(1.0, beta_var) if beta_var > 0 else beta0
        kappa = kappa0 * np.random.normal(1.0, kappa_var) if kappa_var > 0 else kappa0
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var) if epsilon_var > 0 else epsilon0
        xc = xc0 * np.random.normal(1.0, xc_var) if xc_var > 0 else xc0
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        crossed_death = False
        crossed_disease = False
        disease_time = 0.0
        
        # Store individual threshold values
        individual_xc_values.append(xc)
        individual_xd_values.append(xd)
        
        # For damage averaging (sliding window over last avg_window_steps sub-steps)
        if avg_window_steps > 0:
            window_size = avg_window_steps
            window = np.empty(window_size)
            for wi in range(window_size):
                window[wi] = 0.0
            window_sum = 0.0
            window_count = 0
            window_index = 0
        else:
            window_size = 0
        save_idx = 0
        last_save_step = -save_interval_steps  # To trigger save at step 0
        
        while j < s - 1 and not crossed_death:
            for sub_step in range(time_step_multiplier):
                # Standard Euler step
                current_time = t[j] + sub_step * ndt
                drift = eta * current_time - beta * x / (x + kappa)
                noise = sqrt_2epsilon * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Accumulate for averaging
                if window_size > 0:
                    if window_count < window_size:
                        window[window_index] = x_new
                        window_sum += x_new
                        window_count += 1
                    else:
                        window_sum -= window[window_index]
                        window[window_index] = x_new
                        window_sum += x_new
                    window_index += 1
                    if window_index == window_size:
                        window_index = 0
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = xc
                    crossed_death = True
                    if not crossed_disease:
                        crossed_disease = True
                        disease_time = current_time
                    break
                
                # Check disease threshold
                if not crossed_disease and x_new >= xd:
                    crossed_disease = True
                    disease_time = current_time
                
                # Direct death crossing check
                if x_new >= xc:
                    x = x_new
                    crossed_death = True
                    if not crossed_disease:
                        crossed_disease = True
                        disease_time = current_time
                    break
                
                # Brownian bridge crossing test for death
                if (x < xc) and (x_new < xc) and (x > 0):
                    dx1 = xc - x
                    dx2 = xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        var = 2.0 * epsilon * ndt
                        if var > 0.0 and 2.0 * dx1 * dx2 < 30.0 * var:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = xc
                                crossed_death = True
                                if not crossed_disease:
                                    crossed_disease = True
                                    disease_time = current_time
                                break

                # Brownian bridge crossing test for disease (if not yet sick)
                if not crossed_disease and (x < xd) and (x_new < xd) and (x > 0):
                    dx1 = xd - x
                    dx2 = xd - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        var = 2.0 * epsilon * ndt
                        if var > 0.0 and 2.0 * dx1 * dx2 < 30.0 * var:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                crossed_disease = True
                                disease_time = current_time
                
                x = x_new
            
            # Save damage at regular intervals
            if n_save_points > 0 and save_idx < n_save_points:
                if (save_interval_steps == 0 and j == 0) or (
                    save_interval_steps > 0
                    and (j - last_save_step >= save_interval_steps or j == 0)
                ):
                    if window_size > 0 and window_count > 0:
                        damage_trajectories[person, save_idx] = window_sum / window_count
                    else:
                        damage_trajectories[person, save_idx] = x
                    save_idx += 1
                    last_save_step = j
            
            j += 1
        
        # Fill remaining save points with final value (use xc if died)
        if n_save_points > 0:
            fill_value = xc if (crossed_death or x >= xc) else x
            while save_idx < n_save_points:
                damage_trajectories[person, save_idx] = fill_value
                save_idx += 1
        
        death_times.append(j * dt)
        if crossed_death or x >= xc:
            events.append(1)
        else:
            events.append(0)
        
        if crossed_disease:
            disease_onset_times.append(disease_time)
            disease_events.append(1)
        else:
            disease_onset_times.append(j * dt)  # Censored at end
            disease_events.append(0)
    
    return (death_times, events, disease_onset_times, disease_events, 
            damage_trajectories, individual_xc_values, individual_xd_values)


@jit(nopython=jit_nopython)
def death_times_ddd_euler(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                          epsilon0, epsilon_var, xc0, xc_var, xd, sdt, npeople,
                          external_hazard=np.inf, time_step_multiplier=1,
                          x_res=0, x_avg_window=0, n_save_points=0):
    """
    Standard Euler method extended for disease tracking and damage recording.
    """
    death_times = []
    events = []
    disease_onset_times = []
    disease_events = []
    individual_xc_values = []
    individual_xd_values = []
    
    if n_save_points > 0:
        damage_trajectories = np.zeros((npeople, n_save_points))
    else:
        damage_trajectories = np.zeros((1, 1))

    ndt = dt / time_step_multiplier
    nsdt = np.sqrt(ndt)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    else:
        chance_to_die_externally = 0.0

    if x_res > 0:
        save_interval_steps = int(x_res / dt)
        if save_interval_steps < 1:
            save_interval_steps = 1
    else:
        save_interval_steps = 0
    if x_avg_window > 0:
        avg_window_steps = int(x_avg_window / ndt)
        if avg_window_steps < 1:
            avg_window_steps = 1
    else:
        avg_window_steps = 0

    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var) if eta_var > 0 else eta0
        beta = beta0 * np.random.normal(1.0, beta_var) if beta_var > 0 else beta0
        kappa = kappa0 * np.random.normal(1.0, kappa_var) if kappa_var > 0 else kappa0
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var) if epsilon_var > 0 else epsilon0
        xc = xc0 * np.random.normal(1.0, xc_var) if xc_var > 0 else xc0
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        crossed_death = False
        crossed_disease = False
        disease_time = 0.0

        # Store individual threshold values
        individual_xc_values.append(xc)
        individual_xd_values.append(xd)

        # If using averaging over window
        if avg_window_steps > 0:
            window_size = avg_window_steps
            window = np.empty(window_size)
            for wi in range(window_size):
                window[wi] = 0.0
            window_sum = 0.0
            window_count = 0
            window_index = 0
        else:
            window_size = 0
        save_idx = 0
        last_save_step = -save_interval_steps

        while j < s - 1 and not crossed_death:
            for sub_step in range(time_step_multiplier):
                current_time = t[j] + sub_step * ndt
                noise = sqrt_2epsilon * np.random.normal()
                x = x + ndt * (eta * current_time - beta * x / (x + kappa)) + noise * nsdt
                x = max(x, 0.0)

                if avg_window_steps > 0 and window_size > 0:
                    # Rolling window averaging logic as before
                    if window_count < window_size:
                        window[window_index] = x
                        window_sum += x
                        window_count += 1
                    else:
                        window_sum -= window[window_index]
                        window[window_index] = x
                        window_sum += x
                    window_index += 1
                    if window_index == window_size:
                        window_index = 0

                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = xc
                    crossed_death = True
                    if not crossed_disease:
                        crossed_disease = True
                        disease_time = current_time

                if not crossed_disease and x >= xd:
                    crossed_disease = True
                    disease_time = current_time

                if x >= xc:
                    crossed_death = True
                    if not crossed_disease:
                        crossed_disease = True
                        disease_time = current_time
                    break

            if n_save_points > 0 and save_idx < n_save_points:
                if (save_interval_steps == 0 and j == 0) or (
                    save_interval_steps > 0
                    and (j - last_save_step >= save_interval_steps or j == 0)
                ):
                    if avg_window_steps > 0:
                        # Average over window
                        if window_count > 0:
                            damage_trajectories[person, save_idx] = window_sum / window_count
                        else:
                            damage_trajectories[person, save_idx] = x
                    else:
                        # No averaging: just sample current x
                        damage_trajectories[person, save_idx] = x
                    save_idx += 1
                    last_save_step = j

            j += 1

        # Fill remaining save points with final value (use xc if died)
        if n_save_points > 0:
            fill_value = xc if (crossed_death or x >= xc) else x
            while save_idx < n_save_points:
                damage_trajectories[person, save_idx] = fill_value
                save_idx += 1

        death_times.append(j * dt)
        if crossed_death or x >= xc:
            events.append(1)
        else:
            events.append(0)

        if crossed_disease:
            disease_onset_times.append(disease_time)
            disease_events.append(1)
        else:
            disease_onset_times.append(j * dt)
            disease_events.append(0)

    return (death_times, events, disease_onset_times, disease_events,
            damage_trajectories, individual_xc_values, individual_xd_values)


# ==================== Parallel Versions ====================

def death_times_ddd_brownian_bridge_parallel(s, dt, t, eta0, eta_var, beta0, beta_var,
                                              kappa0, kappa_var, epsilon0, epsilon_var,
                                              xc0, xc_var, xd, sdt, npeople,
                                              external_hazard=np.inf, time_step_multiplier=1,
                                              x_res=0, x_avg_window=0, n_save_points=0,
                                              n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_ddd_brownian_bridge.
    """
    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, xd, sdt, external_hazard,
               time_step_multiplier, x_res, x_avg_window, n_save_points):
        return death_times_ddd_brownian_bridge(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, xd, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, x_res, x_avg_window, n_save_points
        )
    
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, xd, sdt, external_hazard,
            time_step_multiplier, x_res, x_avg_window, n_save_points
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )
    
    death_times = np.concatenate([np.array(res[0]) for res in results])
    events = np.concatenate([np.array(res[1]) for res in results])
    disease_onset_times = np.concatenate([np.array(res[2]) for res in results])
    disease_events = np.concatenate([np.array(res[3]) for res in results])
    
    if n_save_points > 0:
        damage_trajectories = np.concatenate([res[4] for res in results], axis=0)
    else:
        damage_trajectories = np.zeros((1, 1))
    
    individual_xc = np.concatenate([np.array(res[5]) for res in results])
    individual_xd = np.concatenate([np.array(res[6]) for res in results])
    
    return (death_times, events, disease_onset_times, disease_events, 
            damage_trajectories, individual_xc, individual_xd)


def death_times_ddd_euler_parallel(s, dt, t, eta0, eta_var, beta0, beta_var,
                                    kappa0, kappa_var, epsilon0, epsilon_var,
                                    xc0, xc_var, xd, sdt, npeople,
                                    external_hazard=np.inf, time_step_multiplier=1,
                                    x_res=0, x_avg_window=0, n_save_points=0,
                                    n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_ddd_euler.
    """
    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, xd, sdt, external_hazard,
               time_step_multiplier, x_res, x_avg_window, n_save_points):
        return death_times_ddd_euler(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, xd, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, x_res, x_avg_window, n_save_points
        )
    
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, xd, sdt, external_hazard,
            time_step_multiplier, x_res, x_avg_window, n_save_points
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )
    
    death_times = np.concatenate([np.array(res[0]) for res in results])
    events = np.concatenate([np.array(res[1]) for res in results])
    disease_onset_times = np.concatenate([np.array(res[2]) for res in results])
    disease_events = np.concatenate([np.array(res[3]) for res in results])
    
    if n_save_points > 0:
        damage_trajectories = np.concatenate([res[4] for res in results], axis=0)
    else:
        damage_trajectories = np.zeros((1, 1))
    
    individual_xc = np.concatenate([np.array(res[5]) for res in results])
    individual_xd = np.concatenate([np.array(res[6]) for res in results])
    
    return (death_times, events, disease_onset_times, disease_events, 
            damage_trajectories, individual_xc, individual_xd)


# ==================== Factory Function ====================

def getSrDDD(theta, n=1000, nsteps=6000, t_end=110, external_hazard=np.inf,
             time_step_multiplier=1, npeople=None, parallel=False,
             eta_var=0, beta_var=0, epsilon_var=0, xc_var=0.2, kappa_var=0,
             hetro=True, bandwidth=3, step_size=None, method='brownian_bridge',
             xd=None, x_res=None, x_avg_window=None, function_ks=None):
    """
    Factory function to create SR_DDD simulation.
    
    Parameters:
        theta: Array of [eta, beta, epsilon, xc] parameters
        n: Number of individuals (default 25000)
        nsteps: Number of time steps (default 6000)
        t_end: End time (default 110)
        external_hazard: External hazard rate (default np.inf)
        time_step_multiplier: Time step multiplier (default 1)
        npeople: Alias for n
        parallel: Whether to use parallel computation
        eta_var, beta_var, epsilon_var, xc_var, kappa_var: Variance parameters
        hetro: Whether to use heterogeneous parameters
        bandwidth: Bandwidth for hazard smoothing
        step_size: Optional step size (overrides nsteps/time_step_multiplier)
        method: Simulation method ('brownian_bridge' or 'euler')
        xd: Disease threshold (optional)
        x_res: Time resolution for damage recording (optional)
        x_avg_window: Averaging window for damage recording (optional)
        function_ks: List of k values for calculating F = 1 - x/(x + k*kappa) (optional)
        
    Returns:
        SR_DDD: Configured simulation object
    """
    if npeople is not None:
        n = npeople
    
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    
    if not hetro:
        eta_var = 0
        beta_var = 0
        epsilon_var = 0
        xc_var = 0
        kappa_var = 0
    
    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf
    
    # Handle step_size logic
    if step_size is not None:
        total_steps = int(np.ceil(t_end / step_size))
        if total_steps <= 6000:
            nsteps = total_steps
            time_step_multiplier = 1
        else:
            time_step_multiplier = int(np.ceil(total_steps / 6000))
            nsteps = int(np.ceil(total_steps / time_step_multiplier))
            time_step_multiplier = max(1, time_step_multiplier)
            nsteps = max(1, nsteps)
    
    sim = SR_DDD(
        eta=eta,
        beta=beta,
        epsilon=epsilon,
        xc=xc,
        eta_var=eta_var,
        beta_var=beta_var,
        kappa_var=kappa_var,
        epsilon_var=epsilon_var,
        xc_var=xc_var,
        kappa=0.5,
        npeople=n,
        nsteps=nsteps,
        t_end=t_end,
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
        bandwidth=bandwidth,
        method=method,
        xd=xd,
        x_res=x_res,
        x_avg_window=x_avg_window,
        function_ks=function_ks
    )
    
    return sim
