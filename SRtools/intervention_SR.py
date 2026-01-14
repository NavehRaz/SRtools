from SRtools import SR_hetro as srh
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from SRtools import deathTimesDataSet as dtds
import os
from SRtools import sr_mcmc as srmc
from SRtools import SRmodellib as sr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

jit_nopython = True
theta_f = [7.26046391e-04, 6.29789339e-01, 1.47904646e+00, 2.97000000e+01]
theta_m = [3.35836076e-04, 1.80921611e-01, 1.30164401e+00, 4.26370000e+01]

class intervention_SR(srh.SR_Hetro):
    """
    Optimized intervention_SR class that always implements:
    - A change
    - Beta change
    - Eta change
    - Xc change
    
    All interventions use time_dependence type 'time_dependence' with different values.
    Each intervention can have different durations.
    """
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, 
                 eta_var=0, beta_var=0, kappa_var=0, epsilon_var=0, xc_var=0, 
                 t_start=0, tscale='years', external_hazard=np.inf, 
                 time_step_multiplier=1, parallel=False, bandwidth=3, method='brownian_bridge',
                 # Fast intervention parameters - always 5 interventions
                 intervention_time=0,
                 A_effect=0.0, A_duration=[0, 0], A_time_dep_value=0.0, A_exit_alpha=0.0, A_exit_tau=1.0,
                 beta_effect=0.0, beta_duration=[0, 0], beta_time_dep_value=0.0, beta_exit_alpha=0.0, beta_exit_tau=1.0,
                 eta_effect=0.0, eta_duration=[0, 0], eta_time_dep_value=0.0, eta_exit_alpha=0.0, eta_exit_tau=1.0,
                 epsilon_effect=0.0, epsilon_duration=[0, 0], epsilon_time_dep_value=0.0, epsilon_exit_alpha=0.0, epsilon_exit_tau=1.0,
                 Xc_effect=0.0, Xc_duration=[0, 0], Xc_time_dep_value=0.0, Xc_exit_alpha=0.0, Xc_exit_tau=1.0,
                 save_trajectory=False, traj_points=500):
        
        self.save_trajectory = save_trajectory
        self.traj_points = traj_points
        
        # Store intervention parameters
        self.intervention_time = intervention_time
        
        # Check if method uses exit effects
        self.use_exit_effect = '_exit' in method.lower()
        
        # Process durations (need intervention_time set first)
        self.A_effect = float(A_effect)
        self.A_duration = self._process_duration(A_duration, intervention_time)
        self.A_time_dep_value = float(A_time_dep_value)
        self.A_exit_alpha = float(A_exit_alpha)
        self.A_exit_tau = float(A_exit_tau)
        
        self.beta_effect = float(beta_effect)
        self.beta_duration = self._process_duration(beta_duration, intervention_time)
        self.beta_time_dep_value = float(beta_time_dep_value)
        self.beta_exit_alpha = float(beta_exit_alpha)
        self.beta_exit_tau = float(beta_exit_tau)
        
        self.eta_effect = float(eta_effect)
        self.eta_duration = self._process_duration(eta_duration, intervention_time)
        self.eta_time_dep_value = float(eta_time_dep_value)
        self.eta_exit_alpha = float(eta_exit_alpha)
        self.eta_exit_tau = float(eta_exit_tau)
        
        self.epsilon_effect = float(epsilon_effect)
        self.epsilon_duration = self._process_duration(epsilon_duration, intervention_time)
        self.epsilon_time_dep_value = float(epsilon_time_dep_value)
        self.epsilon_exit_alpha = float(epsilon_exit_alpha)
        self.epsilon_exit_tau = float(epsilon_exit_tau)
        
        self.Xc_effect = float(Xc_effect)
        self.Xc_duration = self._process_duration(Xc_duration, intervention_time)
        self.Xc_time_dep_value = float(Xc_time_dep_value)
        self.Xc_exit_alpha = float(Xc_exit_alpha)
        self.Xc_exit_tau = float(Xc_exit_tau)
        
        # Extract base method (remove _exit suffix if present)
        base_method = method.replace('_exit', '').replace('_Exit', '')
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, 
                        eta_var, beta_var, kappa_var, epsilon_var, xc_var, 
                        t_start, tscale, external_hazard, time_step_multiplier, 
                        parallel, bandwidth, method=base_method)

    def _process_duration(self, duration, intervention_time):
        """Convert duration to numpy array format [start, stop]."""
        if isinstance(duration, (int, float)):
            # Single duration value - convert to [intervention_time, intervention_time + duration]
            return np.array([float(intervention_time), float(intervention_time + duration)])
        elif isinstance(duration, (list, tuple)):
            if len(duration) == 2:
                return np.array([float(duration[0]), float(duration[1])])
            else:
                raise ValueError(f"Duration must be a single value or [start, stop]. Got: {duration}")
        elif hasattr(duration, 'ndim'):
            if duration.ndim == 1 and len(duration) == 2:
                return np.array([float(duration[0]), float(duration[1])])
            else:
                raise ValueError(f"Duration must be 1D array with 2 elements. Got shape: {duration.shape}")
        else:
            return np.array([0.0, 0.0])

    def _process_traj_points(self):
        """Process traj_points to extract time points for trajectory saving."""
        if not self.save_trajectory:
            return np.array([0.0])
        
        if isinstance(self.traj_points, (int, float)):
            num_points = int(self.traj_points)
            return np.linspace(0, self.t_end, num_points)
        elif hasattr(self.traj_points, '__iter__'):
            return np.array([float(t) for t in self.traj_points])
        else:
            return np.linspace(0, self.t_end, 500)

    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1] - self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        # Process trajectory points for saving
        traj_time_points = self._process_traj_points()
        
        # Build common arguments
        common_args = (s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, 
                      self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, 
                      self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, 
                      self.time_step_multiplier, self.intervention_time)
        
        # Check if using exit effects (method contains "_exit")
        if self.use_exit_effect:
            # Build intervention arguments with exit parameters
            intervention_args = (
                self.A_effect, self.A_duration[0], self.A_duration[1], self.A_time_dep_value,
                self.beta_effect, self.beta_duration[0], self.beta_duration[1], self.beta_time_dep_value,
                self.eta_effect, self.eta_duration[0], self.eta_duration[1], self.eta_time_dep_value,
                self.epsilon_effect, self.epsilon_duration[0], self.epsilon_duration[1], self.epsilon_time_dep_value,
                self.Xc_effect, self.Xc_duration[0], self.Xc_duration[1], self.Xc_time_dep_value,
                self.A_exit_alpha, self.A_exit_tau, self.beta_exit_alpha, self.beta_exit_tau,
                self.eta_exit_alpha, self.eta_exit_tau, self.epsilon_exit_alpha, self.epsilon_exit_tau,
                self.Xc_exit_alpha, self.Xc_exit_tau,
                self.save_trajectory, traj_time_points)
            
            # Use exit-specific functions
            if self.parallel:
                death_times, events, trajectories = death_times_euler_brownian_bridge_parallel_exit_fast(
                    *common_args, *intervention_args)
            else:
                death_times, events, trajectories = death_times_euler_brownian_bridge_exit_fast(
                    *common_args, *intervention_args)
        else:
            # Build intervention arguments without exit parameters
            intervention_args = (
                self.A_effect, self.A_duration[0], self.A_duration[1], self.A_time_dep_value,
                self.beta_effect, self.beta_duration[0], self.beta_duration[1], self.beta_time_dep_value,
                self.eta_effect, self.eta_duration[0], self.eta_duration[1], self.eta_time_dep_value,
                self.epsilon_effect, self.epsilon_duration[0], self.epsilon_duration[1], self.epsilon_time_dep_value,
                self.Xc_effect, self.Xc_duration[0], self.Xc_duration[1], self.Xc_time_dep_value,
                self.save_trajectory, traj_time_points)
            
            if self.method == 'brownian_bridge':
                if self.parallel:
                    death_times, events, trajectories = death_times_euler_brownian_bridge_parallel_fast(
                        *common_args, *intervention_args)
                else:
                    death_times, events, trajectories = death_times_euler_brownian_bridge_fast(
                        *common_args, *intervention_args)
            elif self.method == 'euler':
                if self.parallel:
                    death_times, events, trajectories = death_times_accelerator2_fast(
                        *common_args, *intervention_args)
                else:
                    death_times, events, trajectories = death_times_accelerator_fast(
                        *common_args, *intervention_args)
            else:
                # Default to brownian bridge
                if self.parallel:
                    death_times, events, trajectories = death_times_euler_brownian_bridge_parallel_fast(
                        *common_args, *intervention_args)
                else:
                    death_times, events, trajectories = death_times_euler_brownian_bridge_fast(
                        *common_args, *intervention_args)

        # Store results
        self.death_times = np.array(death_times)
        self.events = np.array(events)
        
        if self.save_trajectory:
            self.trajectories = trajectories
            self.traj_time_points = traj_time_points
        
        return self.death_times, self.events
    
    def plot_trajectories(self, n_trajectories=10, mark_death=True, random_selection=True, 
                         colormap='viridis', ax=None, alpha=0.7, linewidth=1.0, show_death_level=True,
                         title=None, xlabel='Time', ylabel='Damage'):
        """
        Plot damage trajectories from the simulation.
        Same as parent class implementation.
        """
        if not hasattr(self, 'trajectories') or self.trajectories is None:
            raise ValueError("No trajectories available. Set save_trajectory=True when creating the simulation.")
        
        if not hasattr(self, 'traj_time_points') or self.traj_time_points is None:
            raise ValueError("No trajectory time points available.")
        
        n_people = self.trajectories.shape[0]
        if n_trajectories > n_people:
            n_trajectories = n_people
            print(f"Warning: Requested {n_trajectories} trajectories but only {n_people} available. Plotting all {n_people}.")
        
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if random_selection:
            selected_indices = np.random.choice(n_people, size=n_trajectories, replace=False)
        else:
            selected_indices = np.arange(n_trajectories)
        
        cmap = cm.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_trajectories))
        
        for i, idx in enumerate(selected_indices):
            trajectory = self.trajectories[idx, :].copy()
            time_points = self.traj_time_points.copy()
            
            if hasattr(self, 'death_times') and hasattr(self, 'events'):
                if idx < len(self.events) and self.events[idx] == 1:
                    death_time = self.death_times[idx]
                    valid_indices = time_points <= death_time
                    
                    if np.any(valid_indices):
                        time_points = time_points[valid_indices]
                        trajectory = trajectory[valid_indices]
                        
                        if show_death_level:
                            time_points = np.append(time_points, death_time)
                            trajectory = np.append(trajectory, self.xc)
            
            ax.plot(time_points, trajectory, color=colors[i], 
                   alpha=alpha, linewidth=linewidth, label=f'Individual {idx}' if n_trajectories <= 10 else None)
            
            if mark_death and hasattr(self, 'death_times') and hasattr(self, 'events'):
                if idx < len(self.events) and self.events[idx] == 1:
                    death_time = self.death_times[idx]
                    death_damage = self.xc if show_death_level else trajectory[-1]
                    ax.scatter(death_time, death_damage, color=colors[i], s=50, 
                             marker='x', alpha=1.0, zorder=5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if title is None:
            ax.set_title(f'Damage Trajectories (n={n_trajectories})')
        else:
            ax.set_title(title)
            
        ax.grid(True, alpha=0.3)
        
        if n_trajectories <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if mark_death:
            death_label = 'Death time (Xc level)' if show_death_level else 'Death time'
            ax.scatter([], [], color='black', marker='x', s=50, alpha=1.0, label=death_label)
            if n_trajectories <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return ax


@jit(nopython=jit_nopython)
def get_time_dependent_effect_fast(initial_effect, time_dep_value, current_time, duration_start, duration_stop):
    """
    Calculate time-dependent effect using 'time_dependence' type.
    Effect decreases by p% per unit time until 0.
    """
    # Check if we're in the intervention duration FIRST
    if not (duration_start <= current_time <= duration_stop):
        return 0.0
    
    # If no time dependence, return the initial effect
    if time_dep_value == 0.0:
        return initial_effect
    
    # Calculate elapsed time since intervention started
    elapsed_time = current_time - duration_start
    
    if elapsed_time <= 0.0:
        return initial_effect
    
    # Effect decreases by p% every 1 unit of time
    if initial_effect > 0:
        remaining_effect = initial_effect * ((1.0 - time_dep_value / 100.0) ** elapsed_time)
        return max(0.0, remaining_effect)
    elif initial_effect < 0:
        remaining_effect = initial_effect * ((1.0 - time_dep_value / 100.0) ** elapsed_time)
        return min(0.0, remaining_effect)
    else:
        return 0.0


@jit(nopython=jit_nopython)
def get_exit_effect_fast(alpha, tau, current_time, duration_stop):
    """
    Calculate exit effect after intervention stops.
    After the intervention stops, there is an opposite effect of size alpha
    that decays to baseline with time constant tau.
    
    Parameters:
    -----------
    alpha : float
        Size of the opposite effect (positive value, will be negated)
    tau : float
        Time constant for exponential decay
    current_time : float
        Current simulation time
    duration_stop : float
        Time when intervention stops
    
    Returns:
    --------
    float
        Exit effect value (negative, decays to 0)
    """
    # Only apply exit effect after intervention stops
    if current_time <= duration_stop:
        return 0.0
    
    # If tau is 0 or invalid, no decay (constant effect)
    if tau <= 0.0:
        return -alpha
    
    # Calculate time since intervention stopped
    time_since_stop = current_time - duration_stop
    
    # Exponential decay: -alpha * exp(-time_since_stop / tau)
    exit_effect = -alpha * np.exp(-time_since_stop / tau)
    
    return exit_effect


@jit(nopython=jit_nopython)
def get_combined_effect_fast(initial_effect, time_dep_value, exit_alpha, exit_tau, 
                             current_time, duration_start, duration_stop):
    """
    Calculate combined intervention effect (during intervention) and exit effect (after intervention).
    
    Parameters:
    -----------
    initial_effect : float
        Initial intervention effect
    time_dep_value : float
        Time dependence value (p% per unit time)
    exit_alpha : float
        Size of exit effect (opposite to intervention)
    exit_tau : float
        Time constant for exit effect decay
    current_time : float
        Current simulation time
    duration_start : float
        Time when intervention starts
    duration_stop : float
        Time when intervention stops
    
    Returns:
    --------
    float
        Combined effect (intervention effect + exit effect)
    """
    # Get intervention effect during the intervention period
    intervention_eff = get_time_dependent_effect_fast(initial_effect, time_dep_value, 
                                                        current_time, duration_start, duration_stop)
    
    # Get exit effect after intervention stops
    exit_eff = get_exit_effect_fast(exit_alpha, exit_tau, current_time, duration_stop)
    
    # Combine both effects
    return intervention_eff + exit_eff


@jit(nopython=jit_nopython)
def death_times_accelerator_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                 epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                 external_hazard=np.inf, time_step_multiplier=1,
                                 intervention_time=0.0,
                                 A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                 beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                 eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                 epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                 Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                 save_trajectory=False, traj_time_points=np.array([0.0])):
    """
    Optimized death times calculator for 4 fixed interventions: A, beta, eta, Xc.
    All use time_dependence type with different values.
    """
    death_times = []
    events = []
    if save_trajectory:
        trajectories = np.zeros((npeople, len(traj_time_points)))
    else:
        trajectories = np.zeros((1, 1))
    
    for l in range(npeople):
        x = 0.0
        y = 0.0
        a = 1.0
        j = 0
        ndt = dt / time_step_multiplier
        nsdt = sdt / np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard) * ndt
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        
        while j < s - 1 and x < xc:
            current_time = t[j]
            
            # Save trajectory if requested
            if save_trajectory and x < xc:
                for k in range(len(traj_time_points)):
                    if abs(current_time - traj_time_points[k]) < dt / 2:
                        trajectories[l, k] = x
            
            for i in range(time_step_multiplier):
                # Calculate time-dependent effects for each intervention
                A_eff = get_time_dependent_effect_fast(A_effect, A_time_dep_value, current_time, 
                                                       A_duration_start, A_duration_stop)
                beta_eff = get_time_dependent_effect_fast(beta_effect, beta_time_dep_value, current_time,
                                                         beta_duration_start, beta_duration_stop)
                eta_eff = get_time_dependent_effect_fast(eta_effect, eta_time_dep_value, current_time,
                                                         eta_duration_start, eta_duration_stop)
                epsilon_eff = get_time_dependent_effect_fast(epsilon_effect, epsilon_time_dep_value, current_time,
                                                            epsilon_duration_start, epsilon_duration_stop)
                Xc_eff = get_time_dependent_effect_fast(Xc_effect, Xc_time_dep_value, current_time,
                                                        Xc_duration_start, Xc_duration_stop)
                
                # Apply parameter modifications
                current_eta = eta * (1 + eta_eff) if eta_eff != 0.0 else eta
                current_beta = beta * (1 + beta_eff) if beta_eff != 0.0 else beta
                current_epsilon = epsilon * (1 + epsilon_eff) if epsilon_eff != 0.0 else epsilon
                current_xc = xc * (1 + Xc_eff) if Xc_eff != 0.0 else xc
                current_a = a * (1 + A_eff) if A_eff != 0.0 else a
                
                # Update y: dy/dt = a
                y = y + current_a * ndt
                
                noise = np.sqrt(2 * current_epsilon) * np.random.normal(0.0, 1.0)
                x = x + ndt * (current_eta * y - current_beta * x / (x + kappa)) + noise * nsdt
                x = np.maximum(x, 0.0)
                
                if np.random.uniform(0, 1) < chance_to_die_externally:
                    x = current_xc
                if x >= current_xc:
                    break
            j += 1
        
        if x >= xc:
            death_times.append(j * dt)
            events.append(1)
        else:
            death_times.append(j * dt)
            events.append(0)
    
    return death_times, events, trajectories


def death_times_accelerator2_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                  epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                  external_hazard=np.inf, time_step_multiplier=1,
                                  intervention_time=0.0,
                                  A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                  beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                  eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                  epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                  Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                  save_trajectory=False, traj_time_points=np.array([0.0])):
    """
    Parallel version of death_times_accelerator_fast.
    """
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                              epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
                              intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
                              beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
                              eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
                              epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
                              Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
                              save_trajectory, traj_time_points):
        return death_times_accelerator_fast(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
            external_hazard, time_step_multiplier, intervention_time,
            A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
            beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
            eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
            epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
            Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
            save_trajectory, traj_time_points
        )
    
    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
        epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
        intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
        beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
        eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
        epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
        Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
        save_trajectory, traj_time_points
    ) for _ in range(n_jobs))
    
    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    trajectories = np.vstack([sublist[2] for sublist in results])
    return death_times, events, trajectories


@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                          epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                          external_hazard=np.inf, time_step_multiplier=1,
                                          intervention_time=0.0,
                                          A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                          beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                          eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                          epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                          Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                          save_trajectory=False, traj_time_points=np.array([0.0])):
    """
    Euler method with Brownian bridge crossing detection - optimized for 4 fixed interventions.
    """
    death_times = []
    events = []
    if save_trajectory:
        trajectories = np.zeros((npeople, len(traj_time_points)))
    else:
        trajectories = np.zeros((1, 1))
    
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        y = 0.0
        a = 1.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        crossed = False
        
        while j < s - 1 and not crossed:
            current_time = t[j]
            
            # Save trajectory if requested
            if save_trajectory and x < xc:
                for k in range(len(traj_time_points)):
                    if abs(current_time - traj_time_points[k]) < dt / 2:
                        trajectories[person, k] = x
            
            for _ in range(time_step_multiplier):
                # Calculate time-dependent effects
                A_eff = get_time_dependent_effect_fast(A_effect, A_time_dep_value, current_time,
                                                       A_duration_start, A_duration_stop)
                beta_eff = get_time_dependent_effect_fast(beta_effect, beta_time_dep_value, current_time,
                                                          beta_duration_start, beta_duration_stop)
                eta_eff = get_time_dependent_effect_fast(eta_effect, eta_time_dep_value, current_time,
                                                         eta_duration_start, eta_duration_stop)
                epsilon_eff = get_time_dependent_effect_fast(epsilon_effect, epsilon_time_dep_value, current_time,
                                                             epsilon_duration_start, epsilon_duration_stop)
                Xc_eff = get_time_dependent_effect_fast(Xc_effect, Xc_time_dep_value, current_time,
                                                        Xc_duration_start, Xc_duration_stop)
                
                # Apply parameter modifications
                current_eta = eta * (1 + eta_eff) if eta_eff != 0.0 else eta
                current_beta = beta * (1 + beta_eff) if beta_eff != 0.0 else beta
                current_epsilon = epsilon * (1 + epsilon_eff) if epsilon_eff != 0.0 else epsilon
                current_xc = xc * (1 + Xc_eff) if Xc_eff != 0.0 else xc
                current_a = a * (1 + A_eff) if A_eff != 0.0 else a
                
                # Update y: dy/dt = a
                y = y + current_a * ndt
                
                # Standard Euler step
                drift = current_eta * y - current_beta * x / (x + kappa)
                sqrt_2current_epsilon = np.sqrt(2 * current_epsilon)
                noise = sqrt_2current_epsilon * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = current_xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= current_xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test
                if (x < current_xc) and (x_new < current_xc) and (x > 0 * kappa):
                    dx1 = current_xc - x
                    dx2 = current_xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        var = 2.0 * epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = current_xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= current_xc:
            events.append(1)
        else:
            events.append(0)
    
    return np.array(death_times), np.array(events), trajectories


@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge_exit_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                                epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                                external_hazard=np.inf, time_step_multiplier=1,
                                                intervention_time=0.0,
                                                A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                                beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                                eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                                epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                                Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                                A_exit_alpha=0.0, A_exit_tau=1.0, beta_exit_alpha=0.0, beta_exit_tau=1.0,
                                                eta_exit_alpha=0.0, eta_exit_tau=1.0, epsilon_exit_alpha=0.0, epsilon_exit_tau=1.0,
                                                Xc_exit_alpha=0.0, Xc_exit_tau=1.0,
                                                save_trajectory=False, traj_time_points=np.array([0.0])):
    """
    Euler method with Brownian bridge crossing detection - with exit effects.
    Independent function optimized for exit effects (no conditional branching).
    """
    death_times = []
    events = []
    if save_trajectory:
        trajectories = np.zeros((npeople, len(traj_time_points)))
    else:
        trajectories = np.zeros((1, 1))
    
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        y = 0.0
        a = 1.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        crossed = False
        
        while j < s - 1 and not crossed:
            current_time = t[j]
            
            # Save trajectory if requested
            if save_trajectory and x < xc:
                for k in range(len(traj_time_points)):
                    if abs(current_time - traj_time_points[k]) < dt / 2:
                        trajectories[person, k] = x
            
            for _ in range(time_step_multiplier):
                # Calculate combined effects (intervention + exit)
                A_eff = get_combined_effect_fast(A_effect, A_time_dep_value, A_exit_alpha, A_exit_tau,
                                                 current_time, A_duration_start, A_duration_stop)
                beta_eff = get_combined_effect_fast(beta_effect, beta_time_dep_value, beta_exit_alpha, beta_exit_tau,
                                                   current_time, beta_duration_start, beta_duration_stop)
                eta_eff = get_combined_effect_fast(eta_effect, eta_time_dep_value, eta_exit_alpha, eta_exit_tau,
                                                  current_time, eta_duration_start, eta_duration_stop)
                epsilon_eff = get_combined_effect_fast(epsilon_effect, epsilon_time_dep_value, epsilon_exit_alpha, epsilon_exit_tau,
                                                       current_time, epsilon_duration_start, epsilon_duration_stop)
                Xc_eff = get_combined_effect_fast(Xc_effect, Xc_time_dep_value, Xc_exit_alpha, Xc_exit_tau,
                                                 current_time, Xc_duration_start, Xc_duration_stop)
                
                # Apply parameter modifications
                current_eta = eta * (1 + eta_eff) if eta_eff != 0.0 else eta
                current_beta = beta * (1 + beta_eff) if beta_eff != 0.0 else beta
                current_epsilon = epsilon * (1 + epsilon_eff) if epsilon_eff != 0.0 else epsilon
                current_xc = xc * (1 + Xc_eff) if Xc_eff != 0.0 else xc
                current_a = a * (1 + A_eff) if A_eff != 0.0 else a
                
                # Update y: dy/dt = a
                y = y + current_a * ndt
                
                # Standard Euler step
                drift = current_eta * y - current_beta * x / (x + kappa)
                sqrt_2current_epsilon = np.sqrt(2 * current_epsilon)
                noise = sqrt_2current_epsilon * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = current_xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= current_xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test
                if (x < current_xc) and (x_new < current_xc) and (x > 0 * kappa):
                    dx1 = current_xc - x
                    dx2 = current_xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        var = 2.0 * current_epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = current_xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= current_xc:
            events.append(1)
        else:
            events.append(0)
    
    return np.array(death_times), np.array(events), trajectories


def death_times_euler_brownian_bridge_parallel_exit_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                                          epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                                          external_hazard=np.inf, time_step_multiplier=1,
                                                          intervention_time=0.0,
                                                          A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                                          beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                                          eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                                          epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                                          Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                                          A_exit_alpha=0.0, A_exit_tau=1.0, beta_exit_alpha=0.0, beta_exit_tau=1.0,
                                                          eta_exit_alpha=0.0, eta_exit_tau=1.0, epsilon_exit_alpha=0.0, epsilon_exit_tau=1.0,
                                                          Xc_exit_alpha=0.0, Xc_exit_tau=1.0,
                                                          save_trajectory=False, traj_time_points=np.array([0.0]), n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge_exit_fast.
    """
    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
               intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
               beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
               eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
               epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
               Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
               A_exit_alpha, A_exit_tau, beta_exit_alpha, beta_exit_tau,
               eta_exit_alpha, eta_exit_tau, epsilon_exit_alpha, epsilon_exit_tau,
               Xc_exit_alpha, Xc_exit_tau,
               save_trajectory, traj_time_points):
        return death_times_euler_brownian_bridge_exit_fast(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, intervention_time,
            A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
            beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
            eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
            epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
            Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
            A_exit_alpha, A_exit_tau, beta_exit_alpha, beta_exit_tau,
            eta_exit_alpha, eta_exit_tau, epsilon_exit_alpha, epsilon_exit_tau,
            Xc_exit_alpha, Xc_exit_tau,
            save_trajectory, traj_time_points
        )
    
    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
            intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
            beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
            eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
            epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
            Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
            A_exit_alpha, A_exit_tau, beta_exit_alpha, beta_exit_tau,
            eta_exit_alpha, eta_exit_tau, epsilon_exit_alpha, epsilon_exit_tau,
            Xc_exit_alpha, Xc_exit_tau,
            save_trajectory, traj_time_points
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )
    
    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    trajectories = np.vstack([res[2] for res in results])
    return death_times, events, trajectories


def death_times_euler_brownian_bridge_parallel_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                                    epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                                    external_hazard=np.inf, time_step_multiplier=1,
                                                    intervention_time=0.0,
                                                    A_effect=0.0, A_duration_start=0.0, A_duration_stop=0.0, A_time_dep_value=0.0,
                                                    beta_effect=0.0, beta_duration_start=0.0, beta_duration_stop=0.0, beta_time_dep_value=0.0,
                                                    eta_effect=0.0, eta_duration_start=0.0, eta_duration_stop=0.0, eta_time_dep_value=0.0,
                                                    epsilon_effect=0.0, epsilon_duration_start=0.0, epsilon_duration_stop=0.0, epsilon_time_dep_value=0.0,
                                                    Xc_effect=0.0, Xc_duration_start=0.0, Xc_duration_stop=0.0, Xc_time_dep_value=0.0,
                                                    save_trajectory=False, traj_time_points=np.array([0.0]), n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge_fast.
    """
    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
               intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
               beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
               eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
               epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
               Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
               save_trajectory, traj_time_points):
        return death_times_euler_brownian_bridge_fast(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, intervention_time,
            A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
            beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
            eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
            epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
            Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
            save_trajectory, traj_time_points
        )
    
    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier,
            intervention_time, A_effect, A_duration_start, A_duration_stop, A_time_dep_value,
            beta_effect, beta_duration_start, beta_duration_stop, beta_time_dep_value,
            eta_effect, eta_duration_start, eta_duration_stop, eta_time_dep_value,
            epsilon_effect, epsilon_duration_start, epsilon_duration_stop, epsilon_time_dep_value,
            Xc_effect, Xc_duration_start, Xc_duration_stop, Xc_time_dep_value,
            save_trajectory, traj_time_points
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )
    
    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    trajectories = np.vstack([res[2] for res in results])
    return death_times, events, trajectories


def getISR(theta, n=25000, nsteps=6000, t_end=110, external_hazard=np.inf,
                           time_step_multiplier=1, npeople=None, parallel=False,
                           eta_var=0, beta_var=0, epsilon_var=0, xc_var=0.2, kappa_var=0,
                           hetro=False, bandwidth=3, step_size=None, method='brownian_bridge',
                           intervention_time=0,
                           A_effect=0.0, A_duration=[0, 0], A_time_dep_value=0.0, A_exit_alpha=0.0, A_exit_tau=1.0,
                           beta_effect=0.0, beta_duration=[0, 0], beta_time_dep_value=0.0, beta_exit_alpha=0.0, beta_exit_tau=1.0,
                           eta_effect=0.0, eta_duration=[0, 0], eta_time_dep_value=0.0, eta_exit_alpha=0.0, eta_exit_tau=1.0,
                           epsilon_effect=0.0, epsilon_duration=[0, 0], epsilon_time_dep_value=0.0, epsilon_exit_alpha=0.0, epsilon_exit_tau=1.0,
                           Xc_effect=0.0, Xc_duration=[0, 0], Xc_time_dep_value=0.0, Xc_exit_alpha=0.0, Xc_exit_tau=1.0,
                           save_trajectory=False, traj_points=500, theta_intervention=None, durations = None):
    """
    Factory function to generate an intervention SR simulation with flexible interventions.

    This function creates a simulation using the intervention SR model. It supports
    simultaneous, independently configured interventions on up to five parameters: 
    A, eta, beta, epsilon, and Xc. Each intervention can be given an effect size, a time 
    window [start, stop], and a rate for time-dependent interventions.

    Parameters
    ----------
    theta : array-like
        Model parameters in the order [eta, beta, epsilon, xc].
    n : int, optional
        Number of individuals in the simulation (default=25000).
    nsteps : int, optional
        Number of discrete time steps in the simulation (default=6000).
    t_end : float, optional
        End time of simulation (default=110).
    external_hazard : float, optional
        Hazard rate from external causes (default: np.inf).
    time_step_multiplier : float, optional
        Multiplier for the time step size (default=1).
    npeople : int, optional
        Alternative to n as the number of individuals (if specified).
    parallel : bool, optional
        Whether to run simulation in parallel (default=False).
    eta_var, beta_var, epsilon_var, xc_var, kappa_var : float, optional
        Variances for heterogeneity in each parameter (default: see function signature).
    hetro : bool, optional
        If True, include heterogeneity (default=False).
    bandwidth : float, optional
        Bandwidth for kernel density estimate (default=3).
    step_size : float, optional
        Time step size (overrides nsteps if provided).
    method : str, optional
        Simulation method (default='brownian_bridge').
    intervention_time : float, optional
        Time at which interventions start (default=0).

    A_effect, beta_effect, eta_effect, epsilon_effect, Xc_effect : float, optional
        The magnitude of the intervention effect on A, beta, eta, epsilon, and Xc, respectively.
    A_duration, beta_duration, eta_duration, epsilon_duration, Xc_duration : list of two floats, optional
        Time interval as [start, stop] for each intervention.
    A_time_dep_value, beta_time_dep_value, eta_time_dep_value, epsilon_time_dep_value, Xc_time_dep_value : float, optional
        Value for time-dependent intervention for each parameter.
    A_exit_alpha, beta_exit_alpha, eta_exit_alpha, epsilon_exit_alpha, Xc_exit_alpha : float, optional
        Parameter for intervention exit dynamics (default=0.0).
    A_exit_tau, beta_exit_tau, eta_exit_tau, epsilon_exit_tau, Xc_exit_tau : float, optional
        Time constant for intervention exit dynamics (default=1.0).

    save_trajectory : bool, optional
        If True, save and return full trajectories (default=False).
    traj_points : int, optional
        Number of time points in saved trajectory (default=500).
    theta_intervention : array-like, optional
        Parameter values for interventions; see code for order/conventions.
    durations : None, 'all', list, optional
        If 'all', apply all interventions over full t_end. If a list, specifies durations
        for each intervention.

    Returns
    -------
    death_times : np.ndarray
        Simulated death times for all individuals.
    events : np.ndarray
        Event/censoring status for each individual.
    trajectories : np.ndarray
        (Optional) Trajectories if save_trajectory is True.

    Notes
    -----
    - You can specify intervention parameters individually or via `theta_intervention` (see code for conventions).
    - To apply an intervention throughout the simulation, set durations='all' or pass [0, t_end] for any *_duration parameter.
    - For more details on simulation mechanics, see `death_times_euler_brownian_bridge_fast`.

    """

    if theta_intervention is not None:
        if len(theta_intervention) == 8:
            A_effect = theta_intervention[0]
            eta_effect = theta_intervention[1]
            beta_effect = theta_intervention[2]
            Xc_effect = theta_intervention[3]
            A_time_dep_value = theta_intervention[4]
            beta_time_dep_value = theta_intervention[5]
            eta_time_dep_value = theta_intervention[6]
            Xc_time_dep_value = theta_intervention[7]
        elif len(theta_intervention) == 5:
            A_effect = theta_intervention[0]
            eta_effect = theta_intervention[1]
            beta_effect = theta_intervention[2]
            epsilon_effect = theta_intervention[3]
            Xc_effect = theta_intervention[4]
        elif len(theta_intervention) == 10:
            A_effect = theta_intervention[0]
            eta_effect = theta_intervention[1]
            beta_effect = theta_intervention[2]
            epsilon_effect = theta_intervention[3]
            Xc_effect = theta_intervention[4]
            A_time_dep_value = theta_intervention[5]
            beta_time_dep_value = theta_intervention[6]
            eta_time_dep_value = theta_intervention[7]
            epsilon_time_dep_value = theta_intervention[8]
            Xc_time_dep_value = theta_intervention[9]
    
    if durations == 'all':
        A_duration = [0, t_end]
        beta_duration = [0, t_end]
        eta_duration = [0, t_end]
        epsilon_duration = [0, t_end]
        Xc_duration = [0, t_end]
    elif isinstance(durations, list) and len(durations) == 2:
        A_duration = durations
        beta_duration = durations
        eta_duration = durations
        epsilon_duration = durations
        Xc_duration = durations
    elif isinstance(durations, list) and len(durations) == 5:
        A_duration = durations[0]
        beta_duration = durations[1]
        eta_duration = durations[2]
        epsilon_duration = durations[3]
        Xc_duration = durations[4]
    elif durations is None:
        pass
    else:
        raise ValueError("Invalid durations: must be None,'all', a list of 2 elements, or a list of 5 elements.")

    if npeople is not None:
        n = npeople
    
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    
    if not hetro:
        eta_var = 0.0
        beta_var = 0.0
        epsilon_var = 0.0
        xc_var = 0.0
        kappa_var = 0.0
    else:
        eta_var = float(eta_var)
        beta_var = float(beta_var)
        epsilon_var = float(epsilon_var)
        xc_var = float(xc_var)
        kappa_var = float(kappa_var)
    
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
    
    # Create simulation
    sim = intervention_SR(
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
        intervention_time=float(intervention_time),
        A_effect=A_effect,
        A_duration=A_duration,
        A_time_dep_value=A_time_dep_value,
        A_exit_alpha=A_exit_alpha,
        A_exit_tau=A_exit_tau,
        beta_effect=beta_effect,
        beta_duration=beta_duration,
        beta_time_dep_value=beta_time_dep_value,
        beta_exit_alpha=beta_exit_alpha,
        beta_exit_tau=beta_exit_tau,
        eta_effect=eta_effect,
        eta_duration=eta_duration,
        eta_time_dep_value=eta_time_dep_value,
        eta_exit_alpha=eta_exit_alpha,
        eta_exit_tau=eta_exit_tau,
        epsilon_effect=epsilon_effect,
        epsilon_duration=epsilon_duration,
        epsilon_time_dep_value=epsilon_time_dep_value,
        epsilon_exit_alpha=epsilon_exit_alpha,
        epsilon_exit_tau=epsilon_exit_tau,
        Xc_effect=Xc_effect,
        Xc_duration=Xc_duration,
        Xc_time_dep_value=Xc_time_dep_value,
        Xc_exit_alpha=Xc_exit_alpha,
        Xc_exit_tau=Xc_exit_tau,
        save_trajectory=save_trajectory,
        traj_points=traj_points
    )
    
    return sim



def model(theta, n, nsteps, t_end, dataSet, sim=None, metric='baysian', time_range=None, 
          time_step_multiplier=1, parallel=False, dt=1, set_params=None, kwargs=None):
    """
    Evaluate SR model with given parameters and return score according to metric.
    
    This function simulates the Saturating Removal model with the provided parameters
    and compares the simulation results to the observed dataset using the specified
    metric. It handles parameter parsing and simulation setup.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters [eta, beta, epsilon, xc, external_hazard, ...]
    n : int
        Number of individuals to simulate
    nsteps : int
        Number of time steps for simulation
    t_end : float
        End time for simulation
    dataSet : object
        Dataset object containing observed data for comparison
    sim : object, optional
        Pre-computed simulation object. If None, simulation is performed
    metric : str, default='baysian'
        Metric for comparison: 'baysian', 'survival', etc.
    time_range : tuple, optional
        Time range (start, end) for comparison
    time_step_multiplier : float, default=1
        Multiplier for time step size
    parallel : bool, default=False
        Whether to use parallel simulation
    dt : float, default=1
        Time step for distance calculation
    set_params : dict, optional
        Dictionary of fixed parameters
    kwargs : dict, optional
        Additional keyword arguments
    
    Returns:
    --------
    float
        Score according to the specified metric. Returns -inf for invalid parameters.
    
    Notes:
    ------
    The function returns -inf if 1/beta < time_step_size or if any NaN values
    are encountered in the probability calculation.
    """
    if set_params is None:
        set_params = {}
    # parse parameters
    durations = kwargs['durations']
    thetaSR = kwargs['thetaSR']
    theta_indxes = kwargs['theta_indxes']
    thetaIntervention = [0.0] * 8
    for i, index in enumerate(theta_indxes):
        thetaIntervention[index] = theta[i]
    sim = getISR(thetaSR, n, nsteps, t_end, time_step_multiplier=time_step_multiplier, parallel=parallel, theta_intervention=thetaIntervention, durations=durations)
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.isnan(tprob):
        return -np.inf
    return tprob

def plotIntervensions(int_list, int_list_sims=None, ref=None, ax=None, label=None, color=None, alpha=0.5):
    """
    int_list is a list of dictionaries with the following keys: duration (or None), ds, label (if duration is None)
    int_list_sims is a list of dictionaries with the following keys: duration (or None), sim, label (if duration is None)
    ref is a deathTimesDataSet object
    If sims are present:
      - First row: all ds in int_list together + ref.
      - Second row: all sims in int_list_sims together (+ ref).
      - Next N rows: Each with ref, ds, and sim matching the ds's duration or label.
    If sims not present: Original layout.
    Duration is highlighted as a shaded background using the duration (in months) * 30 for days. 
    If duration is None, no shading is applied and label field is used for matching and plotting.
    In the 'all' panels, do NOT show duration backgrounds.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_ds = len(int_list)
    has_sims = int_list_sims is not None

    # Color handling
    if color is None:
        cmap = cm.get_cmap('tab10')
        color_map = {i: cmap(i % 10) for i in range(n_ds)}
        if has_sims:
            sim_color_map = {i: cmap(i % 10) for i in range(len(int_list_sims))}
        else:
            sim_color_map = color_map
    else:
        color_map = {i: color[i % len(color)] for i in range(n_ds)}
        if has_sims:
            sim_color_map = {i: color[i % len(color)] for i in range(len(int_list_sims))}
        else:
            sim_color_map = color_map

    duration_bg_alpha = 0.2
    duration_bg_colors = []
    if color is None:
        for i in range(n_ds):
            c = list(cm.get_cmap('tab10')(i % 10))
            c[3] = duration_bg_alpha
            duration_bg_colors.append(tuple(c))
    else:
        for i in range(n_ds):
            c = list(color[i % len(color)])
            if len(c) == 4:
                c[3] = duration_bg_alpha
            else:
                c.append(duration_bg_alpha)
            duration_bg_colors.append(tuple(c))

    if has_sims:
        nrow = 2 + n_ds  # 1 for all ds, 1 for all sims, rest for matched ds/sims
    else:
        nrow = 1 + n_ds  # all ds + each ds

    fig, axs = plt.subplots(nrow, 2, figsize=(12, 3 * nrow), squeeze=False)

    rowidx = 0

    # Row 0: All real ds together, with ref
    ax_surv_all = axs[rowidx, 0]
    ax_hz_all = axs[rowidx, 1]

    if ref is not None:
        ref.plotSurvival(ax=ax_surv_all, color='k', label='ref', alpha=alpha)
        ref.plotHazard(ax=ax_hz_all, color='k', label='ref', alpha=alpha)
    for idx, item in enumerate(int_list):
        ds = item['ds']
        duration = item.get('duration')
        item_label = item.get('label', f"duration: {duration}" if duration is not None else f"item {idx}")
        c = color_map[idx]
        plot_label = item_label if duration is None else f"duration: {duration}"
        ds.plotSurvival(ax=ax_surv_all, color=c, label=plot_label, alpha=alpha)
        ds.plotHazard(ax=ax_hz_all, color=c, label=plot_label, alpha=alpha)
    ax_surv_all.set_title("All Surv (real)")
    ax_hz_all.set_title("All Hazard (real)")
    ax_surv_all.legend()
    ax_hz_all.legend()

    rowidx += 1

    # If sims present, Row 1: all sims together, with ref
    if has_sims:
        ax_sim_surv_all = axs[rowidx, 0]
        ax_sim_hz_all = axs[rowidx, 1]
        if ref is not None:
            ref.plotSurvival(ax=ax_sim_surv_all, color='k', label='ref', alpha=alpha)
            ref.plotHazard(ax=ax_sim_hz_all, color='k', label='ref', alpha=alpha)
        for idx, sim_item in enumerate(int_list_sims):
            duration = sim_item.get('duration')
            sim = sim_item['sim']
            sim_label = sim_item.get('label', f"sim: {duration}" if duration is not None else f"sim {idx}")
            c = sim_color_map[idx]
            plot_label = sim_label if duration is None else f"sim: {duration}"
            sim.plotSurvival(ax=ax_sim_surv_all, color=c, ls='--', label=plot_label, alpha=alpha)
            sim.plotHazard(ax=ax_sim_hz_all, color=c, ls='--', label=plot_label, alpha=alpha)
        ax_sim_surv_all.set_title("All Surv (sim)")
        ax_sim_surv_all.legend()
        ax_sim_hz_all.set_title("All Hazard (sim)")
        ax_sim_hz_all.legend()
        rowidx += 1

    # Remaining rows: For each ds, show ref, ds, and only the matched sim (by duration or label)
    for idx, item in enumerate(int_list):
        ds = item['ds']
        duration = item.get('duration')
        item_label = item.get('label', f"duration: {duration}" if duration is not None else f"item {idx}")
        c = color_map[idx]
        bg_c = duration_bg_colors[idx]

        ax = axs[rowidx, 0]
        # Survival panel
        if ref is not None:
            ref.plotSurvival(ax=ax, color='k', label='ref', alpha=alpha)
        plot_label = item_label if duration is None else f"duration: {duration}"
        ds.plotSurvival(ax=ax, color=c, label=plot_label, alpha=alpha)
        
        # Only shade duration if it's not None
        if duration is not None:
            dur_start = duration[0] * 30
            dur_end = duration[1] * 30
            xlim = ax.get_xlim()
            end_clipped = min(dur_end, xlim[1])
            ax.axvspan(dur_start, end_clipped, color=bg_c, zorder=0)
        
        # Sims: plot ONLY the one matching this duration or label (if any)
        if has_sims:
            # Find sim with matching duration or label
            matched_sim = None
            for sim_idx, sim_item in enumerate(int_list_sims):
                sim_duration = sim_item.get('duration')
                sim_label = sim_item.get('label')
                
                if duration is not None and sim_duration is not None:
                    # Match by duration
                    if np.allclose(np.array(sim_duration), np.array(duration)):
                        matched_sim = (sim_item, sim_idx)
                        break
                elif duration is None and sim_duration is None:
                    # Match by label
                    if sim_label == item_label:
                        matched_sim = (sim_item, sim_idx)
                        break
            
            if matched_sim is not None:
                sim_item, sim_idx = matched_sim
                sim = sim_item['sim']
                sim_c = sim_color_map[sim_idx]
                sim_duration = sim_item.get('duration')
                sim_label = sim_item.get('label', f"sim: {sim_duration}" if sim_duration is not None else f"sim {sim_idx}")
                sim_plot_label = sim_label if sim_duration is None else f"sim: {sim_duration}"
                sim.plotSurvival(
                    ax=ax,
                    color=sim_c,
                    ls='--',
                    label=sim_plot_label,
                    alpha=alpha
                )
        title_label = item_label if duration is None else f"duration: {duration}"
        ax.set_title(f"DS+ref Survival ({title_label})")
        ax.legend()

        # Hazard panel
        ax2 = axs[rowidx, 1]
        if ref is not None:
            ref.plotHazard(ax=ax2, color='k', label='ref', alpha=alpha)
        ds.plotHazard(ax=ax2, color=c, label=plot_label, alpha=alpha)
        
        # Only shade duration if it's not None
        if duration is not None:
            dur_start = duration[0] * 30
            dur_end = duration[1] * 30
            xlim = ax2.get_xlim()
            end_clipped = min(dur_end, xlim[1])
            ax2.axvspan(dur_start, end_clipped, color=bg_c, zorder=0)
        
        if has_sims and matched_sim is not None:
            sim_item, sim_idx = matched_sim
            sim = sim_item['sim']
            sim_c = sim_color_map[sim_idx]
            sim_duration = sim_item.get('duration')
            sim_label = sim_item.get('label', f"sim: {sim_duration}" if sim_duration is not None else f"sim {sim_idx}")
            sim_plot_label = sim_label if sim_duration is None else f"sim: {sim_duration}"
            sim.plotHazard(
                ax=ax2,
                color=sim_c,
                ls='--',
                label=sim_plot_label,
                alpha=alpha
            )
        ax2.set_title(f"DS+ref Hazard ({title_label})")
        ax2.legend()

        rowidx += 1

    plt.tight_layout()
    plt.show()
    return fig, axs


def get_sims_from_theta_and_ds(theta, theta_sr, ds_dict, n, nsteps, t_end, time_step_multiplier=1, parallel=False, duration_factor=30, bandwidth=3):
    sims =[]
    A_effect = theta[0]
    eta_effect = theta[1]
    beta_effect = theta[2]
    Xc_effect = theta[3]
    A_time_dep_value = theta[4]
    beta_time_dep_value = theta[5]
    eta_time_dep_value = theta[6]
    Xc_time_dep_value = theta[7]
    for ds in ds_dict:
        item = {}
        duration = ds['duration']
        scaled_duration = np.array(duration) * duration_factor
        sim = getISR(theta_sr, n, nsteps, t_end, time_step_multiplier=time_step_multiplier, parallel=parallel,
                A_effect=A_effect, A_duration=scaled_duration, A_time_dep_value=A_time_dep_value,
                beta_effect=beta_effect, beta_duration=scaled_duration, beta_time_dep_value=beta_time_dep_value,
                eta_effect=eta_effect, eta_duration=scaled_duration, eta_time_dep_value=eta_time_dep_value,
                Xc_effect=Xc_effect, Xc_duration=scaled_duration, Xc_time_dep_value=Xc_time_dep_value,
                bandwidth=bandwidth)
        item['duration'] = duration
        item['sim'] = sim
        sims.append(item)
    return sims