import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numba import jit
from scipy.stats import gaussian_kde
from scipy.stats import skew
import os
from scipy.stats import bootstrap
from collections import namedtuple
from scipy.special import gammaincc,gamma,gammainc,gammaln
from scipy.stats import norm, ks_2samp
from scipy.stats import entropy
from scipy.stats import anderson_ksamp
from joblib import Parallel, delayed
from scipy.integrate import quad
from . import deathTimesDataSet as dtds

jit_nopython = True
jit_parallel = True

class SR(dtds.Dataset):
    def __init__(self, eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years', memory_efficient =True, natural_units = True, smoothing = 20,
                      boundary = 'sticking',
                        save_dist = False , dist_years =np.linspace(0,100,101), dist_method = 'hist', dist_nvalues = 40, y_gamma =None, death_times_method = 1,external_hazard = np.inf, time_step_multiplier = 1, parallel =False, method='brownian_bridge'):
        """ 
        An instance of SR simulation for SR model dX/dt = eta*t - beta*X/(X+kappa) + sqrt(2*epsilon)*xci.
        Upon creation the model runs (and saves) the trajectories for a number of agents = npeople, running for the number of timesteps defined by 
        nsteps. Then the modle calculates the survival curve, cumulative hazard and hazrard functions.
        To smooth noise, the Hazard is portrayed with a time axis in the time units of the simulation (there is an assumption of at least one timestep per time unit)
        Parameters:
            eta:     SR model eta (damage production units growth parameter) [damage/time^2]
            beta:    SR model beta (damage removal parameter) [damage/time]
            kappa:   SR model kappa (removal saturation paranmeter) [damage]
            epsilon: SR model noise parameter [damage^2/time]
            xc:      SR model X critical [damage]
            npeople: number of agents the simulation is going to run for
            nsteps:  number of time steps for each tragectory of each agent for the simulation (each timestep can be smaller then a time unit)
            t_end:   the end time of the simulation [simulation time units] (time units is expected to be years/months etc...)
            t_start: the start time of the simulation [simulation time units]  defaults to 0
            tscale:  to keep track of what is the meaning of the simulation time unit. defaults to 'years'. Has no implication on any calculation
            memory_efficient: if set to True damage trajectories are not kept and damage related methods are invalid
            natural_units: assumes there is a natural time unit 'U' such as days or years that times are given in (so if U is years and t_end is 120 then the simulation should run to 120 years ) 
                            currently the simulation doesn't care what U is, but returns survival and hazard on a time line that is given by integer values between t_start and t_end (we sample every 1 U)
            smoothing: relevant only if the natural_units = False, then say smoothing = n, the sampling of hazard and survival is going to be every n*dt
            method:  Method to use for death times calculation. Options:
                - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
                - 'euler': Standard Euler method

        """

        # Validation for deprecated features
        if save_dist:
            raise NotImplementedError(
                "save_dist functionality is temporarily disabled pending refactoring. "
                "The distribution saving system requires trajectory storage which is being redesigned. "
                "Set save_dist=False to continue."
            )
        if y_gamma is not None:
            raise NotImplementedError(
                "y_gamma trajectory calculation is temporarily disabled pending refactoring. "
                "Set y_gamma=None to continue."
            )
        
        if nsteps =='auto':
            nsteps =int(2*(t_end-t_start)/(kappa/beta))
        if eta != np.inf:
            if nsteps <((t_end-t_start) and natural_units):
                raise ValueError('number of time steps should be greater then the number of time units of the simulation') 
            self.natural_units = natural_units
            self.smoothing =smoothing
            self.tscale = tscale
            self.eta = eta
            self.beta = beta
            self.kappa =kappa
            self.epsilon = epsilon
            self.xc = xc
            self.npeople = npeople
            self.nsteps = nsteps
            self.t = np.linspace(t_start,stop = t_end, num = nsteps) #timeline
            self.dt = (t_end-t_start)/nsteps
            self.t_end = t_end
            self.t_start = t_start
            if external_hazard is None:
                external_hazard = np.inf
            self.external_hazard = external_hazard
            self.time_step_multiplier = time_step_multiplier
            self.parallel = parallel
            self.death_times = None
            self.events = None
            self.method = method

            # NOTE: getTrajectories is disabled, always set trajectories to None
            # if death_times_method == 0:
            #     self.trajectories = self.getTrajectories(boundary=boundary)
            # else:
            #     self.trajectories = None
            self.trajectories = None
            self.cum_hazard = None
            self.hazard = None
            self.survival = self.calc_survival_and_hazard(death_times_method=death_times_method)    
            self.memory_efficient = memory_efficient
            self.fi=None
            self.sick_spans =None
            self.hle_over_le = None
            self.frailty_values = None

            # NOTE: y_gamma functionality has been temporarily disabled
            # if y_gamma is not None:
            #     self.y = self.y_traj_calc(y_gamma)
            # else:
            #     self.y = None
            self.y = None  # Always set to None since y_gamma is disabled
            
            # NOTE: save_dist and distribution functionality has been temporarily disabled
            # pending refactoring. The implementation below is commented out.
            # Parameters are kept for backward compatibility with child classes.
            # Attempting to use save_dist=True will raise NotImplementedError above.
            # if memory_efficient:
            #     if save_dist:
            #         self.save_dist = False
            #         self.damage_values, self.distributions = self.create_distributions(dist_years,method =dist_method,nvalues=dist_nvalues)
            #         # self.trans_dist,self.trans_dist_mat = trans_time_dist(self,mat=True)
            #     self.trajectories =None
            #     self.y = None
            #     self.death_times = None
            # elif save_dist: #making sure that save_dist is not used with memory_efficient = False
            #     raise ValueError('save_dist can only be used with memory_efficient=True')
            
            # Always set trajectories to None since trajectory functionality is disabled
            
            # Keep parameter assignments for backward compatibility
            self.dist_years = dist_years
            self.dist_method = dist_method
            self.dist_nvalues = dist_nvalues
            self.save_dist = save_dist
        
    def resample(self, npeople=None, nsteps=None, t_end=None, t_start=None, natural_units=None, smoothing=None):
        if npeople is None:
            npeople = self.npeople
        if nsteps is None:
            nsteps = self.nsteps
        if t_end is None:
            t_end = self.t_end
        if t_start is None:
            t_start = self.t_start
        if natural_units is None:
            natural_units = self.natural_units
        if smoothing is None:
            smoothing = self.smoothing
        return SR(eta=self.eta, beta=self.beta, kappa=self.kappa, epsilon=self.epsilon, xc=self.xc, npeople=npeople, 

                  nsteps=nsteps, t_end=t_end, t_start=t_start)

    @staticmethod
    def loadSim(path):
        #loads a model from a file
        file_path = os.path.join(path, os.listdir(path)[0])
        sim = SR(eta=np.inf,beta=0,kappa=0,epsilon=0,xc=0,npeople=0,nsteps=0,t_end=0)
        sim.load(file_path)
        return sim

    def getTrajectories (self,boundary='sticking'):

        """
        Generate damage trajectories for the SR model.
        
        DEPRECATED: This method is temporarily disabled pending refactoring.
        """
        raise NotImplementedError(
            "Trajectory saving and analysis methods are being refactored for efficiency. "
            "This functionality is temporarily disabled. "
            "The save_dist system needs complete reimplementation. "
            "See SRtools.trajectory_analysis for future implementation."
        )
    
    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        if self.method == 'brownian_bridge':
            if self.parallel:
                death_times, events = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        elif self.method == 'euler':
            if self.parallel:
                death_times, events = death_times_accelerator2(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_accelerator(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        else:

            raise ValueError(f"Unknown method: {self.method}")
        
        self.death_times = death_times
        self.events = events
        return death_times, events

    def calc_survival_and_hazard(self,death_times_method = 0):
        if death_times_method == 0:
            death_times, events = self.calc_death_times()
        else:

            death_times, events = self.calc_death_times()
        
        # Calculate survival and hazard using death times
        from .life_table import Life_table
        lt = Life_table(death_times, events)
        self.survival = lt.getSurvival()
        self.hazard = lt.getHazard()
        self.cum_hazard = lt.getCumHazard()
        return self.survival


    def getHazard(self):
        return self.hazard
    

    def getCumHazard(self):
        return self.cum_hazard
    
    def getDeathTimes(self):
        return self.death_times

    def get_median_damage(self):

        """
        Returns the median damage of the trajectories.
        
        DEPRECATED: This method is temporarily disabled pending refactoring.
        """
        raise NotImplementedError(
            "Trajectory saving and analysis methods are being refactored for efficiency. "
            "This functionality is temporarily disabled. "
            "The save_dist system needs complete reimplementation. "
            "See SRtools.trajectory_analysis for future implementation."
        )
    
    def get_damage_PDF(self, t, nvalues=40, pdf_method = 'kde'):
        """
        Returns the damage distribution for a given time t.


        DEPRECATED: This method is temporarily disabled pending refactoring.
        """
        raise NotImplementedError(
            "Trajectory saving and analysis methods are being refactored for efficiency. "
            "This functionality is temporarily disabled. "
            "The save_dist system needs complete reimplementation. "
            "See SRtools.trajectory_analysis for future implementation."
        )
     
    def create_distributions(self, dist_years, method='kde', nvalues=40):
        """
        Create distributions of damage values for each year in dist_years.


        DEPRECATED: This method is temporarily disabled pending refactoring.
        """
        raise NotImplementedError(
            "Trajectory saving and analysis methods are being refactored for efficiency. "
            "This functionality is temporarily disabled. "
            "The save_dist system needs complete reimplementation. "
            "See SRtools.trajectory_analysis for future implementation."
        )

    # def plotHazard(self,smooth = True, linestyle='-',lw=0.6, ax=None,label='',kwargs={}):
    #     label = label
    #     if (smooth):
    #         t,h = self.hazard
    #     else:
    #         t,h = self.hazard_not_smoothed
    #     if ax is None:

    #         import matplotlib.pyplot as plt
    #         plt.plot(t,h,linewidth=lw,linestyle=linestyle,label = label)
    #         plt.legend()
    #     else:
    #         ax.plot(t,h,linewidth=lw,linestyle=linestyle,label = label,**kwargs)
    #     ax.legend()

    #     return   

    

    def plotCumHazard(self, linestyle='-',lw=0.6, ax=None,label='', scale =1):
        #plots the cumulative hazard.
        #scale scales time
        label = label

        t,h = self.cum_hazard
        t=t*scale
        if ax is None:

            import matplotlib.pyplot as plt
            plt.plot(t,h,linewidth=lw,linestyle=linestyle,label = label)
            plt.legend()
        else:

            ax.plot(t,h,linewidth=lw,linestyle=linestyle,label = label)
            ax.legend()
        return  

    def get_ind_at_age(self,age):
        return np.abs(self.t - age).argmin()
    
    def printParams(self):
        """Print the SR model parameters."""
        print(f'eta = {self.eta}, beta = {self.beta}, kappa = {self.kappa}, epsilon = {self.epsilon}, xc = {self.xc}')

    

# Utility functions moved to utils.py - see imports above

# trajectories_accelerator moved to legacy_classes.py - see imports above

@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta,beta,kappa,epsilon,xc,sdt,npeople,external_hazard = np.inf, time_step_multiplier = 1):
    death_times = []
    events = []

    for i in range(npeople):
        x = 0
        for j in range(s-1):
            x = x + dt * (eta * t[j] - beta * x / (x + kappa)) + sdt * np.sqrt(2 * epsilon) * np.random.normal()
            x = max(x, 0)
            if x >= xc:
                death_times.append(t[j])
                events.append(1)
                break

        else:
            # If we didn't die naturally, check external hazard
            if np.random.random() < 1 - np.exp(-external_hazard * (t[-1] - t[0])):
                death_times.append(t[-1])
                events.append(1)
            else:
                death_times.append(t[-1])
                events.append(0)

    return np.array(death_times), np.array(events)


@jit(nopython=jit_nopython, parallel=jit_parallel)
def death_times_accelerator2(s,dt,t,eta,beta,kappa,epsilon,xc,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):

    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    for i in range(npeople):
        x = 0
        for j in range(s-1):
            x = x + dt * (eta * t[j] - beta * x / (x + kappa)) + sdt * np.sqrt(2 * epsilon) * np.random.normal()
            x = max(x, 0)
            if x >= xc:
                death_times[i] = t[j]
                events[i] = 1
                break
            else:

                # If we didn't die naturally, check external hazard
                if np.random.random() < 1 - np.exp(-external_hazard * (t[-1] - t[0])):
                    death_times[i] = t[-1]
                    events[i] = 1
                else:
                    death_times[i] = t[-1]
                    events[i] = 0
        return death_times, events

def death_times_euler_brownian_bridge(s, dt, t, eta, beta, kappa, epsilon, xc, sdt, npeople,

                                     external_hazard = np.inf, time_step_multiplier = 1):
    """Euler method with Brownian bridge crossing detection"""
    death_times = []
    events = []

    
    for i in range(npeople):
        x = 0
        died = False
        
        for j in range(s-1):
            # Euler step
            drift = dt * (eta * t[j] - beta * x / (x + kappa))
            diffusion = sdt * np.sqrt(2 * epsilon) * np.random.normal()
            x_new = x + drift + diffusion
            x_new = max(x_new, 0)
            
            # Check if we crossed the critical threshold
            if x_new >= xc:

                # Use Brownian bridge to find exact crossing time
                if x < xc:  # We crossed from below
                    # Linear interpolation for crossing time
                    alpha = (xc - x) / (x_new - x) if x_new != x else 0.5
                    crossing_time = t[j] + alpha * dt
                    death_times.append(crossing_time)
                    events.append(1)
                    died = True
                    break
                x = x_new

        
        if not died:
            # Check external hazard
            if np.random.random() < 1 - np.exp(-external_hazard * (t[-1] - t[0])):
                death_times.append(t[-1])
            events.append(1)
        else:
            death_times.append(t[-1])
            events.append(0)
    

    return np.array(death_times), np.array(events)

def death_times_euler_brownian_bridge_parallel(s, dt, t, eta, beta, kappa, epsilon, xc, sdt, npeople,

                                              external_hazard = np.inf, time_step_multiplier = 1):
    """Parallel version of Euler method with Brownian bridge crossing detection"""
    def simulate_single(i):
        x = 0
        died = False
        
        for j in range(s-1):
            # Euler step
            drift = dt * (eta * t[j] - beta * x / (x + kappa))
            diffusion = sdt * np.sqrt(2 * epsilon) * np.random.normal()
            x_new = x + drift + diffusion
            x_new = max(x_new, 0)
            
            # Check if we crossed the critical threshold
            if x_new >= xc:
                # Use Brownian bridge to find exact crossing time
                if x < xc:  # We crossed from below
                    # Linear interpolation for crossing time
                    alpha = (xc - x) / (x_new - x) if x_new != x else 0.5
                    crossing_time = t[j] + alpha * dt
                    return crossing_time, 1
            x = x_new
        
        # Check external hazard
        if np.random.random() < 1 - np.exp(-external_hazard * (t[-1] - t[0])):
            return t[-1], 1
        else:
            return t[-1], 0
    
    results = Parallel(n_jobs=-1)(delayed(simulate_single)(i) for i in range(npeople))
    death_times, events = zip(*results)
    return np.array(death_times), np.array(events)

@jit(nopython=jit_nopython)
def median_accelerator(medians, tragectories,xc):
    for i in range(len(medians)):

        medians[i] = np.median(tragectories[i, :])
    return medians


# All distance function implementations removed - see distance_metrics.py for implementations

def agingStats(t, vals, val_critical=np.inf, nvalues=40, corr_threshold=0.5,calc_error_bars=False,confidence_level=0.95,n_resample =150,pdf_method='hist'):
    """
    Calculate aging statistics for a given time point.


    DEPRECATED: This function is temporarily disabled pending refactoring.
    """
    raise NotImplementedError(
        "Trajectory analysis methods are being refactored for efficiency. "
        "This functionality is temporarily disabled. "
        "The save_dist system needs complete reimplementation. "
        "See SRtools.trajectory_analysis for future implementation."
    )


def karin_params():
    print('Karin params\n','eta = 0.49275, beta = 54.75, kappa = 0.5, epsilon = 51.83, xc = 17')

def karin_mice_params():
    print('Karin mice params\n',f'eta = {0.084/365}, beta = {0.15}, kappa = 0.5, epsilon = 0.16, xc = 17')

def get_hazard_from_survival(t,survival):
    """
    utility function to calculate the hazard function from the survival function
    """
    #first index where the survival is 0
    ind = np.argmax(survival==0)
    ind = ind if ind>0 and ind<len(survival) else len(survival)-1
    t = t[:ind]
    if len(t)<=1:
        return np.array([0]),np.array([0])
    survival = survival[:ind]
    # mid_survival = (survival[1:]+survival[:-1])/2
    h = -(np.diff(survival)/np.diff(t))/survival[:-1]
    h = np.concatenate((h,[h[-1]]))
    return t,h



# Backward compatibility imports
from .legacy_classes import (
    SR_karin_human, SRNDK, SRND_Peckle, SR_GWTW,
    SR_karin_human_GWTW, SRNDK_GWTW, SRND_Peckle_GWTW,
    SR_envelope
)
from .distance_metrics import baysianDistance, baysian_dirichlet_distance, distance, ks_test, trim_to_range
from .utils import (
    gompetz_hazard, get_survival_from_hazard, get_dimless_groups,
    get_hazard_from_survival, karin_params, karin_mice_params
)

# Keep old name for backward compatibility with deprecation warning
# baysianDistance is imported from distance_metrics.py
