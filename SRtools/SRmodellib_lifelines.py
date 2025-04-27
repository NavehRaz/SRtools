import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numba import jit
from scipy.stats import gaussian_kde
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from SRtools import SRmodellib as sr



class SR_lf(sr.SR):
    def __init__(self, eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years',external_hazard=np.inf,time_step_multiplier=1, parallel =False, bandwidth=3,heun=False):
        self.bandwidth = bandwidth
        super().__init__(eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years', memory_efficient =False, natural_units = True, smoothing = 20,
                      boundary = 'sticking',
                        save_dist = False , dist_years =np.linspace(0,100,101), dist_method = 'hist', dist_nvalues = 40, y_gamma =None, death_times_method = 2, external_hazard=external_hazard, time_step_multiplier = time_step_multiplier, parallel=parallel, heun=heun)
        
    
    def calc_survival_and_hazard(self, death_times_method=2):
        T, E = self.calc_death_times()
        self.death_times = T
        self.events = E
        # E = np.ones_like(T)
        if len(T)<self.npeople:
            Tn = np.zeros(self.npeople)
            Tn[:len(T)] = T
            Tn[len(T):] = self.t_end
            En = np.zeros(self.npeople)
            En[:len(T)] = E
            T = Tn
            E = En
        
        kmf = KaplanMeierFitter().fit(T, E)
        self.survival = kmf.timeline, np.array(kmf.survival_function_.values)[:,0]
        self.kmf = kmf
        self.kmf_confidence_interval = [np.array(kmf.confidence_interval_['KM_estimate_lower_0.95'].values), np.array(kmf.confidence_interval_['KM_estimate_upper_0.95'].values)]
        self.median_lifetime = kmf.median_survival_time_


        naf = NelsonAalenFitter().fit(T, event_observed=E)
        self.hazard = naf.timeline, np.array(naf.smoothed_hazard_(bandwidth=self.bandwidth).values)[:,0]
        self.naf = naf

        return self.survival
    
    def plotSurvival(self, ax=None, time_range=None, **kwargs):
        if time_range is not None:
            ax= super().plotSurvival(ax=ax, time_range=time_range, **kwargs)
        else:
            if ax is None:
                fig, ax = plt.subplots()
            self.kmf.plot_survival_function(ax=ax, **kwargs)
        return ax
    
    def plotHazard(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        self.naf.plot_hazard(ax=ax, bandwidth=self.bandwidth, **kwargs)
        return ax
    
    def get_median_lifetime(self):
        return self.kmf.median_survival_time_
    
    def getConfidenceInterval(self):
        return self.kmf_confidence_interval
    
    def getMedianLifetime(self):
        return self.get_median_lifetime()
    
    def getMedianLifetimeCI(self):
        """
        This function returns the 95% confidence interval of the median lifetime of the dataset.
        """
        CI = self.kmf_confidence_interval
        idx = np.argmin(np.abs(self.survival[1]-0.5))
        mCI = [CI[0][idx],CI[1][idx]]
        return mCI
    
    def survivalAtTimes(self, times):
        """
        This function returns the survival probability at a given time using the kmf.
        The functions takes the pd.Series of the kmf and the time at which the survival probability is to be calculated and 
        converts it to a numpy array or a float.
        """
        values = self.kmf.survival_function_at_times(times).values
        if len(values) == 1:
            return values[0]
        else:
            return values
 
    

class SR_lf_karin_human(SR_lf):
    def __init__(self, eta=1, beta=1, kappa=1, epsilon=1, xc=1,
                  npeople=5000, nsteps=5000, t_end=115, t_start = 0,
                    tscale = 'years',bandwidth=3,external_hazard=np.inf, time_step_multiplier=1, parallel =False, heun=False):
        karin_vals = [0.49275,54.75,0.5,51.83,17]
        super().__init__(eta*karin_vals[0], beta*karin_vals[1], kappa*karin_vals[2], epsilon*karin_vals[3], xc*karin_vals[4],
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years',bandwidth=bandwidth,external_hazard=external_hazard, time_step_multiplier=time_step_multiplier, parallel=parallel, heun=heun)
        

def getSr(theta, n=25000,nsteps=4500,t_end=110, external_hazard = np.inf, time_step_multiplier=1,parallel = False):

    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    sim = SR_lf(eta=eta,beta=beta,epsilon=epsilon,xc=xc,kappa=0.5,npeople=n,nsteps=nsteps,t_end=t_end,external_hazard=external_hazard,time_step_multiplier=time_step_multiplier,parallel=parallel)
    return sim