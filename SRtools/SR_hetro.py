from SRtools import SRmodellib_lifelines as srl
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from SRtools import deathTimesDataSet as dtds
import os
from SRtools import sr_mcmc as srmc
from SRtools import SRmodellib as sr

jit_nopython = True

"""
After implementing your class, change sr_mcmc.model so it calls your class instead of the default one and uses your metric function.
"""


class SR_Hetro(srl.SR_lf):
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0, t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, heun=False):
        """
        If you want to add parameters to the __init__ method, you can do so here before the call to super().__init__.
        """
        self.eta_var = eta_var
        self.beta_var = beta_var
        self.kappa_var = kappa_var
        self.epsilon_var = epsilon_var
        self.xc_var = xc_var
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, heun)




    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        if self.parallel:
            death_times, events = death_times_accelerator2(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)
        else:
            death_times, events = death_times_accelerator(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)

        return np.array(death_times), np.array(events)
    


def getSrHetro(theta, n=25000,nsteps=6000,t_end=110, external_hazard = np.inf,time_step_multiplier =1, npeople =None, parallel = False,eta_var = 0, beta_var = 0, epsilon_var = 0, xc_var = 0.2, kappa_var = 0, hetro = True, bandwidth = 3):

    if npeople is not None:
        n = npeople
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    if not hetro:
        eta_var =0
        beta_var =0
        epsilon_var =0
        xc_var =0
        kappa_var =0

    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf
    
    sim = SR_Hetro(eta=eta,beta=beta,epsilon=epsilon,xc=xc,
                   eta_var=eta_var,beta_var=beta_var,kappa_var=kappa_var,epsilon_var=epsilon_var,xc_var=xc_var,
                   kappa=0.5,npeople=n,nsteps=nsteps,t_end=t_end,external_hazard=external_hazard, time_step_multiplier=time_step_multiplier,
                     parallel=parallel, bandwidth=bandwidth)
    
    return sim


def model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysian', time_range=None, time_step_multiplier = 1,parallel = False, dt=1, set_params=None,debug=False, kwargs=None):
    """
    The function accepts the parameters of the SR model and returns score according to the metric.
    """
    if set_params is None:
        set_params = {}
    # parse parameters
    pv = srmc.parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    theta_sr = np.array([eta, beta, epsilon, xc])
    time_step_size = t_end/(nsteps*time_step_multiplier)
    if 1/beta < time_step_size:
        if debug: print("Beta is too large for the time step size")
        return -np.inf
    sim = getSrHetro(theta_sr, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel) if sim is None else sim
    
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.any(np.isnan(tprob)):
        if debug: print("Nan in tprob")
        return -np.inf

    return tprob



#method without parallelization (for cluster usage)
@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta0,eta_var,beta0,beta_var,kappa0,kappa_var,epsilon0,epsilon_var,xc0,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):
    death_times = []
    events = []
    for i in range(npeople):
        x=0
        j=0
        ndt = dt/time_step_multiplier
        nsdt = sdt/np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard)*ndt
        eta = eta0*np.random.normal(loc = 1,scale = eta_var)
        beta = beta0 * np.random.normal(loc = 1, scale = beta_var)
        kappa = kappa0 * np.random.normal(loc = 1, scale = kappa_var)
        epsilon = epsilon0 * np.random.normal(loc = 1, scale = epsilon_var)
        xc = xc0 * np.random.normal(loc = 1, scale = xc_var)
        while j in range(s-1) and x<xc:
            for i in range(time_step_multiplier):
                noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                x = np.maximum(x, 0)
                if np.random.uniform(0,1)<chance_to_die_externally:
                    x = xc
                if x>=xc:
                    break
            j+=1
        if x>=xc:
            death_times.append(j*dt)
            events.append(1)
        else:
            death_times.append(j*dt)
            events.append(0)

    return death_times, events

##method with parallelization (run on your computer)
def death_times_accelerator2(s,dt,t,eta,eta_var,beta,beta_var,kappa,kappa_var,epsilon,epsilon_var,xc,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta0,eta_var,beta0,beta_var,kappa0,kappa_var,epsilon0,epsilon_var,xc0,xc_var, sdt, external_hazard,time_step_multiplier):
        death_times = []
        events =[]
        for i in range(npeople):
            died = False
            x = 0
            j = 0
            ndt = dt/time_step_multiplier
            nsdt = np.sqrt(ndt)
            chance_to_die_externally = np.exp(-external_hazard)*ndt
            eta = eta0*np.random.normal(loc = 1,scale = eta_var)
            beta = beta0 * np.random.normal(loc = 1, scale = beta_var)
            kappa = kappa0 * np.random.normal(loc = 1, scale = kappa_var)
            epsilon = epsilon0 * np.random.normal(loc = 1, scale = epsilon_var)
            xc = xc0 * np.random.normal(loc = 1, scale = xc_var)
            while j in range(s - 1) and x < xc and not died:
                for i in range(time_step_multiplier):
                    noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                    x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                    x = np.maximum(x, 0)
                    if np.random.uniform(0,1)<chance_to_die_externally:
                        x = xc
                    if x>=xc:
                        died = True
                j += 1
            if died:
                death_times.append(j * dt)
                events.append(1)
            else:
                death_times.append(j * dt)
                events.append(0)
        return death_times, events

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta,eta_var, beta,beta_var, kappa,kappa_var, epsilon,epsilon_var, xc,xc_var, sdt, external_hazard,time_step_multiplier
    ) for _ in range(n_jobs))

    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    return death_times, events