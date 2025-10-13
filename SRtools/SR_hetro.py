from . import SRmodellib_lifelines as srl
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from . import deathTimesDataSet as dtds
import os
from . import sr_mcmc as srmc
from . import SRmodellib as sr

jit_nopython = True

"""
After implementing your class, change sr_mcmc.model so it calls your class instead of the default one and uses your metric function.
"""


class SR_Hetro(srl.SR_lf):
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0, t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, method='brownian_bridge'):
        """
        If you want to add parameters to the __init__ method, you can do so here before the call to super().__init__.
        
        Parameters:
            method (str): Method to use for death times calculation. Options:
                - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
                - 'euler': Standard Euler method
        """
        self.eta_var = eta_var
        self.beta_var = beta_var
        self.kappa_var = kappa_var
        self.epsilon_var = epsilon_var
        self.xc_var = xc_var
        self.method = method
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, method=method)

    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        if self.method == 'brownian_bridge':
            if self.parallel:
                death_times, events = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        elif self.method == 'euler':
            if self.parallel:
                death_times, events = death_times_accelerator2(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_accelerator(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        else:
            # Default to brownian bridge if method not recognized
            if self.parallel:
                death_times, events = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)

        return np.array(death_times), np.array(events)
    


def getSrHetro(
    theta,
    n=25000,
    nsteps=6000,
    t_end=110,
    external_hazard=np.inf,
    time_step_multiplier=1,
    npeople=None,
    parallel=False,
    eta_var=0,
    beta_var=0,
    epsilon_var=0,
    xc_var=0.2,
    kappa_var=0,
    hetro=True,
    bandwidth=3,
    step_size=None,
    method='brownian_bridge'
    ):
    """
    Optionally specify step_size. If step_size is given, nsteps and time_step_multiplier are ignored and recalculated so that
    t_end/(nsteps*time_step_multiplier) = step_size. If nsteps*time_step_multiplier <= 6000, time_step_multiplier=1, else
    increase time_step_multiplier until nsteps <= 6000. Both nsteps and time_step_multiplier are integers.
    
    Parameters:
        method (str): Method to use for death times calculation. Options:
            - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
            - 'euler': Standard Euler method
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
            # Find smallest integer time_step_multiplier so that nsteps <= 6000
            time_step_multiplier = int(np.ceil(total_steps / 6000))
            nsteps = int(np.ceil(total_steps / time_step_multiplier))
            # Ensure both are at least 1
            time_step_multiplier = max(1, time_step_multiplier)
            nsteps = max(1, nsteps)

    sim = SR_Hetro(
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
        method=method
    )

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
    for l in range(npeople):
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
        for l in range(npeople):
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



# Euler with Brownian Bridge method
@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                     epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                     external_hazard=np.inf, time_step_multiplier=1):
    """
    Euler method with Brownian bridge crossing detection.
    This method uses the standard Euler scheme but adds Brownian bridge
    crossing probability tests to detect barrier crossings between time steps.
    """
    death_times = []
    events = []
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        crossed = False
        
        while j < s - 1 and not crossed:
            for _ in range(time_step_multiplier):
                # Standard Euler step
                drift = eta * t[j] - beta * x / (x + kappa)
                noise = sqrt_2epsilon * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test if not crossed directly
                if (x < xc) and (x_new < xc) and (x > 0*kappa):
                    dx1 = xc - x
                    dx2 = xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        # Brownian bridge crossing probability
                        # P = exp(-2 * (xc - x) * (xc - x_new) / (2 * epsilon * ndt))
                        var = 2.0 * epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= xc:
            events.append(1)
        else:
            events.append(0)
    
    return np.array(death_times), np.array(events)

# Parallel version of Euler with Brownian Bridge method
def death_times_euler_brownian_bridge_parallel(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                              external_hazard=np.inf, time_step_multiplier=1, n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge.
    Splits npeople into chunks and runs death_times_euler_brownian_bridge on each chunk in parallel.
    """
    from joblib import Parallel, delayed
    import numpy as np

    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier):
        # Call the numba-jitted function for this chunk
        return death_times_euler_brownian_bridge(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, time_step_multiplier
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
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )

    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    return death_times, events



