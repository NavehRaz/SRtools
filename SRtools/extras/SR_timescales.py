from SRtools import SR_hetro as srh
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


class SR_timescales(srh.SR_Hetro):
    def __init__(self, t_eta,t_epsilon, npeople, nsteps, t_end, 
                 t_c=0, kappa=0.5, xc=100,
                 eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0,
                   t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, method='brownian_bridge'):
        """
        If you want to add parameters to the __init__ method, you can do so here before the call to super().__init__. if you add beta2 as aparameter for example then
        add self.beta2=beta2 here.
        """
        self.t_eta = t_eta
        self.t_epsilon = t_epsilon
        self.t_c = t_c

        eta = xc/(t_eta**2)
        epsilon = xc**2/t_epsilon
        beta = t_c*eta
        #this is the call to my class, do not modify it. also, do not earase any of the parameters I cal here unless you give them a default value or somehting
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var, beta_var, kappa_var, epsilon_var, xc_var, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, method)




    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        if self.parallel:
            death_times, events = death_times_euler_brownian_bridge_parallel(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)
        else:
            death_times, events = death_times_euler_brownian_bridge(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)

        return np.array(death_times), np.array(events)
    

def getSRTS(theta, n=25000, nsteps=6000, t_end=110, external_hazard=np.inf, time_step_multiplier=1, npeople=None, parallel=False,
            eta_var=0, beta_var=0, epsilon_var=0, xc_var=0, kappa_var=0, hetro=False, bandwidth=3, xc=100, kappa=0.5):
    """
    Returns an SR_timescales object.
    theta is interpreted as follows:
        - len(theta) == 2: [t_eta, t_epsilon]
        - len(theta) == 3: [t_eta, t_c, t_epsilon]
    xc and kappa can be specified, and default to 100 and 0.5.
    """
    if npeople is not None:
        n = npeople

    # Parse theta based on length
    if len(theta) == 2:
        t_eta = theta[0]
        t_epsilon = theta[1]
        t_c = 0
    elif len(theta) == 3:
        t_eta = theta[0]
        t_c = theta[1]
        t_epsilon = theta[2]
    else:
        raise ValueError("theta must have length 2 ([t_eta, t_epsilon]) or 3 ([t_eta, t_c, t_epsilon])")

    # Disable heterogeneity if hetro is False
    if not hetro:
        eta_var = 0
        beta_var = 0
        epsilon_var = 0
        xc_var = 0
        kappa_var = 0

    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf

    sim = SR_timescales(
        t_eta=t_eta,
        t_epsilon=t_epsilon,
        npeople=n,
        nsteps=nsteps,
        t_end=t_end,
        t_c=t_c,
        kappa=kappa,
        xc=xc,
        eta_var=eta_var,
        beta_var=beta_var,
        kappa_var=kappa_var,
        epsilon_var=epsilon_var,
        xc_var=xc_var,
        t_start=0,
        tscale='years',
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
        bandwidth=bandwidth
    )

    return sim




#example model
def example_model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysian', time_range=None, time_step_multiplier = 1,parallel = False, dt=1, set_params=None,debug=False, kwargs=None):
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
    sim = srh.getSrHetro(theta_sr, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel) if sim is None else sim
    
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.any(np.isnan(tprob)):
        if debug: print("Nan in tprob")
        return -np.inf

    return tprob


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