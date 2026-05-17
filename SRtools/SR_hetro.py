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
        Heterogeneous SR (Saturating Removal) model for mortality simulation.

        Simulates a population of individuals whose internal damage variable X(t)
        follows the SDE:  dX/dt = η·t − β·X/(X+κ) + √(2ε)·ξ.
        Death occurs when X crosses the threshold xc.  Population heterogeneity is
        modelled by drawing each individual's parameters from a normal distribution
        around the population mean (controlled by the *_var arguments).

        Inherits all survival-analysis methods from Dataset (getSurvival, plotSurvival,
        getHazard, getMedianLifetime, etc.).

        Parameters
        ----------
        eta : float
            Damage accumulation rate coefficient (units: damage/time²).
        beta : float
            Maximum repair/removal capacity (units: damage/time).
        kappa : float
            Half-saturation constant for repair (same units as damage X).
        epsilon : float
            Noise amplitude controlling stochastic damage fluctuations.
        xc : float
            Critical damage threshold; an individual dies when X > xc.
        npeople : int
            Number of individuals to simulate.
        nsteps : int
            Number of discrete time steps in [t_start, t_end].
        t_end : float
            End of simulation (in tscale units, e.g. years).
        eta_var : float, optional
            Fractional std of individual variation in eta (0 = homogeneous). Default 0.
        beta_var : float, optional
            Fractional std of individual variation in beta. Default 0.
        kappa_var : float, optional
            Fractional std of individual variation in kappa. Default 0.
        epsilon_var : float, optional
            Fractional std of individual variation in epsilon. Default 0.
        xc_var : float, optional
            Fractional std of individual variation in xc. Default 0.
        t_start : float, optional
            Simulation start time. Default 0.
        tscale : str, optional
            Time unit label ('years', 'days', 'weeks', etc.). Default 'years'.
        external_hazard : float, optional
            Constant background hazard rate (additional death channel).
            Default np.inf (no external mortality).
        time_step_multiplier : int, optional
            Sub-step factor for numerical accuracy; effective dt = dt/time_step_multiplier.
            Default 1.
        parallel : bool, optional
            Use joblib parallelism for the simulation loop. Default False.
        bandwidth : int, optional
            Smoothing bandwidth for the Nelson-Aalen hazard estimate. Default 3.
        method : str, optional
            Simulation algorithm:
            - ``'brownian_bridge'`` (default): Euler + Brownian bridge crossing test.
              More accurate for threshold crossings at coarse time steps.
            - ``'euler'``: Plain Euler scheme (faster, less accurate).
        """
        self.eta_var = eta_var
        self.beta_var = beta_var
        self.kappa_var = kappa_var
        self.epsilon_var = epsilon_var
        self.xc_var = xc_var
        self.method = method
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, method=method)

    def calc_death_times(self):
        """
        Simulate death times for the full population.

        Runs the Euler (or Brownian-bridge) stochastic simulation for every
        individual and records the time at which each one's damage X first
        exceeds xc.  Called automatically during ``__init__``.

        Override this method in subclasses to implement custom dynamics.

        Returns
        -------
        death_times : ndarray, shape (npeople,)
            Age at death (or at end of simulation if censored), in tscale units.
        events : ndarray of int, shape (npeople,)
            1 if the individual died (crossed xc), 0 if censored (reached t_end).
        """
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
    kappa=0.5,
    hetro=True,
    bandwidth=3,
    step_size=None,
    method='brownian_bridge',
    ):
    """
    Factory function for :class:`SR_Hetro` — the recommended way to create a simulation.

    Parameters
    ----------
    theta : array-like, shape (4,)
        ``[eta, beta, epsilon, xc]`` — SR model parameters:
        eta (damage rate), beta (repair capacity), epsilon (noise), xc (death threshold).
    n : int, optional
        Population size. Default 25000.
    nsteps : int, optional
        Number of simulation time steps. Default 6000.  Ignored if *step_size* is given.
    t_end : float, optional
        End of simulation in years. Default 110.
    external_hazard : float, optional
        Constant background hazard (additional death channel). Default np.inf (none).
    time_step_multiplier : int, optional
        Sub-step factor; effective dt = t_end / (nsteps * time_step_multiplier).
        Ignored if *step_size* is given. Default 1.
    npeople : int, optional
        Alias for *n*; overrides *n* if provided.
    parallel : bool, optional
        Use parallel computation. Default False.
    eta_var, beta_var, epsilon_var, xc_var, kappa_var : float, optional
        Fractional std of inter-individual variation in each parameter.
        Set to 0 for a homogeneous population. Default: xc_var=0.2, others 0.
    kappa : float, optional
        Half-saturation constant for repair (fixed, not fitted). Default 0.5.
    hetro : bool, optional
        If False, forces all *_var values to 0 (homogeneous model). Default True.
    bandwidth : int, optional
        Hazard smoothing bandwidth. Default 3.
    step_size : float, optional
        Target simulation time step (years).  When provided, *nsteps* and
        *time_step_multiplier* are recomputed so that
        ``t_end / (nsteps * time_step_multiplier) ≈ step_size`` with
        ``nsteps ≤ 6000``.
    method : str, optional
        Simulation algorithm: ``'brownian_bridge'`` (default) or ``'euler'``.

    Returns
    -------
    SR_Hetro
        Initialised simulation object.  Access results via
        ``sim.getSurvival()``, ``sim.plotSurvival()``, ``sim.getMedianLifetime()``, etc.

    Examples
    --------
    >>> theta = [0.05, 50, 50, 17]   # [eta, beta, epsilon, xc]
    >>> sim = getSrHetro(theta, n=10000, t_end=110)
    >>> sim.plotSurvival()

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
        kappa=kappa,
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
    Evaluate SR model log-likelihood against observed mortality data.

    This is the default ``model_func`` passed to :func:`~SRtools.sr_mcmc.getSampler`.
    It simulates the SR model with the given parameters and returns the log-likelihood
    of the data under that simulation.

    Parameters
    ----------
    theta : array-like
        Free parameters parsed by :func:`~SRtools.sr_mcmc.parse_theta`.
        Typically ``[eta, beta, epsilon, xc]``; fixed parameters are supplied via
        *set_params*.
    n : int
        Simulation population size.
    nsteps : int
        Number of simulation time steps.
    t_end : float
        Simulation end time (years).
    dataSet : Dataset
        Observed mortality data to fit.
    sim : SR_Hetro, optional
        Pre-computed simulation; if None it is created from *theta*. Default None.
    metric : str, optional
        Distance / likelihood metric passed to :func:`~SRtools.SRmodellib.distance`.
        Default ``'baysian'``.
    time_range : tuple, optional
        ``(t_min, t_max)`` window for likelihood evaluation. Default None (full range).
    time_step_multiplier : int, optional
        Sub-step factor for numerical stability. Default 1.
    parallel : bool, optional
        Use parallel simulation. Default False.
    dt : float, optional
        Time bin width for likelihood calculation. Default 1.
    set_params : dict, optional
        Fixed parameters, e.g. ``{'external_hazard': 0.01}``. Default None.
    debug : bool, optional
        Print diagnostics when returning -inf. Default False.
    kwargs : ignored

    Returns
    -------
    float
        Log-likelihood.  Returns ``-np.inf`` if:
        - ``1/beta < time_step_size`` (numerical instability), or
        - the likelihood evaluates to NaN.
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

