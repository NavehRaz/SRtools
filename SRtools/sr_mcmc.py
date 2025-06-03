from SRtools import SRmodellib as sr
import numpy as np
from SRtools import SRmodellib_lifelines as srl
import emcee # a MCMC sampler
import time
import pandas as pd






def parse_theta(theta, set_params):
    """
    Parses the parameter array 'theta' according to the dictionary 'set_params'.
    Any parameter found in 'set_params' is skipped in 'theta' and taken directly from 'set_params'.
    Default parameter order is [eta, beta, epsilon, xc, external_hazard].
    Any remaining items from 'theta' are returned in param_values['extra'].
    """
    param_names = ['eta', 'beta', 'epsilon', 'xc', 'external_hazard']
    param_values = {}
    idx = 0
    for name in param_names:
        if name in set_params:
            # parameter given in set_params, skip reading it from theta
            param_values[name] = set_params[name]
        else:
            # read next position of theta array if available
            if idx < len(theta):
                param_values[name] = theta[idx]
                idx += 1
            else:
                param_values[name] = None
    # store any extra entries in 'extra'
    if idx < len(theta):
        param_values['extra'] = theta[idx:]
    return param_values

def parse_theta_trans(theta_trans, set_params):
    """
    Parses the parameter array 'theta' according to the dictionary 'set_params'.
    Any parameter found in 'set_params' is skipped in 'theta' and taken directly from 'set_params'.
    Default parameter order is [xc_eta, eta_beta, xc2_epsilon, xc, external_hazard].
    Any remaining items from 'theta' are returned in param_values['extra'].
    """
    param_names = ['xc_eta', 'eta_beta', 'xc2_epsilon', 'xc', 'external_hazard']
    param_values = {}
    idx = 0
    for name in param_names:
        if name in set_params:
            # parameter given in set_params, skip reading it from theta
            param_values[name] = set_params[name]
        else:
            # read next position of theta array if available
            if idx < len(theta_trans):
                param_values[name] = theta_trans[idx]
                idx += 1
            else:
                param_values[name] = None
    # store any extra entries in 'extra'
    param_values['extra'] = theta_trans[idx:]
    return param_values

def model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysian', time_range=None, time_step_multiplier = 1,parallel = False, dt=1, set_params=None, kwargs=None):
    """
    The function accepts the parameters of the SR model and returns score according to the metric.
    """
    if set_params is None:
        set_params = {}
    # parse parameters
    pv = parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    theta_sr = np.array([eta, beta, epsilon, xc])
    time_step_size = t_end/(nsteps*time_step_multiplier)
    if 1/beta < time_step_size:
        return -np.inf
    sim = getSr(theta_sr, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel) if sim is None else sim
    
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.any(np.isnan(tprob)):
        return -np.inf

    return tprob



def lnlike(theta , n, nsteps, t_end, dataSet, metric = 'baysian', time_range=None, time_step_multiplier = 1, sim=None, dt=1, set_params=None,model_func=model, kwargs=None):
    """
    The likelihood function for the MCMC sampler.
    """
    if set_params is None:
        set_params = {}
    
    LnLike = model_func(theta, n, nsteps, t_end, dataSet,sim=sim, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params, kwargs=kwargs)
    if metric =='survival':
        LnLike =1/LnLike
    return LnLike

def lnlikeTransformed(theta_trans , n, nsteps, t_end, dataSet, metric = 'baysian', time_range=None, time_step_multiplier = 1, sim=None, dt=1, set_params=None,model_func=model, kwargs=None):
    """
    The likelihood function for the MCMC sampler.
    """
    if set_params is None:
        set_params = {}
    eta = theta_trans[3]/theta_trans[0]
    beta=theta_trans[1]*eta
    epsilon =theta_trans[3]**2/theta_trans[2]
    xc = theta_trans[3]
    theta = np.array([eta,beta,epsilon,xc])
    
    LnLike = model_func(theta, n, nsteps, t_end, dataSet,sim=sim, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params, kwargs=kwargs)
    if metric =='survival':
        LnLike =1/LnLike
    return LnLike

def transform(theta, set_params={}):
    pv = parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    xc_eta = xc/eta
    beta_eta = beta/eta
    xc2_epsilon = xc**2/epsilon
    if 'extra' in pv:
        return np.array([xc_eta, beta_eta, xc2_epsilon,xc]+ pv['extra'].tolist())
    else:
        return np.array([xc_eta, beta_eta, xc2_epsilon,xc])

def inv_transform(theta_trans, set_params={}):
    pv = parse_theta_trans(theta_trans, set_params)
    eta = pv['xc'] / pv['xc_eta']
    beta = pv['eta_beta'] * eta
    epsilon = pv['xc']**2 / pv['xc2_epsilon']
    xc = pv['xc']
    theta = np.array([eta, beta, epsilon, xc])
    if 'extra' in pv:
        return np.concatenate([theta, pv['extra']])
    else:
        return theta

def lnprior(theta,prior):
    """
    The prior function for the MCMC sampler.
    """
    #check theta is same length as prior
    if len(theta) != len(prior):
        raise ValueError("The length of theta should be the same as the length of the prior.")

    
    for i, (param, bounds) in enumerate(zip(theta, prior)):
        if not (bounds[0] < param < bounds[1]):
            return -np.inf
    return 0.0
    


def lnprob(theta , n, nsteps, t_end, dataSet, metric = 'survival', time_range=None, time_step_multiplier = 1,prior = None, dt=1, set_params=None,model_func=model, kwargs=None):
    """
    The posterior function for the MCMC sampler.
    """
    if set_params is None:
        set_params = {}
    lp = lnprior(theta, prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta , n, nsteps, t_end, dataSet=dataSet, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt,model_func=model_func, set_params=set_params, kwargs=kwargs)

def lnprobTransformed(theta_trans , n, nsteps, t_end, dataSet, metric = 'survival', time_range=None, time_step_multiplier = 1,prior = None, dt=1, set_params=None,model_func=model, kwargs=None):
    """
    The posterior function for the MCMC sampler.
    """
    if set_params is None:
        set_params = {}
    lp = lnprior(theta_trans, prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikeTransformed(theta_trans , n, nsteps, t_end, dataSet=dataSet, metric=metric, time_range=time_range, time_step_multiplier = time_step_multiplier, dt=dt, set_params=set_params,model_func=model_func, kwargs=kwargs)

def draw_param(bins, log_sapce = True):
    """
    Draw a random parameter from the bins. The bins are a list of lists of parameter bins.
    First, we randomly choose the bin, then we randomly choose the parameter from the bin with a uniform distribution (same bin index for all parameters).

    Parameters:
    - bins (list): A list of lists of parameter bins.

    Returns:
    - theta (ndarray): An array of the parameters.
    """
    bin_index = np.random.randint(0, len(bins[0]))
    if log_sapce:
        theta = np.array([np.exp(np.random.uniform(np.log(bins[i][bin_index][0]), np.log(bins[i][bin_index][1]))) for i in range(len(bins))])
    else:
        theta = np.array([np.random.uniform(bins[i][bin_index][0], bins[i][bin_index][1]) for i in range(len(bins))])
    return theta

#fix this function so that it take the variations as a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements. the usage of specific var_eta, varbeta etc should be dropped and the variations should be used instead, the function should return a list of ndims lists of n_bins lists of 2 elements.
def get_bins_from_seed(seed, ndims=4, variations=[0.7, 1.3]):
    """
    Get the bins from the seed theta. The bins are the seed multiplied by the variations.
    If variations is a list of 2 elements, the same variations are applied to all parameters.
    If variations is a list of ndims lists of 2 elements, each parameter has its own variations.
    If variations is a list of ndims lists of n_bins lists of 2 elements, each parameter has its own bins.

    Returns:
    - bins (list): A list of ndims lists of n_bins lists of 2 elements.
    """
    if len(variations) == 2:
        variations = [variations] * ndims
    elif len(variations) != ndims:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")

    bins = []
    lengths  = []
    for i in range(ndims):
        if len(variations[i]) == 2 and isinstance(variations[i][0], (int, float)):
            bins.append([[seed[i] * variations[i][0], seed[i] * variations[i][1]]])
        else:
            bins.append([[seed[i] * var[0], seed[i] * var[1]] for var in variations[i]])
            lengths.append(len(variations[i]))
    if len(set(lengths)) > 1 and len(lengths) > 1:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")
    if len(lengths) > 0 and len(lengths)< ndims:
        raise ValueError("The variations should be a list of 2 elements or a list of ndims lists of 2 elements or a list of ndims lists of n_bins lists of 2 elements.")

    return bins

   

def getSampler(nwalkers, num_mcmc_steps, dataSet ,seed =None, npeople=10000, nsteps=5000, t_end=None, ndim = 4,
                bins =None, variations = [0.7, 1.3], draw_params_in_log_space =True, prior_generator = None,
                back_end_file= None, metric = 'baysian', time_range=None, time_step_multiplier = 1, prior = None, 
                restartFromBackEnd = False, progress = False, transformed =False, dt=1, set_params=None ,model_func =model, **kwargs):
    """
    The function to get the MCMC sampler.
    """
    ####THIS IS FOR BACKWARDS COMPATIBILITY.##### 
    eta_bins = kwargs.get('eta_bins', None)
    beta_bins = kwargs.get('beta_bins', None)
    epsilon_bins = kwargs.get('epsilon_bins', None)
    xc_bins = kwargs.get('xc_bins', None)
    
    if prior_generator is not None:
        pass
    elif bins is None and all(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        bins = [eta_bins, beta_bins, epsilon_bins, xc_bins]
    elif bins is None and any(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        raise ValueError("Either all bins should be None or all of them specified.")
    ###########################################

    #CHANGE THIS AFTER YOU SORT VARIABLE LENGTH THETA
    if set_params is None and ndim == 4:
        set_params = {'external_hazard': dataSet.external_hazard}
        
    if prior_generator is None:
        #check if only some of the bins are specified and raise an exception if so.
        if bins is not None and len(bins) != ndim:
            raise ValueError("Either all bins should be specified or None of them.")

        if bins is None and seed is None:
            raise ValueError("Either the seed or the bins should be specified.")
        
        if bins is None:
            bins = get_bins_from_seed(seed, variations = variations)

    if t_end is None:
        t_end = dataSet.t_end

    #check if the prior is specified and generate a default prior if not.
    #  If the prior is a scalar a default prior is generated using it, otherwise 10 is used. 
    if prior is None:
        prior = 10
    if prior_generator is not None:
        if type(prior) is float or type(prior) is int:
            prior = prior_generator.getBounds(expansion_factor=prior)
        else:
            prior = prior_generator.getBounds()
    elif  type(prior) is float or type(prior) is int:
        v= prior
        prior = [[np.min(bins[i])/v,np.max(bins[i])*v] for i in range(ndim)]
    elif len(prior) != ndim:
        raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
    else:
        for i,p in enumerate(prior):
            if type(p) is float or type(p) is int:
                prior[i] = [np.min(bins[i])/p,np.max(bins[i])*p]
            elif len(p) <=2:
                prior[i] = [np.min(bins[i])/p[0],np.max(bins[i])*p[-1]]
            elif len(p) >2:
                raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")

    

    args = [ npeople, nsteps, t_end, dataSet, metric, time_range, time_step_multiplier, prior, dt, set_params,model_func,kwargs]
    # Set the initial positions of the walkers
    if prior_generator is None:
        pos = [draw_param(bins=bins,log_sapce=draw_params_in_log_space) for i in range(nwalkers)]
    else:
        pos = prior_generator.sample(n_samples=nwalkers)
    
    
    if transformed:
        lp = lnprobTransformed
    else:
        lp = lnprob

    start=time.time()
    if back_end_file is not None and not restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args,backend=backend)
        sampler.run_mcmc(pos, num_mcmc_steps, progress=progress)
    elif back_end_file is not None and restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp,backend=backend, args=args)
        pos, prob, state = sampler.run_mcmc(None, num_mcmc_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args)
        sampler.run_mcmc(pos, num_mcmc_steps,progress='notebook')
    end = time.time()
    print("Time elapsed: ", end-start)
    return sampler


def getSamplerAutoCorrMon(nwalkers, num_mcmc_steps, dataSet, seed=None, npeople=10000, nsteps=5000, t_end=None, ndim=4,
                          bins=None, variations=[0.7, 1.3], draw_params_in_log_space=True,
                          back_end_file=None, metric='baysian', time_range=None, time_step_multiplier=1, prior=None,
                          restartFromBackEnd=False, progress=False, plot_correlations=False, transformed=False, dt=1,
                          set_params=None, model_func=model, **kwargs):
    """
    The function to get the MCMC sampler with autocorrelation monitoring.
    """
    import matplotlib.pyplot as plt

    ####THIS IS FOR BACKWARDS COMPATIBILITY.##### 
    eta_bins = kwargs.get('eta_bins', None)
    beta_bins = kwargs.get('beta_bins', None)
    epsilon_bins = kwargs.get('epsilon_bins', None)
    xc_bins = kwargs.get('xc_bins', None)

    if bins is None and all(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        bins = [eta_bins, beta_bins, epsilon_bins, xc_bins]
    elif bins is None and any(bin is not None for bin in [eta_bins, beta_bins, epsilon_bins, xc_bins]):
        raise ValueError("Either all bins should be None or all of them specified.")
    ###########################################

    if set_params is None and ndim == 4:
        set_params = {'external_hazard': dataSet.external_hazard}

    if bins is None and seed is None:
        raise ValueError("Either the seed or the bins should be specified.")

    if bins is None:
        bins = get_bins_from_seed(seed, variations=variations)

    if t_end is None:
        t_end = dataSet.t_end

    if prior is None:
        prior = 10
    if type(prior) is float or type(prior) is int:
        v = prior
        prior = [[np.min(bins[i]) / v, np.max(bins[i]) * v] for i in range(ndim)]
    elif len(prior) != ndim:
        raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
    else:
        for i, p in enumerate(prior):
            if len(p) > 2:
                raise ValueError("The prior should be a scalar or a list of ndim pairs [[eta_min,eta_max],[beta_min,beta_max]...].")
            elif type(p) is float or type(p) is int:
                prior[i] = [np.min(bins[i]) / p, np.max(bins[i]) * p]

    args = [npeople, nsteps, t_end, dataSet, metric, time_range, time_step_multiplier, prior, dt, set_params, model_func, kwargs]
    pos = [draw_param(bins=bins, log_sapce=draw_params_in_log_space) for i in range(nwalkers)]

    if transformed:
        lp = lnprobTransformed
    else:
        lp = lnprob

    start = time.time()
    if back_end_file is not None and not restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args, backend=backend)
    elif back_end_file is not None and restartFromBackEnd:
        backend = emcee.backends.HDFBackend(back_end_file)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, backend=backend, args=args)
        pos, prob, state = sampler.run_mcmc(None, num_mcmc_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, args=args)

    index = 0
    niters = np.empty(num_mcmc_steps // 100)
    autocorr = np.empty(num_mcmc_steps // 100)
    old_tau = np.inf
    if plot_correlations:
        fig, ax = plt.subplots()

    for sample in sampler.sample(pos, iterations=num_mcmc_steps, progress=progress):
        if sampler.iteration % 100:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        niters[index] = sampler.iteration
        index += 1

        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if plot_correlations:
            ax.cla()
            ax.plot(niters[:index], autocorr[:index])
            ax.plot(niters[:index], niters[:index] / 100, linestyle='dashed', color='gray')
            ax.set_xlabel('iteration')
            ax.set_ylabel(r'autocorrelation time estimate')

        if converged:
            break
        old_tau = tau

    end = time.time()
    print("Time elapsed: ", end - start)
    return sampler, autocorr, niters, index



def loadSamples(back_end_file,flat = True, thin = 1, discard = 0):
    backend = emcee.backends.HDFBackend(back_end_file)
    samples = backend.get_chain(flat = flat, thin =thin, discard = discard)
    lnprobs = backend.get_log_prob(flat = flat,thin =thin, discard = discard)
    return samples,lnprobs


def loadSamplesFromDir(dirs,best =True,flat = True, n_per_file = 800, thin = 1, discard = 0, debug = False):
    """
    Load the samples from the directory/s. Loads the samples from all the h5 files in the directory.
    And returns a concatenated array of the samples and the log probabilities.
    """
    import os
    samples = []
    lnprobs = []
    if type(dirs) is not list:
        dirs = [dirs]
    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith(".h5"):
                try:
                    samples_,lnprobs_ = loadSamples(os.path.join(dir,file),flat = flat)
                except:
                    print("Error loading file: ",file)
                    if debug:
                        import traceback
                        print("Error details:")
                        traceback.print_exc()
                        continue
                # If we want the best thetas, we only want the 100 thetas with the highest log probability.
                if best:
                    idx = np.argsort(lnprobs_)[-n_per_file:]
                    samples_ = samples_[idx]
                    lnprobs_ = lnprobs_[idx]
                samples.append(samples_)
                lnprobs.append(lnprobs_)
    if flat:
        samples = np.concatenate(samples)
        lnprobs = np.concatenate(lnprobs)
    return samples,lnprobs


def getSr(theta, n=25000,nsteps=6000,t_end=110, external_hazard = np.inf,time_step_multiplier =1, npeople =None, parallel = False):

    if npeople is not None:
        n = npeople
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    sim = srl.SR_lf(eta=eta,beta=beta,epsilon=epsilon,xc=xc,kappa=0.5,npeople=n,nsteps=nsteps,t_end=t_end,external_hazard=external_hazard, time_step_multiplier=time_step_multiplier, parallel=parallel)
    return sim

def save_best_thetas(samples,lnprobs,save_path,best_threshold=0.9):
    """
    Save the best thetas from the samples.
    """
    best_thetas = samples[np.where(lnprobs>best_threshold)]
    np.save(save_path,best_thetas)
    return best_thetas

def load_thetas(path):
    return np.load(path)

def karin_theta():
    return [0.49275,54.75,51.83,17]

def get_params_from_thetas(thetas):
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])
    return etas,betas,epsilons,xcs


def plot_thetas_and_probs(thetas, probs, marked_thetas = None,marked_probs =None, annotations = None, xscale = 'linear',yscale = 'log',threshold = None):
    import matplotlib.pyplot as plt
    #annotations should be the same length as marked_thetas
    
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    if threshold is not None:
        thetas = thetas[probs>threshold]
        probs = probs[probs>threshold]

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas/etas
    bxe = betas*xcs/epsilons
    Fx = betas**2/(etas*xcs)
    Dx = betas*epsilons/(etas*(xcs**2))

    td = xcs**2/epsilons
    bke = betas*kappa/epsilons
    Fk = betas**2/(etas*kappa)
    Dk = betas*epsilons/(etas*(kappa**2))

    s = (xcs**1.5)*(etas**0.5)/epsilons
    slope = etas*xcs/epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk

    marked_etas = np.array([theta[ETA] for theta in marked_thetas])
    marked_betas = np.array([theta[BETA] for theta in marked_thetas])
    marked_epsilons = np.array([theta[EPSILON] for theta in marked_thetas] )
    marked_xcs = np.array([theta[XC] for theta in marked_thetas])

    marked_t3 = marked_betas/marked_etas
    marked_bxe = marked_betas*marked_xcs/marked_epsilons
    marked_Fx = marked_betas**2/(marked_etas*marked_xcs)
    marked_Dx = marked_betas*marked_epsilons/(marked_etas*(marked_xcs**2))

    marked_td = marked_xcs**2/marked_epsilons
    marked_bke = marked_betas*kappa/marked_epsilons
    marked_Fk = marked_betas**2/(marked_etas*kappa)
    marked_Dk = marked_betas*marked_epsilons/(marked_etas*(kappa**2))

    marked_s = (marked_xcs**1.5)*(marked_etas**0.5)/marked_epsilons
    marked_slope = marked_etas*marked_xcs/marked_epsilons
    marked_Xceps = marked_xcs/marked_epsilons
    marked_Pk = marked_Fk/marked_Dk

    fig, axs = plt.subplots(5, 4,figsize=(35,30))
    arrow_color='red'
    headlength = 0.5
    headwidth = 0.25
    annot_fontsize = 16
    xytext_shift = 5
    for i in range(4):
        axs[0, i].hist([etas, betas, epsilons, xcs][i], bins=50)
        axs[0, i].set_xlabel(['ETA', 'BETA', 'EPSILON', 'XC'][i])
        axs[0, i].set_ylabel('Frequency')

        axs[1, i].scatter(probs, [etas, betas, epsilons, xcs][i], c=probs, cmap='viridis', s=1)
        axs[1, i].set_ylabel(['ETA', 'BETA', 'EPSILON', 'XC'][i])
        axs[1, i].set_xlabel('lnprobs')
        axs[1, i].set_yscale(yscale)
        axs[1, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[1, i].scatter(marked_probs, [marked_etas, marked_betas, marked_epsilons, marked_xcs][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[1, i].annotate(txt, (marked_probs[j], [marked_etas, marked_betas, marked_epsilons, marked_xcs][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                   fontsize=annot_fontsize)

        axs[2, i].scatter(probs, [t3, bxe, Fx, Dx][i], c=probs, cmap='viridis', s=1)
        axs[2, i].set_ylabel(['beta/eta', 'intercept=beta*xc/eps', 'Fx=beta^2/(eta*xc)', 'Dx=beta*eps/(eta*xc^2)'][i])
        axs[2, i].set_xlabel('lnprobs')
        axs[2, i].set_yscale(yscale)
        axs[2, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[2, i].scatter(marked_probs, [marked_t3, marked_bxe, marked_Fx, marked_Dx][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[2, i].annotate(txt, (marked_probs[j], [marked_t3, marked_bxe, marked_Fx, marked_Dx][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.01, headlength=headlength, headwidth=headwidth),
                                   fontsize=annot_fontsize)

        axs[3, i].scatter(probs, [td, bke, Fk, Dk][i], c=probs, cmap='viridis', s=1)
        axs[3, i].set_ylabel(['xc^2/eps', 'beta*kappa/eps', 'Fk=beta^2/(eta*kappa)', 'Dk=beta*eps/(eta*kappa^2)'][i])
        axs[3, i].set_xlabel('lnprobs')
        axs[3, i].set_yscale(yscale)
        axs[3, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[3, i].scatter(marked_probs, [marked_td, marked_bke, marked_Fk, marked_Dk][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[3, i].annotate(txt, (marked_probs[j], [marked_td, marked_bke, marked_Fk, marked_Dk][i][j]),
                                    xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                    fontsize=annot_fontsize)

        axs[4, i].scatter(probs, [s, slope, Xceps, Pk][i], c=probs, cmap='viridis', s=1)
        axs[4, i].set_ylabel(['s=(xc^1.5*eta^0.5)/eps', 'slope=eta*xc/eps', 'xc/eps', 'Fk/Dk'][i])
        axs[4, i].set_xlabel('lnprobs')
        axs[4, i].set_yscale(yscale)
        axs[4, i].set_xscale(xscale)
        #mark the marked thetas:
        axs[4, i].scatter(marked_probs, [marked_s, marked_slope, marked_Xceps, marked_Pk][i], c='red', s=3)
        if annotations is not None:
            for j, txt in enumerate(annotations):
                axs[4, i].annotate(txt, (marked_probs[j], [marked_s, marked_slope, marked_Xceps, marked_Pk][i][j]),
                                   xytext = (xytext_shift, xytext_shift),
                                   textcoords="offset points",
                                   #arrowprops=dict(facecolor=arrow_color, shrink=0.05, headlength=headlength, headwidth=headwidth),
                                    fontsize=annot_fontsize)

def getStats(thetas, probs, threshold = None):
    """
    Prints the stats for the thetas with log probability greater than the threshold.
    gives the valuse of best fit, median, means, std and median absolute deviation.
    The valuse are printed for each parameter, and for 
    beta/eta, beta*xc/eps, Fx, Dx, xc^2/eps, beta*kappa/eps, Fk, Dk, s, slope, xc/eps, Fk/Dk.
    """
    from scipy.stats import median_abs_deviation
    import pandas as pd
    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas/etas
    bxe = betas*xcs/epsilons
    Fx = betas**2/(etas*xcs)
    Dx = betas*epsilons/(etas*(xcs**2))

    td = xcs**2/epsilons
    bke = betas*kappa/epsilons
    Fk = betas**2/(etas*kappa)
    Dk = betas*epsilons/(etas*(kappa**2))

    s = (xcs**1.5)*(etas**0.5)/epsilons
    slope = etas*xcs/epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk

    if threshold is not None:
        etas = etas[probs>threshold]
        betas = betas[probs>threshold]
        epsilons = epsilons[probs>threshold]
        xcs = xcs[probs>threshold]

        t3 = t3[probs>threshold]
        bxe = bxe[probs>threshold]
        Fx = Fx[probs>threshold]
        Dx = Dx[probs>threshold]

        td = td[probs>threshold]
        bke = bke[probs>threshold]
        Fk = Fk[probs>threshold]
        Dk = Dk[probs>threshold]

        s = s[probs>threshold]
        slope = slope[probs>threshold]
        Xceps = Xceps[probs>threshold]
        Pk = Pk[probs>threshold]

    #all valuse should be rounded to 2 decimal places or to 3 non-zero decimal places. (1987.3424 -> 1987.34, 0.0006675 -> 0.000667)
    def round_value(value):
        if value == 0:
            return 0
        elif abs(value) < 1:
            r = int(np.abs(np.log10(abs(value))))
            return round(value, r+2)
        else:
            return round(value, 2)

    data = {
        'Best fit': [round_value(etas[np.argmax(probs)]), round_value(betas[np.argmax(probs)]), round_value(epsilons[np.argmax(probs)]), round_value(xcs[np.argmax(probs)]), round_value(t3[np.argmax(probs)]), round_value(bxe[np.argmax(probs)]), round_value(Fx[np.argmax(probs)]), round_value(Dx[np.argmax(probs)]), round_value(td[np.argmax(probs)]), round_value(bke[np.argmax(probs)]), round_value(Fk[np.argmax(probs)]), round_value(Dk[np.argmax(probs)]), round_value(s[np.argmax(probs)]), round_value(slope[np.argmax(probs)]), round_value(Xceps[np.argmax(probs)]), round_value(Pk[np.argmax(probs)])],
        'Median': [round_value(np.median(etas)), round_value(np.median(betas)), round_value(np.median(epsilons)), round_value(np.median(xcs)), round_value(np.median(t3)), round_value(np.median(bxe)), round_value(np.median(Fx)), round_value(np.median(Dx)), round_value(np.median(td)), round_value(np.median(bke)), round_value(np.median(Fk)), round_value(np.median(Dk)), round_value(np.median(s)), round_value(np.median(slope)), round_value(np.median(Xceps)), round_value(np.median(Pk))],
        'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk))],
        'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk))],
        'Median absolute deviation': [round_value(median_abs_deviation(etas)), round_value(median_abs_deviation(betas)), round_value(median_abs_deviation(epsilons)), round_value(median_abs_deviation(xcs)), round_value(median_abs_deviation(t3)), round_value(median_abs_deviation(bxe)), round_value(median_abs_deviation(Fx)), round_value(median_abs_deviation(Dx)), round_value(median_abs_deviation(td)), round_value(median_abs_deviation(bke)), round_value(median_abs_deviation(Fk)), round_value(median_abs_deviation(Dk)), round_value(median_abs_deviation(s)), round_value(median_abs_deviation(slope)), round_value(median_abs_deviation(Xceps)), round_value(median_abs_deviation(Pk))]
    }

    columns = ['Eta', 'Beta', 'Epsilon', 'Xc', 'Beta/Eta', 'Beta*Xc/Eps', 'Fx=Beta^2/(Eta*Xc)', 'Dx=Beta*Eps/(Eta*Xc^2)', 'Xc^2/Eps', 'Beta*Kappa/Eps', 'Fk=Beta^2/(Eta*Kappa)', 'Dk=Beta*Eps/(Eta*Kappa^2)', 's=(Xc^1.5*Eta^0.5)/Eps', 'Slope=Eta*Xc/Eps', 'Xc/Eps', 'Fk/Dk']

    df = pd.DataFrame(data, index=columns)
    dfr = df.transpose()

    print(dfr.to_string())
    return dfr


def getThreshold(ds,metric = 'survival',time_range = None):
    """
    Get the threshold for the log probability. The threshold is the log probability CI.
    The supplied ds should implement a getConfidenceInterval() method.
    """
    CI = ds.getConfidenceInterval()
    t,s = ds.getSurvival()
    #trim to the time range
    if time_range is not None:
        idx = (t>=time_range[0]) & (t<=time_range[1])
        t = t[idx]
        s = s[idx]
        CI[0] = CI[0][idx]
        CI[1] = CI[1][idx]
    if metric == 'survival':
        d0 =np.mean((s-CI[0])**2)
        d1 = np.mean((s-CI[1])**2)
    else:
        #throw an exception if the metric is not implemented
        raise Exception(f"{metric} Not implemented")
    thresh0 = 1/(0.1*d0)
    thresh1 = 1/(0.1*d1)
    #return the maximum of the two thresholds
    return max(thresh0,thresh1)

def applyThreshold(thetas,lnprobs,ds,metric = 'survival',time_range = None):
    """
    Apply the threshold to the thetas and the log probabilities.
    """
    threshold = getThreshold(ds,metric = metric,time_range = time_range)
    print(f"Threshold: {threshold}")
    #print the number of thetas that are above the threshold
    print(f"Number of thetas above the threshold: {len(thetas[lnprobs>threshold])}")
    print(f"Number of thetas below the threshold: {len(thetas[lnprobs<threshold])}")
    return thetas[lnprobs>threshold],lnprobs[lnprobs>threshold]
    

def custom_corner(samples,lnprobs,labels = ['eta','beta','epsilon','xc','ext h'], truths = None, scale ='log', grid =True,figsize=(15,15),quantiles = [0.16,0.5,0.84], show_color_bar=True):
    """
    A custom corner plot for the samples.
    Plots the samples and colors them according to the log probabilities.
    params:
    - samples (ndarray): The samples.
    - lnprobs (ndarray): The log probabilities.
    - labels (list): The labels for the parameters.
    - truths (list): The true values for the parameters. If None, the median of the samples is used.
    - scale (str): The scale for the axes.
    - grid (bool): Whether to plot the grid.
    - figsize (tuple): The size of the figure.
    - quantiles (list): The quantiles to plot.
    - show_color_bar (bool): Whether to show the color bar for the log probabilities.
    returns:
    - fig: The figure.
    - axes: The axes.
    """
    import matplotlib.pyplot as plt
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)
    fig.subplots_adjust(hspace=0.15, wspace=0.15)

    # Create scatter plot for color bar reference
    scatter = None
    for i in range(ndim):
        for j in range(i+1):
            ax = axes[i, j]
            if i == j:
                if scale == 'log':
                    ax.hist(samples[:, i], bins=np.logspace(np.log10(np.min(samples[:, i])), np.log10(np.max(samples[:, i])), 50), color="k", histtype="step")
                    ax.set_xscale('log')
                else:
                    ax.hist(samples[:, i], bins=100, color="k", histtype="step")
                if truths is not None:
                    ax.axvline(truths[i], color="r")
                ax.set_yticks([])

            else:
                scatter = ax.scatter(samples[:, j], samples[:, i], c=lnprobs, cmap='viridis', s=1)
                if truths is not None:
                    ax.axvline(truths[j], color="r")
                    ax.axhline(truths[i], color="r")
                if scale == 'log':
                    ax.set_xscale('log')
                    ax.set_yscale('log')
            if i < ndim - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(labels[j])
            if j > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(labels[i])
            if grid:
                ax.grid(True)
        for j in range(i+1, ndim):
            axes[i, j].set_visible(False)

    for quantile in quantiles:
        for i in range(ndim):
            axes[i, i].axvline(np.percentile(samples[:, i], quantile * 100), color="k", linestyle="dashed")

    # Add color bar if requested
    if show_color_bar and scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label='Log Probability')

    return fig, axes
