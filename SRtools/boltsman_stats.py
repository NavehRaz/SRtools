"""
This is a helper class to properly calculate the weights for the results of the MCMC simulation.
"""

import numpy as np


def findT(E):
    """
    Find the "temperature" T for the energies E.
    The temperture is given by T = sum(E[i]*exp(-E[i]/T)/sum(exp(-E[i])/T) where the sum is over all i.
    This should be numerically solved self consistently.
    param E: list of energies
    return: temperature T(float)
    """
    def f(T):
        return np.sum(E*np.exp(-E/T))/np.sum(np.exp(-E/T))-T
    from scipy.optimize import fsolve
    T = fsolve(f, 1.0)[0]

    return T


def calc_boltsmann_probs(energies, fact =1):
    """
    Calculate the probabilities for the energies E and temperature T.
    The weights are given by w_i = exp(-E_i/T)/sum(exp(-E_i/T))
    param energies: list of energies
    return: weights
    """
    T = findT(energies)*fact
    weights = np.exp(-energies/T)/np.sum(np.exp(-energies/T))
    return weights



def get_bolts_stats(thetas, lnprobs=None, ds=None,weights =None, threshold=None, file_path=None,fact =1,ref_theta=None, days = False,):
    """
    Prints the stats for the thetas with log probability greater than the threshold.
    gives the valuse of best fit, median, means, std and median absolute deviation.
    The valuse are printed for each parameter, and for 
    beta/eta, beta*xc/eps, Fx, Dx, xc^2/eps, beta*kappa/eps, Fk, Dk, s, slope, xc/eps, Fk/Dk.
    if ref_theta is not None, the values are also printed for the difference between the thetas and the ref_theta in 
    addition to the other stats.
    """
    from scipy.stats import median_abs_deviation
    import pandas as pd

    #ds shouldn't br None:
    if ds is None:
        raise ValueError("ds not specified")
    
    #either lnprobs or weights should be specified:
    if lnprobs is None and weights is None:
        raise ValueError("lnprobs or weights should be specified")
    if lnprobs is not None:
        if threshold is not None:
            thetas = thetas[lnprobs > threshold]
            lnprobs = lnprobs[lnprobs > threshold]

        energies = 1/lnprobs



    if weights is None:
        probs = calc_boltsmann_probs(energies,fact=fact)
    else:
        probs = weights


    ETA =0
    BETA =1
    EPSILON =2
    XC =3

    kappa =0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas] )
    xcs = np.array([theta[XC] for theta in thetas])

    if days:
        etas = etas/(365**2)
        betas = betas/(365)
        epsilons = epsilons/(365)

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
    F2Dk = Fk**2/Dk
    t_eta = np.sqrt(xcs/etas)

    if ref_theta is not None:
        ref_eta = ref_theta[ETA]
        ref_beta = ref_theta[BETA]
        ref_epsilon = ref_theta[EPSILON]
        ref_xc = ref_theta[XC]

        if days:
            ref_eta = ref_eta/(365**2)
            ref_beta = ref_beta/(365)
            ref_epsilon = ref_epsilon/(365)

        ref_t3 = ref_beta/ref_eta
        ref_bxe = ref_beta*ref_xc/ref_epsilon
        ref_Fx = ref_beta**2/(ref_eta*ref_xc)
        ref_Dx = ref_beta*ref_epsilon/(ref_eta*(ref_xc**2))

        ref_td = ref_xc**2/ref_epsilon
        ref_bke = ref_beta*kappa/ref_epsilon
        ref_Fk = ref_beta**2/(ref_eta*kappa)
        ref_Dk = ref_beta*ref_epsilon/(ref_eta*(kappa**2))

        ref_s = (ref_xc**1.5)*(ref_eta**0.5)/ref_epsilon
        ref_slope = ref_eta*ref_xc/ref_epsilon
        ref_Xceps = ref_xc/ref_epsilon
        ref_Pk = ref_Fk/ref_Dk
        ref_F2Dk = ref_Fk**2/ref_Dk
        ref_t_eta = np.sqrt(ref_xc/ref_eta)

    #all valuse should be rounded to 2 decimal places or to 3 non-zero decimal places. (1987.3424 -> 1987.34, 0.0006675 -> 0.000667)
    def round_value(value):
        if value == 0:
            return 0
        elif abs(value) < 1:
            r = int(np.abs(np.log10(abs(value))))
            return round(value, r+2)
        else:
            return round(value, 2)
    
    ml =ds.getMedianLifetime()
    mle = np.max(ds.getMedianLifetimeCI())

    if days:
        ml = ml*365
        mle = mle*365


    if ref_theta is not None:
        data = {
            'Best fit': [round_value(etas[np.argmax(probs)]), round_value(betas[np.argmax(probs)]), round_value(epsilons[np.argmax(probs)]), round_value(xcs[np.argmax(probs)]), round_value(t3[np.argmax(probs)]), round_value(bxe[np.argmax(probs)]), round_value(Fx[np.argmax(probs)]), round_value(Dx[np.argmax(probs)]), round_value(td[np.argmax(probs)]), round_value(bke[np.argmax(probs)]), round_value(Fk[np.argmax(probs)]), round_value(Dk[np.argmax(probs)]), round_value(s[np.argmax(probs)]), round_value(slope[np.argmax(probs)]), round_value(Xceps[np.argmax(probs)]), round_value(Pk[np.argmax(probs)]), round_value(F2Dk[np.argmax(probs)]), round_value(t_eta[np.argmax(probs)]), round_value(ml)],
            'Median': [round_value(weighted_median(etas, probs)), round_value(weighted_median(betas, probs)), round_value(weighted_median(epsilons, probs)), round_value(weighted_median(xcs, probs)), round_value(weighted_median(t3, probs)), round_value(weighted_median(bxe, probs)), round_value(weighted_median(Fx, probs)), round_value(weighted_median(Dx, probs)), round_value(weighted_median(td, probs)), round_value(weighted_median(bke, probs)), round_value(weighted_median(Fk, probs)), round_value(weighted_median(Dk, probs)), round_value(weighted_median(s, probs)), round_value(weighted_median(slope, probs)), round_value(weighted_median(Xceps, probs)), round_value(weighted_median(Pk, probs)), round_value(weighted_median(F2Dk, probs)), round_value(weighted_median(t_eta, probs)), round_value(ml)],
            'Mean': [round_value(weighted_mean(etas, probs)), round_value(weighted_mean(betas, probs)), round_value(weighted_mean(epsilons, probs)), round_value(weighted_mean(xcs, probs)), round_value(weighted_mean(t3, probs)), round_value(weighted_mean(bxe, probs)), round_value(weighted_mean(Fx, probs)), round_value(weighted_mean(Dx, probs)), round_value(weighted_mean(td, probs)), round_value(weighted_mean(bke, probs)), round_value(weighted_mean(Fk, probs)), round_value(weighted_mean(Dk, probs)), round_value(weighted_mean(s, probs)), round_value(weighted_mean(slope, probs)), round_value(weighted_mean(Xceps, probs)), round_value(weighted_mean(Pk, probs)), round_value(weighted_mean(F2Dk, probs)), round_value(weighted_mean(t_eta, probs)), round_value(ml)],
            'Std': [round_value(weighted_std(etas, probs)), round_value(weighted_std(betas, probs)), round_value(weighted_std(epsilons, probs)), round_value(weighted_std(xcs, probs)), round_value(weighted_std(t3, probs)), round_value(weighted_std(bxe, probs)), round_value(weighted_std(Fx, probs)), round_value(weighted_std(Dx, probs)), round_value(weighted_std(td, probs)), round_value(weighted_std(bke, probs)), round_value(weighted_std(Fk, probs)), round_value(weighted_std(Dk, probs)), round_value(weighted_std(s, probs)), round_value(weighted_std(slope, probs)), round_value(weighted_std(Xceps, probs)), round_value(weighted_std(Pk, probs)), round_value(weighted_std(F2Dk, probs)), round_value(weighted_std(t_eta, probs)), round_value(mle)],
            'Median absolute deviation': [round_value(weighted_mad(etas, probs)), round_value(weighted_mad(betas, probs)), round_value(weighted_mad(epsilons, probs)), round_value(weighted_mad(xcs, probs)), round_value(weighted_mad(t3, probs)), round_value(weighted_mad(bxe, probs)), round_value(weighted_mad(Fx, probs)), round_value(weighted_mad(Dx, probs)), round_value(weighted_mad(td, probs)), round_value(weighted_mad(bke, probs)), round_value(weighted_mad(Fk, probs)), round_value(weighted_mad(Dk, probs)), round_value(weighted_mad(s, probs)), round_value(weighted_mad(slope, probs)), round_value(weighted_mad(Xceps, probs)), round_value(weighted_mad(Pk, probs)), round_value(weighted_mad(F2Dk, probs)), round_value(weighted_mad(t_eta, probs)), round_value(mle)],
            'Best fit - ref': [round_value(etas[np.argmax(probs)]-ref_eta), round_value(betas[np.argmax(probs)]-ref_beta), round_value(epsilons[np.argmax(probs)]-ref_epsilon), round_value(xcs[np.argmax(probs)]-ref_xc), round_value(t3[np.argmax(probs)]-ref_t3), round_value(bxe[np.argmax(probs)]-ref_bxe), round_value(Fx[np.argmax(probs)]-ref_Fx), round_value(Dx[np.argmax(probs)]-ref_Dx), round_value(td[np.argmax(probs)]-ref_td), round_value(bke[np.argmax(probs)]-ref_bke), round_value(Fk[np.argmax(probs)]-ref_Fk), round_value(Dk[np.argmax(probs)]-ref_Dk), round_value(s[np.argmax(probs)]-ref_s), round_value(slope[np.argmax(probs)]-ref_slope), round_value(Xceps[np.argmax(probs)]-ref_Xceps), round_value(Pk[np.argmax(probs)]-ref_Pk), round_value(F2Dk[np.argmax(probs)]-ref_F2Dk), round_value(t_eta[np.argmax(probs)]-ref_t_eta), 0], 
            'Median - ref': [round_value(weighted_median(etas, probs)-ref_eta), round_value(weighted_median(betas, probs)-ref_beta), round_value(weighted_median(epsilons, probs)-ref_epsilon), round_value(weighted_median(xcs, probs)-ref_xc), round_value(weighted_median(t3, probs)-ref_t3), round_value(weighted_median(bxe, probs)-ref_bxe), round_value(weighted_median(Fx, probs)-ref_Fx), round_value(weighted_median(Dx, probs)-ref_Dx), round_value(weighted_median(td, probs)-ref_td), round_value(weighted_median(bke, probs)-ref_bke), round_value(weighted_median(Fk, probs)-ref_Fk), round_value(weighted_median(Dk, probs)-ref_Dk), round_value(weighted_median(s, probs)-ref_s), round_value(weighted_median(slope, probs)-ref_slope), round_value(weighted_median(Xceps, probs)-ref_Xceps), round_value(weighted_median(Pk, probs)-ref_Pk), round_value(weighted_median(F2Dk, probs)-ref_F2Dk), round_value(weighted_median(t_eta, probs)-ref_t_eta), 0],
            'Mean - ref': [round_value(weighted_mean(etas, probs)-ref_eta), round_value(weighted_mean(betas, probs)-ref_beta), round_value(weighted_mean(epsilons, probs)-ref_epsilon), round_value(weighted_mean(xcs, probs)-ref_xc), round_value(weighted_mean(t3, probs)-ref_t3), round_value(weighted_mean(bxe, probs)-ref_bxe), round_value(weighted_mean(Fx, probs)-ref_Fx), round_value(weighted_mean(Dx, probs)-ref_Dx), round_value(weighted_mean(td, probs)-ref_td), round_value(weighted_mean(bke, probs)-ref_bke), round_value(weighted_mean(Fk, probs)-ref_Fk), round_value(weighted_mean(Dk, probs)-ref_Dk), round_value(weighted_mean(s, probs)-ref_s), round_value(weighted_mean(slope, probs)-ref_slope), round_value(weighted_mean(Xceps, probs)-ref_Xceps), round_value(weighted_mean(Pk, probs)-ref_Pk), round_value(weighted_mean(F2Dk, probs)-ref_F2Dk), round_value(weighted_mean(t_eta, probs)-ref_t_eta), 0],
            'weighted mean error': [round_value(weighted_mean(etas-ref_eta, probs)), round_value(weighted_mean(betas-ref_beta, probs)), round_value(weighted_mean(epsilons-ref_epsilon, probs)), round_value(weighted_mean(xcs-ref_xc, probs)), round_value(weighted_mean(t3-ref_t3, probs)), round_value(weighted_mean(bxe-ref_bxe, probs)), round_value(weighted_mean(Fx-ref_Fx, probs)), round_value(weighted_mean(Dx-ref_Dx, probs)), round_value(weighted_mean(td-ref_td, probs)), round_value(weighted_mean(bke-ref_bke, probs)), round_value(weighted_mean(Fk-ref_Fk, probs)), round_value(weighted_mean(Dk-ref_Dk, probs)), round_value(weighted_mean(s-ref_s, probs)), round_value(weighted_mean(slope-ref_slope, probs)), round_value(weighted_mean(Xceps-ref_Xceps, probs)), round_value(weighted_mean(Pk-ref_Pk, probs)), round_value(weighted_mean(F2Dk-ref_F2Dk, probs)), round_value(weighted_mean(t_eta-ref_t_eta, probs)), 0]
        }
    else:
        data = {
            'Best fit': [round_value(etas[np.argmax(probs)]), round_value(betas[np.argmax(probs)]), round_value(epsilons[np.argmax(probs)]), round_value(xcs[np.argmax(probs)]), round_value(t3[np.argmax(probs)]), round_value(bxe[np.argmax(probs)]), round_value(Fx[np.argmax(probs)]), round_value(Dx[np.argmax(probs)]), round_value(td[np.argmax(probs)]), round_value(bke[np.argmax(probs)]), round_value(Fk[np.argmax(probs)]), round_value(Dk[np.argmax(probs)]), round_value(s[np.argmax(probs)]), round_value(slope[np.argmax(probs)]), round_value(Xceps[np.argmax(probs)]), round_value(Pk[np.argmax(probs)]), round_value(F2Dk[np.argmax(probs)]), round_value(t_eta[np.argmax(probs)]), round_value(ml)],
            'Median': [round_value(weighted_median(etas, probs)), round_value(weighted_median(betas, probs)), round_value(weighted_median(epsilons, probs)), round_value(weighted_median(xcs, probs)), round_value(weighted_median(t3, probs)), round_value(weighted_median(bxe, probs)), round_value(weighted_median(Fx, probs)), round_value(weighted_median(Dx, probs)), round_value(weighted_median(td, probs)), round_value(weighted_median(bke, probs)), round_value(weighted_median(Fk, probs)), round_value(weighted_median(Dk, probs)), round_value(weighted_median(s, probs)), round_value(weighted_median(slope, probs)), round_value(weighted_median(Xceps, probs)), round_value(weighted_median(Pk, probs)), round_value(weighted_median(F2Dk, probs)), round_value(weighted_median(t_eta, probs)), round_value(ml)],
            'Mean': [round_value(weighted_mean(etas, probs)), round_value(weighted_mean(betas, probs)), round_value(weighted_mean(epsilons, probs)), round_value(weighted_mean(xcs, probs)), round_value(weighted_mean(t3, probs)), round_value(weighted_mean(bxe, probs)), round_value(weighted_mean(Fx, probs)), round_value(weighted_mean(Dx, probs)), round_value(weighted_mean(td, probs)), round_value(weighted_mean(bke, probs)), round_value(weighted_mean(Fk, probs)), round_value(weighted_mean(Dk, probs)), round_value(weighted_mean(s, probs)), round_value(weighted_mean(slope, probs)), round_value(weighted_mean(Xceps, probs)), round_value(weighted_mean(Pk, probs)), round_value(weighted_mean(F2Dk, probs)), round_value(weighted_mean(t_eta, probs)), round_value(ml)],
            'Std': [round_value(weighted_std(etas, probs)), round_value(weighted_std(betas, probs)), round_value(weighted_std(epsilons, probs)), round_value(weighted_std(xcs, probs)), round_value(weighted_std(t3, probs)), round_value(weighted_std(bxe, probs)), round_value(weighted_std(Fx, probs)), round_value(weighted_std(Dx, probs)), round_value(weighted_std(td, probs)), round_value(weighted_std(bke, probs)), round_value(weighted_std(Fk, probs)), round_value(weighted_std(Dk, probs)), round_value(weighted_std(s, probs)), round_value(weighted_std(slope, probs)), round_value(weighted_std(Xceps, probs)), round_value(weighted_std(Pk, probs)), round_value(weighted_std(F2Dk, probs)), round_value(weighted_std(t_eta, probs)), round_value(mle)],
            'Median absolute deviation': [round_value(weighted_mad(etas, probs)), round_value(weighted_mad(betas, probs)), round_value(weighted_mad(epsilons, probs)), round_value(weighted_mad(xcs, probs)), round_value(weighted_mad(t3, probs)), round_value(weighted_mad(bxe, probs)), round_value(weighted_mad(Fx, probs)), round_value(weighted_mad(Dx, probs)), round_value(weighted_mad(td, probs)), round_value(weighted_mad(bke, probs)), round_value(weighted_mad(Fk, probs)), round_value(weighted_mad(Dk, probs)), round_value(weighted_mad(s, probs)), round_value(weighted_mad(slope, probs)), round_value(weighted_mad(Xceps, probs)), round_value(weighted_mad(Pk, probs)), round_value(weighted_mad(F2Dk, probs)), round_value(weighted_mad(t_eta, probs)), round_value(mle)]
        }

    columns = ['Eta', 'Beta', 'Epsilon', 'Xc', 'Beta/Eta', 'Beta*Xc/Eps', 'Fx=Beta^2/(Eta*Xc)', 'Dx=Beta*Eps/(Eta*Xc^2)', 'Xc^2/Eps', 'Beta*Kappa/Eps', 'Fk=Beta^2/(Eta*Kappa)', 'Dk=Beta*Eps/(Eta*Kappa^2)', 's=(Xc^1.5*Eta^0.5)/Eps', 'Slope=Eta*Xc/Eps', 'Xc/Eps', 'Fk/Dk', 'Fk^2/Dk', 't_eta', 'Median Lifetime']

    df = pd.DataFrame(data, index=columns)
    dfr = df.transpose()

    print(dfr.to_string())
    if file_path is not None:
        dfr.to_csv(file_path)
    return dfr


def weighted_median(data, weights):
    """
    Compute the weighted median of data.
    
    Parameters:
    data (array-like): The data points.
    weights (array-like): The weights for each data point.
    
    Returns:
    float: The weighted median.
    """
    data, weights = np.array(data), np.array(weights)
    
    # Sort data and weights by data
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Compute the cumulative sum of the weights
    cumulative_weights = np.cumsum(sorted_weights)
    
    # Find the index where the cumulative weight is greater than or equal to half the total weight
    half_total_weight = 0.5 * np.sum(weights)
    median_index = np.where(cumulative_weights >= half_total_weight)[0][0]
    
    return sorted_data[median_index]

def weighted_mean(data, weights):
    """
    Compute the weighted mean of data.
    
    Parameters:
    data (array-like): The data points.
    weights (array-like): The weights for each data point.
    
    Returns:
    float: The weighted mean.
    """
    data, weights = np.array(data), np.array(weights)
    return np.sum(weights * data) / np.sum(weights)


def weighted_std(data, weights):
    """
    Compute the weighted standard deviation of data.
    
    Parameters:
    data (array-like): The data points.
    weights (array-like): The weights for each data point.
    
    Returns:
    float: The weighted standard deviation.
    """
    data, weights = np.array(data), np.array(weights)
    weights = weights / np.sum(weights)
    mean = weighted_mean(data, weights)
    variance = np.sum(weights * (data - mean)**2) / np.sum(weights)
    return np.sqrt(variance)


def weighted_mad(data, weights):
    """
    Compute the weighted median absolute deviation of data.
    
    Parameters:
    data (array-like): The data points.
    weights (array-like): The weights for each data point.
    
    Returns:
    float: The weighted median absolute deviation.
    """
    data, weights = np.array(data), np.array(weights)
    
    # Calculate the weighted median
    median = weighted_median(data, weights)
    
    # Compute the absolute deviations from the weighted median
    absolute_deviations = np.abs(data - median)
    
    # Calculate the weighted median of the absolute deviations
    return weighted_median(absolute_deviations, weights)


def weighted_cov(thetas, lnprob, fact=1):
    """
    Compute the weighted covariance matrix of samples.
    Thetas is a List of samples, where each sample is a list of parameter values.
    """
    energies = 1/lnprobs
    probs = calc_boltsmann_probs(energies,fact=fact)
    thetas = np.log(np.array(thetas).T)
    cov = np.cov(thetas, aweights=probs)

    #calculate the eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)

    return cov, eigvals, eigvecs

  

def plot_vecs(thetas, lnprobs, vecs, fact=1,xscale_energy='log', xscale_probs='linear', yscale='log'):
    """
    For each vector in ves and each theta in theta, calculate the value along the vector and plots it
    against the probability of the theta, and against the energy of the theta.
    The values in each vector are the powers of the correspondin parameters in the theta, so if vec=[2,1,-0.5,3] 
    the corresponding value would be theta[0]**2*theta[1]**1*theta[2]**-0.5*theta[3]**3.
    The plots are scatter plots in a 2xlen(vecs) grid. Each subplot should have a title that describes the vector:
    so if vec=[2,1,-0.5,3] the title should be "Eta^2*Beta*Epsilon^-0.5*Xc^3" in latex script.
    The x and y axis should be labeled. The y axis should be in log scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    energies = 1/lnprobs
    probs = calc_boltsmann_probs(energies,fact=fact)

    for i, vec in enumerate(vecs):
        values = np.prod([theta**v for theta, v in zip(thetas.T, vec)], axis=0)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(r'$\eta^{%.2f}\beta^{%.2f}\epsilon^{%.2f}x_c^{%.2f}$' % tuple(vec), fontsize=16)
        axs[0].scatter(probs, values, alpha=0.5,c=probs, cmap='viridis', s=1)
        axs[0].set_xlabel('Probability')
        axs[0].set_ylabel(r'$\eta^{%.2f}\beta^{%.2f}\epsilon^{%.2f}x_c^{%.2f}$' % tuple(vec))
        axs[0].set_xscale(xscale_probs)
        axs[0].set_yscale(yscale)
        axs[1].scatter(energies, values, alpha=0.5,c=energies, cmap='viridis', s=1)
        axs[1].set_xlabel('Energy')
        axs[1].set_ylabel(r'$\eta^{%.2f}\beta^{%.2f}\epsilon^{%.2f}x_c^{%.2f}$' % tuple(vec))
        axs[1].set_xscale(xscale_energy)
        axs[1].set_yscale(yscale)
        plt.show()


def get_stats(thetas, lnprobs, ds, threshold=None, file_path=None, ref_theta=None, percent =False, days=False):
    """
    Prints the stats for the thetas with log probability greater than the threshold.
    gives the valuse of best fit, median, means, std and median absolute deviation.
    The valuse are printed for each parameter, and for 
    beta/eta, beta*xc/eps, Fx, Dx, xc^2/eps, beta*kappa/eps, Fk, Dk, s, slope, xc/eps, Fk/Dk, Fk^2/Dk.
    if ref_theta is not None, the values are also printed for the difference between the thetas and the ref_theta in 
    addition to the other stats.
    """
    from scipy.stats import median_abs_deviation
    import pandas as pd

    if threshold is not None:
        thetas = thetas[lnprobs > threshold]
        lnprobs = lnprobs[lnprobs > threshold]



    ETA = 0
    BETA = 1
    EPSILON = 2
    XC = 3

    kappa = 0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas])
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas / etas
    bxe = betas * xcs / epsilons
    Fx = betas ** 2 / (etas * xcs)
    Dx = betas * epsilons / (etas * (xcs ** 2))

    td = xcs ** 2 / epsilons
    bke = betas * kappa / epsilons
    Fk = betas ** 2 / (etas * kappa)
    Dk = betas * epsilons / (etas * (kappa ** 2))

    s = (xcs ** 1.5) * (etas ** 0.5) / epsilons
    slope = etas * xcs / epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk
    F2Dk = Fk**2/Dk
    t_eta = np.sqrt(xcs/etas)



    if ref_theta is not None:
        ref_eta = ref_theta[ETA]
        ref_beta = ref_theta[BETA]
        ref_epsilon = ref_theta[EPSILON]
        ref_xc = ref_theta[XC]

        if days:
            ref_eta = ref_eta/(365**2)
            ref_beta = ref_beta/(365)
            ref_epsilon = ref_epsilon/(365)

        ref_t3 = ref_beta/ref_eta
        ref_bxe = ref_beta*ref_xc/ref_epsilon
        ref_Fx = ref_beta**2/(ref_eta*ref_xc)
        ref_Dx = ref_beta*ref_epsilon/(ref_eta*(ref_xc**2))

        ref_td = ref_xc**2/ref_epsilon
        ref_bke = ref_beta*kappa/ref_epsilon
        ref_Fk = ref_beta**2/(ref_eta*kappa)
        ref_Dk = ref_beta*ref_epsilon/(ref_eta*(kappa**2))

        ref_s = (ref_xc**1.5)*(ref_eta**0.5)/ref_epsilon
        ref_slope = ref_eta*ref_xc/ref_epsilon
        ref_Xceps = ref_xc/ref_epsilon
        ref_Pk = ref_Fk/ref_Dk
        ref_F2Dk = ref_Fk**2/ref_Dk
        ref_t_eta = np.sqrt(ref_xc/ref_eta)

    #all valuse should be rounded to 2 decimal places or to 3 non-zero decimal places. (1987.3424 -> 1987.34, 0.0006675 -> 0.000667)
    def round_value(value):
        if value == 0:
            return 0
        elif abs(value) < 1:
            r = int(np.abs(np.log10(abs(value))))
            return round(value, r+2)
        else:
            return round(value, 2)
    
    ml =ds.getMedianLifetime()
    mle = np.max(ds.getMedianLifetimeCI())

    if days:
        ml = ml*365
        mle = mle*365



    if ref_theta is not None and not percent:
        data = {
            'Best fit': [round_value(etas[np.argmax(lnprobs)]), round_value(betas[np.argmax(lnprobs)]), round_value(epsilons[np.argmax(lnprobs)]), round_value(xcs[np.argmax(lnprobs)]), round_value(t3[np.argmax(lnprobs)]), round_value(bxe[np.argmax(lnprobs)]), round_value(Fx[np.argmax(lnprobs)]), round_value(Dx[np.argmax(lnprobs)]), round_value(td[np.argmax(lnprobs)]), round_value(bke[np.argmax(lnprobs)]), round_value(Fk[np.argmax(lnprobs)]), round_value(Dk[np.argmax(lnprobs)]), round_value(s[np.argmax(lnprobs)]), round_value(slope[np.argmax(lnprobs)]), round_value(Xceps[np.argmax(lnprobs)]), round_value(Pk[np.argmax(lnprobs)]), round_value(F2Dk[np.argmax(lnprobs)]), round_value(t_eta[np.argmax(lnprobs)]), round_value(ml)],
            'Median': [round_value(np.median(etas)), round_value(np.median(betas)), round_value(np.median(epsilons)), round_value(np.median(xcs)), round_value(np.median(t3)), round_value(np.median(bxe)), round_value(np.median(Fx)), round_value(np.median(Dx)), round_value(np.median(td)), round_value(np.median(bke)), round_value(np.median(Fk)), round_value(np.median(Dk)), round_value(np.median(s)), round_value(np.median(slope)), round_value(np.median(Xceps)), round_value(np.median(Pk)), round_value(np.median(F2Dk)), round_value(np.median(t_eta)), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
            'Median absolute deviation': [round_value(median_abs_deviation(etas)), round_value(median_abs_deviation(betas)), round_value(median_abs_deviation(epsilons)), round_value(median_abs_deviation(xcs)), round_value(median_abs_deviation(t3)), round_value(median_abs_deviation(bxe)), round_value(median_abs_deviation(Fx)), round_value(median_abs_deviation(Dx)), round_value(median_abs_deviation(td)), round_value(median_abs_deviation(bke)), round_value(median_abs_deviation(Fk)), round_value(median_abs_deviation(Dk)), round_value(median_abs_deviation(s)), round_value(median_abs_deviation(slope)), round_value(median_abs_deviation(Xceps)), round_value(median_abs_deviation(Pk)), round_value(median_abs_deviation(F2Dk)), round_value(median_abs_deviation(t_eta)), round_value(mle)],
            'Best fit - ref': [round_value(etas[np.argmax(lnprobs)]-ref_eta), round_value(betas[np.argmax(lnprobs)]-ref_beta), round_value(epsilons[np.argmax(lnprobs)]-ref_epsilon), round_value(xcs[np.argmax(lnprobs)]-ref_xc), round_value(t3[np.argmax(lnprobs)]-ref_t3), round_value(bxe[np.argmax(lnprobs)]-ref_bxe), round_value(Fx[np.argmax(lnprobs)]-ref_Fx), round_value(Dx[np.argmax(lnprobs)]-ref_Dx), round_value(td[np.argmax(lnprobs)]-ref_td), round_value(bke[np.argmax(lnprobs)]-ref_bke), round_value(Fk[np.argmax(lnprobs)]-ref_Fk), round_value(Dk[np.argmax(lnprobs)]-ref_Dk), round_value(s[np.argmax(lnprobs)]-ref_s), round_value(slope[np.argmax(lnprobs)]-ref_slope), round_value(Xceps[np.argmax(lnprobs)]-ref_Xceps), round_value(Pk[np.argmax(lnprobs)]-ref_Pk), round_value(F2Dk[np.argmax(lnprobs)]-ref_F2Dk), round_value(t_eta[np.argmax(lnprobs)]-ref_t_eta), 0],
            'Median - ref': [round_value(np.median(etas)-ref_eta), round_value(np.median(betas)-ref_beta), round_value(np.median(epsilons)-ref_epsilon), round_value(np.median(xcs)-ref_xc), round_value(np.median(t3)-ref_t3), round_value(np.median(bxe)-ref_bxe), round_value(np.median(Fx)-ref_Fx), round_value(np.median(Dx)-ref_Dx), round_value(np.median(td)-ref_td), round_value(np.median(bke)-ref_bke), round_value(np.median(Fk)-ref_Fk), round_value(np.median(Dk)-ref_Dk), round_value(np.median(s)-ref_s), round_value(np.median(slope)-ref_slope), round_value(np.median(Xceps)-ref_Xceps), round_value(np.median(Pk)-ref_Pk), round_value(np.median(F2Dk)-ref_F2Dk), round_value(np.median(t_eta)-ref_t_eta), 0],
            'Mean - ref': [round_value(np.mean(etas)-ref_eta), round_value(np.mean(betas)-ref_beta), round_value(np.mean(epsilons)-ref_epsilon), round_value(np.mean(xcs)-ref_xc), round_value(np.mean(t3)-ref_t3), round_value(np.mean(bxe)-ref_bxe), round_value(np.mean(Fx)-ref_Fx), round_value(np.mean(Dx)-ref_Dx), round_value(np.mean(td)-ref_td), round_value(np.mean(bke)-ref_bke), round_value(np.mean(Fk)-ref_Fk), round_value(np.mean(Dk)-ref_Dk), round_value(np.mean(s)-ref_s), round_value(np.mean(slope)-ref_slope), round_value(np.mean(Xceps)-ref_Xceps), round_value(np.mean(Pk)-ref_Pk), round_value(np.mean(F2Dk)-ref_F2Dk), round_value(np.mean(t_eta)-ref_t_eta), 0],
            'mean error': [round_value(np.mean(etas-ref_eta)), round_value(np.mean(betas-ref_beta)), round_value(np.mean(epsilons-ref_epsilon)), round_value(np.mean(xcs-ref_xc)), round_value(np.mean(t3-ref_t3)), round_value(np.mean(bxe-ref_bxe)), round_value(np.mean(Fx-ref_Fx)), round_value(np.mean(Dx-ref_Dx)), round_value(np.mean(td-ref_td)), round_value(np.mean(bke-ref_bke)), round_value(np.mean(Fk-ref_Fk)), round_value(np.mean(Dk-ref_Dk)), round_value(np.mean(s-ref_s)), round_value(np.mean(slope-ref_slope)), round_value(np.mean(Xceps-ref_Xceps)), round_value(np.mean(Pk-ref_Pk)), round_value(np.mean(F2Dk-ref_F2Dk)), round_value(np.mean(t_eta-ref_t_eta)), 0]
        }
    elif ref_theta is not None and percent:
        data = {
            'Best fit': [round_value(etas[np.argmax(lnprobs)]), round_value(betas[np.argmax(lnprobs)]), round_value(epsilons[np.argmax(lnprobs)]), round_value(xcs[np.argmax(lnprobs)]), round_value(t3[np.argmax(lnprobs)]), round_value(bxe[np.argmax(lnprobs)]), round_value(Fx[np.argmax(lnprobs)]), round_value(Dx[np.argmax(lnprobs)]), round_value(td[np.argmax(lnprobs)]), round_value(bke[np.argmax(lnprobs)]), round_value(Fk[np.argmax(lnprobs)]), round_value(Dk[np.argmax(lnprobs)]), round_value(s[np.argmax(lnprobs)]), round_value(slope[np.argmax(lnprobs)]), round_value(Xceps[np.argmax(lnprobs)]), round_value(Pk[np.argmax(lnprobs)]), round_value(F2Dk[np.argmax(lnprobs)]), round_value(t_eta[np.argmax(lnprobs)]), round_value(ml)],
            'Median': [round_value(np.median(etas)), round_value(np.median(betas)), round_value(np.median(epsilons)), round_value(np.median(xcs)), round_value(np.median(t3)), round_value(np.median(bxe)), round_value(np.median(Fx)), round_value(np.median(Dx)), round_value(np.median(td)), round_value(np.median(bke)), round_value(np.median(Fk)), round_value(np.median(Dk)), round_value(np.median(s)), round_value(np.median(slope)), round_value(np.median(Xceps)), round_value(np.median(Pk)), round_value(np.median(F2Dk)), round_value(np.median(t_eta)), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
            'Median absolute deviation': [round_value(median_abs_deviation(etas)), round_value(median_abs_deviation(betas)), round_value(median_abs_deviation(epsilons)), round_value(median_abs_deviation(xcs)), round_value(median_abs_deviation(t3)), round_value(median_abs_deviation(bxe)), round_value(median_abs_deviation(Fx)), round_value(median_abs_deviation(Dx)), round_value(median_abs_deviation(td)), round_value(median_abs_deviation(bke)), round_value(median_abs_deviation(Fk)), round_value(median_abs_deviation(Dk)), round_value(median_abs_deviation(s)), round_value(median_abs_deviation(slope)), round_value(median_abs_deviation(Xceps)), round_value(median_abs_deviation(Pk)), round_value(median_abs_deviation(F2Dk)), round_value(median_abs_deviation(t_eta)), round_value(mle)],
            'Best fit - ref': [round_value(np.abs(etas[np.argmax(lnprobs)]-ref_eta)/ref_eta), round_value(np.abs(betas[np.argmax(lnprobs)]-ref_beta)/ref_beta), round_value(np.abs(epsilons[np.argmax(lnprobs)]-ref_epsilon)/ref_epsilon), round_value(np.abs(xcs[np.argmax(lnprobs)]-ref_xc)/ref_xc), round_value(np.abs(t3[np.argmax(lnprobs)]-ref_t3)/ref_t3), round_value(np.abs(bxe[np.argmax(lnprobs)]-ref_bxe)/ref_bxe), round_value(np.abs(Fx[np.argmax(lnprobs)]-ref_Fx)/ref_Fx), round_value(np.abs(Dx[np.argmax(lnprobs)]-ref_Dx)/ref_Dx), round_value(np.abs(td[np.argmax(lnprobs)]-ref_td)/ref_td), round_value(np.abs(bke[np.argmax(lnprobs)]-ref_bke)/ref_bke), round_value(np.abs(Fk[np.argmax(lnprobs)]-ref_Fk)/ref_Fk), round_value(np.abs(Dk[np.argmax(lnprobs)]-ref_Dk)/ref_Dk), round_value(np.abs(s[np.argmax(lnprobs)]-ref_s)/ref_s), round_value(np.abs(slope[np.argmax(lnprobs)]-ref_slope)/ref_slope), round_value(np.abs(Xceps[np.argmax(lnprobs)]-ref_Xceps)/ref_Xceps), round_value(np.abs(Pk[np.argmax(lnprobs)]-ref_Pk)/ref_Pk), round_value(np.abs(F2Dk[np.argmax(lnprobs)]-ref_F2Dk)/ref_F2Dk), round_value(np.abs(t_eta[np.argmax(lnprobs)]-ref_t_eta)/ref_t_eta), round_value(ml)],
            'Median - ref': [round_value(np.abs(np.median(etas)-ref_eta)/ref_eta), round_value(np.abs(np.median(betas)-ref_beta)/ref_beta), round_value(np.abs(np.median(epsilons)-ref_epsilon)/ref_epsilon), round_value(np.abs(np.median(xcs)-ref_xc)/ref_xc), round_value(np.abs(np.median(t3)-ref_t3)/ref_t3), round_value(np.abs(np.median(bxe)-ref_bxe)/ref_bxe), round_value(np.abs(np.median(Fx)-ref_Fx)/ref_Fx), round_value(np.abs(np.median(Dx)-ref_Dx)/ref_Dx), round_value(np.abs(np.median(td)-ref_td)/ref_td), round_value(np.abs(np.median(bke)-ref_bke)/ref_bke), round_value(np.abs(np.median(Fk)-ref_Fk)/ref_Fk), round_value(np.abs(np.median(Dk)-ref_Dk)/ref_Dk), round_value(np.abs(np.median(s)-ref_s)/ref_s), round_value(np.abs(np.median(slope)-ref_slope)/ref_slope), round_value(np.abs(np.median(Xceps)-ref_Xceps)/ref_Xceps), round_value(np.abs(np.median(Pk)-ref_Pk)/ref_Pk), round_value(np.abs(np.median(F2Dk)-ref_F2Dk)/ref_F2Dk), round_value(np.abs(np.median(t_eta)-ref_t_eta)/ref_t_eta), round_value(ml)],
            'Mean - ref': [round_value(np.abs(np.mean(etas)-ref_eta)/ref_eta), round_value(np.abs(np.mean(betas)-ref_beta)/ref_beta), round_value(np.abs(np.mean(epsilons)-ref_epsilon)/ref_epsilon), round_value(np.abs(np.mean(xcs)-ref_xc)/ref_xc), round_value(np.abs(np.mean(t3)-ref_t3)/ref_t3), round_value(np.abs(np.mean(bxe)-ref_bxe)/ref_bxe), round_value(np.abs(np.mean(Fx)-ref_Fx)/ref_Fx), round_value(np.abs(np.mean(Dx)-ref_Dx)/ref_Dx), round_value(np.abs(np.mean(td)-ref_td)/ref_td), round_value(np.abs(np.mean(bke)-ref_bke)/ref_bke), round_value(np.abs(np.mean(Fk)-ref_Fk)/ref_Fk), round_value(np.abs(np.mean(Dk)-ref_Dk)/ref_Dk), round_value(np.abs(np.mean(s)-ref_s)/ref_s), round_value(np.abs(np.mean(slope)-ref_slope)/ref_slope), round_value(np.abs(np.mean(Xceps)-ref_Xceps)/ref_Xceps), round_value(np.abs(np.mean(Pk)-ref_Pk)/ref_Pk), round_value(np.abs(np.mean(F2Dk)-ref_F2Dk)/ref_F2Dk), round_value(np.abs(np.mean(t_eta)-ref_t_eta)/ref_t_eta), round_value(ml)],
            'mean error': [round_value(np.abs(np.mean(etas-ref_eta))/ref_eta), round_value(np.abs(np.mean(betas-ref_beta))/ref_beta), round_value(np.abs(np.mean(epsilons-ref_epsilon))/ref_epsilon), round_value(np.abs(np.mean(xcs-ref_xc))/ref_xc), round_value(np.abs(np.mean(t3-ref_t3))/ref_t3), round_value(np.abs(np.mean(bxe-ref_bxe))/ref_bxe), round_value(np.abs(np.mean(Fx-ref_Fx))/ref_Fx), round_value(np.abs(np.mean(Dx-ref_Dx))/ref_Dx), round_value(np.abs(np.mean(td-ref_td))/ref_td), round_value(np.abs(np.mean(bke-ref_bke))/ref_bke), round_value(np.abs(np.mean(Fk-ref_Fk))/ref_Fk), round_value(np.abs(np.mean(Dk-ref_Dk))/ref_Dk), round_value(np.abs(np.mean(s-ref_s))/ref_s), round_value(np.abs(np.mean(slope-ref_slope))/ref_slope), round_value(np.abs(np.mean(Xceps-ref_Xceps))/ref_Xceps), round_value(np.abs(np.mean(Pk-ref_Pk))/ref_Pk), round_value(np.abs(np.mean(F2Dk-ref_F2Dk))/ref_F2Dk), round_value(np.abs(np.mean(t_eta-ref_t_eta))/ref_t_eta), round_value(ml)],
        }
    else:
        data = {
            'Best fit': [round_value(etas[np.argmax(lnprobs)]), round_value(betas[np.argmax(lnprobs)]), round_value(epsilons[np.argmax(lnprobs)]), round_value(xcs[np.argmax(lnprobs)]), round_value(t3[np.argmax(lnprobs)]), round_value(bxe[np.argmax(lnprobs)]), round_value(Fx[np.argmax(lnprobs)]), round_value(Dx[np.argmax(lnprobs)]), round_value(td[np.argmax(lnprobs)]), round_value(bke[np.argmax(lnprobs)]), round_value(Fk[np.argmax(lnprobs)]), round_value(Dk[np.argmax(lnprobs)]), round_value(s[np.argmax(lnprobs)]), round_value(slope[np.argmax(lnprobs)]), round_value(Xceps[np.argmax(lnprobs)]), round_value(Pk[np.argmax(lnprobs)]), round_value(F2Dk[np.argmax(lnprobs)]), round_value(t_eta[np.argmax(lnprobs)]), round_value(ml)],
            'Median': [round_value(np.median(etas)), round_value(np.median(betas)), round_value(np.median(epsilons)), round_value(np.median(xcs)), round_value(np.median(t3)), round_value(np.median(bxe)), round_value(np.median(Fx)), round_value(np.median(Dx)), round_value(np.median(td)), round_value(np.median(bke)), round_value(np.median(Fk)), round_value(np.median(Dk)), round_value(np.median(s)), round_value(np.median(slope)), round_value(np.median(Xceps)), round_value(np.median(Pk)), round_value(np.median(F2Dk)), round_value(np.median(t_eta)), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
            'Median absolute deviation': [round_value(median_abs_deviation(etas)), round_value(median_abs_deviation(betas)), round_value(median_abs_deviation(epsilons)), round_value(median_abs_deviation(xcs)), round_value(median_abs_deviation(t3)), round_value(median_abs_deviation(bxe)), round_value(median_abs_deviation(Fx)), round_value(median_abs_deviation(Dx)), round_value(median_abs_deviation(td)), round_value(median_abs_deviation(bke)), round_value(median_abs_deviation(Fk)), round_value(median_abs_deviation(Dk)), round_value(median_abs_deviation(s)), round_value(median_abs_deviation(slope)), round_value(median_abs_deviation(Xceps)), round_value(median_abs_deviation(Pk)), round_value(median_abs_deviation(F2Dk)), round_value(median_abs_deviation(t_eta)), round_value(mle)]
        }

    columns = ['Eta', 'Beta', 'Epsilon', 'Xc', 'Beta/Eta', 'Beta*Xc/Eps', 'Fx=Beta^2/(Eta*Xc)', 'Dx=Beta*Eps/(Eta*Xc^2)', 'Xc^2/Eps', 'Beta*Kappa/Eps', 'Fk=Beta^2/(Eta*Kappa)', 'Dk=Beta*Eps/(Eta*Kappa^2)', 's=(Xc^1.5*Eta^0.5)/Eps', 'Slope=Eta*Xc/Eps', 'Xc/Eps', 'Fk/Dk', 'Fk^2/Dk', 't_eta', 'Median Lifetime']

    df = pd.DataFrame(data, index=columns)
    dfr = df.transpose()

    print(dfr.to_string())
    if file_path is not None:
        dfr.to_csv(file_path)
    return dfr


def get_stats_dynesty(thetas, weights, ds, threshold=None, file_path=None, ref_theta=None, percent =False, days=False):
    """
    Prints the stats for the thetas with log probability greater than the threshold.
    gives the valuse of best fit, median, means, std and median absolute deviation.
    The valuse are printed for each parameter, and for 
    beta/eta, beta*xc/eps, Fx, Dx, xc^2/eps, beta*kappa/eps, Fk, Dk, s, slope, xc/eps, Fk/Dk, Fk^2/Dk.
    if ref_theta is not None, the values are also printed for the difference between the thetas and the ref_theta in 
    addition to the other stats.
    """
    from scipy.stats import median_abs_deviation
    import pandas as pd

    if threshold is not None:
        thetas = thetas[weights > threshold]
        weights = weights[weights > threshold]



    ETA = 0
    BETA = 1
    EPSILON = 2
    XC = 3

    kappa = 0.5

    etas = np.array([theta[ETA] for theta in thetas])
    betas = np.array([theta[BETA] for theta in thetas])
    epsilons = np.array([theta[EPSILON] for theta in thetas])
    xcs = np.array([theta[XC] for theta in thetas])

    t3 = betas / etas
    bxe = betas * xcs / epsilons
    Fx = betas ** 2 / (etas * xcs)
    Dx = betas * epsilons / (etas * (xcs ** 2))

    td = xcs ** 2 / epsilons
    bke = betas * kappa / epsilons
    Fk = betas ** 2 / (etas * kappa)
    Dk = betas * epsilons / (etas * (kappa ** 2))

    s = (xcs ** 1.5) * (etas ** 0.5) / epsilons
    slope = etas * xcs / epsilons
    Xceps = xcs/epsilons
    Pk = Fk/Dk
    F2Dk = Fk**2/Dk

    if ref_theta is not None:
        ref_eta = ref_theta[ETA]
        ref_beta = ref_theta[BETA]
        ref_epsilon = ref_theta[EPSILON]
        ref_xc = ref_theta[XC]

        ref_t3 = ref_beta/ref_eta
        ref_bxe = ref_beta*ref_xc/ref_epsilon
        ref_Fx = ref_beta**2/(ref_eta*ref_xc)
        ref_Dx = ref_beta*ref_epsilon/(ref_eta*(ref_xc**2))

        ref_td = ref_xc**2/ref_epsilon
        ref_bke = ref_beta*kappa/ref_epsilon
        ref_Fk = ref_beta**2/(ref_eta*kappa)
        ref_Dk = ref_beta*ref_epsilon/(ref_eta*(kappa**2))

        ref_s = (ref_xc**1.5)*(ref_eta**0.5)/ref_epsilon
        ref_slope = ref_eta*ref_xc/ref_epsilon
        ref_Xceps = ref_xc/ref_epsilon
        ref_Pk = ref_Fk/ref_Dk
        ref_F2Dk = ref_Fk**2/ref_Dk
        t_eta = np.sqrt(xcs/etas)

        if ref_theta is not None:
            ref_eta = ref_theta[ETA]
            ref_beta = ref_theta[BETA]
            ref_epsilon = ref_theta[EPSILON]
            ref_xc = ref_theta[XC]

            if days:
                ref_eta = ref_eta/(365**2)
                ref_beta = ref_beta/(365)
                ref_epsilon = ref_epsilon/(365)

            ref_t3 = ref_beta/ref_eta
            ref_bxe = ref_beta*ref_xc/ref_epsilon
            ref_Fx = ref_beta**2/(ref_eta*ref_xc)
            ref_Dx = ref_beta*ref_epsilon/(ref_eta*(ref_xc**2))

            ref_td = ref_xc**2/ref_epsilon
            ref_bke = ref_beta*kappa/ref_epsilon
            ref_Fk = ref_beta**2/(ref_eta*kappa)
            ref_Dk = ref_beta*ref_epsilon/(ref_eta*(kappa**2))

            ref_s = (ref_xc**1.5)*(ref_eta**0.5)/ref_epsilon
            ref_slope = ref_eta*ref_xc/ref_epsilon
            ref_Xceps = ref_xc/ref_epsilon
            ref_Pk = ref_Fk/ref_Dk
            ref_F2Dk = ref_Fk**2/ref_Dk
            ref_t_eta = np.sqrt(ref_xc/ref_eta)

    #all valuse should be rounded to 2 decimal places or to 3 non-zero decimal places. (1987.3424 -> 1987.34, 0.0006675 -> 0.000667)
    def round_value(value):
        if value == 0:
            return 0
        elif abs(value) < 1:
            r = int(np.abs(np.log10(abs(value))))
            return round(value, r+2)
        else:
            return round(value, 2)
    
    ml =ds.getMedianLifetime()
    mle = np.max(ds.getMedianLifetimeCI())

    if days:
        ml = ml*365
        mle = mle*365



    if ref_theta is not None and not percent:
        data = {
            'Best fit': [round_value(etas[np.argmax(weights)]), round_value(betas[np.argmax(weights)]), round_value(epsilons[np.argmax(weights)]), round_value(xcs[np.argmax(weights)]), round_value(t3[np.argmax(weights)]), round_value(bxe[np.argmax(weights)]), round_value(Fx[np.argmax(weights)]), round_value(Dx[np.argmax(weights)]), round_value(td[np.argmax(weights)]), round_value(bke[np.argmax(weights)]), round_value(Fk[np.argmax(weights)]), round_value(Dk[np.argmax(weights)]), round_value(s[np.argmax(weights)]), round_value(slope[np.argmax(weights)]), round_value(Xceps[np.argmax(weights)]), round_value(Pk[np.argmax(weights)]), round_value(F2Dk[np.argmax(weights)]), round_value(t_eta[np.argmax(weights)]), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
            'Best fit - ref': [round_value(etas[np.argmax(weights)]-ref_eta), round_value(betas[np.argmax(weights)]-ref_beta), round_value(epsilons[np.argmax(weights)]-ref_epsilon), round_value(xcs[np.argmax(weights)]-ref_xc), round_value(t3[np.argmax(weights)]-ref_t3), round_value(bxe[np.argmax(weights)]-ref_bxe), round_value(Fx[np.argmax(weights)]-ref_Fx), round_value(Dx[np.argmax(weights)]-ref_Dx), round_value(td[np.argmax(weights)]-ref_td), round_value(bke[np.argmax(weights)]-ref_bke), round_value(Fk[np.argmax(weights)]-ref_Fk), round_value(Dk[np.argmax(weights)]-ref_Dk), round_value(s[np.argmax(weights)]-ref_s), round_value(slope[np.argmax(weights)]-ref_slope), round_value(Xceps[np.argmax(weights)]-ref_Xceps), round_value(Pk[np.argmax(weights)]-ref_Pk), round_value(F2Dk[np.argmax(weights)]-ref_F2Dk), round_value(t_eta[np.argmax(weights)]-ref_t_eta), 0],
            'Mean - ref': [round_value(np.mean(etas)-ref_eta), round_value(np.mean(betas)-ref_beta), round_value(np.mean(epsilons)-ref_epsilon), round_value(np.mean(xcs)-ref_xc), round_value(np.mean(t3)-ref_t3), round_value(np.mean(bxe)-ref_bxe), round_value(np.mean(Fx)-ref_Fx), round_value(np.mean(Dx)-ref_Dx), round_value(np.mean(td)-ref_td), round_value(np.mean(bke)-ref_bke), round_value(np.mean(Fk)-ref_Fk), round_value(np.mean(Dk)-ref_Dk), round_value(np.mean(s)-ref_s), round_value(np.mean(slope)-ref_slope), round_value(np.mean(Xceps)-ref_Xceps), round_value(np.mean(Pk)-ref_Pk), round_value(np.mean(F2Dk)-ref_F2Dk), round_value(np.mean(t_eta)-ref_t_eta), 0],
            'mean error': [round_value(np.mean(etas-ref_eta)), round_value(np.mean(betas-ref_beta)), round_value(np.mean(epsilons-ref_epsilon)), round_value(np.mean(xcs-ref_xc)), round_value(np.mean(t3-ref_t3)), round_value(np.mean(bxe-ref_bxe)), round_value(np.mean(Fx-ref_Fx)), round_value(np.mean(Dx-ref_Dx)), round_value(np.mean(td-ref_td)), round_value(np.mean(bke-ref_bke)), round_value(np.mean(Fk-ref_Fk)), round_value(np.mean(Dk-ref_Dk)), round_value(np.mean(s-ref_s)), round_value(np.mean(slope-ref_slope)), round_value(np.mean(Xceps-ref_Xceps)), round_value(np.mean(Pk-ref_Pk)), round_value(np.mean(F2Dk-ref_F2Dk)), round_value(np.mean(t_eta-ref_t_eta)), 0]
        }
    elif ref_theta is not None and percent:
        data = {
            'Best fit': [round_value(etas[np.argmax(weights)]), round_value(betas[np.argmax(weights)]), round_value(epsilons[np.argmax(weights)]), round_value(xcs[np.argmax(weights)]), round_value(t3[np.argmax(weights)]), round_value(bxe[np.argmax(weights)]), round_value(Fx[np.argmax(weights)]), round_value(Dx[np.argmax(weights)]), round_value(td[np.argmax(weights)]), round_value(bke[np.argmax(weights)]), round_value(Fk[np.argmax(weights)]), round_value(Dk[np.argmax(weights)]), round_value(s[np.argmax(weights)]), round_value(slope[np.argmax(weights)]), round_value(Xceps[np.argmax(weights)]), round_value(Pk[np.argmax(weights)]), round_value(F2Dk[np.argmax(weights)]), round_value(t_eta[np.argmax(weights)]), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
            'Best fit - ref': [round_value(np.abs(etas[np.argmax(weights)]-ref_eta)/ref_eta), round_value(np.abs(betas[np.argmax(weights)]-ref_beta)/ref_beta), round_value(np.abs(epsilons[np.argmax(weights)]-ref_epsilon)/ref_epsilon), round_value(np.abs(xcs[np.argmax(weights)]-ref_xc)/ref_xc), round_value(np.abs(t3[np.argmax(weights)]-ref_t3)/ref_t3), round_value(np.abs(bxe[np.argmax(weights)]-ref_bxe)/ref_bxe), round_value(np.abs(Fx[np.argmax(weights)]-ref_Fx)/ref_Fx), round_value(np.abs(Dx[np.argmax(weights)]-ref_Dx)/ref_Dx), round_value(np.abs(td[np.argmax(weights)]-ref_td)/ref_td), round_value(np.abs(bke[np.argmax(weights)]-ref_bke)/ref_bke), round_value(np.abs(Fk[np.argmax(weights)]-ref_Fk)/ref_Fk), round_value(np.abs(Dk[np.argmax(weights)]-ref_Dk)/ref_Dk), round_value(np.abs(s[np.argmax(weights)]-ref_s)/ref_s), round_value(np.abs(slope[np.argmax(weights)]-ref_slope)/ref_slope), round_value(np.abs(Xceps[np.argmax(weights)]-ref_Xceps)/ref_Xceps), round_value(np.abs(Pk[np.argmax(weights)]-ref_Pk)/ref_Pk), round_value(np.abs(F2Dk[np.argmax(weights)]-ref_F2Dk)/ref_F2Dk), round_value(np.abs(t_eta[np.argmax(weights)]-ref_t_eta)/ref_t_eta), round_value(ml)],
            'Mean - ref': [round_value(np.abs(np.mean(etas)-ref_eta)/ref_eta), round_value(np.abs(np.mean(betas)-ref_beta)/ref_beta), round_value(np.abs(np.mean(epsilons)-ref_epsilon)/ref_epsilon), round_value(np.abs(np.mean(xcs)-ref_xc)/ref_xc), round_value(np.abs(np.mean(t3)-ref_t3)/ref_t3), round_value(np.abs(np.mean(bxe)-ref_bxe)/ref_bxe), round_value(np.abs(np.mean(Fx)-ref_Fx)/ref_Fx), round_value(np.abs(np.mean(Dx)-ref_Dx)/ref_Dx), round_value(np.abs(np.mean(td)-ref_td)/ref_td), round_value(np.abs(np.mean(bke)-ref_bke)/ref_bke), round_value(np.abs(np.mean(Fk)-ref_Fk)/ref_Fk), round_value(np.abs(np.mean(Dk)-ref_Dk)/ref_Dk), round_value(np.abs(np.mean(s)-ref_s)/ref_s), round_value(np.abs(np.mean(slope)-ref_slope)/ref_slope), round_value(np.abs(np.mean(Xceps)-ref_Xceps)/ref_Xceps), round_value(np.abs(np.mean(Pk)-ref_Pk)/ref_Pk), round_value(np.abs(np.mean(F2Dk)-ref_F2Dk)/ref_F2Dk), round_value(np.abs(np.mean(t_eta)-ref_t_eta)/ref_t_eta), round_value(ml)],
            'mean error': [round_value(np.abs(np.mean(etas-ref_eta))/ref_eta), round_value(np.abs(np.mean(betas-ref_beta))/ref_beta), round_value(np.abs(np.mean(epsilons-ref_epsilon))/ref_epsilon), round_value(np.abs(np.mean(xcs-ref_xc))/ref_xc), round_value(np.abs(np.mean(t3-ref_t3))/ref_t3), round_value(np.abs(np.mean(bxe-ref_bxe))/ref_bxe), round_value(np.abs(np.mean(Fx-ref_Fx))/ref_Fx), round_value(np.abs(np.mean(Dx-ref_Dx))/ref_Dx), round_value(np.abs(np.mean(td-ref_td))/ref_td), round_value(np.abs(np.mean(bke-ref_bke))/ref_bke), round_value(np.abs(np.mean(Fk-ref_Fk))/ref_Fk), round_value(np.abs(np.mean(Dk-ref_Dk))/ref_Dk), round_value(np.abs(np.mean(s-ref_s))/ref_s), round_value(np.abs(np.mean(slope-ref_slope))/ref_slope), round_value(np.abs(np.mean(Xceps-ref_Xceps))/ref_Xceps), round_value(np.abs(np.mean(Pk-ref_Pk))/ref_Pk), round_value(np.abs(np.mean(F2Dk-ref_F2Dk))/ref_F2Dk), round_value(np.abs(np.mean(t_eta-ref_t_eta))/ref_t_eta), round_value(ml)],
        }
    else:
        data = {
            'Best fit': [round_value(etas[np.argmax(weights)]), round_value(betas[np.argmax(weights)]), round_value(epsilons[np.argmax(weights)]), round_value(xcs[np.argmax(weights)]), round_value(t3[np.argmax(weights)]), round_value(bxe[np.argmax(weights)]), round_value(Fx[np.argmax(weights)]), round_value(Dx[np.argmax(weights)]), round_value(td[np.argmax(weights)]), round_value(bke[np.argmax(weights)]), round_value(Fk[np.argmax(weights)]), round_value(Dk[np.argmax(weights)]), round_value(s[np.argmax(weights)]), round_value(slope[np.argmax(weights)]), round_value(Xceps[np.argmax(weights)]), round_value(Pk[np.argmax(weights)]), round_value(F2Dk[np.argmax(weights)]), round_value(t_eta[np.argmax(weights)]), round_value(ml)],
            'Mean': [round_value(np.mean(etas)), round_value(np.mean(betas)), round_value(np.mean(epsilons)), round_value(np.mean(xcs)), round_value(np.mean(t3)), round_value(np.mean(bxe)), round_value(np.mean(Fx)), round_value(np.mean(Dx)), round_value(np.mean(td)), round_value(np.mean(bke)), round_value(np.mean(Fk)), round_value(np.mean(Dk)), round_value(np.mean(s)), round_value(np.mean(slope)), round_value(np.mean(Xceps)), round_value(np.mean(Pk)), round_value(np.mean(F2Dk)), round_value(np.mean(t_eta)), round_value(ml)],
            'Std': [round_value(np.std(etas)), round_value(np.std(betas)), round_value(np.std(epsilons)), round_value(np.std(xcs)), round_value(np.std(t3)), round_value(np.std(bxe)), round_value(np.std(Fx)), round_value(np.std(Dx)), round_value(np.std(td)), round_value(np.std(bke)), round_value(np.std(Fk)), round_value(np.std(Dk)), round_value(np.std(s)), round_value(np.std(slope)), round_value(np.std(Xceps)), round_value(np.std(Pk)), round_value(np.std(F2Dk)), round_value(np.std(t_eta)), round_value(mle)],
        }

    columns = ['Eta', 'Beta', 'Epsilon', 'Xc', 'Beta/Eta', 'Beta*Xc/Eps', 'Fx=Beta^2/(Eta*Xc)', 'Dx=Beta*Eps/(Eta*Xc^2)', 'Xc^2/Eps', 'Beta*Kappa/Eps', 'Fk=Beta^2/(Eta*Kappa)', 'Dk=Beta*Eps/(Eta*Kappa^2)', 's=(Xc^1.5*Eta^0.5)/Eps', 'Slope=Eta*Xc/Eps', 'Xc/Eps', 'Fk/Dk', 'Fk^2/Dk', 't_eta', 'Median Lifetime']

    df = pd.DataFrame(data, index=columns)
    dfr = df.transpose()

    print(dfr.to_string())
    if file_path is not None:
        dfr.to_csv(file_path)
    return dfr