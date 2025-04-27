import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd




class Posterior:
    def __init__(self, samples, lnprobs, bins, log=False, progress_bar=True):
        
        #if samples is none make empty posterrior
        if samples is None:
            self.samples = None
            self.lnprobs = None
            self.unique_samples = None
            self.posterior = None
            self.lnprobs_density = None
            self.dthetas = None
            self.evidence = None
            self.df = None
            return

        self.samples = samples.copy()
        self.lnprobs = lnprobs.copy()
        #check that bins and log are lists of the correct length (equal to the number of features-samples.shape[1]
        if isinstance(bins, int):
            bins = [bins] * samples.shape[1]
        elif len(bins) != samples.shape[1]:
            raise ValueError("Bins should be an integer or the length of bins should be equal to the number of features")
        if isinstance(log, bool):
            log = [log] * samples.shape[1]
        elif len(log) != samples.shape[1]:
            raise ValueError("log should be a boolean or the length of log should be equal to the number of features")
        self.bins = bins.copy()
        self.log = log
        self.logged_samples  = [np.log(samples[:,i]) if log[i] else samples[:,i] for i in range(samples.shape[1])]
        self.logged_samples = np.array(self.logged_samples).T
        self.progress_bar = progress_bar
        self.unique_samples, self.posterior, self.lnprobs_density, self.dthetas, self.evidence = get_posterior(self.logged_samples, lnprobs, bins, log=False, progress_bar=progress_bar, full_output=True)
        self.df = None

    def marginalize_posterior(self, features, density = True):
        unique_samples, lnprobs , dthetas = marginalized_posterior(self.unique_samples, self.lnprobs_density-self.evidence, features, self.dthetas, density=density)
        return unique_samples, lnprobs, dthetas

    def get_stats(self, stats=['mean','std'],percentiles = [16, 50, 95], center_percentiles=True, smooth_mode = False):
        return getStats(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, stats=stats, percentiles=percentiles, center_percentiles=center_percentiles, smooth_mode=smooth_mode)

    def plot_1d_posteriors(self, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean',"std","percentiles", "mode"],percentiles = [16, 50, 95],smooth_mode=False):
        plot_1d_posteriors(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, ax=ax, colors=colors, labels=labels, truths=truths,smooth_mode=smooth_mode, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log)

    def plot_2d_posteriors(self, features, ax=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean',"std","percentiles", "mode"],percentiles = [16, 50, 95],plot_type = 'contourf', **kwargs):
        return plot_2d_posteriors(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, features, ax=ax, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log,plot_type=plot_type, **kwargs)

    def corner_plot(self, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean',"std","percentiles", "mode"],percentiles = [16, 50, 95],plot_type = 'contourf', **kwargs):
        return corner_plot(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, ax=ax, colors=colors, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log,plot_type=plot_type, **kwargs)
    
    def create_posterior_df(self,transforms = ['default'],labels = None, ds=None,ds_labels=None, kappa=0.5, filepath = None, smooth_mode = True, rescale=None):
        """
        Creates a summerey pandas dataframe of the maximum posterior statistics for the samples.
        For each transfor in the transforms list, the statistics are calculated and added to the dataframe.
        The default transforms are several transforms that assume the samples are in [xc/eta, beta/eta, xc^2/epsilon,xc]
        space and calculate also eta, beta, epsilon,t_eta=sqrt(xc/eta), s= eta^0.5*xc^1.5/epsilon, beta*xc/epsilon,slope = eta*xc/epsilon, 
            Fx = beta^2/(eta*xc), Dx = beta*epsilon/(eta*(xc^2)), Pk=beta*kappa/epsilon, Fk = beta^2/(eta*kappa), 
            Dk = beta*epsilon/(eta*kappa^2), Fk^2/Dk = beta^3/(eta*epsilon).
        For each transform, the statistics sample, mean, std, mode and percentiles are calculated on each marginalized posterior of each parameter
            and added to the dataframe. values of sample with maximum likelihood is also added.
        
        Parameters:
        transforms: list
            The transforms to apply to the samples. Each transform T recieves a sample A=[a1,a2...,an] and a kappa value, and should return 
            a transfored sample of same dimentionality T(A,kappa)=B, B=[b1,b2...bn]. The transform should be reversable (the transformed params should span the 
            same space as the original params).
            Default is ['default'], which is a default list of transforms that assume the samples are in [xc/eta, beta/eta, xc^2/epsilon,xc]
        labels: list
            The labels of the parameters.Should be a list of lists to go with the transforms supplied. If the transforms are [T1,T2...Tn],
            and the samples are of dimentionality n, the labels should be of the form [[label1_T1,label2_T1,...labeln_T1],[label1_T2,label2_T2,...labeln_T2],...]
            If a label repeats itself for several transforms, its assumed to be equivalent and would be added to the dataframe only once.
            If None, and only 'default' is in the transforms list, the default labels will be used for the default transforms.
        ds: list or deathTimesDataset.Dataset
            A deathTimesDataset.Dataset object or several in a list. If provided, the middian and maximal lifetime in each dataset will be added to the dataframe.
        ds_labels: list
            The labels of the datasets. Should be a list of strings. If None, labels are set to 'DS 1', 'DS 2', etc.
        kappa: float
            The value of kappa to use in the transforms that involve kappa. Default is 0.5
        filepath: str
            The path to save the dataframe to as csv. If None, the dataframe will not be saved.
        """
        #if rescale is a float or int, and "default" is in the transforms list, rescale the samples as a rescaling of time:
        n_features = self.unique_samples.shape[1]

        time_rescale = 1
        if 'default' in transforms and isinstance(rescale,(float,int)):
            print(f"Rescaling the samples TIME by {rescale}")
            time_rescale = rescale
            rescale = [rescale**2,rescale,rescale,1]
            if n_features > 4:
                rescale += [1]*(n_features-4)


        if 'default' in transforms:
            transforms = [identity_transform,default_transform1,default_transform2,default_transform3,default_transform4,default_transform5]
            if labels is None:
                labels = [["xc/eta","beta/eta","xc^2/epsilon","xc"],["eta","beta","epsilon","xc"],
                          ["sqrt(xc/eta)","s= eta^0.5*xc^1.5/epsilon","beta*xc/epsilon","xc"],
                          ["eta*xc/epsilon","Fx=beta^2/eta*xc","Dx =beta*epsilon/eta*xc^2","xc"],
                          ["Pk=beta*k/epsilon","Fk=beta^2/eta*k","beta/eta","xc"],
                          ["Dk =beta*epsilon/eta*k^2","Fk^2/Dk=beta^3/eta*epsilon","beta/eta","xc"]]
            if n_features > 4:
                extra_labels = ['ExtH']
                extra_labels += [f'lambda{i}' for i in range(n_features-4)]
                for i,label in enumerate(labels):
                    labels[i] += extra_labels
             

        if ds is not None:
            if not isinstance(ds, list):
                ds = [ds]
            if ds_labels is None:
                ds_labels = [f'DS {i+1}' for i in range(len(ds))]

        

        if rescale is not None:
            self.rescale(rescale)
            
        
        percentiles = [16, 50, 95]
        stats=['mean', 'std', 'mode', 'percentiles']
        summery_dict = {}
        for transform, label_set in zip(transforms, labels):
            trans = lambda x: transform(x,kappa)
            transformed_samples = np.apply_along_axis(trans, 1, self.samples)
            post = Posterior(transformed_samples.copy(), self.lnprobs.copy(), self.bins.copy(), log=self.log, progress_bar=self.progress_bar)
            max_liklihood = transformed_samples[np.argmax(self.lnprobs)]
            for i in range(n_features):
                #check if the label is already calculated
                if label_set[i] in summery_dict.keys():
                    continue
            
                marginalized_samples, marginalized_lnprobs, marginalized_dthetas = post.marginalize_posterior( [j for j in range(n_features) if j != i], density = True)
                stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, percentiles=percentiles,smooth_mode=smooth_mode)
                #if log[i] then exponentiate the mean, mode and percentiles. std is [exp(mean+std)-exp(mean),exp(mean)-exp(mean-std)]
                if self.log[i]:
                    stats_dict['std'] = [np.exp(stats_dict['mean']+stats_dict['std'])-np.exp(stats_dict['mean']),np.exp(stats_dict['mean'])-np.exp(stats_dict['mean']-stats_dict['std'])]
                    stats_dict['mean'] = np.exp(stats_dict['mean'])
                    stats_dict['mode'] = np.exp(stats_dict['mode'])
                    for percentile in percentiles:
                        stats_dict[f'percentile_{percentile}'] = [np.exp(stats_dict[f'percentile_{percentile}'][0]), np.exp(stats_dict[f'percentile_{percentile}'][1])]
                #round all values to 4 non 0 decimal points:
                for key in stats_dict.keys():
                    #if the value is a list, round each element
                    if isinstance(stats_dict[key],list):
                        stats_dict[key] = [round_value(val,3) for val in stats_dict[key]]
                    else:
                        stats_dict[key] = round_value(stats_dict[key],3)
                stats_dict['max_likelihood'] = round_value(max_liklihood[i],3)
                summery_dict[label_set[i]]=stats_dict

        if ds is not None:
            for dataset, ds_label in zip(ds, ds_labels):
                ML = round_value(dataset.getMedianLifetime()*time_rescale)
                mCI = dataset.getMedianLifetimeCI()
                mCI = [round_value(val*time_rescale) for val in mCI]
                mCI = [ML-mCI[0],ML+mCI[1]]
                maxLS = round_value(dataset.getMaxLifetime()*time_rescale)
                stats_dict_ML = {'max_likelihood': ML, 'mean': ML, 'std': mCI[1]-ML, 'mode': ML}
                stats_dict_maxLS = {'max_likelihood': maxLS, 'mean': maxLS, 'std': 0, 'mode': maxLS}
                for percentile in percentiles:
                    stats_dict_ML[f'percentile_{percentile}'] = mCI
                    stats_dict_maxLS[f'percentile_{percentile}'] = [maxLS,maxLS]
                summery_dict[ds_label+'_MedianLifetime'] = stats_dict_ML
                summery_dict[ds_label+'_MaxLifetime'] = stats_dict_maxLS

        self.df = pd.DataFrame(summery_dict).T  

        if filepath is not None:
            #add csv at the end of the file name if not already there
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            self.df.to_csv(filepath, index=True)

        return self.df 
    
    def rescale(self, rescale):
        """
        recreates the posterior object with the samples rescaled by the rescale factor.
        parameters:
        rescale: ndarray
            The rescale factor for each feature. Should be of the same length as the number of features.
        """
        new_samples  = [self.samples[:,i]*rescale[i] for i in range(self.samples.shape[1])]
        new_samples = np.array(new_samples).T
        self.__init__(new_samples, self.lnprobs, self.bins, log=self.log, progress_bar=self.progress_bar)
        return self
    
    def save(self, filepath):
        """
        Save the posterior object to a file.
        parameters:
        filepath: str
            The path to save the file to.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


def load_posterior(filepath):
    """
    Load a posterior object from a file.
    parameters:
    filepath: str
        The path to load the file from.
    returns:
    posterior: Posterior
        The loaded posterior object.
    """
    import pickle
    with open(filepath, 'rb') as f:
        posterior = pickle.load(f)
    return posterior

        

        

def bin_index(samples, bins, index,log =False, return_bins=False):
    """
    Bin the samples for the given index. The bins can be provided as a list of bin edges or as a single integer.
    The samples are of shape (n_samples, n_features) and the index is the feature index along which to bin the samples.
    parameters:
    samples: np.ndarray
        The samples to be binned
    bins: int or list
        The number of bins or the bin edges
    index: int
        The feature index along which to bin the samples
    returns:
    binned_samples: np.ndarray
        The samples, binned along the relevant feature
    bins: np.ndarray
        The bin edges
    binned_index: np.ndarray
        The index of the bin to which each sample belongs
    """
    if isinstance(bins, int):
        if log:
            bins = np.logspace(np.log10(samples[:, index].min()*0.999), np.log10(samples[:, index].max()*1.001), bins + 1)
        else:
            if samples[:, index].min()<0:
                bottom = samples[:, index].min()*1.001
            else:
                bottom = samples[:, index].min()*0.999
            if samples[:, index].max()<0:
                top = samples[:, index].max()*0.999
            else:
                top = samples[:, index].max()*1.001
            bins = np.linspace(bottom,top, bins + 1)

    binned_index = np.digitize(samples[:, index], bins)
    #print the samples that have binned_index = 0 or binned_index = len(bins)
    if (binned_index == 0).any():
        print(f"Samples with binned_index = 0: {samples[binned_index == 0]}")
    if (binned_index == len(bins)).any():
        print(f"Samples with binned_index = len(bins): {samples[binned_index == len(bins)]}")
    binned_values = 0.5 * (bins[binned_index] + bins[binned_index - 1])
    binned_samples  = samples.copy()
    binned_samples[:, index] = binned_values

    if return_bins:
        return binned_samples, bins, binned_index
    else:
        return binned_samples
    

def bin_samples(samples, bins, log=False, return_bins=False, progress_bar=True):
    """
    Bin the samples. The bins can be provided as a list of bin edges or as a single integer.
    The samples are of shape (n_samples, n_features).
    parameters:
    samples: np.ndarray
        The samples to be binned
    bins: int or list
        The number of bins or the bin edges. if int, the same number of bins is used for all features, 
        if list, the length of the list should be equal to the number of features and each element of the list
        should be either an integer or a list of bin edges.
    log: bool or list
        Whether to use log binning for each feature. If a list, the length of the list should be equal to the number of features.
        in this case, the elements of the list should be boolean values indicating whether to use log binning for the corresponding feature. 
    returns:
    binned_samples: np.ndarray
        The samples, binned along all features
    """
    if isinstance(bins, int):
        bins = [bins] * samples.shape[1]
    if isinstance(log, bool):
        log = [log] * samples.shape[1]

    binned_samples = samples.copy()
    binned_index = np.zeros(samples.shape)
    
    if progress_bar:
        iterator = tqdm(range(samples.shape[1]), desc="Binning samples")
    else:
        iterator = range(samples.shape[1])

    if return_bins:
        for i in iterator:
            binned_samples, bins[i], binned_index[:,i] = bin_index(binned_samples, bins[i], i, log=log[i], return_bins=True)
    else:
        for i in iterator:
            binned_samples = bin_index(binned_samples, bins[i], i, log=log[i], return_bins=False)

    if return_bins:
        return binned_samples, bins, binned_index
    else:
        return binned_samples


def avarage_samples(binned_samples,lnprobs, calc_prob_volume=False,bins =None,binned_index=None, progress_bar=True):
    """
    Avarage the lnprobability of the samples in point on a multidimentional grid. The samples are of shape (n_samples, n_features).
    parameters:
    binned_samples: np.ndarray
        The samples, binned along all features
    lnprobs: np.ndarray
        The log-probabilities of the samples
    bins: list
        The bin edges. Should be provided if calc_prob_volume is True
    binned_index: np.ndarray
        The index of the bin to which each sample belongs along each feature. Should be provided if calc_prob_volume is True
    calc_prob_volume: bool
        Whether to calculate the volume of the bins. This will be used to calculate the probability volume.
    returns:
    unique_samples: np.ndarray
        The unique smaple values
    avaraged_lnprobs: np.ndarray
        The avaraged log-probabilities in each bin
    volume: np.ndarray
        if calc_prob_volume is True, the volume of each bin associated with the avaraged log-probabilities
    """
    raw_unique_samples = defaultdict(lambda: [np.zeros(binned_samples.shape[1]), 0, 0, 0])
    if progress_bar:
        iterator = tqdm(range(binned_samples.shape[0]), desc="Processing samples")
    else:
        iterator = range(binned_samples.shape[0])
    
    if calc_prob_volume:
        if bins is None:
            raise ValueError("bins should be provided to calculate the probability volume")
        if binned_index is None:
            raise ValueError("binned_index should be provided to calculate the probability volume")

    for i in iterator:
        key = tuple(binned_samples[i, :])
        raw_unique_samples[key][0] = binned_samples[i]
        raw_unique_samples[key][1] += lnprobs[i]
        raw_unique_samples[key][2] += 1
        if calc_prob_volume and raw_unique_samples[key][3] == 0:
            raw_unique_samples[key][3] = [bins[j][int(binned_index[i,j])] - bins[j][int(binned_index[i,j])-1] for j in range(binned_samples.shape[1])]
            if (np.prod(raw_unique_samples[key][3])<=0):
                message =f"Volume is zero or negative:  {raw_unique_samples[key][3]}  "
                for j in range(binned_samples.shape[1]):
                   message+= f"[bins[{j}][int(binned_index[{i},{j}])] (bin #{binned_index[i,j]}): {bins[j][int(binned_index[i,j])]}, bins[{j}][int(binned_index[{i},{j}])-1], {bins[j][int(binned_index[i,j])-1]})"
                raise ValueError(message)
    raw_unique_samples = list(raw_unique_samples.values())

    avaraged_lnprob_density = []
    unique_samples = []
    volumes = []
    dthetas = []
    
    if progress_bar:
        iterator = tqdm(range(len(raw_unique_samples)), desc="Averaging log-probabilities")
    else:
        iterator = range(len(raw_unique_samples))
    
    for i in iterator:
        avaraged_lnprob_density.append(raw_unique_samples[i][1] / raw_unique_samples[i][2])
        unique_samples.append(raw_unique_samples[i][0])
        dthetas.append(raw_unique_samples[i][3])
        if calc_prob_volume:
            volumes.append(np.prod(raw_unique_samples[i][3]))
    
    if calc_prob_volume:
        return np.array(unique_samples), np.array(avaraged_lnprob_density), np.array(dthetas), np.array(volumes)

    return np.array(unique_samples), np.array(avaraged_lnprob_density), np.array(dthetas)


def get_posterior(samples,lnprobs,bins,log=False, progress_bar=True, full_output=False):
    """
    Get the posterior distribution from the samples. The samples are of shape (n_samples, n_features).
    parameters:
    samples: np.ndarray
        The samples
    lnprobs: np.ndarray
        The log-probabilities of the samples
    bins: int or list
        The number of bins or the bin edges. if int, the same number of bins is used for all features, 
        if list, the length of the list should be equal to the number of features and each element of the list
        should be either an integer or a list of bin edges.
    log: bool or list
        Whether to use log binning for each feature. If a list, the length of the list should be equal to the number of features.
        in this case, the elements of the list should be boolean values indicating whether to use log binning for the corresponding feature. 
    returns:
    unique_samples: np.ndarray
        The unique smaple values
    avaraged_lnprobs: np.ndarray
        The avaraged log-probabilities in each bin
    """
    binned_samples, bins, binned_index = bin_samples(samples, bins, log=log, return_bins=True, progress_bar=progress_bar)
    unique_samples, lnprobs_density,dthetas, volumes = avarage_samples(binned_samples,
                                                                lnprobs,calc_prob_volume=True,
                                                                bins=bins,binned_index=binned_index, 
                                                                progress_bar=progress_bar)
    
    #Multiply the avaraged log-probabilities by the volume of the bins to get the posterior.
    #Because we use lnprobabilities, we can add the log of the volumes to the log-probabilities.
    if(np.isnan(lnprobs_density)).any():
        print("avaraged_lnprobs contains NaNs")
    if(np.isnan(volumes)).any():
        print("volumes contains NaNs")
    if np.any(volumes <= 0):
        print("volumes contains zeros or negative values")
    evidence = calulate_evidence(lnprobs_density,volumes)
    posterior_lnprobs =lnprobs_density+ np.log(volumes)-evidence
    if np.isnan(lnprobs_density).any():
        print("final_lnprobs contains NaNs")
    if full_output:
        return unique_samples, posterior_lnprobs,lnprobs_density,dthetas, evidence
    return unique_samples, posterior_lnprobs

    
def marginalized_posterior(unique_samples,lnprobs_density,features,dthetas, density =True):
    """
    Get the marginalized posterior distribution from the samples. The samples are of shape (n_samples, n_features).
    parameters:
    unique_samples: np.ndarray
        The unique smaple values
    lnprobs: np.ndarray
        The log-probabilities of the samples
    features: list
        The feature indices along which to marginalize the posterior
    returns:
    marginalized_samples: np.ndarray
        The marginalized samples
    marginalized_lnprobs: np.ndarray
        The marginalized log-probabilities
    """
    from scipy.special import logsumexp

    #if we are looking for probabilitoes and not densities, we need to multiply the probability density by the volume of the bins
    if not density:
        volumes = np.prod(dthetas,axis=1)
        lnprobs = lnprobs_density + np.log(volumes)
    else:
        lnprobs = lnprobs_density 

    #sort the feature indices in descending order to make sure that we marginalize the posterior in the correct order.
    features = sorted(features, reverse=True)
    
    #for every feature in features, we need to sum the probabilities of all the samples that have the same value for all the other features
    #if we are working with densities, we need to multiply the probabilities by the dtheta (delta theta) along the marginalized feature
    for feature in features:
        raw_unique_samples = defaultdict(lambda: [np.zeros(unique_samples.shape[1]), [],[]])
        for i in range(unique_samples.shape[0]):
            key = tuple(np.delete(unique_samples[i, :], feature))
            raw_unique_samples[key][0] = np.delete(unique_samples[i, :], feature)
            raw_unique_samples[key][1].append(lnprobs[i]+np.log(dthetas[i,feature]))
            raw_unique_samples[key][2] = np.delete(dthetas[i,:], feature)

        raw_unique_samples = list(raw_unique_samples.values())

        avaraged_lnprobs = []
        unique_samples = []
        dthetas = []
        for i in range(len(raw_unique_samples)):
            unique_samples.append(raw_unique_samples[i][0])
            avaraged_lnprobs.append(logsumexp(np.array((raw_unique_samples[i][1]))))
            dthetas.append(np.array(raw_unique_samples[i][2]))

        unique_samples = np.squeeze(np.array(unique_samples))
        lnprobs = np.squeeze(np.array(avaraged_lnprobs))
        dthetas = np.array(dthetas)
    dthetas = np.squeeze(np.array(dthetas))



    sort_index = np.argsort(lnprobs)
    unique_samples = unique_samples[sort_index]
    lnprobs = lnprobs[sort_index]
    dthetas = dthetas[sort_index]


    return unique_samples, lnprobs , dthetas


def calulate_evidence(lnprobs,volumes):
    """
    Calculate the evidence from the samples. The samples are of shape (n_samples, n_features).
    parameters:
    lnprobs: np.ndarray
        The log-probabilities of the samples
    volumes: np.ndarray
        The volume of the bins associated with the avaraged log-probabilities
    returns:
    evidence: float
        The evidence
    """
    from scipy.special import logsumexp
    evidence = logsumexp(lnprobs + np.log(volumes))
    return evidence

def plot_1d_posteriors(unique_samples,lnprobs_densities,dthetas,ax=None,colors=None,labels=None,truths=None,smooth_mode=False, scale ='log',show_ln_prob = True, stats = ['mean',"std","percentiles"],percentiles = [16, 50, 95],log=None,truth_label = "Best fit"):
    """ 
    Parameters:
    unique_samples : np.ndarray
        The unique sample values.
    lnprobs_densities : np.ndarray
        The log-probabilities or densities of the samples.
    dthetas : np.ndarray
        The differential elements for each parameter.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axes on which to plot the posteriors. If None, new axes will be created.
    colors : list, optional
        The colors of the posteriors. If None, default colors will be used.
    labels : list, optional
        The labels of the posteriors. If None, default labels will be used.
    truths : list, optional
        The true values of the posteriors. If provided, vertical lines will be drawn at these values.
    scale : str, optional
        The scale of the x-axis. Can be 'log' or 'linear'. Default is 'log'.
    show_ln_prob : bool, optional
        If True, the log-probabilities will be shown. If False, the probabilities will be exponentiated.
    stats : list, optional
        The statistics to calculate and dispaly. Can be 'mean', 'std', 'percentiles' or 'mode'

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    n_features = unique_samples.shape[1]
    if log is None:
        log = [False] * n_features
    if ax is None:
        fig, ax = plt.subplots(n_features, 1, figsize=(8, 3 * n_features))
        fig.tight_layout(pad=4.0)
    if n_features == 1:
        ax = [ax]
    for i in range(n_features):
        if colors is None:
            color = "C" + str(i)
        else:
            color = colors[i]
        if labels is None:
            label = f"Feature {i}"
        else:
            label = labels[i]
        
        if n_features == 1:
            marginalized_samples, marginalized_lnprobs, marginalized_dthetas = np.squeeze(unique_samples), np.squeeze(lnprobs_densities), np.squeeze(dthetas)
        else:
            marginalized_samples, marginalized_lnprobs, marginalized_dthetas = marginalized_posterior(unique_samples, lnprobs_densities, [j for j in range(n_features) if j != i],dthetas, density = True)
        # marginalized_samples = marginalized_samples
        #sort the samples
        sort_index = np.argsort(marginalized_samples)
        marginalized_samples = marginalized_samples[sort_index]
        marginalized_lnprobs = marginalized_lnprobs[sort_index]
        marginalized_dthetas = marginalized_dthetas[sort_index]
        stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats,smooth_mode=smooth_mode, percentiles=percentiles)
        if 'mean' in stats:
            mean_i = np.exp(stats_dict['mean']) if log[i] else stats_dict['mean']
            ax[i].axvline(mean_i, color="k", linestyle="--", label="Mean")
        if 'std' in stats:
            pstd_i = np.exp((stats_dict['mean']+stats_dict['std'])) if log[i] else (stats_dict['mean']+stats_dict['std'])
            mstd_i = np.exp((stats_dict['mean']-stats_dict['std'])) if log[i] else (stats_dict['mean']-stats_dict['std'])
            ax[i].axvline(pstd_i, color="gray", linestyle=":", label="Std")
            ax[i].axvline(mstd_i, color="gray", linestyle=":")
        if 'percentiles' in stats:
            last_low = np.exp(stats_dict['mode']) if log[i] else stats_dict['mode']
            last_high = np.exp(stats_dict['mode']) if log[i] else stats_dict['mode']
            alpha = 0.1*(len(percentiles))+0.1
            for percentile in percentiles:
                low = np.exp(stats_dict[f'percentile_{percentile}'][0]) if log[i] else stats_dict[f'percentile_{percentile}'][0]
                high = np.exp(stats_dict[f'percentile_{percentile}'][1]) if log[i] else stats_dict[f'percentile_{percentile}'][1]
                # if percentile == 95: ###DEBUG###
                #     print(f"{label} low: {low}, high: {high}")
                top = np.max(marginalized_lnprobs) if show_ln_prob else np.exp(np.max(marginalized_lnprobs))
                bottom = np.min(marginalized_lnprobs) if show_ln_prob else np.exp(np.min(marginalized_lnprobs))
                ax[i].fill_betweenx([bottom,top], low, last_low, color=color, alpha=alpha, label=f"{percentile}th percentile")
                ax[i].fill_betweenx([bottom,top], last_high, high, color=color, alpha=alpha )
                last_low = low
                last_high = high
                alpha -= 0.1

        if 'mode' in stats:
            mode_i = np.exp(stats_dict['mode']) if log[i] else stats_dict['mode']
            ax[i].axvline(mode_i, color="k", linestyle="-.", label="Mode")
        if not show_ln_prob:
            marginalized_lnprobs = np.exp(marginalized_lnprobs)
        if log[i]:
            marginalized_samples = np.exp(marginalized_samples)
        ax[i].plot(marginalized_samples, marginalized_lnprobs, color=color, label=label)
        if truths is not None:
            ax[i].axvline(truths[i], color="red", linestyle="--", label=truth_label)
        ylabel = "ln Posterior prob" if show_ln_prob else "Posterior prob"
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel(f"{label}")
        if scale == 'log':
            ax[i].set_xscale('log')
        ax[i].legend()
        
def plot_2d_posteriors(unique_samples,lnprobs_densities,dthetas,features, ax=None,labels=None,truths=None, scale ='log',show_ln_prob = True, stats = ['mean',"std","percentiles"],percentiles = [16, 50, 95],log=None,plot_type = 'contourf', **kwargs):
    """ 
    Parameters:
    unique_samples : np.ndarray
        The unique sample values.
    lnprobs_densities : np.ndarray
        The log-probabilities or densities of the samples.
    dthetas : np.ndarray
        The differential elements for each parameter.
    features : list
        The features to plot
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axes on which to plot the posteriors. If None, new axes will be created.
    labels : list, optional
        The labels of the posteriors. If None, default labels will be used.
    truths : list, optional
        The true values of the posteriors. If provided, vertical lines will be drawn at these values.
    scale : str, optional
        The scale of the axis. Can be 'log' or 'linear'. Default is 'log'.
    show_ln_prob : bool, optional
        If True, the log-probabilities will be shown. If False, the probabilities will be exponentiated.
    stats : list, optional
        The statistics to calculate and dispaly. Can be 'mean', 'std', 'percentiles' or 'mode'

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    n_features = unique_samples.shape[1]
    if log is None:
        log = [False] * n_features
    if ax is None:
        fig, ax = plt.subplots()
    
    marginalized_samples, marginalized_lnprobs, marginalized_dthetas = marginalized_posterior(unique_samples, lnprobs_densities, [j for j in range(n_features) if not j in features],dthetas, density = True)
    if log[features[0]]:
        marginalized_samples[:,0] = np.exp(marginalized_samples[:,0])
    if log[features[1]]:
        marginalized_samples[:,1] = np.exp(marginalized_samples[:,1])

    #Plot the 2D posterior on a mesh grid
    x = marginalized_samples[:,0]
    y = marginalized_samples[:,1]
    z = marginalized_lnprobs
    if not show_ln_prob:
        z = np.exp(z)
    X, Y = np.meshgrid(x, y)
    # make the Z values a 2D grid. for eache x,y pair, the z value is the probability density
    Z = np.zeros(X.shape)
    if show_ln_prob:
        Z = Z+np.min(z)
    for i in range(len(x)):
        Z[i,i] = z[i]
    if plot_type == 'contourf':
        #if levels is not provided, the number of levels will be 100
        if 'levels' not in kwargs:
            kwargs['levels'] = 100
        ax.contourf(X, Y, Z, **kwargs)
    elif plot_type == 'pcolormesh':
        ax.pcolormesh(X, Y, Z, **kwargs)
    else:
        raise ValueError(f"plot_type {plot_type} is not supported. Supported types are 'contourf' and 'pcolormesh'")    

    if labels is None:
        labels = [f"Feature {features[0]}", f"Feature {features[1]}"]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    else:
        ax.set_xlabel(labels[features[0]])
        ax.set_ylabel(labels[features[1]])
    if truths is not None:
        ax.axvline(truths[features[0]], color="red", linestyle="--", label="Truth")
        ax.axhline(truths[features[1]], color="red", linestyle="--")
    # ax.legend()
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_aspect('auto')
    ax.set_title("2D Posterior")
    return X, Y


def corner_plot(unique_samples,lnprobs_densities,dthetas,ax=None,colors=None,labels=None,truths=None, scale ='log',show_ln_prob = True, stats = ['mean',"std","percentiles"],percentiles = [16, 50, 95],log=None,plot_type = 'contourf',progress_bar=True, **kwargs):
    """ 
    Parameters:
    unique_samples : np.ndarray
        The unique sample values.
    lnprobs_densities : np.ndarray
        The log-probabilities or densities of the samples.
    dthetas : np.ndarray
        The differential elements for each parameter.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axes on which to plot the posteriors. If None, new axes will be created.
    colors : list, optional
        The colors of the posteriors. If None, default colors will be used.
    labels : list, optional
        The labels of the posteriors. If None, default labels will be used.
    truths : list, optional
        The true values of the posteriors. If provided, vertical lines will be drawn at these values.
    scale : str, optional
        The scale of the x-axis. Can be 'log' or 'linear'. Default is 'log'.
    show_ln_prob : bool, optional
        If True, the log-probabilities will be shown. If False, the probabilities will be exponentiated.
    stats : list, optional
        The statistics to calculate and dispaly. Can be 'mean', 'std', 'percentiles' or 'mode'

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    n_features = unique_samples.shape[1]
    if log is None:
        log = [False] * n_features
    if ax is None:
        fig, ax = plt.subplots(n_features, n_features, figsize=(5 * n_features, 5 * n_features))
        fig.tight_layout(pad=3.0)
    diagonal_axes = [ax[i,i] for i in range(n_features)]
    plot_1d_posteriors(unique_samples,lnprobs_densities,dthetas,ax=diagonal_axes,colors=colors,labels=labels,truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=log)
    
    if progress_bar:
        iterator = tqdm(range(n_features), desc="Creating corner plot")
    else:
        iterator = range(n_features)
    
    for i in iterator:
        for j in range(i):
            plot_2d_posteriors(unique_samples,lnprobs_densities,dthetas,features=[j,i], ax=ax[i,j], labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=log,plot_type=plot_type, **kwargs)
        for j in range(i+1,n_features):
            ax[i,j].axis('off')
    return ax




def getStats(samples, lnprobs, dthetas, stats=['mean','std'],percentiles = [16, 50, 84], center_percentiles=True, smooth_mode = True,debug=False):
    """
    Get the statistics of the samples. The samples are of shape (n_samples, n_features).
    Can be used to get the mean, standard deviation, median, percentiles and mode of the samples.
    parameters:
    samples: np.ndarray
        The samples
    lnprobs: np.ndarray
        The log-probability density of the samples
    dthetas: np.ndarray
        The differential elements for each parameter for each sample.
    stats: list
        The statistics to calculate. Can be 'mean', 'std', 'percentiles' or 'mode'
    percentiles: list
        The percentiles to calculate. Default is [16, 50, 84]
    center_percentiles: bool
        Whether to center the percentiles around the mean. If True each percentile returned as a tuple (low,high) where the
         where the probability in the inteval mean-low = mean-high = 1/2 of the percentile. Default is True
    returns:
    stats_dict: dict
        A dictionary containing the statistics
    """
    #if we are working with densities, we need to multiply the probabilities by the volume of the bins
    #if the data is 1D, the volume is just the dtheta
    if len(samples.shape) == 1 or samples.shape[1] == 1:
        
        #sort the samples and the probabilities
        sort_index = np.argsort(samples)
        samples = samples[sort_index]
        lnprobs = lnprobs[sort_index]
        dthetas = dthetas[sort_index]
        volumes = dthetas
    else:
        volumes = np.prod(dthetas,axis=1)
    lnprobs = lnprobs + np.log(volumes)
    stats_dict = {}

   

    if 'mean' in stats:
        stats_dict['mean'] = np.average(samples, weights=np.exp(lnprobs), axis=0)
    if 'std' in stats:
        stats_dict['std'] = np.sqrt(np.average((samples - stats_dict['mean'])**2, weights=np.exp(lnprobs), axis=0))
    if 'mode' in stats:
        #smooth the log-probabilities
        if smooth_mode:
            from scipy.ndimage import gaussian_filter
            std = len(samples)/50
            lnprobs = gaussian_filter(lnprobs, std)
        stats_dict['mode'] = samples[np.argmax(lnprobs)]
        
        if debug: print(f"mode: {np.exp(stats_dict['mode']), np.exp(lnprobs[np.argmax(lnprobs)])}")

    if 'percentiles' in stats:
        #calculate the percentiles. percentiles should be calculated only if number of features is 1
        if len(samples.shape) == 1 or samples.shape[1] == 1:
            
            #calculate the cumulative probabilities
            cum_probs = np.exp(lnprobs)
            cum_probs = np.cumsum(cum_probs)
            # cum_probs /= cum_probs[-1]
            probs = np.exp(lnprobs)/cum_probs[-1]
            #find the percentiles
            if center_percentiles:
                mean_percentile = np.interp(stats_dict['mode'], samples, cum_probs)
                for i, percentile in enumerate(percentiles):
                    target = percentile / 100.0
                    mode_index = np.argmax(probs)
                    prob_low = mode_index
                    prob_high = mode_index
                    total_prob = probs[mode_index]
                    
                    # if total_prob >= target:
                    #     new_high = ne
                    #     if prob_high+1<len(lnprobs): new_high = prob_high +1
                    #     if prob_low-1>=0: prob_low -= 1
                    
                    while total_prob < target:
                        new_low = prob_low - 1
                        new_high = prob_high + 1
                        if new_low < 0 and new_high >= len(probs):
                            raise ValueError(f"Percentile {percentile} not found, total probability is {total_prob}")
                        elif new_low < 0:
                            prob_high = new_high
                            total_prob += probs[new_high]
                        elif new_high >= len(probs):
                            prob_low = new_low
                            total_prob += probs[new_low]
                        else:
                            # expand towards the side with the higher adjacent probability
                            if probs[new_low] > probs[new_high]:
                                prob_low = new_low
                                total_prob += probs[new_low]
                            else:
                                prob_high = new_high
                                total_prob += probs[new_high]
                                
                    low = samples[prob_low]-0.5*dthetas[prob_low]
                    high = samples[prob_high]+0.5*dthetas[prob_high]
                    stats_dict[f'percentile_{percentile}'] = (low, high)

            else:
                cum_probs /= cum_probs[-1]
                for i, percentile in enumerate(percentiles):
                    stats_dict[f'percentile_{percentiles[i]}'] = np.interp(percentile / 100, cum_probs, samples)
        else:
            print("Percentiles can only be calculated for 1D samples")
    
    return stats_dict


import joint_posterior as jp
class JointPosterior(jp.JointPosterior):
    def __init__(self,samples_list,lnprobs_list,bins,log=False,progress_bar=True):
        return super().__init__(samples_list,lnprobs_list,bins,log,progress_bar)

def posterior_of_joint_distribution(samples_list,lnprobs_list,bins,log=False,progress_bar=True):
    """
    Returns a posterior object of the joint distribution of many samples lists.
    """
    #if bins is an integer or a list of integers, it means we need to build the bins for each feature so all samples sets are on the same grid.
    #this we do by building the grid on the widest range of the samples along each feature.
    
    n_features = samples_list[0].shape[1]

    if isinstance(bins, int):
        bins = [bins] * n_features
    if isinstance(log, bool):
        log = [log] * n_features

    for feature in range(n_features):

        if isinstance(bins[feature], int):
            nbins = bins[feature]
            min_sample = np.min([np.min(samples[:, feature]) for samples in samples_list])
            max_sample = np.max([np.max(samples[:, feature]) for samples in samples_list])
            if log[feature]:
                min_sample = np.log(min_sample)
                max_sample = np.log(max_sample)
            
            if min_sample<0:
                bottom = min_sample*1.001
            else:
                bottom = min_sample.min()*0.999
            if max_sample<0:
                top = max_sample*0.999
            else:
                top = max_sample*1.001
            bins[feature] = np.linspace(bottom,top, nbins + 1)
    
    posts =[]
    raw_unique_samples = defaultdict(lambda: [np.zeros(n_features), 0, 0])

    for i in range(len(samples_list)):
        post = Posterior(samples_list[i],lnprobs_list[i],bins,log=log,progress_bar=progress_bar)
        posts.append(post)
        if progress_bar:
            iterator = tqdm(range(post.unique_samples.shape[0]), desc=f"Processing unique samples for set {i}")
        else:
            iterator = range(post.unique_samples.shape[0])

        for j in iterator:
            key = tuple(post.unique_samples[j, :])
            raw_unique_samples[key][0] = post.unique_samples[j,:]
            raw_unique_samples[key][1] += post.lnprobs_density[j]
            if  isinstance(raw_unique_samples[key][2],int) and raw_unique_samples[key][2]== 0:
                raw_unique_samples[key][2] = post.dthetas[j,:]
            
    
    raw_unique_samples = list(raw_unique_samples.values())

    unique_samples = np.array([sample[0] for sample in raw_unique_samples])
    lnprobs_density = np.array([sample[1] for sample in raw_unique_samples])
    dthetas = np.array([sample[2] for sample in raw_unique_samples])
    volumes = np.prod(dthetas,axis=1)
    
    evidence = calulate_evidence(lnprobs_density,volumes)
    posterior_lnprobs =lnprobs_density+ np.log(volumes)-evidence

    joint_post = Posterior(samples =None,lnprobs=None,bins=bins,log=log,progress_bar=progress_bar)
    
    joint_post.bins = bins
    joint_post.log = log
    joint_post.progress_bar = progress_bar
    joint_post.unique_samples = unique_samples
    joint_post.lnprobs_density = lnprobs_density
    joint_post.posterior = posterior_lnprobs
    joint_post.dthetas = dthetas
    joint_post.evidence = evidence

    return joint_post




    

            


def identity_transform(sample,kappa):
    return sample

def default_transform1(sample,kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    return [eta, beta, epsilon] + list(sample[3:])

def default_transform2(sample, kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    t_eta = np.sqrt(xc_eta)
    s = xc2_epsilon/t_eta
    bxe = beta*xc/epsilon
    
    return [t_eta, s, bxe] + list(sample[3:])

def default_transform3(sample, kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    slope = xc2_epsilon/xc_eta
    Fx = beta ** 2 / (eta * xc)
    Dx = beta * epsilon / (eta * (xc ** 2))
    
    return [slope, Fx, Dx] + list(sample[3:])

def default_transform4(sample, kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    Pk = beta * kappa / epsilon
    Fk = beta ** 2 / (eta * kappa)
    
    return [Pk, Fk, beta_eta] + list(sample[3:])

def default_transform5(sample, kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    Dk = beta * epsilon / (eta * kappa ** 2)
    Fk2_Dk = beta ** 3 / (eta * epsilon)
    return [Dk, Fk2_Dk, beta_eta] + list(sample[3:])



def round_value(value, precision=2):
    if value == 0:
        return 0
    elif abs(value) < 1:
        r = int(np.abs(np.log10(abs(value))))
        return round(value, r+precision)
    else:
        return round(value, precision)