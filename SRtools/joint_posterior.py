import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from . import samples_utils as su

    
    
class JointPosterior(su.Posterior):    
    def __init__(self, samples_list, lnprobs_list, bins, log=False, progress_bar=True, config_params=None, help_text=None, prior=None, sorting=True):
        """
        Returns a posterior object of the joint distribution of many samples lists.
        """
        #if bins is an integer or a list of integers, it means we need to build the bins for each feature so all samples sets are on the same grid.
        #this we do by building the grid on the widest range of the samples along each feature.
        
        if prior is not None:
            import warnings
            warnings.warn("Using joint posteriors with priors is not tested. Marginalizations might not be calculated properly.", UserWarning)

                # Initialize new attributes from Posterior class
        self.config_params = config_params
        self.help_text = help_text
        self.prior = prior
        self.sorting = sorting
        self.prior_unique_samples = None
        self.prior_posterior = None
        self.prior_dthetas = None
        self.prior_lnprobs = None
        self.samples_list = samples_list.copy()
        self.lnprobs_list = lnprobs_list.copy()
        self.raw_bins = bins.copy() if isinstance(bins,list) or isinstance(bins,np.ndarray) else bins
        
        n_features = samples_list[0].shape[1]

        self.n_features = n_features

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
        

        #if the prior is a Posterior object, we need to set the prior attributes, else we assume the prior is a uniformative prior
        if prior is not None and isinstance(prior, su.Posterior):
            self.prior_unique_samples = prior.unique_samples.copy()
            self.prior_posterior = prior.posterior.copy()
            self.prior_dthetas = prior.dthetas.copy()
            self.prior =True
        else:
            self.prior = False # Check if the samples_list and lnprobs_list are lists of lists

        posts =[]
        raw_unique_samples = defaultdict(lambda: [np.zeros(n_features), 0, 0])


        for i in range(len(samples_list)):
            post = su.Posterior(samples_list[i],lnprobs_list[i],bins,log=log,progress_bar=progress_bar,config_params=config_params,help_text=help_text,prior=prior,sorting=sorting)
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
        evidence = su.calulate_evidence(lnprobs_density,volumes)

        if prior:
            # Evaluate the prior probability density at each sample
            prior_lnprobs = []
            iterator = tqdm(range(len(unique_samples)), desc="Calculating prior for posterior bins") if progress_bar else range(len(unique_samples))
            for i in iterator:
                sample = unique_samples[i]
                # we need to calculate the probability density of the prior at the sample point
                # Using the "get_probability_in_region" method allows us to account for different bin sizes between the prior and the posterior
                prob = self.get_probability_in_region(sample, dthetas[i], density=True, log_check=False, prior =True)
                prior_lnprobs.append(prob)
            prior_lnprobs = np.array(prior_lnprobs)
        else:
            #we assume an uniformative prior, so we set the prior to 1
            prior_lnprobs = np.zeros_like(lnprobs_density)
  
        posterior_lnprobs =lnprobs_density+ np.log(volumes)-evidence + prior_lnprobs
        
        self.bins = bins.copy()
        self.log = log
        self.progress_bar = progress_bar
        self.unique_samples = unique_samples
        self.lnprobs_density = lnprobs_density
        self.posterior = posterior_lnprobs
        self.prior_lnprobs = prior_lnprobs
        self.dthetas = dthetas
        self.evidence = evidence
        self.df = None


        

        # Ensure unique_samples is sorted (if not already sorted)
        if sorting:
            sort_indices = np.lexsort(self.unique_samples.T[::-1])  # Sort by all dimensions
            self.unique_samples = self.unique_samples[sort_indices]
            self.lnprobs_density = self.lnprobs_density[sort_indices]
            self.dthetas = self.dthetas[sort_indices]
            if self.posterior is not None:
                self.posterior = self.posterior[sort_indices]
            if self.evidence is not None and type(self.evidence) is np.ndarray:
                if  len(self.evidence) == len(self.lnprobs_density):
                    self.evidence = self.evidence[sort_indices]
            if self.prior_lnprobs is not None and type(self.prior_lnprobs) is np.ndarray:
                if  len(self.prior_lnprobs) == len(self.lnprobs_density):
                    self.prior_lnprobs = self.prior_lnprobs[sort_indices]
        # Call parent class's __make_normalized_samples__ method
        super().__make_normalized_samples__()

    #raise error if self.samples or self.lnprobs are called:
    @property
    def samples(self):
        raise AttributeError("JointPosterior object has no attribute 'samples'")
    
    @property
    def lnprobs(self):
        raise AttributeError("JointPosterior object has no attribute 'lnprobs'")
    
    def create_posterior_df(self,transforms = ['default'],labels = None, ds=None,ds_labels=None, kappa=0.5, filepath = None, smooth_mode = True, rescale=None, set_xc=False, truth=None):
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
        set_xc: bool or float, optional
            If True, uses the xc value from the sample. If a float, uses that fixed xc value for all transforms.
            When set_xc is provided, the transformed samples will have 3 parameters instead of 4 (xc is not returned).
            Default is False.
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
            transforms = [identity_transform,default_transform1,default_transform2,default_transform3,default_transform4,default_transform5,default_transform6,default_transform7]
            if labels is None:
                if set_xc is not False and set_xc is not None:
                    # When set_xc is provided, transformed samples have 3 params instead of 4 (xc is not returned)
                    labels = [["xc/eta","beta/eta","xc^2/epsilon"],["eta","beta","epsilon"],
                              ["sqrt(xc/eta)","s= eta^0.5*xc^1.5/epsilon","beta*xc/epsilon"],
                              ["eta*xc/epsilon","Fx=beta^2/eta*xc","Dx =beta*epsilon/eta*xc^2"],
                              ["Pk=beta*k/epsilon","Fk=beta^2/eta*k","beta/eta"],
                              ["Dk =beta*epsilon/eta*k^2","Fk^2/Dk=beta^3/eta*epsilon","beta/eta"],
                              ["epsilon/beta^2","k/beta","k^2/epsilon"],
                              ["eta/xc","beta/xc","epsilon/xc^2"]]
                else:
                    labels = [["xc/eta","beta/eta","xc^2/epsilon","xc"],["eta","beta","epsilon","xc"],
                              ["sqrt(xc/eta)","s= eta^0.5*xc^1.5/epsilon","beta*xc/epsilon","xc"],
                              ["eta*xc/epsilon","Fx=beta^2/eta*xc","Dx =beta*epsilon/eta*xc^2","xc"],
                              ["Pk=beta*k/epsilon","Fk=beta^2/eta*k","beta/eta","xc"],
                              ["Dk =beta*epsilon/eta*k^2","Fk^2/Dk=beta^3/eta*epsilon","beta/eta","xc"],
                              ["epsilon/beta^2","k/beta","k^2/epsilon","xc"],
                              ["eta/xc","beta/xc","epsilon/xc^2","k/xc"]]
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
        
        # Rescale truth if provided
        if truth is not None:
            truth = np.array(truth)
            if rescale is not None:
                truth_rescaled = truth * np.array(rescale)
            else:
                truth_rescaled = truth.copy()
        else:
            truth_rescaled = None
            
        percentiles = [16, 50, 95]
        stats=['mean', 'std', 'mode', 'percentiles']
        summery_dict = {}
        for transform, label_set in zip(transforms, labels):
            # Handle set_xc logic
            # When set_xc is used, samples are already [xc/eta, beta/eta, xc^2/epsilon] (or with ExtH)
            # and don't contain xc, so we don't trim them - pass samples as-is to transform
            if set_xc is not False and set_xc is not None:
                # Use set_xc value (float) - samples are already without xc, pass them as-is (no trimming)
                trans = lambda x: transform(x, kappa, set_xc=set_xc)
                transformed_samples_list = [np.apply_along_axis(trans, 1, samples) for samples in self.samples_list]
            else:
                trans = lambda x: transform(x, kappa)
                transformed_samples_list = [np.apply_along_axis(trans, 1, samples) for samples in self.samples_list]
            
            # Transform truth value
            if truth_rescaled is not None:
                if set_xc is not False and set_xc is not None:
                    truth_transformed = transform(truth_rescaled, kappa, set_xc=set_xc)
                else:
                    truth_transformed = transform(truth_rescaled, kappa)
            else:
                truth_transformed = None
            
            # n_features is already correct (samples are already in the format without xc when set_xc is used)
            n_features_transformed = transformed_samples_list[0].shape[1] if len(transformed_samples_list) > 0 else n_features
            # Adjust log array to match transformed dimensions
            if isinstance(self.log, list):
                log_transformed = self.log[:n_features_transformed] if len(self.log) >= n_features_transformed else self.log
            elif isinstance(self.log, np.ndarray):
                log_transformed = self.log[:n_features_transformed] if len(self.log) >= n_features_transformed else self.log
            else:
                log_transformed = self.log
            
            post = JointPosterior(transformed_samples_list.copy(), self.lnprobs_list.copy(), self.raw_bins, log=log_transformed, progress_bar=self.progress_bar)
            max_liklihood = [transformed_samples_list[i][np.argmax(self.lnprobs_list[i])] for i in range(len(transformed_samples_list))]
            max_likelihood_overall_index = np.argmax([max(self.lnprobs_list[i]) for i in range(len(self.lnprobs_list))])
            max_liklihood = max_liklihood[max_likelihood_overall_index]
            mode_overall = post.get_mode()
            for i in range(n_features_transformed):
                #check if the label is already calculated
                if label_set[i] in summery_dict.keys():
                    continue
            
                marginalized_post = post.marginalize_posterior([j for j in range(n_features_transformed) if j != i], density=True)
                marginalized_samples = marginalized_post.unique_samples
                marginalized_lnprobs = marginalized_post.lnprobs_density - marginalized_post.evidence
                marginalized_dthetas = marginalized_post.dthetas
                marginalized_prior_lnprobs = marginalized_post.prior_lnprobs if marginalized_post.prior_lnprobs is not None else 0
                marginalized_lnprobs = marginalized_lnprobs + marginalized_prior_lnprobs
                stats_dict = su.getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, percentiles=percentiles, smooth_mode=smooth_mode)
                #if log[i] then exponentiate the mean, mode and percentiles. std is [exp(mean+std)-exp(mean),exp(mean)-exp(mean-std)]
                # Use log_transformed for transformed features
                if isinstance(log_transformed, (list, np.ndarray)):
                    log_val = log_transformed[i]
                else:
                    log_val = log_transformed
                if log_val:
                    stats_dict['std'] = [np.exp(stats_dict['mean'])-np.exp(stats_dict['mean']-stats_dict['std']),np.exp(stats_dict['mean']+stats_dict['std'])-np.exp(stats_dict['mean'])]
                    stats_dict['mean'] = np.exp(stats_dict['mean'])
                    stats_dict['mode'] = np.exp(stats_dict['mode'])
                    for percentile in percentiles:
                        stats_dict[f'percentile_{percentile}'] = [np.exp(stats_dict[f'percentile_{percentile}'][0]), np.exp(stats_dict[f'percentile_{percentile}'][1])]
                #round all values to 4 non 0 decimal points:
                for key in stats_dict.keys():
                    #if the value is a list, round each element
                    if isinstance(stats_dict[key],list):
                        stats_dict[key] = [float(round_value(val,3)) for val in stats_dict[key]]
                    else:
                        stats_dict[key] = float(round_value(stats_dict[key],3))
                    if isinstance(stats_dict[key],np.ndarray):
                        stats_dict[key] = stats_dict[key].tolist()
                stats_dict['max_likelihood'] = float(round_value(max_liklihood[i],3))
                stats_dict['mode_overall'] = float(round_value(mode_overall[i],3))
                
                # Add truth comparison if truth is provided
                if truth_transformed is not None:
                    # Get truth value for this parameter, handling log scaling
                    truth_val = truth_transformed[i]

                    # Get max_likelihood and mode_overall values, handling log scaling
                    max_likelihood_val = max_liklihood[i]
                    mode_overall_val = mode_overall[i]
                    
                    
                    # Handle division by zero for percentage calculations
                    truth_abs = abs(truth_val)
                    if truth_abs < 1e-10:
                        # Use np.inf for division by zero cases
                        error_mode = np.inf
                        error_max_likelihood = np.inf
                        error_mean = np.inf
                        error_mode_overall = np.inf
                        ci95_size_pct = np.inf
                    else:
                        # Compute error columns as percentage of truth value
                        error_mode = (abs(truth_val - stats_dict['mode']) / truth_abs) * 100
                        error_max_likelihood = (abs(truth_val - max_likelihood_val) / truth_abs) * 100
                        error_mean = (abs(truth_val - stats_dict['mean']) / truth_abs) * 100
                        error_mode_overall = (abs(truth_val - mode_overall_val) / truth_abs) * 100
                        
                        # Compute 95% CI size as percentage of truth
                        if 'percentile_95' in stats_dict:
                            ci95_low, ci95_high = stats_dict['percentile_95']
                            ci95_size_pct = ((ci95_high - ci95_low) / truth_abs) * 100
                        else:
                            ci95_size_pct = np.nan
                    
                    # Round error values
                    error_mode = float(round_value(error_mode, 3)) if not np.isinf(error_mode) else error_mode
                    error_max_likelihood = float(round_value(error_max_likelihood, 3)) if not np.isinf(error_max_likelihood) else error_max_likelihood
                    error_mean = float(round_value(error_mean, 3)) if not np.isinf(error_mean) else error_mean
                    error_mode_overall = float(round_value(error_mode_overall, 3)) if not np.isinf(error_mode_overall) else error_mode_overall
                    ci95_size_pct = float(round_value(ci95_size_pct, 3)) if not (np.isinf(ci95_size_pct) or np.isnan(ci95_size_pct)) else ci95_size_pct
                    
                    # Add error columns to stats_dict
                    stats_dict['error_mode'] = error_mode
                    stats_dict['error_max_likelihood'] = error_max_likelihood
                    stats_dict['error_mean'] = error_mean
                    stats_dict['error_mode_overall'] = error_mode_overall
                    stats_dict['ci95_size_pct'] = ci95_size_pct
                    
                    # Compute CI coverage columns (boolean)
                    for percentile in percentiles:
                        percentile_key = f'percentile_{percentile}'
                        if percentile_key in stats_dict:
                            percentile_low, percentile_high = stats_dict[percentile_key]
                            in_ci = (truth_val >= percentile_low) and (truth_val <= percentile_high)
                            stats_dict[f'in_percentile_{percentile}_CI'] = bool(in_ci)
                    
                    # Compute std CI coverage
                    if 'std' in stats_dict:
                        std_val = stats_dict['std']
                        mean_val = stats_dict['mean']
                        if isinstance(std_val, list):
                            # std is [lower, upper] format
                            std_lower, std_upper = std_val
                            std_ci_low = mean_val - std_lower
                            std_ci_high = mean_val + std_upper
                        else:
                            # std is scalar
                            std_ci_low = mean_val - std_val
                            std_ci_high = mean_val + std_val
                        in_std_ci = (truth_val >= std_ci_low) and (truth_val <= std_ci_high)
                        stats_dict['in_std_CI'] = bool(in_std_ci)
                
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
        new_samples =[]
        for j in range(len(self.samples_list)):
            new_samples.append( [self.samples_list[j][:,i]*rescale[i] for i in range(self.n_features)])
            new_samples[j] = np.array(new_samples[j]).T
        self.__init__(new_samples, self.lnprobs_list, self.raw_bins, log=self.log, progress_bar=self.progress_bar)
        return self
    
    def save_to_file(self, filepath, save_raw_data=True):
        """
        Save the joint posterior attributes to a CSV file and optionally save raw data (samples_list and lnprobs_list) to a separate binary file.
        parameters:
        filepath: str
            The base path to save the CSV and binary files.
        save_raw_data: bool, optional
            Whether to save the binary file containing raw data (samples_list and lnprobs_list). Default is True.
        """
        import numpy as np
        import pandas as pd
        import json

        # Save metadata to a CSV file
        data = {
            "log": [self.log] * self.unique_samples.shape[0],
            "unique_samples": self.unique_samples.tolist() if self.unique_samples is not None else None,
            "lnprobs_density": self.lnprobs_density.tolist() if self.lnprobs_density is not None else None,
            "posterior": self.posterior.tolist() if self.posterior is not None else None,
            "dthetas": self.dthetas.tolist() if self.dthetas is not None else None,
            "evidence": self.evidence,
            "config_params": [self.config_params] * self.unique_samples.shape[0] if self.config_params is not None else None,
            "help_text": [self.help_text] * self.unique_samples.shape[0] if self.help_text is not None else None,
            "prior_lnprobs": self.prior_lnprobs.tolist() if self.prior_lnprobs is not None else None,
            "prior_unique_samples": self.prior_unique_samples.tolist() if self.prior_unique_samples is not None else None,
            "prior_posterior": self.prior_posterior.tolist() if self.prior_posterior is not None else None,
            "prior_dthetas": self.prior_dthetas.tolist() if self.prior_dthetas is not None else None,
            "prior": [self.prior] * self.unique_samples.shape[0] if hasattr(self, "prior") else None,
            "raw_bins": [self.raw_bins] * self.unique_samples.shape[0] if self.raw_bins is not None else None,
        }

        # Ensure the file has a .csv extension
        if not filepath.endswith('.csv'):
            filepath += '.csv'
        pd.DataFrame(data).to_csv(filepath, index=False)

        # Save large arrays (samples_list and lnprobs_list) to a separate binary file if save_raw_data is True
        if save_raw_data:
            binary_filepath = filepath.replace('.csv', '_data.npz')
            save_dict = {
                'config_params': json.dumps(self.config_params) if self.config_params is not None else None,
                'help_text': self.help_text.encode('utf-8') if self.help_text is not None else None,
                'prior_lnprobs': self.prior_lnprobs if self.prior_lnprobs is not None else None,
                'n_samples': len(self.samples_list)
            }
            
            # Save each item in samples_list and lnprobs_list separately
            for i, (samples, lnprobs) in enumerate(zip(self.samples_list, self.lnprobs_list)):
                save_dict[f'samples_{i}'] = samples
                save_dict[f'lnprobs_{i}'] = lnprobs
            
            np.savez_compressed(binary_filepath, **save_dict)

    @staticmethod
    def load_from_file(filepath, load_raw_data=True):
        """
        Load joint posterior attributes from a CSV file and optionally load raw data (samples_list and lnprobs_list) from a separate binary file.
        parameters:
        filepath: str
            The base path to load the CSV and binary files from.
        load_raw_data: bool, optional
            Whether to load the binary file containing raw data (samples_list and lnprobs_list). Default is True.
        returns:
        loaded_posterior: JointPosterior
            The loaded joint posterior object.
        """
        import numpy as np
        import pandas as pd
        import ast

        # Check if the file has a .csv extension
        if not filepath.endswith('.csv'):
            filepath += '.csv'

        # Load metadata from the CSV file
        df = pd.read_csv(filepath)
        
        # Helper function to safely evaluate raw_bins
        def safe_eval_raw_bins(value):
            if pd.isna(value):
                return None
            try:
                # Try to evaluate as a literal first
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If that fails, try to convert to int
                try:
                    return int(value)
                except (ValueError, TypeError):
                    # If that fails, return the value as is
                    return value

        posterior_data = {
            "log": ast.literal_eval(df["log"].iloc[0]) if "log" in df.columns else None,
            "unique_samples": np.array([ast.literal_eval(item) for item in df["unique_samples"].dropna()]),
            "lnprobs_density": np.array([item for item in df["lnprobs_density"].dropna()]),
            "posterior": np.array([item for item in df["posterior"].dropna()]),
            "dthetas": np.array([ast.literal_eval(item) for item in df["dthetas"].dropna()]),
            "evidence": df["evidence"].iloc[0] if "evidence" in df.columns else None,
            "config_params": ast.literal_eval(df["config_params"].iloc[0]) if "config_params" in df.columns and pd.notna(df["config_params"].iloc[0]) else None,
            "help_text": df["help_text"].iloc[0] if "help_text" in df.columns and pd.notna(df["help_text"].iloc[0]) else None,
            "prior_lnprobs": np.array([item for item in df["prior_lnprobs"].dropna()]) if "prior_lnprobs" in df.columns else None,
            "prior_unique_samples": np.array([ast.literal_eval(item) for item in df["prior_unique_samples"].dropna()]) if "prior_unique_samples" in df.columns else None,
            "prior_posterior": np.array([item for item in df["prior_posterior"].dropna()]) if "prior_posterior" in df.columns else None,
            "prior_dthetas": np.array([ast.literal_eval(item) for item in df["prior_dthetas"].dropna()]) if "prior_dthetas" in df.columns else None,
            "prior": df["prior"].iloc[0] if "prior" in df.columns else None,
            "raw_bins": safe_eval_raw_bins(df["raw_bins"].iloc[0]) if "raw_bins" in df.columns and pd.notna(df["raw_bins"].iloc[0]) else None,
        }

        # Initialize samples_list, lnprobs_list, and prior_lnprobs as None
        samples_list = None
        lnprobs_list = None
        prior_lnprobs = posterior_data["prior_lnprobs"]

        # Load large arrays (samples_list and lnprobs_list) from the binary file if load_raw_data is True
        if load_raw_data:
            binary_filepath = filepath.replace('.csv', '_data.npz')
            with np.load(binary_filepath, allow_pickle=True) as data:
                n_samples = data['n_samples']
                samples_list = []
                lnprobs_list = []
                
                # Load each item in samples_list and lnprobs_list
                for i in range(n_samples):
                    samples_list.append(data[f'samples_{i}'])
                    lnprobs_list.append(data[f'lnprobs_{i}'])
                
                if 'prior_lnprobs' in data:
                    prior_lnprobs = data['prior_lnprobs']

        # Create and populate the JointPosterior object
        loaded_posterior = JointPosterior(samples_list=samples_list, lnprobs_list=lnprobs_list, bins=posterior_data["raw_bins"], 
                                        log=posterior_data["log"], progress_bar=True)
        
        # Set additional attributes
        loaded_posterior.unique_samples = posterior_data["unique_samples"]
        loaded_posterior.lnprobs_density = posterior_data["lnprobs_density"]
        loaded_posterior.posterior = posterior_data["posterior"]
        loaded_posterior.dthetas = posterior_data["dthetas"]
        loaded_posterior.evidence = posterior_data["evidence"]
        loaded_posterior.config_params = posterior_data["config_params"]
        loaded_posterior.help_text = posterior_data["help_text"]
        loaded_posterior.prior_lnprobs = prior_lnprobs
        loaded_posterior.prior_unique_samples = posterior_data["prior_unique_samples"]
        loaded_posterior.prior_posterior = posterior_data["prior_posterior"]
        loaded_posterior.prior_dthetas = posterior_data["prior_dthetas"]
        loaded_posterior.prior = posterior_data["prior"]
        loaded_posterior.raw_bins = posterior_data["raw_bins"]

        # Normalize the unique samples
        loaded_posterior.__make_normalized_samples__()

        return loaded_posterior

def identity_transform(sample,kappa, set_xc=None):
    return sample

def default_transform1(sample,kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    if set_xc is not None:
        return [eta, beta, epsilon] + list(sample[3:])
    else:
        return [eta, beta, epsilon] + list(sample[3:])

def default_transform2(sample, kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    t_eta = np.sqrt(xc_eta)
    s = xc2_epsilon/t_eta
    bxe = beta*xc/epsilon
    
    if set_xc is not None:
        return [t_eta, s, bxe] + list(sample[3:])
    else:
        return [t_eta, s, bxe] + list(sample[3:])

def default_transform3(sample, kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    slope = xc2_epsilon/xc_eta
    Fx = beta ** 2 / (eta * xc)
    Dx = beta * epsilon / (eta * (xc ** 2))
    
    if set_xc is not None:
        return [slope, Fx, Dx] + list(sample[3:])
    else:
        return [slope, Fx, Dx] + list(sample[3:])

def default_transform4(sample, kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    Pk = beta * kappa / epsilon
    Fk = beta ** 2 / (eta * kappa)
    
    if set_xc is not None:
        return [Pk, Fk, beta_eta] + list(sample[3:])
    else:
        return [Pk, Fk, beta_eta] + list(sample[3:])

def default_transform5(sample, kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta = xc / xc_eta
    beta = beta_eta * eta
    epsilon = (xc ** 2)/xc2_epsilon 
    Dk = beta * epsilon / (eta * kappa ** 2)
    Fk2_Dk = beta ** 3 / (eta * epsilon)
    if set_xc is not None:
        return [Dk, Fk2_Dk, beta_eta] + list(sample[3:])
    else:
        return [Dk, Fk2_Dk, beta_eta] + list(sample[3:])

def default_transform6(sample,kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    epsilon_beta2 = xc_eta**2/(beta_eta**2 * xc2_epsilon)
    eta = xc / xc_eta
    k_beta = kappa/(beta_eta * eta)
    k2_epsilon = kappa**2/((xc ** 2)/xc2_epsilon )
    if set_xc is not None:
        return [epsilon_beta2, k_beta, k2_epsilon] + list(sample[3:])
    else:
        return [epsilon_beta2, k_beta, k2_epsilon] + list(sample[3:])

def default_transform7(sample,kappa, set_xc=None):
    if set_xc is not None:
        xc_eta, beta_eta, xc2_epsilon = sample[:3]
        xc = set_xc
    else:
        xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    eta_xc = 1 / xc_eta
    beta_xc = beta_eta * eta_xc
    epsilon_xc2 = 1/xc2_epsilon 
    kappa_xc = kappa /xc
    if set_xc is not None:
        return [eta_xc, beta_xc, epsilon_xc2] + list(sample[3:])
    elif len(sample) > 4:
        return [eta_xc, beta_xc, epsilon_xc2, kappa_xc] + list(sample[4:])
    else:
        return [eta_xc, beta_xc, epsilon_xc2, kappa_xc]

def round_value(value, precision=2):
    if value == 0:
        return 0
    elif abs(value) < 1:
        r = int(np.abs(np.log10(abs(value))))
        return float(round(value, r+precision))
    else:
        return float(round(value, precision))