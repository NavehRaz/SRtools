import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import ast
import json
import plotly.graph_objects as go





class Posterior:
    def __init__(self, samples, lnprobs, bins, log=False, progress_bar=True, config_params=None, help_text = None, prior=None, sorting=True):
        

        self.mins = None
        self.maxs = None
        self.ranges = None
        self.norm_unique_samples = None

        #if samples is none make empty posterrior
        if samples is None:
            self.samples = None
            self.lnprobs = None
            self.unique_samples = None
            self.posterior = None
            self.lnprobs_density = None
            self.dthetas = None
            self.evidence = None
            self.prior_lnprobs = None
            self.df = None

            return

        self.config_params = config_params
        self.help_text = help_text
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
        self.unique_samples, self.posterior, self.lnprobs_density, self.dthetas, self.evidence, self.prior_lnprobs = get_posterior(self.logged_samples, lnprobs, bins, log=False, progress_bar=progress_bar, full_output=True, prior=prior)
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

            # Normalize the unique samples
            self.__make_normalized_samples__()
        #Print warning if the samples are not sorted
        else:
            print("Warning: The samples and posterior are not sorted. This may cause problems when using the posterior to calculate probabilities.")

    def __make_normalized_samples__(self):
        """
        Normalize the unique samples to the range [0, 1] for each feature.
        """
        self.mins = self.unique_samples.min(axis=0)
        self.maxs = self.unique_samples.max(axis=0)
        self.ranges = self.maxs - self.mins
        # Avoid division by zero for constant features
        self.ranges[self.ranges == 0] = 1.0
        self.norm_unique_samples = (self.unique_samples - self.mins) / self.ranges

    def marginalize_posterior(self, features, density = True):
        unique_samples, lnprobs , dthetas = marginalized_posterior(self.unique_samples, self.lnprobs_density-self.evidence, features, self.dthetas, density=density)
        return unique_samples, lnprobs, dthetas

    def get_stats(self, stats=['mean','std'],percentiles = [16, 50, 95], center_percentiles=True, smooth_mode = False, prior_lnprobs=None):
        if prior_lnprobs is None and hasattr(self, 'prior_lnprobs'):
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
        return getStats(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, stats=stats, percentiles=percentiles, center_percentiles=center_percentiles, smooth_mode=smooth_mode, prior_lnprobs=prior_lnprobs)

    def plot_1d_posteriors(self, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],smooth_mode=False, prior_lnprobs=None):
        if prior_lnprobs is None and hasattr(self, 'prior_lnprobs'):
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
        plot_1d_posteriors(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, ax=ax, colors=colors, labels=labels, truths=truths, smooth_mode=smooth_mode, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log, prior_lnprobs=prior_lnprobs)

    def plot_2d_posteriors(self, features, ax=None, labels=None, truths=None, scale='log', show_ln_prob=False, stats=['mean','std','percentiles', 'mode'], percentiles=[16, 50, 95], plot_type='contourf', prior_lnprobs=None, log=None, smooth_mode=False, **kwargs):
        """
        Plot 2D marginalized posteriors using the standalone plot_2d_posteriors function for consistency.
        """
        if prior_lnprobs is None and hasattr(self, 'prior_lnprobs'):
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
        return plot_2d_posteriors(
            self.unique_samples,
            self.lnprobs_density - self.evidence,
            self.dthetas,
            features,
            ax=ax,
            labels=labels,
            truths=truths,
            scale=scale,
            show_ln_prob=show_ln_prob,
            stats=stats,
            percentiles=percentiles,
            plot_type=plot_type,
            prior_lnprobs=prior_lnprobs,
            log=self.log if log is None else log,
            smooth_mode=smooth_mode,
            **kwargs
        )

    def corner_plot(self, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],plot_type = 'contourf', prior_lnprobs=0, **kwargs):
        if prior_lnprobs is None and hasattr(self, 'prior_lnprobs'):
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
        return corner_plot(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, ax=ax, colors=colors, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log, plot_type=plot_type, prior_lnprobs=prior_lnprobs, **kwargs)
    
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
            transforms = [identity_transform,default_transform1,default_transform2,default_transform3,default_transform4,default_transform5,default_transform6]
            if labels is None:
                labels = [["xc/eta","beta/eta","xc^2/epsilon","xc"],["eta","beta","epsilon","xc"],
                          ["sqrt(xc/eta)","s= eta^0.5*xc^1.5/epsilon","beta*xc/epsilon","xc"],
                          ["eta*xc/epsilon","Fx=beta^2/eta*xc","Dx =beta*epsilon/eta*xc^2","xc"],
                          ["Pk=beta*k/epsilon","Fk=beta^2/eta*k","beta/eta","xc"],
                          ["Dk =beta*epsilon/eta*k^2","Fk^2/Dk=beta^3/eta*epsilon","beta/eta","xc"],
                          ["beta^2/epsilon","k/beta","k/epsilon","xc"]]
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
                stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, percentiles=percentiles, smooth_mode=smooth_mode)
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
    
    
    def get_sample_probability(self, sample, density = False):
        """
        Calculate the probability of a sample given a posterior object using binary search for efficiency.
        
        Parameters:
        posterior: Posterior
            The posterior object containing unique_samples, lnprobs_density, and dthetas.
        sample: np.ndarray
            The sample for which to calculate the probability. Should have the same dimensionality as the posterior's unique_samples.
        
        Returns:
        probability: float
            The probability of the sample.
        """
        unique_samples = self.unique_samples
        lnprobs_density = self.lnprobs_density
        dthetas = self.dthetas
        posterior = self.posterior

        sample =sample.copy()

        #log the sample according to the log flag
        if hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            sample = np.array([
                np.log(val) if log_flag else val
                for val, log_flag in zip(sample, self.log)
            ])

        norm_sample = (sample - self.mins) / self.ranges

        # Find the index of the nearest sample (using Euclidean distance)
        distances = np.linalg.norm(self.norm_unique_samples - norm_sample, axis=1)
        idx = np.argmin(distances)
        print(f"Sample: {sample}, Nearest sample index: {idx}, Distance: {distances[idx]}")
        # Check if the sample lies within the bin boundaries
        if idx < len(unique_samples):
            bin_min = unique_samples[idx] - 0.5 * dthetas[idx]
            bin_max = unique_samples[idx] + 0.5 * dthetas[idx]
            if np.all((sample >= bin_min) & (sample <= bin_max)):
                # Calculate the probability
                if density:
                    probability = lnprobs_density[idx] - self.evidence
                else:
                    probability = posterior[idx]  
                return probability

        # If the sample does not fall into any bin, return 0
        return -np.inf
    
    
    def get_probability_in_region(self, sample, dtheta, density = False, log_check = True):
        """
        Calculate the total probability in the region defined by sample ± dtheta/2,
        counting each bin proportional to the overlap fraction.
        Parameters:
            sample : np.ndarray
                The center of the region (shape: n_features).
            dtheta : np.ndarray or float
                The width of the region for each feature (shape: n_features or scalar).
        Returns:
            total_prob : float
                The total probability mass in the specified region.
        """
        sample = np.asarray(sample)
        sample = sample.copy()

        # log the sample according to the log flag
        if log_check and hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            sample = np.array([
                np.log(val) if log_flag else val
                for val, log_flag in zip(sample, self.log)
            ])

        dtheta = np.asarray(dtheta)
        if dtheta.ndim == 0:
            dtheta = np.full(sample.shape, dtheta)
        region_min = sample - dtheta / 2
        region_max = sample + dtheta / 2

        # Vectorized bin boundaries
        bin_min = self.unique_samples - 0.5 * self.dthetas
        bin_max = self.unique_samples + 0.5 * self.dthetas

        # Vectorized overlap calculation
        overlap_min = np.maximum(region_min, bin_min)
        overlap_max = np.minimum(region_max, bin_max)
        overlap = np.maximum(overlap_max - overlap_min, 0)
        overlap_volume = np.prod(overlap, axis=1)
        bin_volume = np.prod(self.dthetas, axis=1)

        # Only consider bins with nonzero overlap
        mask = overlap_volume > 0
        fractions = overlap_volume[mask] / bin_volume[mask]
        probs = self.posterior[mask] + np.log(fractions)
        from scipy.special import logsumexp
        if probs.size == 0:
            return -np.inf
        total_prob = logsumexp(probs)
        if density:
            total_prob -= np.log(np.prod(dtheta))
        return total_prob
    
    
    def get_mode(self, idx=False):
        """
        Get the mode of the posterior distribution.
        returns:
        mode: np.ndarray
            The mode of the posterior distribution.
        """
        mode_index = np.argmax(self.posterior)
        mode = self.unique_samples[mode_index]
        # Exponentiate only the indexes where self.log is True
        if hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            mode = np.array([
            np.exp(val) if log_flag else val
            for val, log_flag in zip(mode, self.log)
            ])
        if idx:
            return mode, mode_index
        return mode
    

    def plot_posterior3D(self, features, ax=None, labels=None, truths=None, scale ='log',show_ln_prob = True, stats = ['mean',"std","percentiles", "mode"],percentiles = [16, 50, 95],plot_type = 'contourf', **kwargs):
        """
        Plot the posterior distribution in 3D.
        parameters:
        features: list
            The features to plot. Should be a list of 3 integers.
        ax: matplotlib.axes._axes.Axes, optional
            The axes to plot on. If None, a new figure and axes will be created.
        labels: list, optional
            The labels for the axes. If None, the default labels will be used.
        truths: list, optional
            The true values of the parameters. If None, no true values will be plotted.
        scale: str, optional
            The scale of the plot. Default is 'log'.
        show_ln_prob: bool, optional
            Whether to show the log-probability or not. Default is False.
        stats: list, optional
            The statistics to show on the plot. Default is ['mean',"std","percentiles", "mode"].
        percentiles: list, optional
            The percentiles to show on the plot. Default is [16, 50, 95].
        plot_type: str, optional
            The type of plot to create. Default is 'contourf'.
        **kwargs: keyword arguments
            Additional arguments to pass to the plotting function.
        """        

        return plot_3d_posterior(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, features, ax=ax, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log,plot_type=plot_type, **kwargs)

    def plot_posterior3D_interactive(self, features, labels=None, truths=None, show_ln_prob=True):
        """
        Plot the posterior distribution in 3D interactively using Plotly.
        parameters:
        features: list
            The features to plot. Should be a list of 3 integers.
        labels: list, optional
            The labels for the axes. If None, the default labels will be used.
        truths: list, optional
            The true values of the parameters. If None, no true values will be plotted.
        show_ln_prob: bool, optional
            Whether to show the log-probability or not. Default is False.
        """
        return plot_3d_posterior_interactive(self.unique_samples, self.lnprobs_density-self.evidence, features, labels=labels, truths=truths, show_ln_prob=show_ln_prob)

    
    def save_pickle(self, filepath):
        """
        Save the posterior object to a file.
        parameters:
        filepath: str
            The path to save the file to.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def save_to_file(self, filepath, save_raw_data=True):
        """
        Save the posterior attributes to a CSV file and optionally save raw data (samples and lnprobs) to a separate binary file.
        parameters:
        filepath: str
            The base path to save the CSV and binary files.
        save_raw_data: bool, optional
            Whether to save the binary file containing raw data (samples and lnprobs). Default is True.
        """
        import numpy as np
        import pandas as pd
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
        }

        # Ensure the file has a .csv extension
        if not filepath.endswith('.csv'):
            filepath += '.csv'
        pd.DataFrame(data).to_csv(filepath, index=False)

        # Save large arrays (samples and lnprobs) to a separate binary file if save_raw_data is True
        if save_raw_data:
            binary_filepath = filepath.replace('.csv', '_data.npz')
            with open(binary_filepath, 'wb') as f:
                np.savez_compressed(
                    f,
                    samples=self.samples,
                    lnprobs=self.lnprobs,
                    config_params=json.dumps(self.config_params) if self.config_params is not None else None,
                    help_text=self.help_text.encode('utf-8') if self.help_text is not None else None,
                    prior_lnprobs=self.prior_lnprobs if self.prior_lnprobs is not None else None
                )

    @staticmethod
    def load_from_file(filepath, load_raw_data=True):
        """
        Load posterior attributes from a CSV file and optionally load raw data (samples and lnprobs) from a separate binary file.
        parameters:
        filepath: str
            The base path to load the CSV and binary files from.
        load_raw_data: bool, optional
            Whether to load the binary file containing raw data (samples and lnprobs). Default is True.
        returns:
        loaded_posterior: Posterior
            The loaded posterior object.
        """
        import numpy as np
        import pandas as pd
        import ast

        # Check if the file has a .csv extension
        if not filepath.endswith('.csv'):
            #add .csv to the end of the file name
            filepath += '.csv'

        # Load metadata from the CSV file
        df = pd.read_csv(filepath)
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
        }

        # Initialize samples, lnprobs, and prior_lnprobs as None
        samples = None
        lnprobs = None
        prior_lnprobs = posterior_data["prior_lnprobs"]

        # Load large arrays (samples and lnprobs) from the binary file if load_raw_data is True
        if load_raw_data:
            binary_filepath = filepath.replace('.csv', '_data.npz')
            with np.load(binary_filepath) as data:
                samples = data['samples']
                lnprobs = data['lnprobs']
                if 'prior_lnprobs' in data:
                    prior_lnprobs = data['prior_lnprobs']

        # Create and populate the Posterior object
        loaded_posterior = Posterior(samples=None, lnprobs=None, bins=None, log=False, progress_bar=True)
        loaded_posterior.unique_samples = posterior_data["unique_samples"]
        loaded_posterior.lnprobs_density = posterior_data["lnprobs_density"]
        loaded_posterior.posterior = posterior_data["posterior"]
        loaded_posterior.dthetas = posterior_data["dthetas"]
        loaded_posterior.evidence = posterior_data["evidence"]
        loaded_posterior.log = posterior_data["log"]
        loaded_posterior.samples = samples
        loaded_posterior.lnprobs = lnprobs
        loaded_posterior.config_params = posterior_data["config_params"]
        loaded_posterior.help_text = posterior_data["help_text"]
        loaded_posterior.prior_lnprobs = prior_lnprobs
        # Normalize the unique samples
        loaded_posterior.__make_normalized_samples__()

        return loaded_posterior
    
def load_raw_data_from_npz(filepath):
    """
    Load raw data (samples, lnprobs, config_params, and help_text) from an npz file.
    parameters:
    filepath: str
        The path to the npz file.
    returns:
    data: dict
        A dictionary containing the raw data.
    """
    with np.load(filepath, allow_pickle=True) as data:
        help_text = data['help_text']
        if help_text.item() is not None and help_text.size > 0:
            help_text = help_text.tobytes().decode('utf-8')
        raw_data = {
            "samples": data['samples'].tolist(),
            "lnprobs": data['lnprobs'].tolist(),
            "config_params": json.loads(data['config_params'].item()) if 'config_params' in data and data['config_params'] is not None and isinstance(data['config_params'], np.ndarray) and data['config_params'].item() is not None else None,
            "help_text": help_text
        }
    return raw_data


def load_posterior_pickle(filepath):
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


def get_posterior(samples,lnprobs,bins,log=False, progress_bar=True, full_output=False, prior=None):
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
    #Because we use lnproabilities, we can add the log of the volumes to the log-probabilities.
    if(np.isnan(lnprobs_density)).any():
        print("avaraged_lnprobs contains NaNs")
    if(np.isnan(volumes)).any():
        print("volumes contains NaNs")
    if np.any(volumes <= 0):
        print("volumes contains zeros or negative values")

    # if the prior is not None, we need to calculate the unnormalized posterior first:
    if prior is not None:
        # Evaluate the prior probability density at each sample
        prior_lnprobs = []
        iterator = tqdm(range(len(unique_samples)), desc="Calculating prior for posterior bins") if progress_bar else range(len(unique_samples))
        for i in iterator:
            sample = unique_samples[i]
            # we need to calculate the probability density of the prior at the sample point
            # Using the "get_probability_in_region" method allows us to account for different bin sizes between the prior and the posterior
            prob = prior.get_probability_in_region(sample, dthetas[i], density=True, log_check=False)
            prior_lnprobs.append(prob)
        prior_lnprobs = np.array(prior_lnprobs)
    else:
        #we assume an uniformative prior, so we set the prior to 1
        prior_lnprobs = np.zeros_like(lnprobs_density)

    evidence = calulate_evidence(lnprobs_density,volumes=volumes, prior_lnprobs=prior_lnprobs)
    

    posterior_lnprobs =lnprobs_density+prior_lnprobs+ np.log(volumes)-evidence
    if np.isnan(lnprobs_density).any():
        print("final_lnprobs contains NaNs")
    if full_output:
        return unique_samples, posterior_lnprobs,lnprobs_density,dthetas, evidence, prior_lnprobs
    return unique_samples, posterior_lnprobs

    
def marginalized_posterior(unique_samples, lnprobs_density, features, dthetas, density=True, prior_lnprobs=None):
    """
    Get the marginalized posterior distribution from the samples. The samples are of shape (n_samples, n_features).
    parameters:
    unique_samples: np.ndarray
        The unique sample values
    lnprobs: np.ndarray
        The log-probabilities of the samples
    features: list
        The feature indices along which to marginalize the posterior
    dthetas: np.ndarray
        The differential elements for each parameter for each sample.
    density: bool
        Whether to return densities (True) or probabilities (False)
    prior_lnprobs: np.ndarray or None
        Optional. The log prior probabilities for each sample. If provided, will be marginalized in the same way as lnprobs.
    returns:
    marginalized_samples: np.ndarray
        The marginalized samples
    marginalized_lnprobs: np.ndarray
        The marginalized log-probabilities
    marginalized_dthetas: np.ndarray
        The marginalized dthetas
    marginalized_prior_lnprobs: np.ndarray or None
        The marginalized prior log-probabilities (if prior_lnprobs was provided)
    """
    from scipy.special import logsumexp

    #if we are looking for probabilities and not densities, we need to multiply the probability density by the volume of the bins
    if not density:
        volumes = np.prod(dthetas, axis=1)
        lnprobs = lnprobs_density + np.log(volumes)
    else:
        lnprobs = lnprobs_density
    if prior_lnprobs is not None:
        prior_lnprobs = np.asarray(prior_lnprobs)
        has_prior = True
    else:
        has_prior = False

    features = sorted(features, reverse=True)
    for feature in features:
        raw_unique_samples = defaultdict(lambda: [np.zeros(unique_samples.shape[1]), [], [], []])
        for i in range(unique_samples.shape[0]):
            key = tuple(np.delete(unique_samples[i, :], feature))
            raw_unique_samples[key][0] = np.delete(unique_samples[i, :], feature)
            raw_unique_samples[key][1].append(lnprobs[i] + np.log(dthetas[i, feature]))
            raw_unique_samples[key][2] = np.delete(dthetas[i, :], feature)
            if has_prior:
                raw_unique_samples[key][3].append(prior_lnprobs[i] + np.log(dthetas[i, feature]))
        raw_unique_samples = list(raw_unique_samples.values())

        avaraged_lnprobs = []
        marginalized_samples = []
        marginalized_dthetas = []
        marginalized_prior_lnprobs = [] if has_prior else None
        for i in range(len(raw_unique_samples)):
            marginalized_samples.append(raw_unique_samples[i][0])
            avaraged_lnprobs.append(logsumexp(np.array(raw_unique_samples[i][1])))
            marginalized_dthetas.append(np.array(raw_unique_samples[i][2]))
            if has_prior:
                marginalized_prior_lnprobs.append(logsumexp(np.array(raw_unique_samples[i][3])))
        unique_samples = np.squeeze(np.array(marginalized_samples))
        lnprobs = np.squeeze(np.array(avaraged_lnprobs))
        dthetas = np.array(marginalized_dthetas)
        if has_prior:
            prior_lnprobs = np.squeeze(np.array(marginalized_prior_lnprobs))
    dthetas = np.squeeze(np.array(dthetas))
    sort_index = np.argsort(lnprobs)
    unique_samples = unique_samples[sort_index]
    lnprobs = lnprobs[sort_index]
    dthetas = dthetas[sort_index]
    if has_prior:
        prior_lnprobs = prior_lnprobs[sort_index]
        return unique_samples, lnprobs, dthetas, prior_lnprobs
    else:
        return unique_samples, lnprobs, dthetas


def calulate_evidence(lnprobs, volumes=None, prior_lnprobs=None):
    """
    Calculate the evidence from the samples.

    Parameters:
    lnprobs: np.ndarray
        The log-probabilities of the samples.
    volumes: np.ndarray, optional
        The volume of the bins associated with the averaged log-probabilities.
    prior_lnprobs: np.ndarray, optional
        The log-prior probabilities (including log(bin_volume)) for each bin.

    Returns:
    evidence: float
        The evidence (log-sum of probabilities).
    """
    if prior_lnprobs is not None:
        evidence = logsumexp(lnprobs + prior_lnprobs)+ np.log(volumes)
    elif volumes is not None:
        evidence = logsumexp(lnprobs + np.log(volumes))
    else:
        raise ValueError("'volumes' must be provided to calculate the evidence.")
    return evidence

def plot_1d_posteriors(unique_samples,lnprobs_densities,dthetas,ax=None,colors=None,labels=None,truths=None,smooth_mode=False, scale ='log',show_ln_prob = True, stats = ['mean','std','percentiles'],percentiles = [16, 50, 95],log=None,truth_label = "Best fit", prior_lnprobs=0):
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
        The true values of the posteriors. If None, no true values will be plotted.
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
            marginalized_prior_lnprobs = np.squeeze(prior_lnprobs) if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape == marginalized_lnprobs.shape else 0
        else:
            # Marginalize both lnprobs and prior_lnprobs in one call
            if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape[0] == unique_samples.shape[0]:
                marginalized_samples, marginalized_lnprobs, marginalized_dthetas, marginalized_prior_lnprobs = marginalized_posterior(
                    unique_samples, lnprobs_densities, [j for j in range(n_features) if j != i], dthetas, density=True, prior_lnprobs=prior_lnprobs)
            else:
                marginalized_samples, marginalized_lnprobs, marginalized_dthetas = marginalized_posterior(
                    unique_samples, lnprobs_densities, [j for j in range(n_features) if j != i], dthetas, density=True)
                marginalized_prior_lnprobs = 0
        stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, smooth_mode=smooth_mode, percentiles=percentiles, prior_lnprobs=marginalized_prior_lnprobs)
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
        
def plot_2d_posteriors(unique_samples,lnprobs_densities,dthetas,features, ax=None,labels=None,truths=None, scale ='log',show_ln_prob = True, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],plot_type = 'contourf', prior_lnprobs=0, log=None, smooth_mode=False, **kwargs):
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
    
    marginalized_samples, marginalized_lnprobs, marginalized_dthetas = marginalized_posterior(unique_samples, lnprobs_densities, [j for j in range(n_features) if j not in features], dthetas, density = True)
    # Marginalize prior_lnprobs if it is an array of the correct shape
    if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape[0] == unique_samples.shape[0]:
        marginalized_samples, marginalized_lnprobs, marginalized_dthetas, marginalized_prior_lnprobs = marginalized_posterior(
            unique_samples, lnprobs_densities, [j for j in range(n_features) if j not in features], dthetas, density=True, prior_lnprobs=prior_lnprobs)
    else:
        marginalized_samples, marginalized_lnprobs, marginalized_dthetas = marginalized_posterior(
            unique_samples, lnprobs_densities, [j for j in range(n_features) if j not in features], dthetas, density=True)
        marginalized_prior_lnprobs = 0
    stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, smooth_mode=smooth_mode, percentiles=percentiles, prior_lnprobs=marginalized_prior_lnprobs)

    if 'mean' in stats:
        mean_x = np.exp(stats_dict['mean'][0]) if log[features[0]] else stats_dict['mean'][0]
        mean_y = np.exp(stats_dict['mean'][1]) if log[features[1]] else stats_dict['mean'][1]
        ax.axvline(mean_x, color="k", linestyle="--", label="Mean")
        ax.axhline(mean_y, color="k", linestyle="--")
    if 'std' in stats:
        std_x = np.exp(stats_dict['mean'][0]+stats_dict['std'][0]) if log[features[0]] else (stats_dict['mean'][0]+stats_dict['std'][0])
        std_y = np.exp(stats_dict['mean'][1]+stats_dict['std'][1]) if log[features[1]] else (stats_dict['mean'][1]+stats_dict['std'][1])
        ax.axvline(std_x, color="gray", linestyle=":", label="Std")
        ax.axhline(std_y, color="gray", linestyle=":")
        std_x = np.exp(stats_dict['mean'][0]-stats_dict['std'][0]) if log[features[0]] else (stats_dict['mean'][0]-stats_dict['std'][0])
        std_y = np.exp(stats_dict['mean'][1]-stats_dict['std'][1]) if log[features[1]] else (stats_dict['mean'][1]-stats_dict['std'][1])
        ax.axvline(std_x, color="gray", linestyle=":")
        ax.axhline(std_y, color="gray", linestyle=":")
    if 'percentiles' in stats:
        last_low = np.exp(stats_dict['mode']) if log[features[0]] or log[features[1]] else stats_dict['mode']
        last_high = np.exp(stats_dict['mode']) if log[features[0]] or log[features[1]] else stats_dict['mode']
        alpha = 0.1*(len(percentiles))+0.1
        for percentile in percentiles:
            low = np.exp(stats_dict[f'percentile_{percentile}'][0]) if log[features[0]] or log[features[1]] else stats_dict[f'percentile_{percentile}'][0]
            high = np.exp(stats_dict[f'percentile_{percentile}'][1]) if log[features[0]] or log[features[1]] else stats_dict[f'percentile_{percentile}'][1]
            # if percentile == 95: ###DEBUG###
            #     print(f"{label} low: {low}, high: {high}")
            top = np.max(marginalized_lnprobs) if show_ln_prob else np.exp(np.max(marginalized_lnprobs))
            bottom = np.min(marginalized_lnprobs) if show_ln_prob else np.exp(np.min(marginalized_lnprobs))
            ax.fill_betweenx([bottom,top], low, last_low, color='C0', alpha=alpha, label=f"{percentile}th percentile")
            ax.fill_betweenx([bottom,top], last_high, high, color='C0', alpha=alpha )
            last_low = low
            last_high = high
            alpha -= 0.1

    if 'mode' in stats:
        mode_x = np.exp(stats_dict['mode'][0]) if log[features[0]] else stats_dict['mode'][0]
        mode_y = np.exp(stats_dict['mode'][1]) if log[features[1]] else stats_dict['mode'][1]
        ax.axvline(mode_x, color="k", linestyle="-.", label="Mode")
        ax.axhline(mode_y, color="k", linestyle="-.")

    if not show_ln_prob:
        marginalized_lnprobs = np.exp(marginalized_lnprobs)
    if log[features[0]]:
        marginalized_samples[:,0] = np.exp(marginalized_samples[:,0])
    if log[features[1]]:
        marginalized_samples[:,1] = np.exp(marginalized_samples[:,1])

    x = marginalized_samples[:,0]
    y = marginalized_samples[:,1]
    z = marginalized_lnprobs
    if not show_ln_prob:
        z = np.exp(z)
    X, Y = np.meshgrid(x, y)
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_posterior(unique_samples, lnprobs_density, dthetas, features, ax=None, labels=None, truths=None, scale='log', show_ln_prob=False, stats=None, percentiles=None, log=None, plot_type='scatter', **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    x = unique_samples[:, features[0]]
    y = unique_samples[:, features[1]]
    z = unique_samples[:, features[2]]
    c = lnprobs_density if show_ln_prob else np.exp(lnprobs_density)
    img = ax.scatter(x, y, z, c=c, cmap='viridis', **kwargs)
    if labels is not None:
        ax.set_xlabel(labels[features[0]])
        ax.set_ylabel(labels[features[1]])
        ax.set_zlabel(labels[features[2]])
    else:
        ax.set_xlabel(f'Feature {features[0]}')
        ax.set_ylabel(f'Feature {features[1]}')
        ax.set_zlabel(f'Feature {features[2]}')
    if truths is not None:
        ax.scatter(truths[features[0]], truths[features[1]], truths[features[2]], color='red', marker='x', s=100, label='Truth')
    ax.set_title("3D Posterior")
    #set the scale of the axes

    plt.colorbar(img, ax=ax, shrink=0.5, aspect=5)
    
    return ax

def plot_3d_posterior_interactive(unique_samples, lnprobs_density, features, labels=None, truths=None, show_ln_prob=False):
    x = unique_samples[:, features[0]]
    y = unique_samples[:, features[1]]
    z = unique_samples[:, features[2]]
    c = lnprobs_density if show_ln_prob else np.exp(lnprobs_density)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=c,
            colorscale='Viridis',
            colorbar=dict(title='lnprob' if show_ln_prob else 'prob'),
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title=r"$x_c/\eta$",
            yaxis_title=r"$\beta/\eta$",
            zaxis_title=r"$x_c^2/\epsilon$"
            # xaxis_title=labels[features[0]] if labels else f'Feature {features[0]}',
            # yaxis_title=labels[features[1]] if labels else f'Feature {features[1]}',
            # zaxis_title=labels[features[2]] if labels else f'Feature {features[2]}',
        ),
        title="Interactive 3D Posterior"
    )

    if truths is not None:
        fig.add_trace(go.Scatter3d(
            x=[truths[features[0]]],
            y=[truths[features[1]]],
            z=[truths[features[2]]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='x'),
            name='Truth'
        ))

    fig.show()


def corner_plot(unique_samples,lnprobs_densities,dthetas,ax=None,colors=None,labels=None,truths=None, scale ='log',show_ln_prob = True, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],log=None,plot_type = 'contourf',progress_bar=True, prior_lnprobs=0, **kwargs):
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
        The true values of the posteriors. If None, no true values will be plotted.
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
    # Marginalize prior_lnprobs for 1D plots
    if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape[0] == unique_samples.shape[0]:
        marginalized_prior_lnprobs_1d = [
            marginalized_posterior(unique_samples, prior_lnprobs, [j for j in range(n_features) if j != i], dthetas, density=True)[1]
            for i in range(n_features)
        ]
    else:
        marginalized_prior_lnprobs_1d = [0] * n_features
    for i, ax1d in enumerate(diagonal_axes):
        plot_1d_posteriors(unique_samples, lnprobs_densities, dthetas, ax=[ax1d], colors=colors, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=log, prior_lnprobs=marginalized_prior_lnprobs_1d[i])
    if progress_bar:
        iterator = tqdm(range(n_features), desc="Creating corner plot")
    else:
        iterator = range(n_features)
    for i in iterator:
        for j in range(i):
            # Marginalize prior_lnprobs for 2D plots
            if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape[0] == unique_samples.shape[0]:
                marginalized_prior_lnprobs_2d = marginalized_posterior(unique_samples, prior_lnprobs, [k for k in range(n_features) if not k in [j,i]], dthetas, density=True)[1]
            else:
                marginalized_prior_lnprobs_2d = 0
            plot_2d_posteriors(unique_samples, lnprobs_densities, dthetas, features=[j,i], ax=ax[i,j], labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=log, plot_type=plot_type, prior_lnprobs=marginalized_prior_lnprobs_2d, **kwargs)
        for j in range(i+1, n_features):
            ax[i,j].axis('off')
    return ax




def getStats(samples, lnprobs, dthetas, stats=['mean','std'],percentiles = [16, 50, 84], center_percentiles=True, smooth_mode = True,debug=False, prior_lnprobs= 0):
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
    lnprobs = lnprobs + np.log(volumes) + prior_lnprobs
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


from SRtools import joint_posterior as jp
import numpy as np
from scipy.special import logsumexp
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

def default_transform6(sample,kappa):
    xc_eta, beta_eta, xc2_epsilon, xc = sample[:4]
    beta2_eta = beta_eta^2 * xc2_epsilon / xc_eta^2
    eta = xc / xc_eta
    k_beta = kappa/(beta_eta * eta)
    k_epsilon = kappa/((xc ** 2)/xc2_epsilon )
    return [beta2_eta, k_beta, k_epsilon] + list(sample[3:])

def round_value(value, precision=2):
    if value == 0:
        return 0
    elif abs(value) < 1:
        r = int(np.Abs(np.log10(abs(value))))
        return round(value, r+precision)
    else:
        return round(value, precision)



