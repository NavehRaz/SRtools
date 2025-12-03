import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import ast
import json
import plotly.graph_objects as go





class Posterior:
    def __init__(self, samples, lnprobs, bins, log=False, progress_bar=True, config_params=None, help_text = None, prior=None, sorting=True):
        """
        Initialize a Posterior object to store and analyze posterior samples.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of samples from the posterior distribution, shape (n_samples, n_features)
        lnprobs : numpy.ndarray 
            Log probabilities corresponding to each sample, shape (n_samples,)
        bins : int or list
            Number of bins for discretizing each feature. If int, same number used for all features.
            If list, should have length equal to number of features.
        log : bool or list, optional
            Whether to use log scale for each feature. If bool, same value used for all features.
            If list, should have length equal to number of features. Default is False.
        progress_bar : bool, optional
            Whether to show progress bars during computations. Default is True.
        config_params : dict, optional
            Configuration parameters used to generate the samples. Default is None.
        help_text : str, optional
            Help text describing the posterior. Default is None.
        prior : Posterior or None, optional
            A Posterior object representing the prior distribution. If provided and prior.prior=True,
            a warning will be issued about potential inaccuracies in marginalization. Default is None.
        sorting : bool, optional
            Whether to sort samples by probability. Default is True.

        Attributes
        ----------
        samples : numpy.ndarray
            Original samples array
        lnprobs : numpy.ndarray
            Original log probabilities array
        bins : list
            Number of bins for each feature
        log : list 
            Log scale flag for each feature
        logged_samples : numpy.ndarray
            Samples after applying log transform where specified
        unique_samples : numpy.ndarray
            Unique sample points after binning
        posterior : numpy.ndarray
            Posterior probability for each unique sample
        lnprobs_density : numpy.ndarray
            Log probability density at each unique sample
        dthetas : numpy.ndarray
            Bin widths for each feature
        evidence : float
            Evidence (marginal likelihood) estimate
        config_params : dict
            Configuration parameters
        help_text : str
            Help text description
        prior_* : various
            Attributes for storing prior distribution information

        Warns
        -----
        UserWarning
            If the provided prior is a Posterior object with prior.prior=True, warning about
            potential inaccuracies in marginalization calculations.
        """
        if prior is not None and hasattr(prior, 'prior') and prior.prior:
            import warnings
            warnings.warn("The provided prior has prior.prior=True. This may cause inaccuracies when marginalizing the posterior.")

        self.mins = None
        self.maxs = None
        self.ranges = None
        self.norm_unique_samples = None
        self.prior_object = prior

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
            self.prior_unique_samples = None
            self.prior_posterior = None
            self.prior_dthetas = None
            self.prior = False
            self.config_params = None
            self.help_text = None
            self.bins = None
            self.log = None
            self.logged_samples = None
            self.progress_bar = None
            self.prior_object = None
            return

        self.config_params = config_params
        self.help_text = help_text
        self.samples = samples.copy()
        self.lnprobs = lnprobs.copy()
        #check that bins and log are lists of the correct length (equal to the number of features-samples.shape[1]
        if isinstance(bins, int):
            bins = [bins] * samples.shape[1]
        elif len(bins) != samples.shape[1]:
            raise ValueError(f"Bins should be an integer or the length of bins should be equal to the number of features, got bins: {bins} and samples.shape[1]: {samples.shape[1]}")
        if isinstance(log, bool):
            log = [log] * samples.shape[1]
        elif len(log) != samples.shape[1]:
            raise ValueError("log should be a boolean or the length of log should be equal to the number of features")
        self.bins = bins.copy()
        self.log = log
        self.logged_samples  = [np.log(samples[:,i]) if log[i] else samples[:,i] for i in range(samples.shape[1])]
        self.logged_samples = np.array(self.logged_samples).T
        self.progress_bar = progress_bar
        # Handle prior logic
        self.prior_unique_samples = None
        self.prior_posterior = None
        self.prior_dthetas = None
        
        #if the prior is a Posterior object, we need to set the prior attributes, else we assume the prior is a uniformative prior
        if prior is not None and isinstance(prior, Posterior):
            self.prior_unique_samples = prior.unique_samples.copy()
            self.prior_posterior = prior.posterior.copy()
            self.prior_dthetas = prior.dthetas.copy()
            self.prior =True
        else:
            self.prior = False

        self.unique_samples, self.posterior, self.lnprobs_density, self.dthetas, self.evidence, self.prior_lnprobs = self.get_posterior(
            self.logged_samples, lnprobs, bins.copy(), log=False, progress_bar=progress_bar, full_output=True, prior=prior)
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

        if isinstance(self.ranges, np.ndarray):
            self.ranges[self.ranges == 0] = 1.0
        elif isinstance(self.ranges, float) and self.ranges == 0:
            self.ranges = 1.0
        self.norm_unique_samples = (self.unique_samples - self.mins) / self.ranges


    def get_posterior(self, samples, lnprobs, bins, log=False, progress_bar=True, full_output=False, prior=None):
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
        unique_samples, lnprobs_density, dthetas, volumes = avarage_samples(binned_samples,
                                                                lnprobs, calc_prob_volume=True,
                                                                bins=bins, binned_index=binned_index, 
                                                                progress_bar=progress_bar)
        if(np.isnan(lnprobs_density)).any():
            print("avaraged_lnprobs contains NaNs")
        if(np.isnan(volumes)).any():
            print("volumes contains NaNs")
        if np.any(volumes <= 0):
            print("volumes contains zeros or negative values")

        # if the prior is not None, we need to calculate the unnormalized posterior first:
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

        evidence = calulate_evidence(lnprobs_density, volumes=volumes, prior_lnprobs=prior_lnprobs)
        

        posterior_lnprobs =lnprobs_density+prior_lnprobs+ np.log(volumes)-evidence
        if np.isnan(lnprobs_density).any():
            print("final_lnprobs contains NaNs")
        if full_output:
            return unique_samples, posterior_lnprobs,lnprobs_density,dthetas, evidence, prior_lnprobs
        return unique_samples, posterior_lnprobs
    
    def update_posterior(self):
        """
        Update the posterior distribution, used as helper function for the marginalize_posterior function.
        """
        if self.prior:
            # Evaluate the prior probability density at each sample
            prior_lnprobs = []
            iterator = tqdm(range(len(self.unique_samples)), desc="Calculating prior for posterior bins") if self.progress_bar else range(len(self.unique_samples))
            for i in iterator:
                sample = self.unique_samples[i]
                # we need to calculate the probability density of the prior at the sample point
                # Using the "get_probability_in_region" method allows us to account for different bin sizes between the prior and the posterior
                prob = self.get_probability_in_region(sample, self.dthetas[i], density=True, log_check=False, prior =True)
                prior_lnprobs.append(prob)
            self.prior_lnprobs = np.array(prior_lnprobs)
        else:
            #we assume an uniformative prior, so we set the prior to 1
            self.prior_lnprobs = np.zeros_like(self.lnprobs_density)

        # If dthetas is 1D, volumes is just dthetas; otherwise, it's the product along axis=1
        if self.dthetas.ndim == 1:
            volumes = self.dthetas
        else:
            volumes = np.prod(self.dthetas, axis=1)
        self.evidence = calulate_evidence(self.lnprobs_density, volumes=volumes, prior_lnprobs=self.prior_lnprobs)
        

        self.posterior =self.lnprobs_density+self.prior_lnprobs+ np.log(volumes)-self.evidence

    def marginalize_posterior_values(self, features, density = True):
        unique_samples, lnprobs , dthetas = marginalized_posterior(self.unique_samples, self.lnprobs_density-self.evidence, features, self.dthetas, density=density)
        return unique_samples, lnprobs, dthetas
    
    def marginalize_posterior(self, features, density = True):
        """
        
        Marginalize the posterior distribution over the specified features.

        Returns a new Posterior object with the marginalized samples, lnprobs, and dthetas.
        If prior is not None, the prior is also marginalized.

        Parameters:
        features: list
            The features to marginalize over. Should be a list of integers.
        density: bool
            Whether to return the density (True) or probability (False). Default is True.

        Returns:
        Posterior
            A new Posterior object marginalized over the specified features.
        """
        # Marginalize posterior and prior if available
        
        marginalized_unique_samples, marginalized_lnprobs_density, marginalized_dthetas = marginalized_posterior(
        self.unique_samples, self.lnprobs_density, features, self.dthetas, density=density
        )
        if self.prior:
            # Marginalize prior
            marginalized_prior_unique_samples, marginalized_prior_posterior, marginalized_prior_dthetas = marginalized_posterior(
                self.prior_unique_samples, self.prior_posterior, features, self.prior_dthetas, density=density
            )


        # Determine bins and log for marginalized features
        keep_features = [i for i in range(self.unique_samples.shape[1]) if i not in features]
        if self.bins is not None and type(self.bins) is not int:
            marginalized_bins = [self.bins[i] for i in keep_features]
        else:
            marginalized_bins = self.bins
        marginalized_log = [self.log[i] for i in keep_features]

        #creat new empty posterior object
        marginalized_post = Posterior(None, None, None)

        #feel in the fields (samples and lnprobs will remain None)
        marginalized_post.unique_samples = marginalized_unique_samples
        marginalized_post.lnprobs_density = marginalized_lnprobs_density
        marginalized_post.dthetas = marginalized_dthetas
        marginalized_post.bins = marginalized_bins
        marginalized_post.log = marginalized_log
        marginalized_post.progress_bar = self.progress_bar
        marginalized_post.__make_normalized_samples__()
        marginalized_post.prior = self.prior
        marginalized_post.prior_unique_samples = marginalized_prior_unique_samples if self.prior else None
        marginalized_post.prior_posterior = marginalized_prior_posterior if self.prior else None
        marginalized_post.prior_dthetas = marginalized_prior_dthetas if self.prior else None
        marginalized_post.update_posterior()
        return marginalized_post

            
        

    def get_stats(self, stats=['mean','std'],percentiles = [16, 50, 95], center_percentiles=True, smooth_mode = False, prior_lnprobs=None):
        if prior_lnprobs is None and hasattr(self, 'prior_lnprobs'):
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
        return getStats(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, stats=stats, percentiles=percentiles, center_percentiles=center_percentiles, smooth_mode=smooth_mode, prior_lnprobs=prior_lnprobs)

    def plot_1d_posteriors(self,features =None, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],smooth_mode=False, truth_label = "Best fit"):
        import matplotlib.pyplot as plt
        if features is None:
            features = [i for i in range(self.unique_samples.shape[1])]
        n_features = len(features)
        

        log= self.log
        if log is None:
            log = [False] * n_features
        if features is None:
            features = [i for i in range(n_features)]
        if ax is None:
            fig, ax = plt.subplots(n_features, 1, figsize=(8, 3 * n_features))
            fig.tight_layout(pad=4.0)
        if n_features == 1:
            ax = [ax]
        for i,feature in enumerate(features):
            if colors is None:
                color = "C" + str(feature)
            else:
                color = colors[feature]
            if labels is None:
                label = f"Feature {feature}"
            else:
                label = labels[feature]
            
            prior_lnprobs = self.prior_lnprobs if self.prior_lnprobs is not None else 0
            

            if n_features == 1 and self.unique_samples.shape[1]==1:
                marginalized_samples, marginalized_lnprobs, marginalized_dthetas = np.squeeze(self.unique_samples), np.squeeze(self.lnprobs_density-self.evidence), np.squeeze(self.dthetas)
                marginalized_prior_lnprobs = np.squeeze(prior_lnprobs) if isinstance(prior_lnprobs, np.ndarray) and prior_lnprobs.shape == marginalized_lnprobs.shape else 0
            else:
                # Marginalize both lnprobs and prior_lnprobs in one call
                marginalized_post = self.marginalize_posterior([j for j in range(self.unique_samples.shape[1]) if j != feature], density=True)
                marginalized_samples = np.squeeze(marginalized_post.unique_samples)
                marginalized_lnprobs = np.squeeze(marginalized_post.lnprobs_density- marginalized_post.evidence)
                marginalized_dthetas = np.squeeze(marginalized_post.dthetas)
                marginalized_prior_lnprobs = np.squeeze(marginalized_post.prior_lnprobs) if marginalized_post.prior_lnprobs is not None else 0
                marginalized_lnprobs = np.squeeze(marginalized_lnprobs + marginalized_prior_lnprobs)
                
            stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, smooth_mode=smooth_mode, percentiles=percentiles, prior_lnprobs=0)
            
            if 'mean' in stats:
                mean_i = np.exp(stats_dict['mean']) if self.log[feature] else stats_dict['mean']
                ax[i].axvline(mean_i, color="k", linestyle="--", label="Mean")
            if 'std' in stats:
                pstd_i = np.exp((stats_dict['mean']+stats_dict['std'])) if self.log[feature] else (stats_dict['mean']+stats_dict['std'])
                mstd_i = np.exp((stats_dict['mean']-stats_dict['std'])) if self.log[feature] else (stats_dict['mean']-stats_dict['std'])
                ax[i].axvline(pstd_i, color="gray", linestyle=":", label="Std")
                ax[i].axvline(mstd_i, color="gray", linestyle=":")
            if 'percentiles' in stats:
                last_low = np.exp(stats_dict['mode']) if self.log[feature] else stats_dict['mode']
                last_high = np.exp(stats_dict['mode']) if self.log[feature] else stats_dict['mode']
                alpha = 0.1*(len(percentiles))+0.1
                for percentile in percentiles:
                    low = np.exp(stats_dict[f'percentile_{percentile}'][0]) if self.log[feature] else stats_dict[f'percentile_{percentile}'][0]
                    high = np.exp(stats_dict[f'percentile_{percentile}'][1]) if self.log[feature] else stats_dict[f'percentile_{percentile}'][1]
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
                mode_i = np.exp(stats_dict['mode']) if self.log[i] else stats_dict['mode']
                ax[i].axvline(mode_i, color="k", linestyle="-.", label="Mode")
            if not show_ln_prob:
                marginalized_lnprobs = np.exp(marginalized_lnprobs)
            if self.log[i]:
                marginalized_samples = np.exp(marginalized_samples)
            #sort the samples and lnprobs
            sort_index = np.argsort(marginalized_samples)
            marginalized_samples = marginalized_samples[sort_index]
            marginalized_lnprobs = marginalized_lnprobs[sort_index]
            ax[i].plot(marginalized_samples, marginalized_lnprobs, color=color, label=label)
            if truths is not None:
                ax[i].axvline(truths[feature], color="red", linestyle="--", label=truth_label)
            ylabel = "ln Posterior prob" if show_ln_prob else "Posterior prob"
            ax[i].set_ylabel(ylabel)
            ax[i].set_xlabel(f"{label}")
            if scale == 'log':
                ax[i].set_xscale('log')
            ax[i].legend()

    def plot_2d_posteriors(self, features, ax=None, labels=None, truths=None, scale='log', show_ln_prob=False, stats=['mean','std','percentiles', 'mode'], percentiles=[16, 50, 95], plot_type='contourf', prior_lnprobs=0, smooth_mode=False, **kwargs):
        """
        Plot 2D marginalized posteriors using the standalone plot_2d_posteriors function for consistency.
        """
        import matplotlib.pyplot as plt
        n_features = self.unique_samples.shape[1]
        log = self.log
        if self.log is None:
            log = [False] * n_features
        if ax is None:
            fig, ax = plt.subplots()
        
        # Marginalize prior_lnprobs if it is an array of the correct shape
       
        marginalized_post = self.marginalize_posterior(
            [j for j in range(n_features) if j not in features], density=True)
        marginalized_samples = marginalized_post.unique_samples
        marginalized_lnprobs = marginalized_post.lnprobs_density- marginalized_post.evidence
        marginalized_dthetas = marginalized_post.dthetas
        marginalized_prior_lnprobs = marginalized_post.prior_lnprobs if marginalized_post.prior_lnprobs is not None else 0
        marginalized_lnprobs = marginalized_lnprobs + marginalized_prior_lnprobs
        
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
        return X, Y

    def corner_plot(self, ax=None, colors=None, labels=None, truths=None, scale ='log',show_ln_prob = False, stats = ['mean','std','percentiles', 'mode'],percentiles = [16, 50, 95],plot_type = 'contourf', prior_lnprobs=0,smooth_mode=False, show_colorbar=False, **kwargs):
            
            n_features = self.unique_samples.shape[1]
            
            if ax is None:
                fig, ax = plt.subplots(n_features, n_features, figsize=(5 * n_features, 5 * n_features))
                fig.tight_layout(pad=3.0)
            diagonal_axes = [ax[i,i] for i in range(n_features)]
            # Marginalize prior_lnprobs for 1D plots
            
            for i, ax1d in enumerate(diagonal_axes):
                _=self.plot_1d_posteriors(ax=ax1d,features=[i], colors=colors, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles)
            if self.progress_bar:
                iterator = tqdm(range(n_features), desc="Creating corner plot")
            else:
                iterator = range(n_features)
            for i in iterator:
                for j in range(i):
                    # Marginalize prior_lnprobs for 2D plots
                    _=self.plot_2d_posteriors(features=[j,i], ax=ax[i,j], labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, plot_type=plot_type, **kwargs)
                for j in range(i+1, n_features):
                    ax[i,j].axis('off')
            
            # Add colorbar if requested
            if show_colorbar and plot_type in ['contourf', 'pcolormesh']:
                # Get the last 2D plot to use for colorbar
                last_plot = ax[1,0].collections[0] if plot_type == 'contourf' else ax[1,0].collections[0]
                cbar_ax = ax[0,0].figure.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = plt.colorbar(last_plot, cax=cbar_ax)
                cbar.set_label('Probability Density' if not show_ln_prob else 'Log Probability Density', fontsize=14)

            # Only show labels on outer panels and make them bigger
            for i in range(n_features):
                for j in range(n_features):
                    if i < n_features-1:  # Not bottom row
                        ax[i,j].set_xlabel('')
                    if j > 0:  # Not leftmost column
                        ax[i,j].set_ylabel('')
                    if i == n_features-1 and labels is not None:  # Bottom row
                        ax[i,j].set_xlabel(labels[j], fontsize=14)
                    if j == 0 and labels is not None:  # Leftmost column
                        ax[i,j].set_ylabel(labels[i], fontsize=14)
                    # Make tick labels bigger
                    ax[i,j].tick_params(axis='both', which='major', labelsize=12)
            
            # Add main title
            if labels is not None:
                ax[0,0].figure.suptitle('2D Marginalized Posterior', fontsize=20, y=1.02)
            
            return ax


    def create_posterior_df(self,transforms = ['default'],labels = None, ds=None,ds_labels=None, kappa=0.5, filepath = None, smooth_mode = False, rescale=None, set_xc=False, truth=None):
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
            if n_features > 4 or (set_xc is not False and set_xc is not None and n_features > 3):
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
            resecaled_posterior = self.rescale(rescale)
        else:
            resecaled_posterior = self
        
        # Rescale truth if provided
        if truth is not None:
            truth = np.array(truth)
            if rescale is not None:
                truth_rescaled = truth * np.array(rescale)
            else:
                truth_rescaled = truth.copy()
        else:
            truth_rescaled = None
            
        bins = resecaled_posterior.bins if isinstance(resecaled_posterior.bins, int) else resecaled_posterior.bins.copy()
        percentiles = [16, 50, 95]
        stats=['mean', 'std', 'mode', 'percentiles']
        summery_dict = {}
        for transform, label_set in zip(transforms, labels):
            if set_xc is not False and set_xc is not None:
                trans = lambda x: transform(x,kappa,set_xc=set_xc)
            else:
                trans = lambda x: transform(x,kappa)
            transformed_samples = np.apply_along_axis(trans, 1, resecaled_posterior.samples)
            # Transform truth value
            if truth_rescaled is not None:
                truth_transformed = trans(truth_rescaled)
            else:
                truth_transformed = None
            post = Posterior(transformed_samples.copy(), resecaled_posterior.lnprobs.copy(), bins=bins, log=resecaled_posterior.log, progress_bar=resecaled_posterior.progress_bar, config_params=resecaled_posterior.config_params, help_text=resecaled_posterior.help_text, prior=resecaled_posterior.prior_object)
            max_liklihood = transformed_samples[np.argmax(resecaled_posterior.lnprobs)]
            mode_overall = post.get_best_sample_in_mode()
            for i in range(n_features):
                #check if the label is already calculated
                if label_set[i] in summery_dict.keys():
                    continue
                

                marginalized_post = post.marginalize_posterior(
                    [j for j in range(n_features) if j !=i], density=True)
                marginalized_samples = marginalized_post.unique_samples
                marginalized_lnprobs = marginalized_post.lnprobs_density- marginalized_post.evidence
                marginalized_dthetas = marginalized_post.dthetas
                marginalized_prior_lnprobs = marginalized_post.prior_lnprobs if marginalized_post.prior_lnprobs is not None else 0
                marginalized_lnprobs = marginalized_lnprobs + marginalized_prior_lnprobs
                stats_dict = getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, percentiles=percentiles, smooth_mode=smooth_mode)
                #if log[i] then exponentiate the mean, mode and percentiles. std is [exp(mean+std)-exp(mean),exp(mean)-exp(mean-std)]
                if self.log[i]:
                    stats_dict['std'] = [np.exp(stats_dict['mean'])-np.exp(stats_dict['mean']-stats_dict['std']),np.exp(stats_dict['mean']+stats_dict['std'])-np.exp(stats_dict['mean'])]
                    stats_dict['mean'] = np.exp(stats_dict['mean'])
                    stats_dict['mode'] = np.exp(stats_dict['mode'])
                    for percentile in percentiles:
                        stats_dict[f'percentile_{percentile}'] = [np.exp(stats_dict[f'percentile_{percentile}'][0]), np.exp(stats_dict[f'percentile_{percentile}'][1])]
                #round all values to 4 non 0 decimal points:
                for key in stats_dict.keys():
                    #if the value is a list, round each element
                    if isinstance(stats_dict[key], (list, tuple)):
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
        
        # Add ML_lnprob row with highest lnprob value
        ml_lnprob_value = float(np.max(self.lnprobs))
        ml_lnprob_row = {}
        for col in self.df.columns:
            # For CI columns (std, percentile columns), use [value, value]
            if col.startswith('percentile_') or col == 'std':
                ml_lnprob_row[col] = [ml_lnprob_value, ml_lnprob_value]
            else:
                # For other columns, use the value as is
                ml_lnprob_row[col] = ml_lnprob_value
        
        # Add the row to the dataframe
        ml_lnprob_series = pd.Series(ml_lnprob_row, name='ML_lnprob')
        self.df = pd.concat([self.df, ml_lnprob_series.to_frame().T])

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
        if self.prior_object is not None:
            scaled_prior = self.prior_object.rescale(rescale)
        else:
            scaled_prior = None
        new_posterior = Posterior(new_samples, self.lnprobs, self.bins, log=self.log, progress_bar=self.progress_bar, prior=scaled_prior, config_params=self.config_params, help_text=self.help_text)
        return new_posterior
    
    
    
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
    
    
    def get_probability_in_region(self, sample, dtheta, density = False, log_check = True, prior = False):
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

        if prior:
            unique_samples = self.prior_unique_samples
            dthetas = self.prior_dthetas
            posterior = self.prior_posterior
        else:
            unique_samples = self.unique_samples
            dthetas = self.dthetas
            posterior = self.posterior

        # Vectorized bin boundaries
        bin_min = unique_samples - 0.5 * dthetas
        bin_max = unique_samples + 0.5 * dthetas

        # Vectorized overlap calculation
        overlap_min = np.maximum(region_min, bin_min)
        overlap_max = np.minimum(region_max, bin_max)
        overlap = np.maximum(overlap_max - overlap_min, 0)
        # Compute overlap volume: if overlap is 1D, just use overlap; else, take product along axis=1
        if overlap.ndim == 1:
            overlap_volume = overlap
        else:
            overlap_volume = np.prod(overlap, axis=1)
        # Compute bin volumes: if dthetas is 1D, use as is; else, take product along axis=1
        if dthetas.ndim == 1:
            bin_volume = dthetas
        else:
            bin_volume = np.prod(dthetas, axis=1)

        # Only consider bins with nonzero overlap
        mask = overlap_volume > 0
        fractions = overlap_volume[mask] / bin_volume[mask]
        

        probs = posterior[mask] + np.log(fractions)
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
        mode = self.unique_samples[mode_index].copy()
        # Exponentiate only the indexes where self.log is True
        if hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            mode = np.array([
            np.exp(val) if log_flag else val
            for val, log_flag in zip(mode, self.log)
            ])
        if idx:
            return mode, mode_index
        return mode
    
    def get_samples_in_region(self, bin_sample, dtheta, raw=True, return_type='lnprobs'):
        """
        Get the samples and probabilities in the region defined by bin_sample ± dtheta/2.
        
        Parameters
        ----------
        bin_sample : numpy.ndarray
            The center point of the region
        dtheta : numpy.ndarray
            The width of the region
        raw : bool, optional
            Whether to use raw samples or binned samples. Default is True.
        return_type : str, optional
            Type of probability to return. Options are:
            - 'lnprobs': log probabilities (raw=True only)
            - 'lnprobs_density': log probability density
            - 'posterior': posterior probability
            - 'priorlnprobs': prior log probabilities
            - 'indices': indices of samples in region
            Default is 'lnprobs'.
            
        Returns
        -------
        tuple
            (samples, probabilities) or (samples, indices) if return_type='indices'
        """
        
        # Ensure bin_sample and dtheta are broadcastable with the samples
        bin_sample = np.asarray(bin_sample)
        dtheta = np.asarray(dtheta)
        if bin_sample.ndim == 1:
            bin_sample = bin_sample.reshape(1, -1)
        if dtheta.ndim == 0:
            dtheta = np.full(bin_sample.shape, dtheta)
        elif dtheta.ndim == 1:
            dtheta = dtheta.reshape(1, -1)
        
        # Check if self.samples is None before calculating the mask
        if raw and self.samples is None:
            print("WARNING: self.samples is None, and raw is True.")
            return None
        
        if raw  and hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            samples = np.where(self.log, np.log(self.samples), self.samples)

        # Compute the mask using broadcasting
        mask = np.all(np.abs((samples if raw else self.unique_samples) - bin_sample) <= dtheta / 2, axis=1)
        if raw:
            if self.samples is None or self.lnprobs is None:
                print("WARNING: self.samples or self.lnprobs are not set, and raw is True.")
                return None
            samples = self.samples[mask]
            if return_type == 'indices':
                return samples, np.where(mask)[0]
            return samples, self.lnprobs[mask]
        else:
            samples = self.unique_samples[mask]
            if return_type == 'indices':
                return samples, np.where(mask)[0]
            elif return_type == 'lnprobs_density':
                return samples, self.lnprobs_density[mask] - self.evidence
            elif return_type == 'posterior':
                return samples, self.posterior[mask]
            elif return_type == 'priorlnprobs':
                if not hasattr(self, 'prior_lnprobs'):
                    raise ValueError("Prior probabilities not available")
                return samples, self.prior_lnprobs[mask]
            else:
                raise ValueError(f"Invalid return_type: {return_type}")
            
    def get_best_sample_in_mode(self):
        """
        Get the best sample in the mode of the posterior distribution.
        """
        mode, mode_index = self.get_mode(idx=True)
        if hasattr(self, "log") and isinstance(self.log, (list, np.ndarray)):
            mode = np.where(self.log, np.log(mode), mode)
        samples, lnprobs = self.get_samples_in_region(mode, self.dthetas[mode_index], raw=True, return_type='lnprobs')
        return samples[np.argmax(lnprobs)]
    
    def best_raw_sample(self):
        """
        Get the best sample in the posterior distribution.
        """
        if self.samples is None or self.lnprobs is None:
            print("WARNING: self.samples or self.lnprobs are not set.")
            return None
        return self.samples[np.argmax(self.lnprobs)]

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
        print("WARNING: plot_posterior3D is deprecated and incomplete.")
        return plot_3d_posterior(self.unique_samples, self.lnprobs_density-self.evidence, self.dthetas, features, ax=ax, labels=labels, truths=truths, scale=scale, show_ln_prob=show_ln_prob, stats=stats, percentiles=percentiles, log=self.log,plot_type=plot_type, **kwargs)
    
    def plot_posterior3D_interactive(self, features, labels=None, truths=None, show_ln_prob=True, density=True, truth_label="Truth", fig=None, row=None, col=None):
        """
        Plot the posterior distribution in 3D interactively using Plotly.
        parameters:
        features: list
            The features to plot. Should be a list of 3 integers.
        labels: list, optional
            The labels for the axes. If None, the default labels will be used.
            For math symbols, use HTML entities or Unicode.
        truths: list, optional
            The true values of the parameters. If None, no true values will be plotted.
        show_ln_prob: bool, optional
            Whether to show the log-probability or not. Default is True.
        density: bool, optional
            Whether to show probability density (True) or probability (False). Default is True.
        truth_label: str, optional
            The label to use for the truth marker in the legend. Default is "Truth".
        fig: plotly.graph_objects.Figure or None, optional
            If provided, the plot will be added to this figure as a subplot. If None, a new figure will be created.
        row: int, optional
            The row number for the subplot. Required if fig is provided.
        col: int, optional
            The column number for the subplot. Required if fig is provided.
        """
        import plotly.graph_objects as go
        import numpy as np

        n_features = self.unique_samples.shape[1]
        marginalized_post = self.marginalize_posterior([j for j in range(n_features) if j not in features], density=density)

        x = marginalized_post.unique_samples[:, 0]
        y = marginalized_post.unique_samples[:, 1]
        z = marginalized_post.unique_samples[:, 2]
        c = marginalized_post.lnprobs_density-marginalized_post.evidence + marginalized_post.prior_lnprobs

        if not show_ln_prob:
            c = np.exp(c)

        # Create new figure if none provided
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title='3D Posterior Distribution',
                margin=dict(l=0, r=0, t=30, b=0),  # Adjust margins to maximize plot space
                font=dict(family="Computer Modern")
            )

        # Determine scene name based on row and col
        scene_name = f"scene{row}{col}" if row is not None and col is not None else "scene"

        # Calculate colorbar position based on subplot position
        colorbar_x = 1.0
        if col is not None:
            # Adjust colorbar position based on column
            colorbar_x = 0.95 if col == 1 else 0.45

        # Main posterior points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=c,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title=dict(
                        text='Log Probability' if show_ln_prob else 'Probability',
                        side='right',
                        font=dict(size=16)
                    ),
                    x=colorbar_x,
                    len=0.4,  # Make colorbar shorter
                    thickness=20,  # Make colorbar thinner
                    y=0.5  # Center colorbar vertically
                ),
            ),
            name='Posterior',
            showlegend=False,  # Don't show in legend
            scene=scene_name
        ))

        # Truth marker if provided
        if truths is not None:
            fig.add_trace(go.Scatter3d(
                x=[truths[features[0]]],
                y=[truths[features[1]]],
                z=[truths[features[2]]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=3, color='black')
                ),
                name=truth_label,
                showlegend=True,
                scene=scene_name
            ))

        # Update layout with proper axis labels and legend
        if labels is not None:
            # Convert LaTeX labels to HTML/Unicode
            def convert_label(label):
                # Remove LaTeX delimiters
                label = label.replace('$', '')
                
                # Replace common LaTeX commands with HTML/Unicode
                replacements = {
                    '\\alpha': 'α',
                    '\\beta': 'β',
                    '\\gamma': 'γ',
                    '\\delta': 'δ',
                    '\\epsilon': 'ε',
                    '\\zeta': 'ζ',
                    '\\eta': 'η',
                    '\\theta': 'θ',
                    '\\iota': 'ι',
                    '\\kappa': 'κ',
                    '\\lambda': 'λ',
                    '\\mu': 'μ',
                    '\\nu': 'ν',
                    '\\xi': 'ξ',
                    '\\pi': 'π',
                    '\\rho': 'ρ',
                    '\\sigma': 'σ',
                    '\\tau': 'τ',
                    '\\upsilon': 'υ',
                    '\\phi': 'φ',
                    '\\chi': 'χ',
                    '\\psi': 'ψ',
                    '\\omega': 'ω'
                }
                
                # First replace all Greek letters
                for tex, html in replacements.items():
                    label = label.replace(tex, html)
                
                # Handle subscripts and powers
                import re
                
                # Handle subscripts: match _x where x is any character
                def subscript_replace(match):
                    subscript = match.group(1)
                    return f'<sub>{subscript}</sub>'
                label = re.sub(r'_([a-zA-Z0-9]+)', subscript_replace, label)
                
                # Handle powers
                def power_replace(match):
                    # Get the power value from either group 1 or 2
                    power = match.group(1) or match.group(2)
                    if power:
                        return f'<sup>{power}</sup>'
                    return match.group(0)  # Return original if no match
                
                # Replace both ^{number} and ^number patterns
                label = re.sub(r'\^{([^}]+)}|\^(\d+)', power_replace, label)
                
                return label

            # Update scene properties for the specific subplot
            scene_dict = dict(
                aspectmode='cube',
                xaxis=dict(
                    title=dict(
                        text=convert_label(labels[features[0]]),
                        font=dict(size=14)
                    ),
                    showexponent='all',
                    exponentformat='e'
                ),
                yaxis=dict(
                    title=dict(
                        text=convert_label(labels[features[1]]),
                        font=dict(size=14)
                    ),
                    showexponent='all',
                    exponentformat='e'
                ),
                zaxis=dict(
                    title=dict(
                        text=convert_label(labels[features[2]]),
                        font=dict(size=14)
                    ),
                    showexponent='all',
                    exponentformat='e'
                )
            )

            if row is not None and col is not None:
                fig.update_scenes(scene_dict, row=row, col=col)
            else:
                fig.update_layout(scene=scene_dict)
        else:
            if row is not None and col is not None:
                fig.update_scenes(
                    dict(
                        xaxis_title=f'Feature {features[0]}',
                        yaxis_title=f'Feature {features[1]}',
                        zaxis_title=f'Feature {features[2]}',
                        aspectmode='cube'
                    ),
                    row=row,
                    col=col
                )
            else:
                fig.update_layout(
                    scene=dict(
                        xaxis_title=f'Feature {features[0]}',
                        yaxis_title=f'Feature {features[1]}',
                        zaxis_title=f'Feature {features[2]}',
                        aspectmode='cube'
                    )
                )

        return fig

    
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

        if self.prior:
            import warnings
            warnings.warn("Saving with prior=True is deprecated and will be removed in a future version. Please save the prior separately if needed.", DeprecationWarning)
        
        # Save metadata to a CSV file
        data = {
            "log": [self.log] * self.unique_samples.shape[0],
            "bins": [self.bins] * self.unique_samples.shape[0],  # Add bins information
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
        from . import joint_posterior as jp

        # Check if the file has a .csv extension
        if not filepath.endswith('.csv'):
            filepath += '.csv'

        # If load_raw_data is True, check if this is a joint posterior file
        if load_raw_data:
            binary_filepath = filepath.replace('.csv', '_data.npz')
            try:
                with np.load(binary_filepath, allow_pickle=True) as data:
                    if 'n_samples' in data:
                        # This is a joint posterior file, delegate to JointPosterior.load_from_file
                        return jp.JointPosterior.load_from_file(filepath, load_raw_data)
            except (FileNotFoundError, IOError):
                pass  # Continue with normal Posterior loading if file doesn't exist or can't be read

        # Load metadata from the CSV file
        df = pd.read_csv(filepath)
        posterior_data = {
            "log": ast.literal_eval(df["log"].iloc[0]) if "log" in df.columns else None,
            "bins": ast.literal_eval(df["bins"].iloc[0]) if "bins" in df.columns else None,  # Add bins information
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
        loaded_posterior.bins = posterior_data["bins"]  # Add bins information
        loaded_posterior.samples = samples
        loaded_posterior.lnprobs = lnprobs
        loaded_posterior.config_params = posterior_data["config_params"]
        loaded_posterior.help_text = posterior_data["help_text"]
        loaded_posterior.prior_lnprobs = prior_lnprobs
        loaded_posterior.prior_unique_samples = posterior_data["prior_unique_samples"]
        loaded_posterior.prior_posterior = posterior_data["prior_posterior"]
        loaded_posterior.prior_dthetas = posterior_data["prior_dthetas"]
        loaded_posterior.prior = posterior_data["prior"]
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
            # Use natural logarithm for binning
            min_val = np.log(samples[:, index].min()*0.999)
            max_val = np.log(samples[:, index].max()*1.001)
            if min_val == max_val:
                raise ValueError(f"Min value and max value are the same: {min_val}")
            bins = np.exp(np.linspace(min_val, max_val, bins + 1))
        else:
            if samples[:, index].min()<0:
                bottom = samples[:, index].min()*1.001
            else:
                bottom = samples[:, index].min()*0.999
            if samples[:, index].max()<0:
                top = samples[:, index].max()*0.999
            else:
                top = samples[:, index].max()*1.001
            if bottom == top:
                raise ValueError(f"Bottom value and top value are the same: {bottom}")
            bins = np.linspace(bottom,top, bins + 1)


    binned_index = np.digitize(samples[:, index], bins)
    #if any binned index = len(bins), set it to len(bins) - 1
    binned_index[binned_index == len(bins)] = len(bins) - 1
    
    #print the samples that have binned_index = 0 or binned_index = len(bins)
    if (binned_index == 0).any():
        print(f"Samples with binned_index = 0: {samples[binned_index == 0]}")
    if (binned_index == len(bins)).any():
        print(f"Samples with binned_index = len(bins): {samples[binned_index == len(bins)]}")
    binned_values = 0.5 * (bins[binned_index] + bins[binned_index - 1])
    binned_samples  = samples.copy()
    binned_samples[:, index] = binned_values
    if np.any(np.isnan(binned_values)):
        print(f"binned_values are NaN: {binned_values}")
        print(f"bins: {bins}")
        print(f"binned_index: {binned_index}")
        print(f"samples: {samples}")
        print(f"index: {index}")
        raise ValueError(f"binned_values are NaN: {binned_values}")

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
            if (np.isnan(np.prod(raw_unique_samples[key][3]))):
                print('i:', i, 'key:', key)
                for j in range(binned_samples.shape[1]):
                    print('j:', j, 'binned_index:', binned_index[i,j], 'bin:', bins[j][int(binned_index[i,j])], 'bin-1:', bins[j][int(binned_index[i,j])-1])
                print('bins:\n', bins)
                raise ValueError(f"Volume is NaN: {raw_unique_samples[key][3]}")
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
            if (np.prod(raw_unique_samples[i][3]) is None) or np.isnan(np.prod(raw_unique_samples[i][3])):
                raise ValueError(f"Volume is None: {raw_unique_samples[i][3]}")
            volumes.append(np.prod(raw_unique_samples[i][3]))
    
    if calc_prob_volume:
        return np.array(unique_samples), np.array(avaraged_lnprob_density), np.array(dthetas), np.array(volumes)

    return np.array(unique_samples), np.array(avaraged_lnprob_density), np.array(dthetas)


    

    
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
        evidence = logsumexp(lnprobs + prior_lnprobs+ np.log(volumes))
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
        #sort the samples and lnprobs
        sort_index = np.argsort(marginalized_samples)
        marginalized_samples = marginalized_samples[sort_index]
        marginalized_lnprobs = marginalized_lnprobs[sort_index]
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


from . import joint_posterior as jp
import numpy as np
from scipy.special import logsumexp
class JointPosterior(jp.JointPosterior):
    def __init__(self, samples_list, lnprobs_list, bins, log=False, progress_bar=True, config_params=None, help_text=None, prior=None, sorting=True):
        return super().__init__(samples_list, lnprobs_list, bins, log=log, progress_bar=progress_bar, config_params=config_params, help_text=help_text, prior=prior, sorting=sorting)

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



