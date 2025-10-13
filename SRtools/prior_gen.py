import numpy as np
import pandas as pd
from . import sr_mcmc
import ast

class PriorGen:
    """
    This class is used to generate prior values for the model. The class recieves a vaules vector (of thetas) and a corrsponding lnprobs vector.
    Values then can be randomly sampled from the prior distribution using the lnprobs as weights. For examples, if the values are [[1,2,3],[0.1,0.2,0.3]], the lnprobs are [0.25,0.75] 
    then we will sample [1,2,3] with probability 0.25 and [0.1,0.2,0.3] with probability 0.75.
    """
    def __init__(self, values, lnprobs):
        self.values = values
        self.lnprobs = lnprobs

    def sample(self, n_samples=1, temperature=1.0):
        """
        Sample from the prior distribution with optional temperature-based noise.
        
        Args:
            n_samples (int): Number of samples to generate
            temperature (float): Controls the amount of noise added to samples. 
                               Higher temperature = more noise. Default is 1.0 (no noise).
        """
        # Convert probabilities to 1D array
        probs = np.exp(self.lnprobs)
        probs = probs / np.sum(probs)  # Normalize probabilities
        
        # Sample indices
        indices = np.random.choice(len(self.values), size=n_samples, p=probs)
        
        # Get base samples
        samples = self.values[indices]
        
        # Add temperature-based noise if temperature > 0
        if temperature > 0:
            # Calculate standard deviation for each dimension
            # Calculate standard deviations in log space
            stds = np.std(np.log(self.values), axis=0)
            # Add noise in log space and exponentiate back
            noise = np.random.normal(0, temperature * stds, size=samples.shape)
            samples = np.exp(np.log(samples) + noise)
        
        # Return the corresponding values
        if n_samples == 1:
            return samples[0]
        return samples
    
    @staticmethod
    def prior_from_posterior_csv(csv_path, transform=True):
        """
        This function is used to generate a prior from a posterior csv file.
        The csv file should have columns unique_samples and posterior.
        """
        df = pd.read_csv(csv_path)
        values = df['unique_samples'].values
        #convert to numpy array with ast.literal_eval
        values = np.exp(np.array([ast.literal_eval(val) for val in values]))
        if not transform:
            values =  np.array([sr_mcmc.inv_transform(value) for value in values])

        lnprobs = df['posterior'].values
        return PriorGen(values, lnprobs)
    

    def getBounds(self,expansion_factor = 2):
        """
        This function is used to get the bounds of the prior distribution.
        """
        bounds = []
        for i in range(len(self.values[0])):
            bounds.append([np.min(self.values[:,i])*expansion_factor,np.max(self.values[:,i])*expansion_factor])
        return bounds
    
  

class PriorGenExtended(PriorGen):
    
    """
    This class is used when we want to extend the thetas vector to more dimentions. It uses a seed value and variations
    to generate bins and draw params using sr_mcmc.get_bins_from_seed and sr_mcmc.draw_param.
    """
    def __init__(self, values, lnprobs, seed, variations, n_extra_dims, draw_params_in_log_space = True):
        self.seed = seed
        self.variations = variations
        self.n_extra_dims = n_extra_dims
        self.bins = sr_mcmc.get_bins_from_seed(seed, n_extra_dims, variations)
        self.draw_params_in_log_space = draw_params_in_log_space
        super().__init__(values, lnprobs)


    def sample(self, n_samples=1, temperature=1.0):
        """
        Sample from the extended prior distribution with optional temperature-based noise.
        
        Args:
            n_samples (int): Number of samples to generate
            temperature (float): Controls the amount of noise added to samples. 
                               Higher temperature = more noise. Default is 1.0 (no noise).
        """
        # Get base theta values from parent class
        theta = super().sample(n_samples, temperature)
        
        # Handle single sample case
        if n_samples == 1:
            theta = theta.reshape(1, -1)
            
        # Create extended theta array
        theta_extended = np.zeros((n_samples, theta.shape[1] + self.n_extra_dims))
        theta_extended[:, :theta.shape[1]] = theta
        
        # Add additional parameters for each sample
        if self.n_extra_dims > 0:
            for i in range(n_samples):
                theta_extended[i, theta.shape[1]:] = sr_mcmc.draw_param(self.bins, self.draw_params_in_log_space)
            
        # Return single sample as 1D array if n_samples=1
        if n_samples == 1:
            return theta_extended[0]
        return theta_extended
    
    def getBounds(self,expansion_factor = 2):
        """
        This function is used to get the bounds of the prior distribution.
        """
        bounds = []
        for i in range(len(self.values[0])):
            bounds.append([np.min(self.values[:,i])/expansion_factor,np.max(self.values[:,i])*expansion_factor])
        for i in range(self.n_extra_dims):
            bin = self.bins[i][0]
            bounds.append([bin[0]/expansion_factor,bin[1]*expansion_factor])
        return bounds


    @staticmethod
    def prior_from_posterior_csv_extended(csv_path, seed, variations, n_extra_dims,ndims_from_csv =4, transform=True, draw_params_in_log_space = True):
        """
        This function is used to generate a prior from a posterior csv file.
        The csv file should have columns unique_samples and posterior.
        """
        df = pd.read_csv(csv_path)
        values = df['unique_samples'].values
        #convert to numpy array with ast.literal_eval
        values = np.exp(np.array([ast.literal_eval(val) for val in values]))
        if not transform:
            values =  np.array([sr_mcmc.inv_transform(value) for value in values])
        if ndims_from_csv:
            values = values[:,:ndims_from_csv]
        
        lnprobs = df['posterior'].values
        return PriorGenExtended(values, lnprobs, seed, variations, n_extra_dims, draw_params_in_log_space)







    
    