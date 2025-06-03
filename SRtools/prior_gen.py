import numpy as np
import pandas as pd
from SRtools import sr_mcmc
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

    def sample(self, n_samples=1):
        # Convert probabilities to 1D array
        probs = np.exp(self.lnprobs)
        probs = probs / np.sum(probs)  # Normalize probabilities
        
        # Sample indices
        indices = np.random.choice(len(self.values), size=n_samples, p=probs)
        
        # Return the corresponding values
        if n_samples == 1:
            return self.values[indices]
        return self.values[indices]
    
    @staticmethod
    def prior_from_posterior_csv(csv_path):
        """
        This function is used to generate a prior from a posterior csv file.
        The csv file should have columns unique_samples and posterior.
        """
        df = pd.read_csv(csv_path)
        values = df['unique_samples'].values
        #convert to numpy array with ast.literal_eval
        values = np.exp(np.array([ast.literal_eval(val) for val in values]))
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
    def __init__(self, values, lnprobs, seed, variations, ndims, draw_params_in_log_space = True):
        self.seed = seed
        self.variations = variations
        self.ndims = ndims
        self.bins = sr_mcmc.get_bins_from_seed(seed, ndims, variations)
        self.draw_params_in_log_space = draw_params_in_log_space
        super().__init__(values, lnprobs)


    def sample(self, n_samples=1):
        """
        This function is used to draw parameters from the prior distribution. The first part of the thetas vector is drawn from the prior distribution,
        the rest of the thetas vector is drawn from the bins.
        """
        # Get base theta values from parent class
        theta = super().sample(n_samples)
        
        # Handle single sample case
        if n_samples == 1:
            theta = theta.reshape(1, -1)
            
        # Create extended theta array
        theta_extended = np.zeros((n_samples, theta.shape[1] + self.ndims))
        theta_extended[:, :theta.shape[1]] = theta
        
        # Add additional parameters for each sample
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
        for i in range(self.ndims):
            bin = self.bins[i][0]
            bounds.append([bin[0]/expansion_factor,bin[1]*expansion_factor])
        return bounds


    @staticmethod
    def prior_from_posterior_csv_extended(csv_path, seed, variations, ndims, draw_params_in_log_space = True):
        """
        This function is used to generate a prior from a posterior csv file.
        The csv file should have columns unique_samples and posterior.
        """
        df = pd.read_csv(csv_path)
        values = df['unique_samples'].values
        #convert to numpy array with ast.literal_eval
        values = np.exp(np.array([ast.literal_eval(val) for val in values]))
        values =  np.array([sr_mcmc.inv_transform(value) for value in values])
        
        lnprobs = df['posterior'].values
        return PriorGenExtended(values, lnprobs, seed, variations, ndims, draw_params_in_log_space)







    
    