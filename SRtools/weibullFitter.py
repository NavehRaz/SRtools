from . import deathTimesDataSet as dtds
import autograd.numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullFitter as WBF
from lifelines.fitters import ParametricUnivariateFitter



class WeibullFitter(dtds.Dataset):
    def __init__(self, death_times,events, external_hazard = np.inf, bandwidth=None, properties = None, data_dt=1):
        """
        Initialize the WeibullFitter class.

        Parameters:
        - death_times: array-like, shape (n_samples,)
            The observed death times.
        - events: array-like, shape (n_samples,)
            The event indicators (1 if event occurred, 0 if censored).
        - external_hazard: float, optional
            The external hazard rate. Default is np.inf.
        - bandwidth: int, optional
            The bandwidth for the kernel density estimation. Default is 3.
        - properties: dict, optional
            Additional properties for the dataset. Default is None.
        - data_dt: int, optional
            The time step for the data. Default is 1.
        """
        super().__init__(death_times, events, external_hazard, bandwidth, properties, data_dt)


    #make self.wf alias for self.kmf
    @property
    def wf(self):
        """
        Return the WeibullFitter object.

        Returns:
        - WeibullFitter object.
        """
        return self.kmf
    
    @wf.setter
    def wf(self, value):
        """
        Set the WeibullFitter object.

        Parameters:
        - value: WeibullFitter object
            The WeibullFitter object to set.
        """
        self.mkf = value
        self.naf = value
    

    def calc_survival_and_hazard(self, events = None):
        """
        This function calculates the survival and hazard functions for the dataset.
        """
        T = self.death_times
        if events is not None:
            E = events
        else:
            E = np.ones_like(T)
            self.events = E

        
        # kmf = ExtendedWeibullFitter().fit(T, E)
        kmf = WBF().fit(T, E)

        self.survival = kmf.timeline, np.array(kmf.survival_function_.values)[:,0]
        #95% confidence interval
        try:
            self.kmf_confidence_interval = [np.array(kmf.confidence_interval_['KM_estimate_lower_0.95'].values), np.array(kmf.confidence_interval_['KM_estimate_upper_0.95'].values)]
        except:
            print("Confidence interval not available for this dataset")
            self.kmf_confidence_interval = None
        self.median_lifetime = kmf.median_survival_time_
        self.kmf = kmf

        # naf = ExtendedWeibullFitter().fit(T, event_observed=E)
        naf = WBF().fit(T, event_observed=E)
        self.hazard = naf.timeline, np.array(naf.hazard_.values)[:,0]
        self.naf = naf 

    def BIC(self):
        """
        Calculate the Bayesian Information Criterion (BIC) for the Weibull model.
        """
        return self.wf.BIC_
    
    def print_summary(self):
        """
        Print the summary of the WeibullFitter object.
        """
        self.wf.print_summary()
    
    def print_params(self):
        """
        Print the parameters of the WeibullFitter object.
        """
        self.wf.print_params()
    
    def getParams(self):
        """
        Get the parameters of the WeibullFitter object.
        """
        return self.wf.params_
    

class ExtendedWeibullFitter(ParametricUnivariateFitter):
    """
    A custom fitter for the extended Weibull model with an external hazard.
    
    Hazard: h(t) = c + (rho / lambda) * (t / lambda)^(rho - 1)
    Cumulative Hazard: H(t) = c*t + (t / lambda)^rho
    """
    _fitted_parameter_names = ["c", "lambda", "rho"]
    # Parameter bounds: c>=0, lambda>0, rho>0.
    _bounds = [(0, None), (1e-8, None), (1e-8, None)]
    _n_parameters = 3
    
    def _cumulative_hazard(self, params, times):
        c, lam, rho = params
        return c * times + (times / lam) ** rho

    def _log_hazard(self, params, times):
        c, lam, rho = params
        # Hazard: h(t) = c + (rho/lam)*(t/lam)^(rho-1)
        hazard = c + (rho / lam) * (times / lam) ** (rho - 1)
        return np.log(hazard)

    def _log_likelihood(self, params, times, event_observed, weights):
        H = self._cumulative_hazard(params, times)
        log_h = self._log_hazard(params, times)
        return np.sum(weights * (event_observed * log_h - H))

    def predict_survival_function(self, times, params=None):
        if params is None:
            params = self._fitted_params.values
        H = self._cumulative_hazard(params, times)
        return np.exp(-H)