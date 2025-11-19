from . import deathTimesDataSet as dtds
import autograd.numpy as np
import matplotlib.pyplot as plt
from lifelines.fitters import ParametricUnivariateFitter


class GompertzMakehamFitter(dtds.Dataset):
    def __init__(self, death_times, events, external_hazard=np.inf, bandwidth=None, properties=None, data_dt=1):
        """
        Initialize the GompertzMakehamFitter class.

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

    @property
    def gmf(self):
        """
        Return the GompertzMakehamFitter object.

        Returns:
        - GompertzMakehamFitter object.
        """
        return self.kmf

    @gmf.setter
    def gmf(self, value):
        """
        Set the GompertzMakehamFitter object.

        Parameters:
        - value: GompertzMakehamFitter object
            The GompertzMakehamFitter object to set.
        """
        self.mkf = value
        self.naf = value

    def calc_survival_and_hazard(self, events=None):
        """
        This function calculates the survival and hazard functions for the dataset.
        """
        T = self.death_times
        if events is not None:
            E = events
        else:
            E = np.ones_like(T)
            self.events = E

        # gmf = MakehamGompertzFitter().fit(T, E)
        gmf = GompertzFitter().fit(T, E)

        self.survival = gmf.timeline, np.array(gmf.survival_function_.values)[:, 0]
        # 95% confidence interval
        try:
            self.kmf_confidence_interval = [
                np.array(gmf.confidence_interval_['KM_estimate_lower_0.95'].values),
                np.array(gmf.confidence_interval_['KM_estimate_upper_0.95'].values)
            ]
        except:
            print("Confidence interval not available for this dataset.")
            self.kmf_confidence_interval = None
        self.median_lifetime = gmf.median_survival_time_
        self.kmf = gmf

        # naf = MakehamGompertzFitter().fit(T, event_observed=E)
        naf = GompertzFitter().fit(T, event_observed=E)
        self.hazard = naf.timeline, np.array(naf.hazard_.values)[:, 0]
        self.naf = naf

    def BIC(self):
        """
        Calculate the Bayesian Information Criterion (BIC) for the Gompertz-Makeham model.
        """
        return self.gmf.BIC_
    
    def print_summary(self):
        """
        Print the summary of the fitted model.
        """
        self.gmf.print_summary()

    def print_parmas(self):
        """
        Print the parameters of the fitted model.
        """
        print("Parameters: ", self.gmf.params_)

    def getParams(self):
        """
        Get the parameters of the fitted model.

        Returns:
        - params: dict
            The parameters of the fitted model.
        """
        return self.gmf.params_


# --- Custom Makeham-Gompertz Fitter ---
class MakehamGompertzFitter(ParametricUnivariateFitter):
    """
    A custom fitter for the Makeham-Gompertz model.
    Hazard: h(t) = c + eta * exp(alfa * t)
    Cumulative Hazard: H(t) = c*t + (eta/alfa)*(exp(alfa*t) - 1)
    """
    _fitted_parameter_names = ["c", "eta", "alfa"]
    # Ensure c and eta are non-negative.
    _bounds = [(0, None), (0, None), (None, None)]
    _n_parameters = 3

    def _cumulative_hazard(self, params, times):
        c, eta, alfa = params
        # Handle near-zero alfa to avoid division by zero.
        if np.abs(alfa) < 1e-6:
            return c * times + eta * times
        return c * times + (eta / alfa) * (np.exp(alfa * times) - 1)

    def _log_hazard(self, params, times):
        c, eta, alfa = params
        hazard = c + eta * np.exp(alfa * times)
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
    


class GompertzFitter(ParametricUnivariateFitter):
    """
    A custom fitter for the Makeham-Gompertz model.
    Hazard: h(t) = c + eta * exp(alfa * t)
    Cumulative Hazard: H(t) = c*t + (eta/alfa)*(exp(alfa*t) - 1)
    """
    _fitted_parameter_names = [ "eta", "alfa"]
    # Ensure c and eta are non-negative.
    _bounds = [(0, None), (None, None)]
    _n_parameters = 2

    def _cumulative_hazard(self, params, times):
        eta, alfa = params
        # Handle near-zero alfa to avoid division by zero.
        if np.abs(alfa) < 1e-6:
            return  eta * times
        return  (eta / alfa) * (np.exp(alfa * times) - 1)

    def _log_hazard(self, params, times):
        eta, alfa = params
        hazard =  eta * np.exp(alfa * times)
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
