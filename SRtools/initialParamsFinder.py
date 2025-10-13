from . import SRmodellib_lifelines as srl
from . import SRmodellib as sr
import numpy as np
from . import deathTimesDataSet as dtds


class Guess:
    """
    This class is used to find the initial parameters for the SR model. To initialize it,
    you need to provide an initial theta and a deathTimesDataSet object. 
    """
    def __init__(self, theta, ds,t_end=None, nsteps = 5000, npeople = 5000, time_step_multiplier = 1,external_hazard =np.inf,time_range =None,dt=1, parallel =True):
        self.theta = theta
        self.ds = ds
        self.nsteps = nsteps
        self.npeople = npeople
        self.time_step_multiplier = time_step_multiplier
        self.parallel = parallel
        self.external_hazard = external_hazard
        self.dt = dt
        self.time_range = time_range
        self.t_end = self.ds.t_end if t_end is None else t_end
        self.guess = srl.getSr(self.theta, n=self.npeople,nsteps=self.nsteps, t_end=self.t_end, external_hazard = self.external_hazard, time_step_multiplier=self.time_step_multiplier, parallel=self.parallel)

    def calibrateEta(self, update_t_end = True, update_nsteps = True):
        """
        This function is used to calibrate the eta parameter of the SR model to get approximate ML. 
        """
        ratio = self.guess.getMedianLifetime()/self.ds.getMedianLifetime()
        self.theta[0] = self.theta[0] * ratio
        if update_t_end:
            self.t_end = self.ds.t_end
            if update_nsteps:
                self.nsteps = int(self.nsteps // ratio)
        self.guess = srl.getSr(self.theta, n=self.npeople, nsteps=self.nsteps, t_end=self.ds.t_end, external_hazard=self.external_hazard, time_step_multiplier=self.time_step_multiplier, parallel=self.parallel)
        print(f'The eta parameter has been calibrated by ratio {ratio} to:', self.theta[0])

    def plotAll(self,dt=1):
        """
        This function plots the survival, hazard and death times distribution of the guess and the dataset.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        self.guess.plotSurvival(ax[0], label='Guess')
        self.ds.plotSurvival(ax[0], label='Data')
        ax[0].legend()
        ax[0].set_title('Survival')

        self.guess.plotHazard(ax[1], label='Guess')
        self.ds.plotHazard(ax[1], label='Data')
        ax[1].legend()
        ax[1].set_title('Hazard')

        self.guess.plotDeathTimesDistribution(ax[2],dt=dt, label='Guess')
        self.ds.plotDeathTimesDistribution(ax[2],dt=dt, label='Data')
        ax[2].legend()
        ax[2].set_title('Death Times Distribution')

        plt.tight_layout()
        plt.show()

    def make_guess(self, niter = 5 , step_size = 1.5, skip_params = [], plot = True,print_thetas =True,plot_dt=1, method='one_at_a_time'):
        """
        This function is used to find the initial parameters for the SR model. 
        The function takes the number of iterations, the step size and the parameters to skip as input.
        """
        import matplotlib.pyplot as plt

        # if step_size is a float, it will be converted to a list of the same size as the number of parameters
        if isinstance(step_size, (int, float)):
            step_size = [step_size] * len(self.theta)
        elif len(step_size) != len(self.theta):
            raise ValueError('The step_size should be a float or a list of the same size as the number of parameters')
        
        old_guess = self.guess
        old_theta = self.theta
        improved = False
        if method not in ['one_at_a_time', 'all_at_once']:
            raise ValueError('The method should be one of the following: one_at_a_time, all_at_once')

        for i in range(niter):
            work_theta = self.theta.copy()
            for j in range(len(self.theta)):
                if j not in skip_params:
                    theta = work_theta.copy()
                    theta[j] = theta[j] * np.random.uniform(1 / step_size[j], step_size[j])
                    if method == 'one_at_a_time':
                        guess = srl.getSr(theta, n=self.npeople, nsteps=self.nsteps, t_end=self.ds.t_end, external_hazard=self.external_hazard, time_step_multiplier=self.time_step_multiplier, parallel=self.parallel)
                        if sr.baysianDistance(self.ds,guess, time_range=self.time_range, dt=self.dt) > sr.baysianDistance( self.ds,self.guess, time_range=self.time_range, dt=self.dt):
                            work_theta = theta
                            self.theta = theta
                            self.guess = guess
                            improved = True
                    else:
                        work_theta = theta
            if method == 'all_at_once':
                guess = srl.getSr(theta, n=self.npeople, nsteps=self.nsteps, t_end=self.ds.t_end, external_hazard=self.external_hazard, time_step_multiplier=self.time_step_multiplier, parallel=self.parallel)
                if sr.baysianDistance(self.ds,guess, time_range=self.time_range, dt=self.dt) > sr.baysianDistance( self.ds,self.guess, time_range=self.time_range, dt=self.dt):
                    self.theta = theta
                    self.guess = guess
                    improved = True
        if print_thetas and improved:
            print('The guess has been improved:')
            print('Old theta:', old_theta)
            print('New theta:', self.theta)
        if plot and improved:
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))
            self.guess.plotSurvival(ax[0], label='Current Guess')
            old_guess.plotSurvival(ax[0], label='Old Guess')
            self.ds.plotSurvival(ax[0], label='Data')
            ax[0].legend()
            ax[0].set_title('Survival')

            self.guess.plotHazard(ax[1], label='Current Guess')
            old_guess.plotHazard(ax[1], label='Old Guess')
            self.ds.plotHazard(ax[1], label='Data')
            ax[1].legend()
            ax[1].set_title('Hazard')

            self.guess.plotDeathTimesDistribution(ax[2],dt=plot_dt, label='Current Guess')
            old_guess.plotDeathTimesDistribution(ax[2],dt=plot_dt, label='Old Guess')
            self.ds.plotDeathTimesDistribution(ax[2],dt=plot_dt, label='Data')
            ax[2].legend()
            ax[2].set_title('Death Times Distribution')

            plt.tight_layout()
            plt.show()
        if not improved:
            print('The guess has NOT been improved')

        return self.guess
    

    
    def guessToSeed(self, filename):
        """
        This function saves the guess object params as a seed csv file.
        The file has the following columns: Eta, Beta, Epsilon, Xc and an index row: 'Estimate'.
        the values are the current guess theta values.
        """
        import pandas as pd
        df = pd.DataFrame(self.theta, columns=['Estimate'], index=['Eta', 'Beta', 'Epsilon', 'Xc'])
        df=df.T
        df.to_csv(filename)



def defaultGuess(ds, animal ='Mice',t_end =None, nsteps =None, npeople = None, time_step_multiplier = None, external_hazard = None,time_range=None,dt=1, parallel = True):
    """
    This function returns default Guess objects to start the optimization process, based on fitted organisms.
    If an additional parameter is not provided, it will overwrite the default value.

    for Mice:
        theta = [0.00023014, 0.15, 0.16, 17]
        npeople = 5000
        nsteps = 5000
        time_step_multiplier = 3
        external_hazard = np.inf
        t_end =70

    for Humans:
        theta = [0.49275, 54.75, 51.83, 17]
        npeople = 5000
        nsteps = 5000
        time_step_multiplier = 30
        external_hazard = np.inf
        t_end =115

    for Yeast:
        theta = [0.314, 0.121 , 223.651, 185.273]
        npeople = 5000
        nsteps = 5000
        time_step_multiplier = 1
        external_hazard = np.inf
        t_end = 1500
    """

    if animal == 'Mice':
        theta = [0.00023014, 0.15, 0.16, 17]
        nsteps = 5000 if nsteps is None else nsteps
        npeople = 5000 if npeople is None else npeople
        time_step_multiplier = 3 if time_step_multiplier is None else time_step_multiplier
        external_hazard = np.inf if external_hazard is None else external_hazard
        t_end = 70 if t_end is None else t_end

    elif animal == 'Humans':
        theta = [0.49275, 54.75, 51.83, 17]
        nsteps = 5000 if nsteps is None else nsteps
        npeople = 5000 if npeople is None else npeople
        time_step_multiplier = 30 if time_step_multiplier is None else time_step_multiplier
        external_hazard = np.inf if external_hazard is None else external_hazard
        t_end = 115 if t_end is None else t_end


    elif animal == 'Yeast':
        theta = [0.314, 0.121, 223.651, 185.273]
        nsteps = 5000 if nsteps is None else nsteps
        npeople = 5000 if npeople is None else npeople
        time_step_multiplier = 1 if time_step_multiplier is None else time_step_multiplier
        external_hazard = np.inf if external_hazard is None else external_hazard
        t_end = 1500 if t_end is None else t_end

    return Guess(theta, ds,t_end, nsteps, npeople, time_step_multiplier, external_hazard,time_range,dt, parallel)
    