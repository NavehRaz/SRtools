import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numba import jit
from scipy.stats import gaussian_kde
from scipy.stats import skew
import os
from scipy.stats import bootstrap
from collections import namedtuple
from scipy.special import gammaincc,gamma,gammainc,gammaln
from scipy.stats import norm, ks_2samp
from scipy.stats import entropy
from scipy.stats import anderson_ksamp
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from . import deathTimesDataSet as dtds
from .distance_metrics import ks_test, trim_to_range, baysianDistance, baysian_dirichlet_distance, distance

jit_nopython = True
jit_parallel = True

class SR(dtds.Dataset):
    def __init__(self, eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years', memory_efficient =False, natural_units = True, smoothing = 20,
                      boundary = 'sticking',
                        save_dist = False , dist_years =np.linspace(0,100,101), dist_method = 'hist', dist_nvalues = 40, y_gamma =None, death_times_method = 1,external_hazard = np.inf, time_step_multiplier = 1, parallel =False, method='brownian_bridge'):
        """ 
        An instance of SR simulation for SR model dX/dt = eta*t - beta*X/(X+kappa) + sqrt(2*epsilon)*xci.
        Upon creation the model runs (and saves) the trajectories for a number of agents = npeople, running for the number of timesteps defined by 
        nsteps. Then the modle calculates the survival curve, cumulative hazard and hazrard functions.
        To smooth noise, the Hazard is portrayed with a time axis in the time units of the simulation (there is an assumption of at least one timestep per time unit)
        Parameters:
            eta:     SR model eta (damage production units growth parameter) [damage/time^2]
            beta:    SR model beta (damage removal parameter) [damage/time]
            kappa:   SR model kappa (removal saturation paranmeter) [damage]
            epsilon: SR model noise parameter [damage^2/time]
            xc:      SR model X critical [damage]
            npeople: number of agents the simulation is going to run for
            nsteps:  number of time steps for each tragectory of each agent for the simulation (each timestep can be smaller then a time unit)
            t_end:   the end time of the simulation [simulation time units] (time units is expected to be years/months etc...)
            t_start: the start time of the simulation [simulation time units]  defaults to 0
            tscale:  to keep track of what is the meaning of the simulation time unit. defaults to 'years'. Has no implication on any calculation
            memory_efficient: if set to True damage trajectories are not kept and damage related methods are invalid
            natural_units: assumes there is a natural time unit 'U' such as days or years that times are given in (so if U is years and t_end is 120 then the simulation should run to 120 years ) 
                            currently the simulation doesn't care what U is, but returns survival and hazard on a time line that is given by integer values between t_start and t_end (we sample every 1 U)
            smoothing: relevant only if the natural_units = False, then say smoothing = n, the sampling of hazard and survival is going to be every n*dt
            method:  Method to use for death times calculation. Options:
                - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
                - 'euler': Standard Euler method

        """
        if nsteps =='auto':
            nsteps =int(2*(t_end-t_start)/(kappa/beta))
        if eta != np.inf:
            if nsteps <((t_end-t_start) and natural_units):
                raise ValueError('number of time steps should be greater then the number of time units of the simulation') 
            self.natural_units = natural_units
            self.smoothing =smoothing
            self.event_is_censored = False
            self.tscale = tscale
            self.eta = eta
            self.beta = beta
            self.kappa =kappa
            self.epsilon = epsilon
            self.xc = xc
            self.npeople = npeople
            self.nsteps = nsteps
            self.t = np.linspace(t_start,stop = t_end, num = nsteps) #timeline
            self.dt = (t_end-t_start)/nsteps
            self.t_end = t_end
            self.t_start = t_start
            if external_hazard is None:
                external_hazard = np.inf
            self.external_hazard = external_hazard
            self.time_step_multiplier = time_step_multiplier
            self.parallel = parallel
            self.death_times = None
            self.events = None
            self.method = method
            if death_times_method == 0:
                self.trajectories = self.getTrajectories(boundary=boundary)
            else:
                self.trajectories = None
            self.cum_hazard = None
            self.hazard = None
            self.survival = self.calc_survival_and_hazard(death_times_method=death_times_method)    
            self.memory_efficient = memory_efficient
            self.fi=None
            self.sick_spans =None
            self.hle_over_le = None
            self.frailty_values = None
            if y_gamma is not None:
                self.y = self.y_traj_calc(y_gamma)
            else:
                self.y = None
            if memory_efficient:
                if save_dist:
                    self.save_dist = False
                    self.damage_values, self.distributions = self.create_distributions(dist_years,method =dist_method,nvalues=dist_nvalues)
                    # self.trans_dist,self.trans_dist_mat = trans_time_dist(self,mat=True)
                self.trajectories =None
                self.y = None
                self.death_times = None
            elif save_dist: #making sure that save_dist is not used with memory_efficient = False
                raise ValueError('save_dist can only be used with memory_efficient=True')
            self.dist_years = dist_years
            self.dist_method = dist_method
            self.dist_nvalues = dist_nvalues
            self.save_dist = save_dist

            


        
    def resample(self, npeople=None, nsteps=None, t_end=None, t_start=None, natural_units=None, smoothing=None):
        if npeople is None:
            npeople = self.npeople
        if nsteps is None:
            nsteps = self.nsteps
        if t_end is None:
            t_end = self.t_end
        if t_start is None:
            t_start = self.t_start
        if natural_units is None:
            natural_units = self.natural_units
        if smoothing is None:
            smoothing = self.smoothing
        return SR(eta=self.eta, beta=self.beta, kappa=self.kappa, epsilon=self.epsilon, xc=self.xc, npeople=npeople, 
                  nsteps=nsteps, t_end=t_end, t_start=t_start)#,
                   # natural_units=natural_units, smoothing=smoothing,memory_efficient=self.memory_efficient,
                    #    save_dist = self.save_dist , dist_years =self.dist_years, dist_method = self.dist_method, dist_nvalues = self.dist_nvalues)

    @staticmethod
    def loadSim(path):
        #loads a model from a file
        file_path = os.path.join(path, os.listdir(path)[0])
        sim = SR(eta=np.inf,beta=0,kappa=0,epsilon=0,xc=0,npeople=0,nsteps=0,t_end=0)
        sim.load(file_path)
        return sim

    def getTrajectories (self,boundary='sticking'):
        #simulates the SDE dx/dt=eta*t-b(x/(x+k))+N(0,sig)
        # where N(0,sig) is normally distributed gaussian noise with std eps
        # The assumption is x0 = 0. The model returns the vector (x(t))
        
        s = len(self.t) #size of vector

        xt = np.zeros((s,self.npeople))
        
        dt = self.t[1:s]-self.t[0:s-1]
        sdt = np.sqrt(dt)
    
        noise = np.sqrt(2*self.epsilon)*np.random.normal(loc = 0,scale = 1, size =(s-1,self.npeople))

        # for i in tqdm(range(s-1)):
        
        #     xt[i+1,:] = xt[i,:]+dt[i]*(self.eta*self.t[i]-self.beta*xt[i,:]/(xt[i,:]+self.k))+noise[i,:]*sdt[i]
        #     xt[xt<0]=0

        xt = trajectories_accelerator(noise=noise,xt=xt,s=s,dt=dt,eta=self.eta,t=self.t,beta=self.beta,k=self.kappa,sdt=sdt,boundary=boundary)
        return xt
    
    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        if self.method == 'brownian_bridge':
            if self.parallel:
                death_times, events = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        elif self.method == 'euler':
            if self.parallel:
                death_times, events = death_times_accelerator2(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_accelerator(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
        else:
            # Default to brownian bridge if method not recognized
            if self.parallel:
                death_times, events = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)
            else:
                death_times, events = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.beta, self.kappa, self.epsilon, self.xc, sdt, self.npeople, self.external_hazard, self.time_step_multiplier)

        return np.array(death_times), np.array(events)
    
    def shape(self):
        np = self.npeople
        nstep = self.nsteps
        return [np,nstep]
    
    def calc_survival_and_hazard(self,death_times_method = 1):
        #Gets a Matrix x where each row represents the damage trajectory of an individual, where
        #   the resolution is in timestep dt.
        #Calculates the survival probability where "death" is defined as crossing the threshold xc
        #Returns the "death times and probabilities"
        if death_times_method == 0:
            x = self.trajectories
            ind = x>self.xc
            deaths = np.argmax(ind,0) #first index each trajectory crossed xc
            death_times = deaths*self.dt
        elif death_times_method == 1:
            death_times, events = self.calc_death_times()
            raise ValueError('death_times_method = 1 is not supported becuase censorships are not implemented')
        self.death_times = death_times[:]
        death_times = death_times[death_times!=0] #remove times of people that didn't die
        sorted_dtimes = np.sort(death_times)
        times, death_counts = np.unique(sorted_dtimes, return_counts=True)
        times = np.insert(times,0,0)
        death_counts = np.insert(death_counts,0,0)
        cum_deaths = np.cumsum(death_counts)
        # unnormalized_hazard = death_counts/(self.npeople-cum_deaths)

        if self.natural_units:
            #Note that I now will work with dt = 1 time unit, so everything is naturally normalized. If dt is changed then should add normalization
            time_in_tunits = np.linspace(self.t_start,self.t_end,self.t_end-self.t_start+1)
            cum_deaths_per_time_unit = np.interp(time_in_tunits,times,cum_deaths)
            deaths_per_time_unit = np.diff(cum_deaths_per_time_unit, prepend=0)
            dt =1
        else:
            time_in_tunits = np.linspace(self.t_start,stop = self.t_end, num = int(self.nsteps/self.smoothing))
            dt = time_in_tunits[1]-time_in_tunits[0]
            cum_deaths_per_time_unit = np.interp(time_in_tunits,times,cum_deaths)/dt
            deaths_per_time_unit = np.diff(cum_deaths_per_time_unit, prepend=0)
        
        #treating the case all agents died
        deaths_per_time_unit_truncated = deaths_per_time_unit[self.npeople-(cum_deaths_per_time_unit*dt)>0]
        cum_deaths_per_time_unit_truncated = cum_deaths_per_time_unit[self.npeople-(cum_deaths_per_time_unit*dt)>0]
        time_in_tunits_truncated = time_in_tunits[self.npeople-(cum_deaths_per_time_unit*dt)>0]

        #calculating the hazard
        hazard_per_time_unit = deaths_per_time_unit_truncated/(self.npeople-(cum_deaths_per_time_unit_truncated*dt))
        
        
        self.hazard = [time_in_tunits_truncated, hazard_per_time_unit]
        # self.hazard = np.interp(htimes,times,self.hazard)
        cum_hazard = np.cumsum(hazard_per_time_unit*dt)
        self.cum_hazard = [time_in_tunits_truncated, np.cumsum(hazard_per_time_unit)] #Nelson Aalen estimator
        # cum_prob_deaths = np.arange(1, len(sorted_dtimes) + 1) / len(sorted_dtimes)
        sur_prob = np.exp(-1*cum_hazard)
        return [time_in_tunits_truncated,sur_prob]

    def getHazard(self):
        #Returns the hazard in format ([times,hazard])
        return self.hazard
    
    def get_cum_hazard(self):
        #Returns the hazard in format ([times,hazard])
        return self.cum_hazard
    
    def getSurvival(self,complete_timeline = False,time_range=None):
        #Returns the hazard in format ([times,survival])
        if complete_timeline:
            t,s = self.survival
            if len(t)>1:
                dt = t[1]-t[0]
            else:
                self.printParams()
            if dt!=1:
                raise ValueError('complete_timeline is only valid for natural_units = True')
            t_full = np.linspace(self.t_start,self.t_end,self.t_end-self.t_start+1)
            s_full = np.interp(t_full,t,s)
            s_full[s_full<0]=0
            return t_full,s_full
        elif time_range is not None:
            t,s = self.survival
            t,s = trim_to_range(t,s,time_range,renormalize_survival=True)
        else:
            t,s = self.survival
        return t,s
    
    def get_time_of_death_distribution(self):
        #returns the time of death distribution calculated from the survival curve
        t,s = self.survival
        cum_prob_death = 1-s
        prob_death = np.diff(cum_prob_death,prepend=0)/np.diff(t,prepend=-self.dt)
        return t,prob_death
    
    def getDeathTimes(self):
        return self.death_times



    def get_median_damage(self):
        #retunrns the median damage of the trajectories
        medians =np.zeros_like(self.t)
        medians = median_accelerator(medians,self.trajectories,self.xc)
        return medians
    
    def get_damage_PDF(self, t, nvalues=40, pdf_method = 'kde'):
        """
        Returns the damage distribution for a given time t.

        Parameters:
        - t (float): The time for which the damage distribution is calculated.
        - nvalues (int): The number of values in the damage distribution.
        - pdf_method (str): The method used to calculate the probability density function (PDF).
            Options are 'kde' (kernel density estimation) or 'hist' (histogram). Default is 'kde'.

        Returns:
        - damage_values (ndarray): An array of damage values ranging from 0 to self.xc.
        - pdf (ndarray): The probability density function (PDF) of the damage distribution.
        """
        if self.memory_efficient and self.save_dist:
            return self.damage_values,self.distributions[str(t)]
        elif self.trajectories is None:
            raise ValueError('No trajectories were saved or calculated. Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        
        i = np.argmin(np.abs(self.t - t))
        trag = self.trajectories[i, :]
        trag = trag[trag < self.xc]

        if pdf_method == 'kde':
            if len(trag) < 2:
                trag = np.array([self.xc+1,self.xc+1,self.xc+1])
            kde = gaussian_kde(trag)
            damage_values = np.linspace(0, self.xc, nvalues)
            pdf = kde(damage_values)
            return damage_values, np.array(pdf)
        elif pdf_method == 'hist':
            damage_values = np.linspace(0, self.xc, nvalues)
            pdf, _ = np.histogram(trag, bins=damage_values, density=True)
            return damage_values, np.array(pdf)
        else:
            raise ValueError('pdf_method should be either "kde" or "hist"')
     
    def create_distributions(self, dist_years, method='kde', nvalues=40):
        """
        Create distributions of damage values for each year in dist_years.

        Parameters:
        - dist_years (ndarray): An array of years for which to calculate the damage distributions.

        Returns:
        - distributions (dict): A dictionary containing the damage distributions for each year.
        """
        distributions = {}
        for year in dist_years:
            damage_values, pdf = self.get_damage_PDF(year, nvalues=nvalues, pdf_method=method)
            damage_values,distributions[str(year)] = (damage_values, pdf)
        return damage_values, distributions
    
    def age_distribution_by_damage(self, x_bin, nvalues=40, pdf_method='kde',age_limits = [50,90]):
        """
        Calculate the age distribution of individuals with damage values in the range x_bin.

        Parameters:
        - x_bin (list): A list of two values representing the range of damage values.
        - nvalues (int): The number of values in the damage distribution.
        - pdf_method (str): The method used to calculate the probability density function (PDF).
            Options are 'kde' (kernel density estimation) or 'hist' (histogram). Default is 'kde'.

        Returns:
        - age_values (ndarray): An array of age values.
        - pdf (ndarray): The probability density function (PDF) of the age distribution.
        """
        if self.memory_efficient and self.save_dist:
            return self.damage_values,self.distributions[str(t)]
        elif self.trajectories is None:
            raise ValueError('No trajectories were saved or calculated. Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        
        traj = self.trajectories.copy()
        t =self.t
        ages =np.ones_like(traj)
        #traj is an array with damage values, here I want to create an array with the age of each damage value
        for i in range(len(t)):
            ages[i,:] = t[i]



        #keep only the ages of the people with damage values in the range x_bin
        ind = (traj[:, :] >= x_bin[0]) & (traj[:, :] <= x_bin[1])
        ind = ind.flatten()
        ages = ages.flatten()
        ages = ages[ind]
        ages = ages[(ages>=age_limits[0]) & (ages<=age_limits[1])]

       

        #create the age distribution
        age_values = np.linspace(0, self.t_end, nvalues)
        if pdf_method == 'kde':
            kde = gaussian_kde(ages)
            pdf = kde(age_values)
        elif pdf_method == 'hist':
            pdf, _ = np.histogram(ages, bins=age_values, density=True)
        else:
            raise ValueError('pdf_method should be either "kde" or "hist"')

        return age_values, pdf
        

    def damage_transition_probabilities(self,t_interval,xbins,t_averaging_interval,min_age = 0,max_age = 'Auto'):
        """
        Calculates the transition probabilities between damage bins.
        The time of transition is t_interval and the averaging interval is t_averaging_interval (current damage is considered a time average over this interval)
        note that the averaging interval should be smaller or equal to t_interval, and an integer multiplication of it.
        Returns a matrix of transition probabilities.
        if less then 10% of the population survived to the min_age, trans matrix is not valid and None is returned

        
        Parameters:
        - t_interval(float) is the time interval for the transition. Should be an integer multiplication of t_averaging_interval
        - xbins(list) is the damage bins to calculate the transition probabilities to
        - t_averaging_interval(float) is the time averaging interval to calculate the current damage
        - min_age(int) is the minimum age to consider in the transition probabilities calculation

        Returns:
        - transition_matrix (ndarray): A matrix of transition probabilities between damage bins, the last column indicates the death probabilities.
        
        """
        if t_averaging_interval>t_interval:
            raise ValueError('t_averaging_interval should be smaller or equal to t_interval')
        if self.trajectories is None:
            raise ValueError('Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        # if t_interval%t_averaging_interval!=0:
        #     raise ValueError('t_interval should be an integer multiplication of t_averaging_interval')
        if max_age == 'Auto':
            max_age = self.t_end
        
        t = self.t.copy()
        if self.y is not None:
            x=self.y.copy()
            right =True
        else:
            x = self.trajectories.copy()
            right = False
        dt = self.dt
        npeople = self.npeople
        nxbins = len(xbins)
        transition_matrix = np.zeros((nxbins-1,nxbins))
        #reduce the time and trajectories to the time_avraging_interval:
        n_average = int(t_averaging_interval//dt)
        x_averaged = np.zeros((len(t)//n_average,npeople))

        for i in range(len(t)//n_average):
            t_interval_ind = np.where((t>=i*n_average*dt) & (t<(i+1)*n_average*dt))
            t_interval_ind = t_interval_ind[0]
            x_interval = x[t_interval_ind,:]
            x_averaged[i,:] = np.mean(x_interval,axis=0)
        
        #redefine the variables to avraged time snd damage bins
        x = x_averaged
        x_bins_with_deaths = list(xbins.copy()) # Convert xbins to a list
        x_bins_with_deaths.append(np.inf)
        x_bins_with_deaths = np.array(x_bins_with_deaths)
        # print('before digitation',x[:,0:10])
        x = np.digitize(x,x_bins_with_deaths,right=right) #convert damage values to bins indices
        

        # print('after digitation',x[:,0:10])
        t = t[::n_average]
        dt = t[1]-t[0]
        ind_min_t = np.argmin(np.abs(t-min_age))
        t = t[ind_min_t:]
        x = x[ind_min_t:,:]

        ind_max_t = np.argmin(np.abs(t-max_age))
        t = t[:ind_max_t]
        x = x[:ind_max_t,:]

        #remove people that died before the min_age
        has_died = self.has_died_in_interval([0,min_age+dt])
        # print(np.count_nonzero(~has_died))
        # print(np.count_nonzero(has_died))
        if np.count_nonzero(~has_died)<self.npeople/10:
            return None




        time_interval_index_increment = int(t_interval//t_averaging_interval)
        #calculate the transition matrix
        for i in range(len(t)//time_interval_index_increment-1):
            x1 = x[i*time_interval_index_increment,:]
            x2 = x[(i+1)*time_interval_index_increment,:]
            current_time_interval = [0,t[(i+1)*time_interval_index_increment]+dt]
            # print(x1)
            for j in range(npeople):
                # print(x1[j],x2[j])
                if ~has_died[j]:
                    if x1[j]==nxbins:
                        print('x1[j]',x1[j])
                        print(self.has_died_in_interval(current_time_interval)[j])
                    transition_matrix[x1[j]-1,x2[j]-1]+=1
            #remove people that died in the interval
            has_died = self.has_died_in_interval(current_time_interval)
  
        #normalize the transition matrix
        np.set_printoptions(suppress=True)
        # print('transition_matrix\n', transition_matrix)
        transition_matrix = transition_matrix/np.sum(transition_matrix,axis=1)[:,np.newaxis]
        transition_matrix[transition_matrix==0]=5e-9
        return transition_matrix

        
    def has_died_in_interval(self,t_interval):
        """
        Returns a boolean array indicating if an individual has died in the interval t_interval.

        Parameters:
        - t_interval (list): A list of two values indicating the time interval for which to check if an individual has died.

        Returns:
        - died (ndarray): A boolean array indicating if an individual has died in the interval t_interval.
        """
        if self.death_times is None:
            raise ValueError('No death times were saved or calculated. Run the simulation first.')
        death_times = self.death_times.copy()
        death_times[death_times==0]=np.inf
        return (death_times >= t_interval[0]) & (death_times <= t_interval[1])
        



    def plotHazard(self, ax=None,label='', smooth =True, scale=1,title='Survival', xlabel='time [years]', ylabel='survival', **kwargs):
        #plots the hazard.
        #smooth is provition for inhereting classes (meaning less for the regular SR class)
        #scale scales time, since the hazard is given per unit time scaling the t axis by a means we need to scale the hazard by 1/a as well
        t,h = self.hazard
        t = t*scale
        h = h/scale 
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t,h, **kwargs)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plotSurvival(self, ax=None, scale =1,time_range = None,title='Survival', xlabel='time [years]', ylabel='survival', **kwargs):
        #plots the survival.
        #scale scales time
        t,s = self.getSurvival(time_range=time_range)
        t=t*scale
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t,s,**kwargs)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    
    def plotCumHazard(self, linestyle='-',lw=0.6, ax=None,label='', scale=1):
        label = label
        t,s = self.survival
        t = t*scale
        if ax is None:
            plt.plot(t,-np.log(s),linewidth=lw,linestyle=linestyle,label = label)
            plt.legend()
        else:
            ax.plot(t,-np.log(s),linewidth=lw,linestyle=linestyle,label = label)
            ax.legend()
        return  
    
    
    def calcHmax(self,percentage=1):
        # returns the median of the top "percentage" of the hazard
        t,h = self.hazard
        length = len(h)
        i = int((100-percentage)*length/100)
        toph = h[i:]
        return np.median(toph)
    
    def gompSlope(self):
        uri = self.eta*self.xc/self.epsilon
        kramer = self.eta*self.xc/self.epsilon - self.eta/self.beta

        return uri, kramer
    
    def h0(self):
        uri = np.exp(-self.beta*self.xc/self.epsilon)*(self.beta**2)/self.epsilon
        pref =0.7
        p = self.kappa*self.beta/self.epsilon
        kramer = pref*((self.kappa+self.xc)**p)*(self.kappa**(-p-0.5))*np.sqrt(self.beta)*np.exp(-self.beta*self.xc/self.epsilon)
        return uri, kramer

    def lifeTimeAtSurvival(self, probDeath):
        t,s = self.survival
        i_sp = np.argmin(np.abs(s-probDeath))
        return t[i_sp]
    
    def getLifeSpan(self):
        #returns the median lifetime as calculated from the simulation, and the therotical one from the simulations known parameters.
        #the theoretical calculation is based on Avi's work which assumes full gompertz hazard
        L_simulated = self.lifeTimeAtSurvival(0.5)
        L_theoretical = (self.beta/self.eta )+(self.epsilon/(self.xc*self.eta))*np.log(self.xc*self.eta*np.log(2)/(self.beta**2))
        # L_theoretical = (self.beta/self.eta )+((self.epsilon/(self.xc*self.eta))*np.log(self.kappa*self.xc*self.eta*np.log(2)/(self.beta*self.epsilon)))
        return L_simulated, L_theoretical
      
    
    def printParams(self):
        print(f'eta = {self.eta}, beta = {self.beta}, kappa = {self.kappa}, epsilon = {self.epsilon}, xc = {self.xc}')



    def get_sickspan(self,xd,min_age=0):

        if self.trajectories is None:
                raise ValueError('No trajectories were saved or calculated. Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        x = self.trajectories
        ind_xc = x>self.xc
        deaths = np.argmax(ind_xc,0) #first index each trajectory crossed xc
        
        death_times = deaths[deaths!=0] #remove times of people that didn't die
        
        min_age_index = np.abs(self.t - min_age).argmin() #index of min_age
        ind = (deaths!=0) & (deaths>min_age_index)
        death_times = death_times[death_times>min_age_index]#remove deaths before min_age

        ind_xd = x>xd

        ill_start = np.argmax(ind_xd,0) #first index each trajectory crossed xd
        ill_start = ill_start[ind] 
        ill_start = np.maximum(ill_start,min_age_index)
        # print('mean age at xd: ',np.mean(ill_start)*self.dt)
        # print('mean age at death: ',np.mean(death_times)*self.dt)


        dt = self.t[1]-self.t[0]
        healthy_span = (ill_start-min_age_index)*dt #time from min_age to xd
        life_exp_from_min_age = (death_times-min_age_index)*dt #time from min_age to death
        print('mean life exp from min age: ',np.mean(life_exp_from_min_age))
        print('mean healthy span: ',np.mean(healthy_span))

        sick_spans = (death_times-ill_start)*dt
        self.sick_spans = sick_spans
        self.hle_over_le = healthy_span/life_exp_from_min_age
        return sick_spans,healthy_span/life_exp_from_min_age
    
    def mean_sickspan(self,xd):
        sick_spans,_ = self.get_sickspan(xd)
        return np.mean(sick_spans)

    def mean_hle_over_le(self,xd,min_age=65):
        _,hle_over_le = self.get_sickspan(xd,min_age)
        return np.mean(hle_over_le)


    def kramer_damage_distribution(self, t, nvalues=40, stats = False):
        """
        Returns the damage distribution for a given time t.

        Parameters:
        - t (float): The time for which the damage distribution is calculated.
        - nvalues (int): The number of values in the damage distribution.

        Returns:
        - damage_values (ndarray): An array of damage values ranging from 0 to self.xc.
        - pdf (ndarray): The probability density function (PDF) of the damage distribution.
        """
        if (t-int(t)!=0):
            raise ValueError('t should be an integer')
        x= np.linspace(0,self.xc,nvalues)
        p = self.kappa*self.beta/self.epsilon
        a = (self.kappa+x)**p
        b = (-self.eta*t/self.epsilon-p)**(p+1)
        k=1/(self.kappa**(1+p))
        ex = np.exp(-(x+self.kappa)*(-self.eta*t+self.beta)/self.epsilon)
        gam =gamma(1+p)*gammaincc(1+p,-(self.eta*t-self.beta)*self.kappa/self.epsilon)
        dx = x[1]-x[0]
        pdf = np.real(k*a*b*ex/gam)
        N = np.sum(pdf)*dx
        pdf = pdf/N
        if stats:
            mean = np.sum(pdf*x)*dx
            std = np.sqrt(np.sum((pdf*(x-mean)**2)*dx))
            # print(t,N,np.sum(pdf*x),mean,std)
            return x, pdf, np.real(mean), std
        else:
            return x, pdf


    def plotTrajectoriies(self,n_trajectories,randomize_index=False,resample_traj=10, mark_deaths = True,ax=None, fix_looks = True,kwargs={},y=False):
        if self.trajectories is None:
            raise ValueError('No trajectories were saved or calculated. Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        if randomize_index:
            indices = np.random.choice(self.npeople, n_trajectories, replace=False)
        else:
            indices = np.arange(n_trajectories)
        j=0
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']  # Add more colors as needed
        color_array = [colors[i % len(colors)] for i in range(n_trajectories)]
        for i in indices:
            if y:
                traj = self.y[0::resample_traj, i]
            else:
                traj = self.trajectories[0::resample_traj, i]
            t = self.t[0::resample_traj]
            if mark_deaths:
                death_index = np.argmax(traj>self.xc)+1
            else:
                death_index = len(traj)
            if ax is None:
                ax=plt.gca()
            
            if death_index>0:
                ax.plot(t[0:death_index], traj[0:death_index],color=(0.0, 0.2+0.3*(j%3), 0.5, 1.0), **kwargs)
                # ax.plot(t[death_index:], traj[death_index:], color='gray',alpha=0.2, **kwargs)
                #remove top and right lines from the graph:
                if fix_looks:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            j+=1
        if mark_deaths:
            plt.axhline(y=self.xc, color='gray', linestyle='--')
            #add red dots at the death points
            for i in indices:
                if y:
                    traj = self.y[0::resample_traj, i]
                else:
                    traj = self.trajectories[0::resample_traj, i]
                death_index = np.argmax(traj>self.xc)
                if death_index>0:
                    plt.plot(t[death_index-1]/2+t[death_index]/2, self.xc, 'ro')
        if fix_looks:
            ax.set_xlabel('Age')
            ax.set_ylabel('Damage')
        return ax
        
    def meanDamageTrajectories(self,time):
        if self.trajectories is None:
            raise ValueError('No trajectories were saved or calculated. Trajectories are not kept for initiating with "memory_efficient=True" or for SR_GWTW')
        window_size = time/self.dt
        window_size = int(window_size)
        
        trajectories_ma = np.zeros_like(self.trajectories)
        for i in range(len(self.trajectories)):
            if i < window_size:
                trajectories_ma[i] = np.mean(self.trajectories[:i+1,:],axis=0)
            else:
                trajectories_ma[i] = np.mean(self.trajectories[i-window_size+1:i+1,:],axis=0)
        return trajectories_ma

    def get_ind_at_age(self,age):
        return np.abs(self.t - age).argmin()

    
def gompetz_hazard(t,xc,k,beta,eps,eta):
    p = k*beta/eps
    a = (k+xc)**p
    b = (k*beta)**(-p-0.5)
    c = (beta-eta*t)**(p+1)
    e =np.exp(((k+xc)*eta*t-(xc*beta))/eps)
    gh = a*b*c*e
    return gh   

def get_survival_from_hazard(h,t):
    dt = np.zeros_like(t)
    dt[1:] = t[1:]-t[0:-1]
    ih = np.cumsum(h*dt)
    s=np.exp(-ih)
    return s

def get_dimless_groups(eta,beta,kappa,epsilon,xc):
    #return the dimetionless groups scaled by kappa
    D31 = beta**2 /(kappa*eta)
    D32 = beta*epsilon/(eta*(kappa**2))
    D21 = D31/D32
    Dx = xc/kappa
    return D31,D32,D21,Dx

@jit(nopython=jit_nopython, parallel=jit_parallel)
def trajectories_accelerator(noise,xt,s,dt,eta,t,beta,k,sdt,boundary ='sticking'):
    for i in range(s-1):
        
            if boundary == 'sticking':
                xt[i+1,:] = xt[i,:]+dt[i]*(eta*t[i]-beta*xt[i,:]/(xt[i,:]+k))+noise[i,:]*sdt[i]
                xt[i + 1, :] = np.maximum(xt[i + 1, :], 0)
            elif boundary == 'reflecting':
                xt[i+1,:] = xt[i,:]+dt[i]*(eta*t[i]-beta*xt[i,:]/(xt[i,:]+k))+noise[i,:]*sdt[i]
                xt[i + 1, :] = np.abs(xt[i + 1, :])
            elif boundary == 'RBM':
                Y = noise[i, :] *  sdt[i] 
                U = np.random.uniform(0, 1, len(xt[i,:]))
                M = (Y + np.sqrt(Y**2 - 2 * dt[i] * np.log(U))) / 2
                delta_X = dt[i] * (eta * t[i] - beta * xt[i, :] / (xt[i, :] + k))
                xt[i + 1, :] = np.maximum(M - Y, xt[i, :] + delta_X - Y)
            elif boundary == 'custom':
                thresh = 3
                mask =np.maximum(0,thresh-xt[i,:])
                mask2 = np.maximum(0,noise[i,:])
                fix = (noise[i,:]-mask2)*mask*(1/thresh)
                xt[i+1,:] = xt[i,:]+dt[i]*(eta*t[i]-beta*xt[i,:]/(xt[i,:]+k))+(noise[i,:]-fix)*sdt[i]
                xt[i + 1, :] = np.maximum(xt[i + 1, :], 0)
            elif boundary == 'custom2':
                #NOT USABLE
                # mask_neg = np.minimum(0,xt[i,:])
                # mask_noise =noise[i,:]
                # mask_noise[mask_neg==0] =0 
                # fix = dt[i]*(-beta*mask_neg/(mask_neg+k))+mask_noise*sdt[i]
                # xt[i+1,:] = xt[i,:]+dt[i]*(eta*t[i]-beta*xt[i,:]/(xt[i,:]+k))+(noise[i,:])*sdt[i]-fix
                mask =np.zeros_like(xt[i,:])
                mask[xt[i,:]>0] = 1
                mask_neg = np.zeros_like(xt[i,:])
                mask_neg[xt[i,:]<0] = 1
                xt[i+1,:] = xt[i,:]+dt[i]*(eta*t[i]-beta*xt[i,:]*mask/(xt[i,:]*mask+k)-xt[i,:]*mask_neg)+(noise[i,:])*sdt[i]*mask

            else:
                raise ValueError('boundary should be either "sticking" or "reflecting"')

        
    return xt

@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta,beta,kappa,epsilon,xc,sdt,npeople,external_hazard = np.inf, time_step_multiplier = 1):
    death_times = []
    events = []
    for l in range(npeople):
        x=0
        j=0
        ndt = dt/time_step_multiplier
        nsdt = sdt/np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard)*ndt
        while j in range(s-1) and x<xc:
            for i in range(time_step_multiplier):
                noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                x = np.maximum(x, 0)
                if np.random.uniform(0,1)<chance_to_die_externally:
                    x = xc
                if x>=xc:
                    break
            j+=1
        if x>=xc:
            death_times.append(j*dt)
            events.append(1)
        else:
            death_times.append(j*dt)
            events.append(0)

    return death_times, events

@jit(nopython=jit_nopython)
def death_times_accelerator2(s,dt,t,eta,beta,kappa,epsilon,xc,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta, beta, kappa, epsilon, xc, sdt, external_hazard,time_step_multiplier):
        death_times = []
        events =[]
        for l in range(npeople):
            died = False
            x = 0
            j = 0
            ndt = dt/time_step_multiplier
            nsdt = np.sqrt(ndt)
            chance_to_die_externally = np.exp(-external_hazard)*ndt
            while j in range(s - 1) and x < xc and not died:
                for i in range(time_step_multiplier):
                    noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                    x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                    x = np.maximum(x, 0)
                    if np.random.uniform(0,1)<chance_to_die_externally:
                        x = xc
                    if x>=xc:
                        died = True
                j += 1
            if died:
                death_times.append(j * dt)
                events.append(1)
            else:
                death_times.append(j * dt)
                events.append(0)
        return death_times, events

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta, beta, kappa, epsilon, xc, sdt, external_hazard, time_step_multiplier
    ) for _ in range(n_jobs))

    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    return death_times, events

# Euler method with Brownian Bridge (without heterogeneity)
@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge(s, dt, t, eta, beta, kappa, epsilon, xc, sdt, npeople,
                                     external_hazard=np.inf, time_step_multiplier=1):
    """
    Euler method with Brownian bridge crossing detection.
    This method uses the standard Euler scheme but adds Brownian bridge
    crossing probability tests to detect barrier crossings between time steps.
    """
    death_times = []
    events = []
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        j = 0
        crossed = False
        
        while j < s - 1 and not crossed:
            for i in range(time_step_multiplier):
                current_time = t[j] + i * ndt
                
                # Standard Euler step
                drift = eta * current_time - beta * x / (x + kappa)
                noise = np.sqrt(2 * epsilon) * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test if not crossed directly
                if (x < xc) and (x_new < xc):
                    dx1 = xc - x
                    dx2 = xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        # Brownian bridge crossing probability
                        var = 2.0 * epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= xc:
            events.append(1)
        else:
            events.append(0)
    
    return death_times, events

# Parallel version of Euler with Brownian Bridge (without heterogeneity)
def death_times_euler_brownian_bridge_parallel(s, dt, t, eta, beta, kappa, epsilon, xc, sdt, npeople,
                                              external_hazard=np.inf, time_step_multiplier=1, n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge.
    Splits npeople into chunks and runs death_times_euler_brownian_bridge on each chunk in parallel.
    """
    from joblib import Parallel, delayed
    import numpy as np

    def worker(npeople_chunk, s, dt, t, eta, beta, kappa, epsilon, xc, sdt, external_hazard, time_step_multiplier):
        # Call the numba-jitted function for this chunk
        return death_times_euler_brownian_bridge(
            s, dt, t, eta, beta, kappa, epsilon, xc, sdt, npeople_chunk,
            external_hazard, time_step_multiplier
        )

    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta, beta, kappa, epsilon, xc, sdt, external_hazard, time_step_multiplier
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )

    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    return death_times, events

@jit(nopython=jit_nopython, parallel=jit_parallel)
def median_accelerator(medians, tragectories,xc):
    for i in range(len(medians)):
        tragi = tragectories[i,:]
        tragi = tragi[tragi<xc]
        medians[i] = np.median(tragi)
            
        
    return medians


############################################
class SR_karin_human(SR):
    def __init__(self,eta=1,beta=1,kappa=1,epsilon=1,xc=1,
                 npeople=10000, nsteps=6000,
                   t_end=120, t_start = 0, tscale = 'years',
                     memory_efficient =False,natural_units=False, smoothing =20,boundary = 'sticking',
                        save_dist = False , dist_years =np.linspace(0,100,101), dist_method = 'hist', dist_nvalues = 40,y_gamma =None,death_times_method = 0,external_hazard = np.inf):
        super().__init__(eta=0.00135 *365*eta, beta=0.15*365*beta, kappa=0.5*kappa, epsilon=0.142 *365*epsilon, xc=17*xc,
                         npeople=npeople, nsteps=nsteps, t_end=t_end, t_start = t_start, tscale = tscale,
                           memory_efficient = memory_efficient, natural_units=natural_units, smoothing=smoothing,boundary = boundary,
                        save_dist = save_dist , dist_years =dist_years, dist_method = dist_method, dist_nvalues = dist_nvalues,y_gamma=y_gamma,death_times_method = death_times_method,external_hazard = external_hazard)
############################################

class SRNDK(SR):
    #Non dimentionalized SR model D, F, XC
    def __init__(self,kappa,t3,D31,D32,Dx,npeople, nsteps, t_end, t_start = 0, tscale = 'years', memory_efficient =False,natural_units=False, smoothing =20):
        self.t3 = t3
        t_end = t_end*t3
        self.D31 = D31
        self.D32 = D32
        self.D21 = D31/D32
        self.Dx =Dx
        xc = Dx*kappa
        beta = D31*kappa/t3
        epsilon = self.D32*(kappa**2)/t3
        eta = D31*kappa/(t3**2)
        super().__init__(eta, beta, kappa, epsilon, xc,npeople, nsteps, t_end, t_start = 0, tscale = tscale, memory_efficient = memory_efficient, natural_units=natural_units, smoothing=smoothing)


class SRND_Peckle(SR):
    #Non dimentionalized SR model Lambda, F, XC (Lambda is Peckles number = F/D)
    def __init__(self,kappa,t3,F,Lambda,Xc,npeople, nsteps, t_end, t_start = 0, tscale = 'years', memory_efficient =False,natural_units=False, smoothing =20):
        self.t3 = t3
        t_end = t_end*t3
        self.F = F
        self.Lambda = Lambda
        self.Xc =Xc
        xc = Xc*kappa
        beta = F*kappa/t3
        epsilon = (F/(Lambda))*(kappa**2)/t3
        eta = F*kappa/(t3**2)
        super().__init__(eta, beta, kappa, epsilon, xc,npeople, nsteps, t_end, t_start = 0, tscale = tscale, memory_efficient = memory_efficient, natural_units=natural_units, smoothing=smoothing)


############################################

class SR_GWTW(SR):
    #SR class with hazard and survival calculated via "go with the winner" algorithem to probe later times. Note that tragectories are not kept for this version. 
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start=0, tscale='years', memory_efficient=False,natural_units=False, smoothing=20):
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start, tscale, memory_efficient,natural_units=natural_units, smoothing=smoothing)
        self.hazard_not_smoothed = self.hazard
        self.hazard = self.smooth_hazard()

    def getTrajectories(self,boundary='None'):
        return None
    
    # Step function for handling x > 0 condition
    def step(self, x):
        return 1 * (x > 0)
    
    # Drift function in the classic stochastic model
    def drift_classic(self, x, t):
        prod = self.eta * t
        remov = self.step(x) * x * (self.beta / (x + self.kappa))
        return prod - remov

    # Noise function representing stochastic fluctuation
    def noise(self, x, t):
        return self.step(x) * np.sqrt(2 * self.epsilon)
    
    def calc_survival_and_hazard(self,death_times_method = 0):

        x0 = 1e-10
        tspan = self.t
        Xs = x0*np.ones((int(self.npeople)))
        weights = np.ones((int(self.npeople)))
        hazard = np.zeros(len(tspan))
        survival = np.zeros(len(tspan))
        pruning_counts = 0
        method=3

        for i_t in range(1, np.size(tspan)):
            Xsprev = Xs
            tcur = tspan[i_t]

            if method == 3:
                # method 3 refers to algorithm for reflecting Brownian Motion
                Y = self.noise(Xsprev, tcur) * np.sqrt(self.dt) * np.random.normal(size=int(self.npeople))
                U = np.random.uniform(0, 1, self.npeople)
                M = (Y + np.sqrt(Y**2 - 2 * self.dt * np.log(U))) / 2
                delta_X = self.dt * self.drift_classic(Xsprev, tcur)
                Xs = np.maximum(M-Y, Xsprev + delta_X - Y)
            else:
                # rever to simple method (probably incorrect)
                Xs = np.maximum(0, Xsprev + self.dt * self.drift_classic(Xsprev, tcur)
                                       + self.noise(Xsprev, tcur) * np.sqrt(self.dt) * np.random.normal(size=int(self.npeople)))


        
            
            who_died = np.where(Xs >= self.xc)[0]
            who_survived = np.where(Xs < self.xc)[0]
        # calculate hazard based on weighted deaths and weighted survival
        # Avoid division by zero
            if len(who_survived) > 0:
                hazard[i_t] = (np.sum(weights[who_died]) / np.sum(weights[who_survived]))/self.dt
            else:
                self.hazard = [tspan, hazard]
                return tspan, survival
            
            normalized_weights = weights[who_survived] / np.sum(weights[who_survived])
            
            # Choose a random index from those that haven't died, bias towards high weights
            if np.any(np.isnan(normalized_weights)):
                print('We got NaN!')
                self.hazard = [tspan, hazard]
                return tspan, survival
                
            ## clone those who died from those still alive        
            replace_indices = np.random.choice(who_survived, size=len(who_died), p=normalized_weights)
            Xs[who_died] = Xs[replace_indices]
            
            # Adjust weights
            weights[who_died] /= 2
            weights[replace_indices] /= 2
            # Prune weights that are too small  
            if (weights<1e-100).all():
                weights = weights*1e50
                pruning_counts += 1
                #print('pruned!', tspan[i_t])

            # Calculate survival probability
            weight_sum =  sum(weights)
            for _ in range(pruning_counts):
                weight_sum = weight_sum*(1e-50)
            survival[i_t] = weight_sum / self.npeople
            #print(tspan[i_t], len(who_died) , len(who_survived))
        self.hazard = [tspan, hazard]
        return tspan, survival
    
    def get_cum_hazard(self):
        #Returns the hazard in format ([times,hazard])
        t,s = self.survival
        return t,-np.log(s)
    
    def smooth_survival(self):
        t=np.linspace(0,self.t_end,self.t_end+1)
        ts,s=self.survival
        s_smoothed = np.interp(t,ts,s)
        return t,s_smoothed

    def smooth_hazard(self, res=5):
        t_old,s = self.survival
        t_new = np.linspace(self.t_start,self.t_end,int(res*(self.t_end-self.t_start+1)))
        # t_new = t_old
        dt = t_new[1]-t_new[0]
        s_smoothed = np.interp(t_new,t_old,s)
        # h_smoothed = -np.diff(s_smoothed)/(dt*0.5*(s_smoothed[1:]+s_smoothed[:-1]))
        h_smoothed = -np.diff(s_smoothed)/(dt*s_smoothed[:-1])
        t_h = t_new[:-1] 
        return t_h, h_smoothed
    
    def plotHazard(self,smooth = True, linestyle='-',lw=0.6, ax=None,label='',kwargs={}):
        label = label
        if (smooth):
            t,h = self.hazard
        else:
            t,h = self.hazard_not_smoothed
        if ax is None:
            plt.plot(t,h,linewidth=lw,linestyle=linestyle,label = label)
            plt.legend()
        else:
            ax.plot(t,h,linewidth=lw,linestyle=linestyle,label = label,**kwargs)
            ax.legend()
        return   

#####################################################
class SR_karin_human_GWTW(SR_GWTW):
    def __init__(self,eta=1,beta=1,kappa=1,epsilon=1,xc=1,
                 npeople=10000, nsteps=6000,
                   t_end=120, t_start = 0, tscale = 'years',
                     memory_efficient =False,natural_units=False, smoothing =20):
        super().__init__(eta=0.00135 *365*eta, beta=0.15*365*beta, kappa=0.5*kappa, epsilon=0.142 *365*epsilon, xc=17*xc,
                         npeople=npeople, nsteps=nsteps, t_end=t_end, t_start = t_start, tscale = tscale,
                           memory_efficient = memory_efficient, natural_units=natural_units, smoothing=smoothing)
    ################################################################################

class SRNDK_GWTW(SR_GWTW):
    #Non dimentionalized SR model
    def __init__(self,kappa,t3,D31,D32,Dx,npeople, nsteps, t_end, t_start = 0, tscale = 'years', memory_efficient =False,natural_units=False, smoothing=20):
        self.t3 = t3
        t_end = t_end*t3
        self.D31 = D31
        self.D32 = D32
        self.D21 = D31/D32
        self.Dx =Dx
        xc = Dx*kappa
        beta = D31*kappa/t3
        epsilon = self.D32*(kappa**2)/t3
        eta = D31*kappa/(t3**2)
        super().__init__(eta, beta, kappa, epsilon, xc,npeople, nsteps, t_end, t_start = 0, tscale = tscale, memory_efficient = memory_efficient,natural_units=natural_units, smoothing=smoothing)

class SRND_Peckle_GWTW(SR_GWTW):
    #Non dimentionalized SR_GWTW model Lambda, F, XC (Lambda is Peckles number = F/D)
    def __init__(self,kappa,t3,F,Lambda,Xc,npeople, nsteps, t_end, t_start = 0, tscale = 'years', memory_efficient =False, natural_units=False, smoothing =20):
        self.t3 = t3
        t_end = t_end*t3
        self.F = F
        self.Lambda = Lambda
        self.Xc =Xc
        xc = Xc*kappa
        beta = F*kappa/t3
        epsilon = (F/(Lambda))*(kappa**2)/t3
        eta = F*kappa/(t3**2)
        super().__init__(eta, beta, kappa, epsilon, xc,npeople, nsteps, t_end, t_start = 0, tscale = tscale, memory_efficient = memory_efficient, natural_units=natural_units, smoothing=smoothing)

    def print_peckle_params(self):
        print('Peckle parameters: F = ', self.F, ', Lambda = ', self.Lambda, ', Xc = ', self.Xc, ', t3 = ', self.t3)
        return
    
    def print_relative_peckle_params(self):
        print('Peckle parameters: F = ', self.F/12166.66, ', Lambda = ', self.Lambda/0.528169014084507,
               ', Xc = ', self.Xc/34, ', t3 = ', self.t3/111)
        return


def agingStats(t, vals, val_critical=np.inf, nvalues=40, corr_threshold=0.5,calc_error_bars=False,confidence_level=0.95,n_resample =150,pdf_method='hist'):
    """
    Calculate aging statistics for a given time point.

    Parameters:
    t (array-like): Time points.
    vals (array-like): Values at each time point for each person indexed as [time,person].
    val_critical (float, optional): Critical value. Defaults to np.inf.
    nvalues (int, optional): Number of value bins for distribution . Defaults to 40.

    Returns:
    tuple: A tuple containing mean, standard deviation, coefficient of variation,
           autocorrelations, 
           [distributions,bin vals] where dsitributions is a tupel of disstribution for each time point in t, and bun_vals is the bin values for the distributions   
           and skewness.

    """
    error_bars ={}
    #TEST
    #preprocess values so to not take values after death
    np.set_printoptions(suppress=True)

    for i in range(len(vals[0,:])):
        if np.any(vals[:,i] >= val_critical):
            # if i<10:
                # print(f'vals[{i} pre]:',np.round(vals[:,i],2))
            death_indice = np.argmax(vals[:,i] >= val_critical)
            vals[death_indice:,i] = np.nan
            # if i<10:
                # print(f'vals[{i} mod]:',np.round(vals[:,i],2))
    # Calculate the mean and std of a value at a given time point
    #END
    np.set_printoptions(suppress=False)

    mean = np.nanmean(vals, axis=1)
    std = np.nanstd(vals, axis=1)
    cv = std / mean
        

    if calc_error_bars:
        error_bars['mean'] = bootstrap(data=[vals],statistic= np.nanmean,n_resamples=n_resample,axis=1,confidence_level= confidence_level,method ='percentile')
        error_bars['std'] = bootstrap(data=[vals],statistic= np.nanstd,n_resamples=n_resample,axis=1,confidence_level= confidence_level,method ='percentile')
        fcv= lambda data,axis: np.nanstd(data,axis=axis)/np.nanmean(data,axis=axis)
        error_bars['cv'] = bootstrap(data=[vals],statistic= fcv,n_resamples=n_resample,axis=1,confidence_level= confidence_level,method ='percentile')

    #preprocess values so to not take values after death
    for i in range(len(vals[0,:])):
        if np.any(vals[:,i] >= val_critical):
            death_indice = np.argmax(vals[:,i] >= val_critical)
            vals[death_indice:,i] = np.nan

    # Calculate the autocorrelation
    Autocorrelations = np.corrcoef(np.nan_to_num(vals))
    distributions = []
    skewness = []
    
    if np.any(np.isnan(vals)):
        max_val = np.max(vals[~np.isnan(vals)])
        min_val = np.min(vals[~np.isnan(vals)])
    else:
        max_val = np.max(vals)
        min_val = np.min(vals)
    max_val = np.minimum(max_val, val_critical)
    min_val = np.min(vals[~np.isnan(vals)])
    print(min_val,max_val)
    bin_values = np.linspace(min_val, max_val, nvalues)
    skewness_err = namedtuple('skewness_err', ['confidence_interval'])

    skewness_err.confidence_interval = namedtuple('confidence_interval', ['low', 'high'])
    skewness_err.confidence_interval.low = []
    skewness_err.confidence_interval.high = []

    for i in range(len(t)):
        trag = vals[i, :]
        # print(trag)
        # print(i)
        trag_proccessed_for_NaNs = trag[~np.isnan(trag)]
        if len(trag_proccessed_for_NaNs) >= 2:
            if pdf_method == 'kde':
                kde = gaussian_kde( trag_proccessed_for_NaNs)
                pdf = kde(bin_values)
            elif pdf_method == 'hist': 
                pdf, _ = np.histogram(trag_proccessed_for_NaNs, bins=bin_values, density=True)
                # pdf = pdf / (np.sum(pdf)*(bin_values[1] - bin_values[0]))
            
            distributions.append(pdf)
            skew_bs = bootstrap(data=[trag_proccessed_for_NaNs],statistic= skew,n_resamples=n_resample,confidence_level= confidence_level,method ='percentile')
            skewness.append(skew( trag_proccessed_for_NaNs))
            skewness_err.confidence_interval.low.append(skew_bs.confidence_interval.low)
            skewness_err.confidence_interval.high.append(skew_bs.confidence_interval.high)
           
        # if len(trag[trag<val_critical]) >= 2:
        #     kde = gaussian_kde(trag[trag<val_critical])
        #     pdf = kde(bin_values)
        #     distributions.append(pdf)
        #     # skewness.append(skew( trag_proccessed_for_NaNs))
        #     skewness.append(skew( trag[trag<val_critical]))

        else:
            distributions.append(np.zeros_like(bin_values))
            skewness.append(0)
    error_bars['skewness'] = skewness_err 
    autocorrelation_time = np.empty_like(Autocorrelations[:,0], dtype=float)
    autocorrelation_time[:] = np.nan

    for i in range(len(Autocorrelations)):
        j = np.where(Autocorrelations[i, i+1:] < corr_threshold)[0]
        if len(j) > 0:
            autocorrelation_time[i] = j[0] + 1

    autocorrelation = np.empty_like(Autocorrelations[:-1,0], dtype=float)
    for i in range(len(Autocorrelations)-1):
        autocorrelation[i] = Autocorrelations[i, i+1]

    return mean, std, cv, Autocorrelations, autocorrelation_time, autocorrelation, [distributions, bin_values], skewness,error_bars




def karin_params():
    print('Karin params\n','eta = 0.49275, beta = 54.75, kappa = 0.5, epsilon = 51.83, xc = 17')

def karin_mice_params():
    print('Karin mice params\n',f'eta = {0.084/365}, beta = {0.15}, kappa = 0.5, epsilon = 0.16, xc = 17')

def get_hazard_from_survival(t,survival):
    """
    utility function to calculate the hazard function from the survival function
    """
    #first index where the survival is 0
    ind = np.argmax(survival==0)
    ind = ind if ind>0 and ind<len(survival) else len(survival)-1
    t = t[:ind]
    if len(t)<=1:
        return np.array([0]),np.array([0])
    survival = survival[:ind]
    # mid_survival = (survival[1:]+survival[:-1])/2
    h = -(np.diff(survival)/np.diff(t))/survival[:-1]
    h = np.concatenate((h,[h[-1]]))
    return t,h

class SR_envelope():
    """
    This is the envelope class for the hazards kept from cluster simulations
    I assume that the hazards are with one point per year
    """
    def __init__(self,theta, survival,t =None,npeople =500000, kappa =0.5):
        survival = np.array(survival)
        self.theta = theta
        self.eta = theta[0]
        self.beta = theta[1]
        self.kappa = kappa
        self.epsilon = theta[2]
        self.xc = theta[3]
        self.npeople = npeople
        if t is None:
            t = np.arange(0,len(survival))
        self.t = t
        self.survival = (self.t,survival)
        self.t_end = self.t[-1]
        self.hazard = get_hazard_from_survival(self.t,survival)

    def getSurvival(self):
        return self.survival
    
    def getHazard(self):
        return self.hazard
    
    def smooth_survival(self):
        return self.survival
    
    def plotHazard(self, linestyle='-',lw=0.6, ax=None,label='', smooth =True, scale=1,kwargs={}):
        #plots the hazard.
        #smooth is provition for inhereting classes (meaning less for the regular SR class)
        #scale scales time, since the hazard is given per unit time scaling the t axis by a means we need to scale the hazard by 1/a as well
        label = label
        t,h = self.hazard
        t = t*scale
        h = h/scale 
        if ax is None:
            plt.plot(t,h,linewidth=lw,linestyle=linestyle,label = label,**kwargs)
            plt.legend()
        else:
            ax.plot(t,h,linewidth=lw,linestyle=linestyle,label = label,**kwargs)
            ax.legend()
        return   

    def plotSurvival(self, linestyle='-',lw=0.6, ax=None,label='', scale =1):
        #plots the survival.
        #scale scales time
        label = label
        t,s = self.survival
        t=t*scale
        if ax is None:
            plt.plot(t,s,linewidth=lw,linestyle=linestyle,label = label)
            plt.legend()
        else:
            ax.plot(t,s,linewidth=lw,linestyle=linestyle,label = label)
            ax.legend()
        return  



        
        



