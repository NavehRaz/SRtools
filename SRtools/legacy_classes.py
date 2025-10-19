"""
Legacy SR model classes and functions.

This module contains specialized SR subclasses and trajectory-related functions
that have been moved from SRmodellib.py for better code organization.
These classes are preserved for backward compatibility but are not actively
used in the main LifeTables01 workflow.

Classes moved here:
- SR_karin_human: Human-specific SR parameters
- SR_GWTW: "Go With The Winner" algorithm variant
- SRNDK, SRND_Peckle: Non-dimensionalized variants
- SR_envelope: Envelope class for cluster simulations
- Various GWTW variants of the above

Functions moved here:
- trajectories_accelerator: Numba-accelerated trajectory generation
"""

import numpy as np
from numba import jit
from . import SRmodellib as sr

# Numba configuration
jit_nopython = True
jit_parallel = True

# Import the base SR class from the main module
SR = sr.SR

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
            import matplotlib.pyplot as plt
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
        self.hazard = self.get_hazard_from_survival(self.t,survival)

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
            import matplotlib.pyplot as plt
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
            import matplotlib.pyplot as plt
            plt.plot(t,s,linewidth=lw,linestyle=linestyle,label = label)
            plt.legend()
        else:
            ax.plot(t,s,linewidth=lw,linestyle=linestyle,label = label)
            ax.legend()
        return  

    def get_hazard_from_survival(self,t,survival):
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
