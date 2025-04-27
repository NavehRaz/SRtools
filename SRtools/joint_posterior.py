import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from . import samples_utils as su

    
    
class JointPosterior(su.Posterior):    
    def __init__(self,samples_list,lnprobs_list,bins,log=False,progress_bar=True):
        """
        Returns a posterior object of the joint distribution of many samples lists.
        """
        #if bins is an integer or a list of integers, it means we need to build the bins for each feature so all samples sets are on the same grid.
        #this we do by building the grid on the widest range of the samples along each feature.
        
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
        
        posts =[]
        raw_unique_samples = defaultdict(lambda: [np.zeros(n_features), 0, 0])

        for i in range(len(samples_list)):
            post = su.Posterior(samples_list[i],lnprobs_list[i],bins,log=log,progress_bar=progress_bar)
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
        posterior_lnprobs =lnprobs_density+ np.log(volumes)-evidence
        
        self.bins = bins
        self.log = log
        self.progress_bar = progress_bar
        self.unique_samples = unique_samples
        self.lnprobs_density = lnprobs_density
        self.posterior = posterior_lnprobs
        self.dthetas = dthetas
        self.evidence = evidence
        self.df = None



    #raise error if self.samples or self.lnprobs are called:
    @property
    def samples(self):
        raise AttributeError("JointPosterior object has no attribute 'samples'")
    
    @property
    def lnprobs(self):
        raise AttributeError("JointPosterior object has no attribute 'lnprobs'")
    
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
            transforms = [identity_transform,default_transform1,default_transform2,default_transform3,default_transform4,default_transform5]
            if labels is None:
                labels = [["xc/eta","beta/eta","xc^2/epsilon","xc"],["eta","beta","epsilon","xc"],
                          ["sqrt(xc/eta)","s= eta^0.5*xc^1.5/epsilon","beta*xc/epsilon","xc"],
                          ["eta*xc/epsilon","Fx=beta^2/eta*xc","Dx =beta*epsilon/eta*xc^2","xc"],
                          ["Pk=beta*k/epsilon","Fk=beta^2/eta*k","beta/eta","xc"],
                          ["Dk =beta*epsilon/eta*k^2","Fk^2/Dk=beta^3/eta*epsilon","beta/eta","xc"]]
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
            transformed_samples_list = [np.apply_along_axis(trans, 1, samples) for samples in self.samples_list]
            post = JointPosterior(transformed_samples_list.copy(), self.lnprobs_list.copy(), self.raw_bins, log=self.log, progress_bar=self.progress_bar)
            max_liklihood = [transformed_samples_list[i][np.argmax(self.lnprobs_list[i])] for i in range(len(transformed_samples_list))]
            max_likelihood_overall_index = np.argmax([max(self.lnprobs_list[i]) for i in range(len(self.lnprobs_list))])
            max_liklihood = max_liklihood[max_likelihood_overall_index]
            for i in range(n_features):
                #check if the label is already calculated
                if label_set[i] in summery_dict.keys():
                    continue
            
                marginalized_samples, marginalized_lnprobs, marginalized_dthetas = post.marginalize_posterior( [j for j in range(n_features) if j != i], density = True)
                stats_dict = su.getStats(marginalized_samples, marginalized_lnprobs, marginalized_dthetas, stats=stats, percentiles=percentiles,smooth_mode=smooth_mode)
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
                    if isinstance(stats_dict[key],list) or isinstance(stats_dict[key],tuple):
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
        new_samples =[]
        for j in range(len(self.samples_list)):
            new_samples.append( [self.samples_list[j][:,i]*rescale[i] for i in range(self.n_features)])
            new_samples[j] = np.array(new_samples[j]).T
        self.__init__(new_samples, self.lnprobs_list, self.raw_bins, log=self.log, progress_bar=self.progress_bar)
        return self
    

        

        





    

            


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



def round_value(value, precision=2):
    if value == 0:
        return 0
    elif abs(value) < 1:
        r = int(np.abs(np.log10(abs(value))))
        return round(value, r+precision)
    else:
        return round(value, precision)