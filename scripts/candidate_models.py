from prep_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
from scipy.stats import pearsonr

'''

 PyMC3 models used for analysis are defined here

'''


def test_model():

    '''
     Simple model with normally distributed random data
     for sanity check
    '''

    #data = np.random.randn(100)
    #np.savetxt('generated_test_data.txt',data,delimiter=',')
    data = np.loadtxt('generated_test_data.txt')

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sd=1, testval=0)
        sd = pm.HalfNormal('sd', sd=1)
        y = pm.Normal('y', mu=mu, sd=sd, observed=data)

    return model

def dd_null_model(**kwargs):

    '''
     Model for IP3 with priors as estimated by the TD model.
        We take TD mean and stdev and use this for priors on group distributions for each condition
    '''

    data = load_dataset(group=kwargs['group'],roi=kwargs['roi'],task=kwargs['task'],
                        condition=kwargs['condition'],path=kwargs['path'])

    subject_ids = data.subject.unique()
    subject_idx = np.array([idx-1 for idx in data.subject.values])
    subject_idx = handle_irregular_idx(subject_idx)
    n_subjects = len(subject_ids)

    with pm.Model() as dd_null_model:


        if kwargs['task'] == '_add':
            # Hyperpriors for group nodes
            mu_a = pm.Normal('mu_alpha',  mu=2.60984070181, sd=0.217110334357)
            mu_b = pm.Normal('mu_beta', mu=0.00740967217064, sd=0.0018325644004)
            
            sigma_a = pm.Uniform('sigma_alpha', 0, 100)
            sigma_b = pm.Uniform('sigma_beta', 0, 100)

            # Parameters for each subject
            BoundedNormalAlpha = pm.Bound(pm.Normal, lower=0, upper=5)
            BoundedNormalBeta = pm.Bound(pm.Normal, lower=0, upper=1)

            a = BoundedNormalAlpha('alpha', mu=mu_a, sd=sigma_a, shape=n_subjects)
            b = BoundedNormalBeta('beta', mu=mu_b, sd=sigma_b, shape=n_subjects)

            # Model error
            eps = pm.Uniform('eps', 0, 100)

        elif kwargs['task'] == '_sub':
            # Hyperpriors for group nodes
            mu_a = pm.Normal('mu_alpha', mu=2.78420270477, sd=0.315299147824)
            mu_b = pm.Normal('mu_beta', mu=0.012556158372, sd=0.00217034614933)
            
            sigma_a = pm.Uniform('sigma_alpha', 0, 100)
            sigma_b = pm.Uniform('sigma_beta', 0, 100)

            # Parameters for each subject
            BoundedNormalAlpha = pm.Bound(pm.Normal, lower=0, upper=5)
            BoundedNormalBeta = pm.Bound(pm.Normal, lower=0, upper=1)

            a = BoundedNormalAlpha('alpha', mu=mu_a, sd=sigma_a, shape=n_subjects)
            b = BoundedNormalBeta('beta', mu=mu_b, sd=sigma_b, shape=n_subjects)

            # Model error
            eps = pm.Uniform('eps', 0, 100)

        # Model prediction of voxel activation
        x_est = data.thresh_x.values*a[subject_idx] - data.n_activity.values*b[subject_idx] + data.noise.values

        # Data likelihood
        y = pm.Normal('y', mu=x_est, sd=eps, observed=data.raw_x.values)

    return dd_null_model

def hierarchical_model(**kwargs):

    '''
     Hierarchical model for roi
    '''

    data = load_dataset(group=kwargs['group'],roi=kwargs['roi'],task=kwargs['task'],
                        condition=kwargs['condition'],path=kwargs['path'])

    subject_ids = data.subject.unique()
    subject_idx = np.array([idx-1 for idx in data.subject.values])
    subject_idx = handle_irregular_idx(subject_idx)
    n_subjects = len(subject_ids)

    with pm.Model() as hierarchical_model:

        # Hyperpriors for group nodes
        mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
        mu_b = pm.Normal('mu_beta', mu=0., sd=1)
        
        #sigma_a = pm.HalfCauchy('sigma_alpha', 1)
        #sigma_b = pm.HalfCauchy('sigma_b', 1)
        
        sigma_a = pm.Uniform('sigma_alpha', 0, 100)
        sigma_b = pm.Uniform('sigma_beta', 0, 100)

        # Parameters for each subject
        BoundedNormalAlpha = pm.Bound(pm.Normal, lower=0, upper=5)
        BoundedNormalBeta = pm.Bound(pm.Normal, lower=0, upper=1)

        a = BoundedNormalAlpha('alpha', mu=mu_a, sd=sigma_a, shape=n_subjects)
        b = BoundedNormalBeta('beta', mu=mu_b, sd=sigma_b, shape=n_subjects)

        # Model error
        #eps = pm.HalfCauchy('eps', 1)
        
        eps = pm.Uniform('eps', 0, 100)

        # Model prediction of voxel activation
        x_est = data.thresh_x.values*a[subject_idx] - data.n_activity.values*b[subject_idx] + data.noise.values

        # Data likelihood
        y = pm.Normal('y', mu=x_est, sd=eps, observed=data.raw_x.values)

    return hierarchical_model

def unpooled_model(**kwargs):

    '''
     Unpooled model for roi
    '''

    data = load_dataset(group=kwargs['group'],roi=kwargs['roi'],task=kwargs['task'],
                        condition=kwargs['condition'],path=kwargs['path'])

    subject_ids = data.subject.unique()
    subject_idx = np.array([idx-1 for idx in data.subject.values])
    subject_idx = handle_irregular_idx(subject_idx)
    n_subjects = len(subject_ids)

    with pm.Model() as unpooled_model:

        # Independent parameters for each subject
        BoundedNormalAlpha = pm.Bound(pm.Normal, lower=0, upper=5)
        BoundedNormalBeta = pm.Bound(pm.Normal, lower=0, upper=1)

        a = BoundedNormalAlpha('alpha', mu=0, sd=1, shape=n_subjects)
        b = BoundedNormalBeta('beta', mu=0, sd=1, shape=n_subjects)

        # Model error
        #eps = pm.HalfCauchy('eps', 1)
        eps = pm.Uniform('eps', 0, 100)

        # Model prediction of voxel activation
        x_est = data.thresh_x.values*a[subject_idx] - data.n_activity.values*b[subject_idx] + data.noise.values

        # Data likelihood
        y = pm.Normal('y', mu=x_est, sd=eps, observed=data.raw_x.values)

    return unpooled_model

def handle_irregular_idx(subject_idx):
    new_subject_idx = [0] * len(subject_idx)
    curr_idx = 0
    for sidx in range(len(subject_idx)):
        if sidx != 0:
            if subject_idx[sidx] == subject_idx[sidx-1]:
                new_subject_idx[sidx] = curr_idx            
            else:
                curr_idx+=1
                new_subject_idx[sidx] = curr_idx
        else:
            new_subject_idx[sidx] = curr_idx

    subject_idx = new_subject_idx
    return subject_idx
    
