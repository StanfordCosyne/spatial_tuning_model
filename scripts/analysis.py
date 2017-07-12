from run_models import get_models
from prep_data import load_dataset
from candidate_models import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import itertools
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, ttest_ind

def load_ei(args):
    '''
     Loads parameters of interest into dataframe for further analysis
    '''

    rois = args['rois']
    rois = [roi.replace('.nii','') for roi in rois]
    groups = args['subjectlists']
    groups = [group.replace('.txt','').split('/')[-1] for group in groups]
    tasks = args['tasks']
    tasks = ['_'+task for task in tasks]
    conditions = args['conditions']
    model_names = args['model_names']
    model_path = args['model_path']

    models, dirs = get_models(rois=rois,groups=groups,
                              tasks=tasks,conditions=conditions,
                              model_names=model_names,path=model_path)

    output = {'model':[],'roi':[],'group':[],'task':[],'condition':[],'alpha':[],'beta':[],'ei':[]}
    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]

        model_name = [m for m in model_names if m in dir][0]
        roi = [r for r in rois if r in dir][0]
        group = [g for g in groups if g in dir][0]
        task = [t for t in tasks if t in dir][0]
        condition = [c for c in conditions if c in dir][0]

        with model:
            trace = pm.backends.text.load(dir)
            alpha = trace['mu_alpha']
            beta = trace['mu_beta']
            ei = np.divide(beta,alpha)
            for i in range(len(ei)):
                output['model'].append(model_name)
                output['roi'].append(roi)
                output['group'].append(group)
                output['task'].append(task)
                output['condition'].append(condition)
                output['alpha'].append(alpha[i])
                output['beta'].append(beta[i])
                output['ei'].append(ei[i])

    output = pd.DataFrame.from_dict(output)
    output.to_csv(model_path+'/results/ei_mu_distributions.csv')

    return output


def compute_ppc(args):

    rois = args['rois']
    rois = [roi.replace('.nii','') for roi in rois]
    groups = args['subjectlists']
    groups = [group.replace('.txt','').split('/')[-1] for group in groups]
    tasks = args['tasks']
    tasks = ['_'+task for task in tasks]
    conditions = args['conditions']
    model_names = args['model_names']
    model_path = args['model_path']

    models, dirs = get_models(rois=rois,groups=groups,
                              tasks=tasks,conditions=conditions,
                              model_names=model_names,path=model_path)

    n_datasets = 500
    output = {'model':[],'roi':[],'group':[],'task':[],'condition':[],
              'ppc_mean':[],'ppc_stdev':[],'real_mean':[],'real_stdev':[],'pearsonr':[]}
    full_output = {'model':[],'roi':[],'group':[],'task':[],'condition':[],
                   'predicted_data':[],'real_data':[]}
    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]

        model_name = [m for m in model_names if m in dir][0]
        roi = [r for r in rois if r in dir][0]
        group = [g for g in groups if g in dir][0]
        task = [t for t in tasks if t in dir][0]
        condition = [c for c in conditions if c in dir][0]

        data = load_dataset(group=group,roi=roi,task=task,condition=condition,path=model_path)

        with model:
            trace = pm.backends.text.load(dir)
            ppc = pm.sample_ppc(trace, samples=n_datasets)
    
            #r_vals = [pearsonr(data.raw_x, y)[0] for y in ppc['y']]

            for y in ppc['y']:
                r = pearsonr(data.raw_x,y)[0]
                ppc_mean = y.mean()
                ppc_stdev = np.std(y)
                real_mean = data.raw_x.mean()
                real_stdev = np.std(data.raw_x)

                output['model'].append(model_name)
                output['roi'].append(roi)
                output['group'].append(group)
                output['task'].append(task)
                output['condition'].append(condition)
                output['ppc_mean'].append(ppc_mean)
                output['ppc_stdev'].append(ppc_stdev)
                output['real_mean'].append(real_mean)
                output['real_stdev'].append(real_stdev)
                output['pearsonr'].append(r)

            predicted_data = np.mean(ppc['y'],axis=0)
            real_data = data.raw_x
            for d in range(len(predicted_data)):
                full_output['model'].append(model_name)
                full_output['roi'].append(roi)
                full_output['group'].append(group)
                full_output['task'].append(task)
                full_output['condition'].append(condition)
                full_output['predicted_data'].append(predicted_data[d])
                full_output['real_data'].append(real_data[d])


            y_means = np.zeros(len(data.raw_x))
            for k in range(n_datasets):            
                for y_ind in range(len(ppc['y'][k])):
                    y_means[y_ind] = y_means[y_ind]+ppc['y'][k][y_ind]

            for y_ind in range(len(y_means)):
                y_means[y_ind] = y_means[y_ind]/n_datasets


    output = pd.DataFrame.from_dict(output)
    output.to_csv(model_path+'/results/ppc_summary_conditions.csv')

    full_output = pd.DataFrame.from_dict(full_output)
    full_output.to_csv(model_path+'/results/ppc_avgprediction_conditions.csv')
