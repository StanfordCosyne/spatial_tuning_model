from prep_data import *
from candidate_models import *
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import itertools
from scipy.stats import pearsonr
from joblib import Parallel, delayed

'''
 Runs models in model_names in parallel
'''



def run_models(args):

    rois = args['rois']
    rois = [roi.replace('.nii','') for roi in rois]
    groups = args['subjectlists']
    groups = [group.replace('.txt','').split('/')[-1] for group in groups]
    tasks = args['tasks']
    tasks = ['_'+task for task in tasks]
    conditions = args['conditions']
    model_names = args['model_names']
    model_path = args['model_path']
    n_iters = args['n_iters']
    n_chains = args['n_chains']

    models, dirs = get_models(rois=rois,groups=groups,
                              tasks=tasks,conditions=conditions,
                              model_names=model_names,path=model_path)

    inputs = [(m,d,n_iters,n_chains) for m,d in zip(models,dirs)]

    print('Running models in parallel.')

    n_cores = 12

    Parallel(n_jobs=n_cores)(delayed(runner)(i) for i in inputs)



# define inputs and run in parallel
def runner(vals):
    model = vals[0]
    dir = vals[1]
    n_iters = vals[2]
    n_chains = vals[3]

    print('Running model %s with %s chains and %s iters'%(dir,n_chains,n_iters))

    with model:
        db = pm.backends.Text(dir)
        trace = pm.sample(draws=n_iters, chain=0, njobs=n_chains, trace=db)



def get_models(**kwargs):
    '''
     Get list of models and directories for specified arguments
    '''
    
    print('Creating model objects for all models...')

    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']
    conditions = kwargs['conditions']
    path = kwargs['path']

    if conditions == []:

        # get all models and dirs
        model_args = [dict(zip(('group','task','roi'), (i,j,k))) \
                      for i,j,k in itertools.product(groups, tasks, rois)]
        dir_endings = [i+j+'_'+k for i,j,k in itertools.product(groups, tasks, rois)]

        models, dirs = [], []
        for model in model_names:
            if model == 'unpooled':
                curr_mods = [unpooled_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition='',path=path) for a in model_args]
            elif model == 'hierarchical':
                curr_mods = [hierarchical_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition='',path=path) for a in model_args]
            elif model == 'dd_null':
                curr_mods = [dd_null_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition='',path=path) for a in model_args]
            models.append(curr_mods)
            dirs.append([m+d for m,d in zip([path+'/model_fits/'+model+'/']*len(curr_mods), dir_endings)])
        models = flatten(models)
        dirs = flatten(dirs)

    else:

        # get all models and dirs
        model_args = [dict(zip(('group','task','roi','condition'), (i,j,k,c))) \
                      for i,j,k,c in itertools.product(groups, tasks, rois, conditions)]
        dir_endings = [i+j+'_'+c+'_'+k for i,j,k,c in itertools.product(groups, tasks, rois, conditions)]

        models, dirs = [], []
        for model in model_names:
            if model == 'unpooled':
                curr_mods = [unpooled_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition=a['condition'],path=path) for a in model_args]
            elif model == 'hierarchical':
                curr_mods = [hierarchical_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition=a['condition'],path=path) for a in model_args]
            elif model == 'dd_null':
                curr_mods = [dd_null_model(group=a['group'],task=a['task'],
                             roi=a['roi'],condition=a['condition'],path=path) for a in model_args]
            models.append(curr_mods)
            dirs.append([m+d for m,d in zip([path+'model_fits/'+model+'/']*len(curr_mods), dir_endings)])
        models = flatten(models)
        dirs = flatten(dirs)

    return models, dirs


