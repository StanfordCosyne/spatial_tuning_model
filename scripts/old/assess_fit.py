from candidate_models import *
import os
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import itertools
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def main():
    '''
    This script contains functions to compute fit statistics
    '''
    # Which rois, groups, and tasks to run models for
    #rois = ['09-6mm_right_hIP3_26_-74_54',
     #       '01-6mm_left_Planum_Temporale_-60_-34_14',
     #       '10-6mm_right_SPL_8_70_54',
     #       'Bilateral_V1']
    #rois = ['09-6mm_right_hIP3_26_-74_54']
    rois = ['03-6mm_right_SPL_Knops_saccades_24_-70_61']
    groups = ['td','md']#['td','md']
    tasks = ['_add','_sub']
    conditions = ['comp']
    model_names = ['hierarchical']


    compute_ppc(rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)
    plot_ppc(pd.read_csv('../results/SPL_Knops_ppc_summary_conditions.csv'))
    #model_fit_complete = compute_fit(rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)
    #plot_trace(rois=rois,groups=groups,tasks=tasks,model_names=model_names)
    #plot_fit_vals(pd.read_csv('model_fit_condition_complete_summary.csv'))
    #assess_prediction(pd.read_csv('model_fit_complete_summary.csv'))
    #assess_loo_predicted(data=pd.read_csv('loo_condition_predicted.csv'),rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)
    #compare_vars(rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)

def compare_vars(**kwargs):
    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']
    conditions = kwargs['conditions']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,conditions=conditions,model_names=model_names)

    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]

        model_name = [m for m in model_names if m in dir][0]
        roi = [r for r in rois if r in dir][0]
        group = [g for g in groups if g in dir][0]
        task = [t for t in tasks if t in dir][0]
        condition = [c for c in conditions if c in dir][0]

        data = load_dataset(group=group,roi=roi,task=task,condition=condition)

        print(roi, group, task, condition)

        plt.scatter(data.n_activity,np.sqrt(data.raw_x))
        plt.show()


def assess_loo_predicted(**kwargs):
    data = kwargs['data']
 
    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']
    conditions = kwargs['conditions']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,conditions=conditions,model_names=model_names)

    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]

        model_name = [m for m in model_names if m in dir][0]
        roi = [r for r in rois if r in dir][0]
        group = [g for g in groups if g in dir][0]
        task = [t for t in tasks if t in dir][0]
        condition = [c for c in conditions if c in dir][0]

        print('Assessing loo prediction for: \nroi: %s \ngroup: %s \ntask: %s \ncondition: %s'%(roi,group,task,condition))

        real_data = load_dataset(group=group,roi=roi,task=task,condition=condition)
        pred_data = data[(data.task == task) & (data.roi == roi) & (data.condition == condition) & (data.group == group)]

        #print(pearsonr(real_data.raw_x,pred_data.loo_i))

        print(np.median(real_data.raw_x))
        print(np.median(pred_data.loo_i))

        bins=np.arange(-4,4,0.1)
        sns.distplot(real_data.raw_x,kde_kws={'linewidth':0},color='#929591',label='Actual',bins=bins)
        sns.kdeplot(pred_data.loo_i,shade=False,color='#8c000f',label='Mean PPC Estimate',linewidth=2)
        plt.show()


def plot_ppc(data):

    #plot_order = ['09-6mm_right_hIP3_26_-74_54',
    #              '10-6mm_right_SPL_8_70_54',
    #              '01-6mm_left_Planum_Temporale_-60_-34_14',
    #              'Bilateral_V1']
    plot_order = ['03-6mm_right_SPL_Knops_saccades_24_-70_61']

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1,2, sharey=True)
    for i in range(1):#(2):
        if i == 0:
            group = 'md'
        else:
            group = 'td'

        hier_data = data[data.model == 'dd_null']#'hierarchical']
        plot_data = hier_data[(hier_data.group == group) & (hier_data.condition == 'comp')]# * (hier_data.task == '_add')]

        rois = plot_data.roi.unique()
        flatten = lambda l : [i for j in l for i in j]
        mean_ppc_sub = pd.Series(flatten([[plot_data[plot_data.roi == roi].ppc_mean.mean()] * len(plot_data[plot_data.roi == roi]) for roi in rois]),
                                 index=plot_data.index)


        red = sns.color_palette("Reds")[-2]
        blue = sns.color_palette("Blues")[-2]

        #MEAN OF EACH OF 500 PPC ESTIMATES

        #sns.violinplot(x='ppc_mean',y='roi',hue='task',data=plot_data, order=plot_order,
        #               palette=[blue,red],ax=axes[i],split=True,linewidth=0.2,saturation=1, inner=None)#0.2

        real_means = plot_data.groupby(['condition','model','task','group','roi'], as_index=False)['real_mean'].mean()

        cond = 'comp'
        model = 'hierarchical'
        for roi in plot_data.roi.unique():
            for task in plot_data.task.unique():
                real_mean = real_means[(real_means.condition == cond) & (real_means.model == model) & (real_means.task == task) & (real_means.group == group) & (real_means.roi == roi)]
                pred_mean = np.mean(plot_data[(plot_data.condition == cond) & (plot_data.model == model) & (plot_data.task == task) & (plot_data.group == group) & (plot_data.roi == roi)].ppc_mean)
                std = np.std(plot_data[(plot_data.condition == cond) & (plot_data.model == model) & (plot_data.task == task) & (plot_data.group == group) & (plot_data.roi == roi)].ppc_mean)

                if np.float(real_mean.real_mean) < pred_mean-std:
                    print('real mean > 1 std of ppc_means for %s %s %s'%(roi, task, group))
                elif np.float(real_mean.real_mean) > pred_mean+std:
                    print('real mean > 1 std of ppc_means for %s %s %s'%(roi, task, group))
      

        #sns.stripplot(x='real_mean',y='roi',hue='task',data=real_means,size=7, order=plot_order,
        #              palette=[blue,red],ax=axes[i],linewidth=0.5)

        #PEARSON R FOR 500 PPC ESTIMATES
        sns.violinplot(x='pearsonr',y='roi',hue='task',data=plot_data, order=plot_order,
                       palette=[blue,red],ax=axes[i],split=True,linewidth=0.2,saturation=0.8)

    titles = ['DD','TD']
    axcount = 0
    for ax in axes:
        ax.set(xlim=[0,1]) #0,0.8
        ax.set_yticklabels(['IP3'])#,'SPL','PT','V1'])
        ax.set_title(titles[axcount])
        axcount+=1
    sns.despine(fig=fig, left=True)

    #plt.savefig('../results/ppc_pearsonr.svg')
    plt.show()

def plot_fit_vals(stats):

    for s in ['loo','waic','dic','bpic']:
        sns.set_style("white")
        fig, axes = plt.subplots(1, 4, sharey=True)

        plot_order = ['09-6mm_right_hIP3_26_-74_54',
                    '10-6mm_right_SPL_8_70_54',
                    '01-6mm_left_Planum_Temporale_-60_-34_14',
                    'Bilateral_V1']

        axes[0].plot(stats[(stats.group == 'td') & (stats.model == 'hierarchical') & (stats.task == '_add') & (stats.condition == 'comp')][s],marker="o",label='hierarchical')
        axes[1].plot(stats[(stats.group == 'td') & (stats.model == 'hierarchical') & (stats.task == '_sub') & (stats.condition == 'comp')][s],marker="o",label='hierarchical')
        axes[2].plot(stats[(stats.group == 'md') & (stats.model == 'hierarchical') & (stats.task == '_add') & (stats.condition == 'comp')][s],marker="o",label='hierarchical')
        axes[3].plot(stats[(stats.group == 'md') & (stats.model == 'hierarchical') & (stats.task == '_sub') & (stats.condition == 'comp')][s],marker="o",label='hierarchical')

        axes[0].set_ylabel(s)
        titles = ['TD Add','TD Sub','DD Add','DD Sub']
        for ax in range(len(axes)):
            axes[ax].set_xticklabels(['IP3','PT','SPL','V1'])
            axes[ax].set_title(titles[ax])
    
        legend = axes[0].legend()
        plt.show()


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


def plot_trace(**kwargs):
    '''
     Plots trace for specified arguments
    '''
    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,model_names=model_names)

    output = {'model':[],'roi':[],'group':[],'task':[],'waic':[],'bpic':[],'dic':[],'max_rhat':[],'min_rhat':[]}
    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]
        with model:
            print(dir)
            trace = pm.backends.text.load(dir)

            pm.traceplot(trace);
            plt.show()

def compute_fit(**kwargs):
    '''
     Saves and returns a dataframe of fit statistics for specified arguments
    '''
    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']
    conditions = kwargs['conditions']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,conditions=conditions,model_names=model_names)

    output = {'model':[],'roi':[],'group':[],'task':[],'condition':[],'loo':[],'waic':[],'bpic':[],'dic':[],'max_rhat':[],'min_rhat':[]}
    loo_is = {'model':[],'roi':[],'group':[],'task':[],'condition':[],'loo_i':[]}
    for m in range(len(models)):
        model = models[m]
        dir = dirs[m]
        with model:
            trace = pm.backends.text.load(dir)

            model_name = [m for m in model_names if m in dir][0]
            roi = [r for r in rois if r in dir][0]
            group = [g for g in groups if g in dir][0]
            task = [t for t in tasks if t in dir][0]
            condition = [c for c in conditions if c in dir][0]

            print('Computing fit stats for %s model: roi %s group %s task %s condition %s'%(model_name, roi, group, task, condition))

            loo_full = pm.stats.loo(trace,pointwise=True)
            loo = loo_full[0]
            loo_i = loo_full[3]
            waic = pm.stats.waic(trace)[0]
            bpic = pm.stats.bpic(trace)
            dic = pm.stats.dic(trace)
            rhats = pm.diagnostics.gelman_rubin(trace)
            rhat_vals = np.hstack(list(rhats.values()))
            max_rhat = max(rhat_vals)
            min_rhat = min(rhat_vals)

            output['model'].append(model_name)
            output['roi'].append(roi)
            output['group'].append(group)
            output['task'].append(task)
            output['condition'].append(condition)
            output['loo'].append(loo)
            output['waic'].append(waic)
            output['bpic'].append(bpic)
            output['dic'].append(dic)
            output['max_rhat'].append(max_rhat)
            output['min_rhat'].append(min_rhat)

            for i in loo_i:
                loo_is['model'].append(model_name)
                loo_is['roi'].append(roi)
                loo_is['group'].append(group)
                loo_is['task'].append(task)
                loo_is['condition'].append(condition)
                loo_is['loo_i'].append(i)

    loo_i_output = pd.DataFrame.from_dict(loo_is)
    loo_i_output.to_csv('../results/loo_condition_unpooled_predicted.csv')

    output = pd.DataFrame.from_dict(output)
    output.to_csv('../results/model_fit_condition_complete_unpooled_summary.csv')

    return output   


def compute_ppc(**kwargs):
    rois = kwargs['rois']
    groups = kwargs['groups']
    tasks = kwargs['tasks']
    model_names = kwargs['model_names']
    conditions = kwargs['conditions']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,conditions=conditions,model_names=model_names)
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

        data = load_dataset(group=group,roi=roi,task=task,condition=condition)

        with model:
            trace = pm.backends.text.load(dir)
            ppc = pm.sample_ppc(trace, samples=n_datasets)
            
            print(model_name,group,roi,task,condition)    
    
            r_vals = [pearsonr(data.raw_x, y)[0] for y in ppc['y']]

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
    output.to_csv('../results/SPL_Knops_ppc_summary_conditions.csv')

    full_output = pd.DataFrame.from_dict(full_output)
    full_output.to_csv('../results/SPL_Knops_ppc_avgprediction_conditions.csv')


if __name__ == "__main__":
    main()


