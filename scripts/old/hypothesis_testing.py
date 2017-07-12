from assess_fit import get_models
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

def main():
    '''
    This script performs hypothesis testing on variables on interest
    '''
    # Which rois, groups, and tasks to run models for
    #rois = ['09-6mm_right_hIP3_26_-74_54',
    #        '01-6mm_left_Planum_Temporale_-60_-34_14',
    #        '10-6mm_right_SPL_8_70_54',
    #        'Bilateral_V1']

    #rois = ['03-6mm_right_hIP3_44_-58_52',#]
    #        '01-6mm_left_Planum_Temporale_-62_-34_14',
    #        '02-6mm_right_Inferior_Parietal_lobe_44_-60_56',
    #        'Bilateral_V1',
    #        '04-6mm_right_FG_38_-64_-20']

    #rois = ['R_RSA_IPS_CytohIP3',
    #        'R_RSA_IPS_CytoSPL_7A',
    #        'R_RSA_IPS__R_RSA_SPL',
    #        'R_RSA_IPS_R_RSA_SPL_CytoIP3']

    #rois = ['01-6mm_right_SPL_Knops_enumeration_27_-58_49',
    #        '02-6mm_analog_right_SPL_Knops_vSTM_15_-64_55',
    #        '03-6mm_right_SPL_Knops_saccades_24_-70_61']

    #rois = ['01-6mm_right_SPL_Knops_enumeration_27_-58_49',
    #        '02-6mm_analog_right_SPL_Knops_vSTM_15_-64_55',
    #        '03-6mm_right_SPL_Knops_saccades_24_-70_61',
    #        '01-6mm_right_IPS_Cantlon_33_-34_49',
    #        '02-6mm_left_IPS_Cantlon_-24_-55_49',
    #        'R_RSA_IPS_CytohIP3',
    #        'R_RSA_IPS_CytoSPL_7A']
    '''
    rois = ['COMBINED_R_Knops_Enum_first_FIND_peak',
            'COMBINED_R_Knops_Enum_second_FIND_peak',
            'COMBINED_R_Knops_saccades_with6',
            'COMBINED_R_Knops_saccades_with4',
            'COMBINED_L_Knops_vSTM1',
            '01-6mm_right_hIP3_FIND_GLM_peak_34_-58_46',
            '02-6mm_right_hIP3_FIND_GLM_subpeak1_34_-64_52',
            '03-6mm_right_hIP3_FIND_GLM_subpeak2_44_-54_46',
            '04-6mm_left_hIP3_FIND_GLM_peak_-28_-60_44']
    '''
    rois = ['02-6mm_right_hIP3_FIND_GLM_subpeak1_34_-64_52']
    groups = ['td','md']
    tasks = ['_add','_sub']
    conditions = ['comp']
    model_names = ['hierarchical']

    ei_df = load_ei(rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)
    #ei_df = pd.read_csv('../results/final_group_hierarchical_condition_ei_mu_distributions.csv')


    for roi in rois:
        print(roi)
        td_add_comp = ei_df[(ei_df.group == 'td') & (ei_df.task == '_add') & (ei_df.roi == roi) & (ei_df.condition == 'comp')]
        td_sub_comp = ei_df[(ei_df.group == 'td') & (ei_df.task == '_sub') & (ei_df.roi == roi) & (ei_df.condition == 'comp')]
        md_add_comp = ei_df[(ei_df.group == 'md') & (ei_df.task == '_add') & (ei_df.roi == roi) & (ei_df.condition == 'comp')]
        md_sub_comp = ei_df[(ei_df.group == 'md') & (ei_df.task == '_sub') & (ei_df.roi == roi) & (ei_df.condition == 'comp')]

        # Bayesian hypothesis testing
        print('TD P(sub > add) for comp beta: %s'%((np.array(td_sub_comp.beta) > np.array(td_add_comp.beta)).mean()))
        print('DD P(sub > add) for comp beta: %s'%((np.array(md_sub_comp.beta) > np.array(md_add_comp.beta)).mean()))
        print('TD P(sub > add) for comp alpha: %s'%((np.array(td_sub_comp.alpha) > np.array(td_add_comp.alpha)).mean()))
        print('DD P(sub > add) for comp alpha: %s'%((np.array(md_sub_comp.alpha) > np.array(md_add_comp.alpha)).mean()))
        print('TD P(sub > add) for comp ei: %s'%((np.array(td_sub_comp.ei) > np.array(td_add_comp.ei)).mean()))
        print('DD P(sub > add) for comp ei: %s'%((np.array(md_sub_comp.ei) > np.array(md_add_comp.ei)).mean()))

        print('add P(TD > DD) for comp beta: %s'%((np.array(td_add_comp.beta) > np.array(md_add_comp.beta)).mean()))
        print('add P(TD > DD) for comp alpha: %s'%((np.array(td_add_comp.alpha) > np.array(md_add_comp.alpha)).mean()))
        print('add P(TD > DD) for comp ei: %s'%((np.array(td_add_comp.ei) > np.array(md_add_comp.ei)).mean()))
        print('sub P(TD > DD) for comp beta: %s'%((np.array(td_sub_comp.beta) > np.array(md_sub_comp.beta)).mean()))
        print('sub P(TD > DD) for comp alpha: %s'%((np.array(td_sub_comp.alpha) > np.array(md_sub_comp.alpha)).mean()))
        print('sub P(TD > DD) for comp ei: %s'%((np.array(td_sub_comp.ei) > np.array(md_sub_comp.ei)).mean()))



        '''
        plot_2dist(md_sub_alpha=md_sub_comp.alpha,
                   md_add_alpha=md_add_comp.alpha,
                   td_sub_alpha=td_sub_comp.alpha,
                   td_add_alpha=td_add_comp.alpha,
                   md_sub_beta=md_sub_comp.beta,
                   md_add_beta=md_add_comp.beta,
                   td_sub_beta=td_sub_comp.beta,
                   td_add_beta=td_add_comp.beta,
                   roi=roi)
        '''

def plot_2dist(**kwargs):

    sns.set_style('white')

    fig, axes = plt.subplots(1,2, sharey=True)

    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14

    sns.kdeplot(kwargs['md_sub_alpha'],kwargs['md_sub_beta'],
                cmap="Reds", shade=True, shade_lowest=False, ax=axes[0])
    sns.kdeplot(kwargs['md_add_alpha'],kwargs['md_add_beta'],
                cmap="Blues", shade=True, shade_lowest=False, ax=axes[0])
    axes[0].set_title('DD',fontsize=20)

    sns.kdeplot(kwargs['td_sub_alpha'],kwargs['td_sub_beta'],
                cmap="Reds", shade=True, shade_lowest=False, ax=axes[1])
    sns.kdeplot(kwargs['td_add_alpha'],kwargs['td_add_beta'],
                cmap="Blues", shade=True, shade_lowest=False, ax=axes[1])
    axes[1].set_title('TD',fontsize=20)

    for ax in axes:
        ax.set(xlim=[1,4],ylim=[0,0.02])
        ax.set_ylabel('β',fontsize=20)
        ax.set_xlabel('α',fontsize=20)

    #plt.savefig('../figures/kde_%s_comp.svg'%(kwargs['roi']))
    #plt.savefig('../figures/kde_%s_comp.svg'%(kwargs['roi']))
    plt.show()


def load_ei(**kwargs):
    '''
     Loads parameters of interest into dataframe for further analysis
    '''

    rois=kwargs['rois']
    groups=kwargs['groups']
    tasks=kwargs['tasks']
    model_names=kwargs['model_names']
    conditions=kwargs['conditions']

    models, dirs = get_models(rois=rois,groups=groups,tasks=tasks,model_names=model_names,conditions=conditions)

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
    output.to_csv('../results/final_group_hierarchical_condition_ei_mu_distributions.csv')
    return output

if __name__ == "__main__":
    main()
