'''
Main file for tuning model
Copy this, fill it out, run it

1. start python environment with:
    - seaborn (and all dependencies)
    - pymc3
    - nibabel
2. python model_main.py

'''

###################################
###### Fill out the following #####
###################################

# directory for model output (will create a directory called 'tuning_model' here)
model_path = '/mnt/apricot1_share1/Longitudinal_TD_MD/RSA_td_md/saliency_map_model/'

# path to the folder containing your roi files
roi_path = '/fs/apricot1_share1/Longitudinal_TD_MD/RSA_td_md/ROISignalLevel/Teresa_NEW_RSA_results_ROIs_May2017/16MD_17TD/'

# the name of your individual stats directory
stats_dir = 'addsub_stats_spm8_swaor'

# the roi nii files you want to run this for
rois = ['03-6mm_right_hIP3_FIND_GLM_subpeak2_44_-54_46.nii',
        '02-6mm_right_hIP3_FIND_GLM_subpeak1_34_-64_52.nii']

# list of subject lists (include full path)
subjectlists = ['/mnt/apricot1_share1/Longitudinal_TD_MD/RSA_td_md/saliency_map_model/final/md.txt',
                '/mnt/apricot1_share1/Longitudinal_TD_MD/RSA_td_md/saliency_map_model/final/td.txt']

# the images from which to
# extract input data in the model
contrasts = ['spmT_0017.img','spmT_0021.img']

# labels for your contrasts (for output file naming)
tasks = ['add','sub']

# whether to run separate models for
# the conditions within your contrasts
# i.e. extracted data will be:
#    - cond1_volume = voxels > 0
#    - cond2_volume = voxels < 0
# If contrast is cond1 - cond2 set (max 2):
#    conditions = [cond1, cond2]
# If you don't want to run separate models set:
#    conditions = []
conditions = ['comp','simp']
#conditions = []

# list of which models to run
# options are:
#   - hierarchical
#   - unpooled
model_names = ['hierarchical']

# number of samples in each chain
n_iters = 1000

# number of chains
n_chains = 3

# Project directory
project_dir = '' #If not running on sherlock
#project_dir = '/oak/stanford/groups/menon/jnichola/dynamic_tuning/'

####################################
####################################
####################################

import sys
source_code = '/oak/stanford/groups/menon/projects/jnichola/spatial_tuning_model/scripts'
sys.path.append(source_code)
import prep_data as pd
import run_models as rm
import analysis as an

if model_path[-1] != '/':
    model_path = model_path+'/'
model_path = model_path+'tuning_model/'

args={'model_path':model_path,
       'roi_path':roi_path,
       'stats_dir':stats_dir,
       'rois':rois,
       'subjectlists':subjectlists,
       'contrasts':contrasts,
       'tasks':tasks,
       'conditions':conditions,
       'model_names':model_names,
       'n_iters':n_iters,
       'n_chains':n_chains,
       'project_dir':project_dir}

# create the directory structure
pd.create_directory(args)

# load the data
pd.get_indiv_volumes(args)

# run models in parallel
rm.run_models(args)

# save a summary csv for the parameters of interest
an.load_ei(args)

# run posterior predictive checks and save csv files with results
an.compute_ppc(args)






