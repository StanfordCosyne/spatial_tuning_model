# Spatial Tuning Model

This repository holds code for the tuning model originally used for the study:

    Computational modeling reveals weak spatial tuning of problem representations in children with dyscalculia
    Teresa Iuculano, Jonathan Nicholas, Ting-Ting Chang, Arron, W. S. Metcalfe, Vinod Menon

It is a "spatial tuning" version of the dynamical salience map models used in the following:

    Roggeman et al. (2010) Neuroimage
    Knops et al. (2014) Journal of Neuroscience

Please see **documentation/tuningmodelmethods.pdf** for an in-depth description of how this works.

The code is designed to be run on an roi-by-roi basis. Below are descriptions of the necessary files:

### scripts/model_main.py
This is the main file for running the model. It effectively functions as a config file, with several
variables to fill out. The model needs contrast images from first level stats in order to run. A confusing
parameter might be the 'conditions' variable; this determines whether separate models will be run for the
two (max two) conditions in your contrast image. The reason for this is that the model will treat all values
<0 as 0 and will therefore seriously underfit any negative voxels in your image. To get around this, we
fit one model for each condition using only the voxels that show an effect for that condition, which allows
us to take the absolute value of any negative effects. In addition, through testing it seems that the bare
minimum number of iterations should be 1000 and number of chains should be 3.

Once you have run the model, a directory called 'tuning_model' will be created in the location specified
by the 'model_path' variable. This will contain the extracted data, the fit model objects, a summary csv
containing the parameters of interest (alpha, beta), and a csv with results from posterior predictive checks

### scripts/prep_data.py
This script contains functions to prepare your data to run in the model. It will create a new directory
for your output, extract voxel values from each image for each subject, and preloads the data prior to
any models being run

### scripts/run_models.py
This script will load all of the models you are going to run (for each subject, roi, task, and condition)
and then run them in parallel using the joblib.Parallel package. NOTE: This is not configured for job
submission on sherlock! It might be a good idea to replace these functions with separate job submissions
on the cluster, but this is not necessary if your initial job requests enough memory for your models

### scripts/candidate_models.py
This file holds each of the model definitions and is called by run_models.py. The models are:
	- test_model: a simple model with normally distributed random data for sanity checks
	- unpooled_model: a non-hierarchical version of the spatial tuning model
	- hierarchical_model: the primary hierarchical spatial tuning model 
	- dd_null_model: A model with hardcoded priors from the hierarchical_model; experimental!

### scripts/analysis.py
This script is called after models are fit to extract values for alpha (excitation), beta (inhibition),
and excitatory-inhibitory balance (beta/alpha). It will also run posterior predictive checks by 
generating 500 datasets from the fit models and then saves the mean and standard deviation of these
datasets for comparison with the original data.

### scripts/old/*
This directory has several analysis and plotting functions that have been previously useful but are
not integrated with the current version of the code
