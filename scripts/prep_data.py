import sys
import os
import numpy as np
import scipy.io as sio
import pickle
import random
import itertools
import nibabel as nib
from scipy.stats import *
import pandas as pd

'''
 
 Functions to prepare and clean data prior to model run
 
'''


def get_neighbors(r,xloc,yloc,zloc):
    lims = range(-r,r)
    x_coord, y_coord, z_coord = [], [], []
    for x in range(len(lims)):
        for y in range(len(lims)):
            for z in range(len(lims)):
                if np.linalg.norm([lims[x], lims[y], lims[z]]) <= r and (lims[x] != 0 or lims[y] != 0 or lims[z] != 0):
                    x_coord.append(lims[x]+xloc)
                    y_coord.append(lims[y]+yloc)
                    z_coord.append(lims[z]+zloc)
    coords = [x_coord, y_coord, z_coord]
    return np.array(coords)


def get_indiv_volumes(args):

    subjectlists = args['subjectlists']
    roi_path = args['roi_path']
    rois = args['rois']
    roi_names = [roi.replace('.nii','') for roi in rois]
    contrasts = args['contrasts']
    tasks = args['tasks']
    stats_dir = args['stats_dir']
    conditions = args['conditions']
    model_path = args['model_path']
    project_dir = args['project_dir']

    nVoxels = 119
    nNeighbors = 3

    print('Extracting data for model fitting...')

    for group in subjectlists:
        if '.txt' in group: #if you're using an old subject list
            subjects = np.loadtxt(group,dtype='str')
            subjects = [subj.replace("b'",'').replace("'",'') \
                        for subj in subjects]
        elif '.csv' in group: #if you're using a new subject list
            sub_df = pd.read_csv(group)
            subjects = sub_df['PID']
            subjects = [str(s).zfill(4) for s in subjects]

        for subject_i, subject in enumerate(subjects):

            for itask, task in enumerate(tasks):
                img = contrasts[itask]
                
                for iroi, roi in enumerate(rois):

                    roi_file = os.path.join(roi_path,roi)
                    roi_img = nib.load(roi_file)
                    roi_data = roi_img.get_data()
                    roi_coords = np.where(roi_data != 0)

                    if '.txt' in group:
                        year = '20'+subject[:2]
                        sub_file = '/mnt/musk2/%s/%s/fmri/stats_spm8/%s/%s'%(year,subject,stats_dir,img)
                    elif '.csv' in group:
                        pid = subject
                        visit = 'visit'+str(sub_df['Visit'][subject_i])
                        session = 'session'+str(sub_df['Session'][subject_i])
                        subject = pid+'_'+visit+'_'+session
                        sub_file = os.path.join(project_dir,'/results/taskfmri/participants',pid,visit,session,stats_dir,img)

                    sub_img = nib.load(sub_file)
                    sub_data = sub_img.get_data()

                    neighbors = np.zeros([len(roi_coords[0]),nVoxels])
                    voxel_activity = []
                    for c in range(len(roi_coords[0])):
                        x = roi_coords[0][c]
                        y = roi_coords[1][c]
                        z = roi_coords[2][c]
                        neighbor_coords = get_neighbors(nNeighbors,x,y,z)
                        voxel_activity.append(sub_data[x,y,z])
                        
                        for nc in range(len(neighbor_coords[0])):
                            nx = neighbor_coords[0][nc]
                            ny = neighbor_coords[1][nc]
                            nz = neighbor_coords[2][nc]
                            neighbors[c,nc] = sub_data[nx,ny,nz]
                
                    voxel_activity = np.array(voxel_activity)
                    if conditions == []:
                        volout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                              subject+'_'+task+'_vol.mat')
                        sio.savemat(volout,{'roi_beta':voxel_activity})

                        neighborout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                                   subject+'_'+task+'_neighbors.mat')
                        sio.savemat(neighborout,{'neighbors':neighbors})
                    else:
                        cond1_ind = np.where(voxel_activity > 0)
                        cond2_ind = np.where(voxel_activity < 0)

                        cond1vox = voxel_activity[cond1_ind]
                        cond2vox = voxel_activity[cond2_ind]
                        cond1neighbor = neighbors[cond1_ind,:]
                        cond2neighbor = neighbors[cond2_ind,:]

                        cond1volout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                                   subject+'_'+task+'_'+conditions[0]+'_vol.mat')
                        cond2volout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                                   subject+'_'+task+'_'+conditions[1]+'_vol.mat')
                        cond1neighborout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                                        subject+'_'+task+'_'+conditions[0]+'_neighbors.mat')
                        cond2neighborout = os.path.join(model_path,'extracted_volumes',roi.replace('.nii',''),
                                                        subject+'_'+task+'_'+conditions[1]+'_neighbors.mat')

                        sio.savemat(cond1volout,{'roi_beta':cond1vox})
                        sio.savemat(cond2volout,{'roi_beta':cond2vox})
                        sio.savemat(cond1neighborout,{'neighbors':cond1neighbor})
                        sio.savemat(cond2neighborout,{'neighbors':cond2neighbor})

def create_directory(args):
    '''
        This will create a new directory for model output in the location
        specified by model_path. 

    '''
    import os
    from subprocess import call

    model_path = args['model_path']

    print('Creating new directory in %s'%(model_path))

    if not os.path.isdir(model_path):

        dirs = [model_path,
                model_path+'model_fits',
                model_path+'extracted_volumes',
                model_path+'results',
                model_path+'figures']

        for d in dirs:
            os.mkdir(d)

        for model in args['model_names']:
            os.mkdir(model_path+'model_fits/'+model)

        for roi in args['rois']:
            roi = roi.replace('.nii','')
            os.mkdir(model_path+'extracted_volumes/'+roi)

    else:

        print('Model directory already exists.')


def activation_fun(val):
    if val <= 0 or np.isnan(val):
        return 0.0
    else:
        return val/(val + 1.0)


def flatten(array):
    flattened = [j for i in array for j in i]
    return flattened


def load_dataset(group=None,roi=None,task=None,condition=None,path=None):

    sublist = open(group+".txt", "r")
    subjects = sublist.read().split()

    raw_data, thresh_data, neighbor_data, noise_data, sid, sub_num = [], [], [], [], [], 1
    for sub in subjects:

        if condition != '':
            vol_data = sio.loadmat(path + 'extracted_volumes/'+ roi + '/' + sub + task + '_' + condition + '_vol.mat')
            neighbor_dat = sio.loadmat(path + 'extracted_volumes/'+ roi + '/' + sub + task + '_' + condition + '_neighbors.mat')
            if neighbor_dat['neighbors'].shape == (0,0,0):
                continue
            else:
                neighbor_dat = neighbor_dat['neighbors'][0]
        else:
            vol_data = sio.loadmat(path + 'extracted_volumes/'+ roi + '/' + sub + task + '_vol.mat')
            neighbor_dat = sio.loadmat(path + 'extracted_volumes/'+ roi + '/' + sub + task + '_neighbors.mat')
            neighbor_dat = neighbor_dat['neighbors']

        if vol_data['roi_beta'].shape == (0,0):
            continue

        mat = vol_data['roi_beta'][0]

        if condition != '':
            for m in range(len(mat)):
                if mat[m] < 0:
                    mat[m] *= -1

        dim = len(mat)

        thresh_mat = np.zeros(dim)
        noise_mat = np.zeros(dim)
        raw_mat = np.zeros(dim)
        neighbor_sum = []
        for x in range(dim):
            if np.isnan(mat[x]):
                mat[x] = 0
            thresh_mat[x] = activation_fun(mat[x])
            raw_mat[x] = mat[x]
            noise_mat[x] = np.random.normal(0,0.5)
            neighbors = []
            for y in range(len(neighbor_dat[x])):
                neighbors.append(activation_fun(neighbor_dat[x][y]))
            neighbor_sum.append(np.sum(neighbors))
        neighbor_sum = np.asarray(neighbor_sum)

        sid.append([sub_num] * len(mat))
        thresh_data.append(thresh_mat)   
        raw_data.append(raw_mat)             
        neighbor_data.append(neighbor_sum)
        noise_data.append(noise_mat)
        sub_num += 1

    thresh_x = flatten(thresh_data)
    raw_x = flatten(raw_data)  
    neighbor_activity = flatten(neighbor_data)
    noise = flatten(noise_data)
    subject_id = flatten(sid)
    nSubjects = len(subjects)
    nData = len(raw_x)

    dat = {'N':nData,'S':nSubjects,'subject':subject_id,'raw_x':raw_x,'thresh_x':thresh_x,'n_activity':neighbor_activity,'dim':dim,'noise':noise}
    df = pd.DataFrame.from_dict(dat)
    return df
