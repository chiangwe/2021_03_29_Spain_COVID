#%% ----- Import packages -----#

import sys
print("This script is running on server :", sys.executable)
#import matlab

import os
os.environ["OMP_NUM_THREADS"]           = "6" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"]      = "6" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"]           = "6" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"]    = "6" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"]       = "6" # export NUMEXPR_NUM_THREADS=1

import numpy as np
np.random.seed(0)
import pandas as pd
import argparse

import itertools
import tensorflow as tf
from scipy.stats import weibull_min

import os, sys
import shelve
import copy
from tqdm import tqdm
from sklearn import linear_model
import shelve
from scipy.io import savemat, loadmat
from set_size import set_size
#import rpy2
import pickle
import time
from platform import python_version
print("python_version: ", python_version())

import multiprocessing

# Force to use CUP since I don't have access
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#from sklearnex import patch_sklearn
#patch_sklearn()

# conda install -c intel scikit-learn
import daal4py.sklearn
daal4py.sklearn.patch_sklearn()
from sklearn.metrics import ndcg_score

#%% ----- Read Data-----#

df_raw = pd.read_csv('../../Raw_Dataset/Castilla/01March_2020_27April_2021_CYL_Combined.csv');
#display(df_raw)

#----- Read-in

df_demo   = pd.read_csv('./Input/Demo.csv')
df_infect = pd.read_csv('./Input/Infect.csv')
df_death  = pd.read_csv('./Input/Death.csv')

d = shelve.open('./Input/move_metrics.pkl');
ls_arr_move = d['ls_arr_move'];

# Scale the movment data
ls_arr_move = np.log(ls_arr_move+1)

ls_hzone_code_move = d['ls_hzone_code_move'];
ls_date_move = d['ls_date_move'];
d.close()

#----- Align the mobility dates
ls_date_move = [ pd.to_datetime(each).strftime('%Y-%m-%d')  for each in ls_date_move];
date_covid = df_infect.keys()[6:]

min_date = max( pd.to_datetime(ls_date_move[0]), pd.to_datetime(date_covid[0]) ).strftime('%Y-%m-%d')
max_date = min( pd.to_datetime(ls_date_move[-1]), pd.to_datetime(date_covid[-1]) ).strftime('%Y-%m-%d')

df_infect = pd.merge( df_infect.iloc[:, :6], \
            df_infect.iloc[:, df_infect.keys().tolist().index(min_date): df_infect.keys().tolist().index(max_date)+1],\
             left_index=True, right_index=True);

df_death = pd.merge( df_death.iloc[:, :6], \
            df_death.iloc[:, df_death.keys().tolist().index(min_date): df_death.keys().tolist().index(max_date)+1],\
             left_index=True, right_index=True);

ls_date_move = ls_date_move[ls_date_move.index(min_date):ls_date_move.index(max_date)+1];
ls_arr_move = ls_arr_move[:, :, ls_date_move.index(min_date):ls_date_move.index(max_date)+1];

# Get all dates ranges
date_ranges = pd.date_range(start = '2020-10-04', end =max_date, freq ='7D')

#%% -- Evaluation function ------#
# Alphas for wis
alphas = np.array( [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] )

def wis(y_true, y_all):
    y_true = np.expand_dims(y_true, axis=1)
    l_is = np.quantile( y_all, alphas/2, axis=1).T
    u_is = np.quantile( y_all, 1-alphas/2, axis=1).T
    IS_a = (u_is - l_is) +  (l_is - y_true )*(l_is > y_true ) * (2/np.expand_dims(alphas, 0)) + \
                            (y_true - u_is)*(u_is < y_true ) * (2/np.expand_dims(alphas, 0))
    abs_out = np.abs(y_true - np.expand_dims(y_all.mean(1), 1) );
    WIS = (  np.expand_dims((IS_a * (np.expand_dims(alphas, 0)/2)).sum(1), 1) + abs_out/2 ) / (alphas.shape[0]+0.5)
    return WIS, abs_out

#%% ----- Set initial parameters -----#

ls_method = [ 'UHPOnly', 'MHPsOnly', 'MHPsPRs', 'MHPsTsVarCoef', 'MHPsTsVarGeoVarCoef'];
#ls_method = [ 'UHPOnly'];

ls_pred = [7, 14, 21];

df_eval = pd.DataFrame(columns=['methods', 'type', 'predat', 'week_ahead','para', 'abs', 'wis', 'ndcg']);

for each_method in ls_method:
    for file in os.listdir( './results/' + each_method ):
        print(file)
        if '.mat' not in file:
            continue;
        else:
            mat_dict = loadmat( './results/' + each_method + '/' + file );
            covids_te = mat_dict['covids_te']
            output_cnt = mat_dict['output_cnt']

            ls_str = file.replace('.mat', '').split('_');

            pred_day = ls_str[-1]
            d_type = ls_str[3];

            index = df_infect.keys().tolist().index(pred_day)

            gt_cnt = np.array( [ covids_te[:, (each-1)*7: each*7].sum(1) for each in [1, 2, 3] ] ).T;
            gt_cnt = np.cumsum(gt_cnt, 1);

            pred_cnt = np.array([output_cnt[:, (each - 1) * 7: each * 7, :].sum(1) for each in [1, 2, 3]]);

            for itr_d_pred in range(0, len(ls_pred) ):
                WIS, abs_out = wis(gt_cnt[:, itr_d_pred], pred_cnt[itr_d_pred, :, :])
                ndcg_out = ndcg_score( np.expand_dims( gt_cnt[:, itr_d_pred], axis=0 ), np.expand_dims( pred_cnt[0, :, :].mean(1), axis=0) )

                abs_out_avg = abs_out.mean()
                WIS_avg = WIS.mean()

                new_data = {'methods':each_method, 'type':d_type, 'predat':pred_day, 'week_ahead':itr_d_pred, 'para':'_'.join(ls_str[4:-2]), 'abs':abs_out_avg, 'wis':WIS_avg, 'ndcg':ndcg_out}
                df_eval = df_eval.append(new_data, ignore_index=True)

print(df_eval)
