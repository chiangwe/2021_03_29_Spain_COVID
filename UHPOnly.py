# %% ----- Import packages -----#

import sys

print("This script is running on server :", sys.executable)
# import matlab

import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=1

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
from scipy.io import savemat
from set_size import set_size
# import rpy2
import pickle
import time
from platform import python_version

print("python_version: ", python_version())

# Force to use CUP since I don't have access
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# %% ----- Read Data-----#

df_raw = pd.read_csv('../../Raw_Dataset/Castilla/01March_2020_27April_2021_CYL_Combined.csv');
# display(df_raw)

# ----- Read-in

df_demo = pd.read_csv('./Input/Demo.csv')
df_infect = pd.read_csv('./Input/Infect.csv')
df_death = pd.read_csv('./Input/Death.csv')

d = shelve.open('./Input/move_metrics.pkl');
ls_arr_move = d['ls_arr_move'];

# Scale the movment data
ls_arr_move = np.log(ls_arr_move + 1)

ls_hzone_code_move = d['ls_hzone_code_move'];
ls_date_move = d['ls_date_move'];
d.close()

# ----- Align the mobility dates
ls_date_move = [pd.to_datetime(each).strftime('%Y-%m-%d') for each in ls_date_move];
date_covid = df_infect.keys()[6:]

min_date = max(pd.to_datetime(ls_date_move[0]), pd.to_datetime(date_covid[0])).strftime('%Y-%m-%d')
max_date = min(pd.to_datetime(ls_date_move[-1]), pd.to_datetime(date_covid[-1])).strftime('%Y-%m-%d')

df_infect = pd.merge(df_infect.iloc[:, :6], \
                     df_infect.iloc[:,
                     df_infect.keys().tolist().index(min_date): df_infect.keys().tolist().index(max_date) + 1], \
                     left_index=True, right_index=True);

df_death = pd.merge(df_death.iloc[:, :6], \
                    df_death.iloc[:,
                    df_death.keys().tolist().index(min_date): df_death.keys().tolist().index(max_date) + 1], \
                    left_index=True, right_index=True);

ls_date_move = ls_date_move[ls_date_move.index(min_date):ls_date_move.index(max_date) + 1];
ls_arr_move = ls_arr_move[:, :, ls_date_move.index(min_date):ls_date_move.index(max_date) + 1];

# Get all dates ranges
date_ranges = pd.date_range(start='2020-10-04', end=max_date, freq='7D')


# %% ----- Utilities Functions -----#

def para_diff(prev_arr, curr_arr):
    n_ele = np.size(prev_arr)
    diff_out = np.sqrt(np.sum(np.abs(prev_arr.flatten() - curr_arr.flatten()) ** 2) / n_ele)
    return diff_out


# %% ----- Set initial parameters -----#

# Parse the passed in parameters
parser = argparse.ArgumentParser()
#
parser.add_argument("--ds_crt", type=int, help="Days for boundary correction", default=14)
parser.add_argument("--tol", type=float, help="Tolerance for convergence", default=10 ** -4)
parser.add_argument("--max_itr", type=int, help="# of maximum iterations", default=60)
parser.add_argument("--mode", type=str, help="# of maximum iterations", default='server')
#
parser.add_argument("--mdl_name", type=str, help="# of maximum iterations", default='UHPOnly')
parser.add_argument("--case_type", type=str, help="# of maximum iterations", default='confirm')
parser.add_argument("--alpha_shape", type=str, help="Shape parameters for wbl", default=2)
parser.add_argument("--beta_scale", type=str, help="bandwidth for the kernel", default=5)
args = parser.parse_args()

# Assign to global
dict_parser = [(each, getattr(args, each)) for each in args.__dict__.keys()]
globals().update(dict(dict_parser))

# %% ----- Set initial parameters -----#

for each_date in tqdm(date_ranges):

    pred_st = (each_date - pd.Timedelta(1, unit='D')).strftime("%Y-%m-%d")
    itr_d_st = ls_date_move.index(pred_st)
    d_pred_ahead = 28;

    # ----- Set initial parameters -----#
    para_str = 'type_' + case_type  + '_alpha_' + str(alpha_shape) + '_beta_' + str(
        beta_scale) + '_predat_';

    mdl_path_save = './mdl/' + mdl_name + '/' + mdl_name + '_Spain_' + para_str + each_date.strftime(
        "%Y-%m-%d") + '.mat';

    # ------- Check trained ---------#
    print(mdl_path_save)
    if os.path.exists(mdl_path_save):
        continue;
    # ----- Set initial parameters -----#

    n_hzone = len(ls_hzone_code_move);

    n_dates_tr = itr_d_st + 1
    n_dates_te = len(ls_date_move) - n_dates_tr
    if n_dates_te >= d_pred_ahead:
        n_dates_te = copy.deepcopy(d_pred_ahead)

    n_dates = n_dates_tr + n_dates_te;

    # ----- Make covariates  -----#

    mus = np.random.uniform(0.2, 0.5,    [n_hzone, 1]);

    R0 = np.random.uniform(5, 10, [n_hzone, 1]);
    #R0 = np.expand_dims( np.eye(n_hzone) * R0, axis=2 );
    #R0 = np.tile(R0, [1, 1 , n_dates_tr]);


    if case_type == 'confirm':
        covids = df_infect.iloc[:, 6:].values[:, 0:n_dates_tr]
        covids_te = df_infect.iloc[:, 6:].values[:, n_dates_tr:]
    else:
        covids = df_death.iloc[:, 6:].values[:, 0:n_dates_tr]
        covids_te = df_death.iloc[:, 6:].values[:, n_dates_tr:]

    # Initial intermediate
    di_m_dj = np.tril(np.expand_dims(np.arange(0, n_dates_tr), 1) - np.expand_dims(np.arange(0, n_dates_tr), 0), -1);

    # dict for checking the convergence, difference between previous variable and current variables
    dict_delta = {'delta_mu': [], 'delta_R': []}


    #sample_weight = np.tile(covids[:, 0:n_dates_tr - ds_crt], [n_hzone, 1, 1]).flatten(order='F')


    # ================ EM step for all ==================================================#

    for itr_em in range(0, max_itr):
        # ================================ E step ================================#
        # ----- Calculate Lambda -----#

        wbl_val = weibull_min.pdf(di_m_dj, alpha_shape, 0, beta_scale);  # print("wbl_val: ", type(wbl_val), wbl_val.shape)
        #
        #
        tf_sp_wbl_val = tf.sparse.from_dense(wbl_val, name='tf_sp_wbl_val');  # print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
        tf_sp_wbl_val = tf.sparse.expand_dims(tf_sp_wbl_val, axis=0);  # print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
        #
        tf_mus = tf.convert_to_tensor(mus, name='tf_mus');  # print("tf_mus: ", type(tf_mus), tf_mus.shape)
        tf_R0 = tf.convert_to_tensor(R0,  name='tf_R0');  # print("tf_R0: ", type(tf_R0), tf_R0.shape)
        #
        tf_covid_dj = tf.cast(covids, tf.float64);  # print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
        tf_covid_dj = tf.expand_dims(tf_covid_dj, axis=1, name='tf_covid_dj');  # print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
        #
        tf_dj_wbl = tf.sparse.concat(0, [tf_sp_wbl_val] * n_hzone);  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
        tf_dj_wbl = tf.sparse.SparseTensor.__mul__(tf_dj_wbl, tf_covid_dj);  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
        tf_dj_wbl = tf.sparse.to_dense(tf_dj_wbl, name='tf_dj_wbl');  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)

        #
        tf_lamb = tf.math.reduce_sum(tf.math.multiply( tf.expand_dims(tf_R0, axis= 2), tf_dj_wbl ), axis=2) + tf_mus;

        # print(tf_lamb)

        # ----- Calculate P_j -----#

        tf_covid_di = tf.expand_dims(tf.cast(covids, tf.float64), axis=(2),name='tf_covid_di');
        tf_di_wbl = tf.sparse.to_dense( tf.sparse.SparseTensor.__mul__(tf.sparse.concat(0, [tf_sp_wbl_val] * n_hzone), tf_covid_di), name='tf_di_wbl')

        new_epi = tf_lamb.numpy()[tf_lamb.numpy()!=0].min() * tf.keras.backend.epsilon()**2;
        tf_lamb = tf.dtypes.cast((tf_lamb==0), dtype=tf.float64) * new_epi + tf_lamb;
		
        ##pdb.set_trace()
        P_j = tf.math.reduce_sum( \
            tf.math.divide( tf.math.multiply( tf.expand_dims(tf_R0, axis=2), tf_di_wbl), \
                            tf.expand_dims(tf_lamb, axis=2) ), axis=1)

        # pdb.set_trace()
        # ================================ M step ================================#

        # ----- Update mu -----#

        tf_mus = tf.math.multiply(tf.math.divide(tf_mus, tf_lamb), tf.cast(covids, tf.float64))[:, 0:n_dates - ds_crt];
        tf_mus = tf.math.reduce_sum(tf_mus, axis=1) / (n_dates - ds_crt);
        mus_est = np.expand_dims( tf_mus.numpy(), 1)

        # ----- R -----#

        R0_est = tf.math.reduce_sum(P_j[:, 0:n_dates-ds_crt], axis=1)/(n_dates-ds_crt)
        R0_est = np.expand_dims( R0_est.numpy(), axis=1)
        #R0_est[R0_est > 10] = 10;
        #R0_est = np.multiply(R0_est, np.eye(n_hzone))
        #R0_est = np.tile( np.expand_dims(R0_est, 2), [1, 1, n_dates_tr] )

        # ----- Update WBL -----#

        # alpha_shape_est = alpha_shape
        # beta_scale_est = beta_scale
        # ================================ Calculate the difference parameter ================================#

        dict_delta['delta_mu'].append(para_diff(mus, mus_est));
        dict_delta['delta_R'].append(para_diff(R0, R0_est));

        # ================================ Update all parameter ================================#

        mus = mus_est.copy();

        R0 = R0_est.copy();

        #R0_mean = [ np.diagonal(R0[:, :, each]).mean()  for each in range(0, R0.shape[2])];
        R0_mean = R0.mean()

        print("Itr: ", "{:04d}".format(itr_em), \
              ", mus_mean: ", "{:6f}".format(mus.mean()), \
              ", mus_diff: ", "{:6e}".format(dict_delta['delta_mu'][-1]), \
              ", R0_mean: ", "{:.6f}".format(R0_mean.mean()), \
              ", R0_diff: ", "{:.6e}".format(dict_delta['delta_R'][-1]), \
              alpha_shape, beta_scale)

        # print(mus.mean(), R0.mean(), alpha_shape, beta_scale_est)

        # ----- Early Stop -------#
        if itr_em > 5:
            if ( np.all(np.array(dict_delta['delta_mu'][-5:]) < tol) & np.all(np.array(dict_delta['delta_R'][-5:]) < tol)):
                break;

    # ================================ Save what I think that is important ================================#
    mdic = { \
        'mus': mus, 'R0': R0, 'alpha_shape': alpha_shape, 'beta_scale': beta_scale, \
        'COV_name': COV_name\
        }
    savemat(mdl_path_save, mdic)

    # Save to file in the current working directory
    # pkl_filename = './Model_MultiHPs/' + 'r0_death_model_' + each_date.strftime("%Y-%m-%d") + '.pkl'
    # with open(pkl_filename, 'wb') as file:
    #    pickle.dump(clf, file)
