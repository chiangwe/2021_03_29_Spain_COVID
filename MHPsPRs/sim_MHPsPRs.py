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
from collections import Counter
import multiprocessing
from scipy.sparse import csr_matrix

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

import multiprocessing


#%% ----- Utilities Functions -----#

def para_diff(prev_arr, curr_arr):
    n_ele = np.size( prev_arr )
    diff_out = np.sqrt( np.sum( np.abs( prev_arr.flatten() - curr_arr.flatten() ) ** 2 ) / n_ele )
    return diff_out

def off_sim( q_sim, q_job):

    while(True):

        itr_sim, itr_pred, r_0s, cnts, dict_para = q_job.get()
        np.random.seed(itr_sim)

        n_hzone = dict_para['n_hzone']
        offspring = [np.random.poisson(r_0, int(cnt)).sum() for r_0, cnt in zip(r_0s, cnts)];

        offspring = np.reshape(offspring, [n_hzone, n_hzone]).sum(1);

        ts_offsprint = [
            np.round(np.random.weibull(alpha_shape, offspring[itr_zone]) * beta_scale) + itr_pred \
            for itr_zone in range(0, n_hzone) ];

        itr_zones = [ [itr_zone] * len(ts_offsprint[itr_zone]) \
                     for itr_zone in range(0, n_hzone)];

        ts_offsprint = np.hstack(ts_offsprint)
        itr_zones = np.hstack(itr_zones)

        ts_offsprint = [(itrzone, ts) for itrzone, ts in zip(itr_zones, ts_offsprint) if ts < d_pred];

        dict_cnt = Counter(ts_offsprint);
        dict_cnt_keys = list(dict_cnt.keys())

        data = [dict_cnt[each] for each in dict_cnt_keys]
        row = [each[0] for each in dict_cnt_keys]
        col = [each[1] for each in dict_cnt_keys]

        #output_cnt[:, :, itr_sim] = output_cnt[:, :, itr_sim] + csr_matrix((data, (row, col)),
        #                                                                   shape=(n_hzone, d_pred)).toarray();

        out_sparse = csr_matrix((data, (row, col)), shape=(n_hzone, d_pred));

        q_sim.put( (itr_sim, out_sparse) )


if __name__ == '__main__':

    #%% ----- Set initial parameters -----#

    # Parse the passed in parameters
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--ds_crt", type=int, help="Days for boundary correction", default=14)
    parser.add_argument("--tol", type=float, help="Tolerance for convergence", default=10**-4)
    parser.add_argument("--max_itr", type=int, help="# of maximum iterations", default=60)
    parser.add_argument("--mode", type=str, help="# of maximum iterations", default='server')
    parser.add_argument("--n_proc", type=str, help="Number of processes", default=20)
    #
    parser.add_argument("--mdl_name", type=str, help="# of maximum iterations", default='MHPsPRs')
    parser.add_argument("--case_type", type=str, help="# of maximum iterations", default='confirm')
    parser.add_argument("--n_sim", type=str, help="Number of simulations", default=100)
    parser.add_argument("--d_pred", type=str, help="Number of simulations", default=21)
    parser.add_argument("--alpha_shape", type=str, help="Shape parameters for wbl", default=14)
    parser.add_argument("--beta_scale", type=str, help="bandwidth for the kernel", default=5)

    args = parser.parse_args()

    # Assign to global
    dict_parser = [ ( each, getattr(args, each) ) for each in args.__dict__.keys() ]
    globals().update( dict(dict_parser) )

    q_job = multiprocessing.Queue();
    q_sim = multiprocessing.Queue();

    ls_p = [];
    for each_proc in range(0, n_proc):
        p = multiprocessing.Process(target=off_sim, args=(q_sim, q_job))
        p.start()
        ls_p.append(p)

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

    #

    for each_date in tqdm(date_ranges[0:20], desc='processes on dates'):

        pred_st = (each_date - pd.Timedelta(1, unit='D')).strftime("%Y-%m-%d")
        itr_d_st = ls_date_move.index(pred_st)

        # ----- Set initial parameters -----#
        para_str = 'type_' + case_type + '_alpha_' + str(alpha_shape) + '_beta_' + str(beta_scale) + '_predat_';

        mdl_path_save = './mdl/' + mdl_name + '/' + mdl_name + '_Spain_' + para_str + each_date.strftime("%Y-%m-%d") + '.mat';
        sim_path_save = './results/' + mdl_name + '/' + mdl_name + '_Spain_' + para_str + each_date.strftime("%Y-%m-%d") + '.mat';

        # ------- Check trained ---------#
        print(mdl_path_save)
        if os.path.exists(sim_path_save):
            continue;

        if os.path.exists(mdl_path_save):
            mdic = loadmat( mdl_path_save )

            # Fix alpha beta
            mdic['alpha_shape'] = mdic['alpha_shape'][0][0]
            mdic['beta_scale'] = mdic['beta_scale'][0][0]
            globals().update(mdic)
        else:
            continue;

        # ----- Set initial parameters -----#

        n_hzone = len(ls_hzone_code_move);

        n_dates_tr = itr_d_st + 1
        n_dates_te = len(ls_date_move) - n_dates_tr
        if n_dates_te >= d_pred:
            n_dates_te = copy.deepcopy(d_pred)

        n_dates = n_dates_tr + n_dates_te;

        # ----- Make covariates  -----#

        COV_name = ['movement'] + df_demo.keys().tolist()[6:]
        n_feat = len(COV_name)

        COV = np.concatenate([ \
            np.expand_dims(np.swapaxes(ls_arr_move[:, :, 0:n_dates ], 0, 1), axis=3), \
            np.tile(np.expand_dims(np.tile(df_demo.iloc[:, 6:].values, [n_hzone, 1, 1]), axis=2), [1, 1, n_dates, 1])],
            axis=3)

        COV[:, :, :, 0] = np.log(COV[:, :, :, 0] + 1)

        COV_X = COV[:, :, 0:n_dates_tr, :];
        #COV_te = COV[:, :, n_dates_tr:, :];

        # Training set with boundary correction
        COV_X = np.reshape(COV_X[:, :, 0:n_dates_tr - ds_crt, :], [n_hzone * n_hzone * (n_dates_tr - ds_crt), n_feat],
                           order='F')
        COV_X_mean = np.expand_dims(COV_X.mean(0), axis=0);
        COV_X_std = np.expand_dims(COV_X.std(0), axis=0);
        COV_X = (COV_X - COV_X_mean) / COV_X_std

        # All for prediction
        COV = np.reshape(COV, [n_hzone * n_hzone * n_dates, n_feat], order='F')
        COV = (COV - COV_X_mean) / COV_X_std

        COV_te = COV[n_hzone*n_hzone*n_dates_tr:, : ]

        #R0 = mdic['R0']

        if case_type == 'confirm':
            covids =    df_infect.iloc[:, 6:].values[:, 0:n_dates_tr]
            covids_te = df_infect.iloc[:, 6:].values[:, n_dates_tr:]
        else:
            covids = df_death.iloc[:, 6:].values[:, 0:n_dates_tr]
            covids_te = df_death.iloc[:, 6:].values[:, n_dates_tr:]

            # ================================ Work on Simulation ================================#

        # Initial intermediate
        sim_dates = n_dates_tr + n_dates_te;

        # == Get R0 first ==#

        R0_est = np.zeros((n_hzone , n_hzone , sim_dates ));
        for each_d in range(0, sim_dates ):
            feat = COV[ n_hzone * n_hzone* each_d : (n_hzone * n_hzone* (each_d+1)), : ]
            r0_temp = np.exp( np.matmul(np.hstack((np.ones((feat.shape[0], 1)), feat)), mdic['tolcoef'].T ))
            R0_est[:, :, each_d] = np.reshape(r0_temp, [n_hzone, n_hzone], order='F')

        R0_est[R0_est>1] = 1

        # == Calculate Lambda first ==#

        # Initial intermediate
        di_m_dj = np.tril(np.expand_dims(np.arange(0, sim_dates), 1) - np.expand_dims(np.arange(0, sim_dates), 0), -1);

        # COVIDsim
        covids_sim = np.hstack([covids, np.zeros((covids.shape[0], n_dates_te))])
        R0_est = R0_est[:, :, 0:sim_dates];

        # print(covids.shape, covids_sim.shape)

        wbl_val = weibull_min.pdf(di_m_dj, alpha_shape, 0,
                                  beta_scale);  # print("wbl_val: ", type(wbl_val), wbl_val.shape)
        #
        #
        tf_sp_wbl_val = tf.sparse.from_dense(wbl_val,
                                             name='tf_sp_wbl_val');  # print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
        tf_sp_wbl_val = tf.sparse.expand_dims(tf_sp_wbl_val,
                                              axis=0);  # print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
        tf_sp_wbl_val = tf.sparse.expand_dims(tf_sp_wbl_val,
                                              axis=0);  # print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
        #
        tf_mus = tf.convert_to_tensor(mus, name='tf_mus');  # print("tf_mus: ", type(tf_mus), tf_mus.shape)
        tf_R0 = tf.expand_dims(R0_est, axis=(2), name='tf_R0');  # print("tf_R0: ", type(tf_R0), tf_R0.shape)
        #
        tf_covid_dj = tf.cast(covids_sim, tf.float64);  # print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
        tf_covid_dj = tf.expand_dims(tf_covid_dj,
                                     axis=0);  # print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
        tf_covid_dj = tf.expand_dims(tf_covid_dj, axis=2,
                                     name='tf_covid_dj');  # print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
        #
        tf_dj_wbl = tf.sparse.concat(1, [
            tf_sp_wbl_val] * n_hzone);  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
        tf_dj_wbl = tf.sparse.SparseTensor.__mul__(tf_dj_wbl,
                                                   tf_covid_dj);  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
        tf_dj_wbl = tf.sparse.to_dense(tf_dj_wbl,
                                       name='tf_dj_wbl');  # print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)

        #
        tf_lamb = tf.concat([ \
            tf.math.reduce_sum(
                tf.math.reduce_sum(tf.math.multiply(tf_R0[itr_hzone:itr_hzone + 1, :, :, :], tf_dj_wbl), axis=3),
                axis=1) \
            for itr_hzone in range(0, n_hzone)], axis=0) + tf_mus;

        # =============== Simulation  Drawing samples for background rate ============== #

        output_cnt = np.zeros((n_hzone, d_pred, n_sim))

        #R0_est_sim = R0_est
        R0_est = np.tile(np.expand_dims(R0_est, axis=3), [1, 1, 1, n_sim])

        new_mu = tf_lamb.numpy()[:, n_dates_tr: n_dates_tr + d_pred]
        new_mu = np.tile(np.expand_dims(new_mu, 2), [1, 1, n_sim])

        output_cnt = output_cnt + np.random.poisson(new_mu.flatten(order='F')).reshape((n_hzone, d_pred, n_sim),
                                                                                       order='F')
        tim_indx = np.expand_dims(np.expand_dims(np.expand_dims(np.arange(0, d_pred), axis=0), axis=1), axis=3)
        tim_indx = np.tile(tim_indx, [n_hzone, n_hzone, 1, n_sim])

        row_indx = np.expand_dims(np.expand_dims(np.expand_dims(np.arange(0, n_hzone), axis=1), axis=2), axis=3)
        row_indx = np.tile(row_indx, [1, n_hzone, d_pred, n_sim])

        sim_indx = np.expand_dims(np.expand_dims(np.expand_dims(range(0, n_sim), axis=0), axis=0), axis=0)
        sim_indx = np.tile(sim_indx, [n_hzone, n_hzone, d_pred, 1])

        # print(tim_indx.shape, row_indx.shape, R0_est_sim.shape, output_cnt.shape )

        for itr_pred in range(0, d_pred):

            output_cnt_in = np.expand_dims(output_cnt, 0);
            output_cnt_in = np.tile(output_cnt_in, [n_hzone, 1, 1, 1])

            # Put each t in que
            for itr_sim in range(0, n_sim):

                r_0s = R0_est[:, :, itr_pred, itr_sim].flatten(order='F')
                cnts  = output_cnt_in[:, :, itr_pred, itr_sim].flatten(order='F')

                dict_para = {}
                dict_para['n_hzone'] = n_hzone

                q_job.put( ( itr_sim, itr_pred , r_0s, cnts, dict_para) )

            for itr_sim in range(0, n_sim):
                itr_sim, out_sparse = q_sim.get()  # Returns output or blocks until ready

                output_cnt[:, :, itr_sim] = output_cnt[:, :, itr_sim] + out_sparse.toarray()
        '''
        for itr_pred in tqdm(range(0, d_pred), desc='processes on itr_pred'):
            # output_cnt_sim = output_cnt[:, :, d_pred, :]
    
            output_cnt_in = np.expand_dims(output_cnt, 0);
            output_cnt_in = np.tile(output_cnt_in, [n_hzone, 1, 1, 1])
            
            for itr_sim in range(0, n_sim):
                offspring = [np.random.poisson(r_0, int(cnt)).sum() \
                             for r_0, cnt in zip(R0_est[:, :, itr_pred, itr_sim].flatten(order='F'), \
                                                 output_cnt_in[:, :, itr_pred, itr_sim].flatten(order='F'))];
    
                offspring = np.reshape(offspring, [n_hzone, n_hzone]).sum(1);
    
                ts_offsprint = [
                    np.round(np.random.weibull(alpha_shape[0], offspring[itr_zone]) * beta_scale[0]) + itr_pred \
                    for itr_zone in range(0, n_hzone)];
    
                itr_zones = [[itr_zone] * len(ts_offsprint[itr_zone]) \
                             for itr_zone in range(0, n_hzone)];
    
                ts_offsprint = np.hstack(ts_offsprint)
                itr_zones = np.hstack(itr_zones)
    
                ts_offsprint = [(itrzone, ts) for itrzone, ts in zip(itr_zones, ts_offsprint) if ts < d_pred];
    
                dict_cnt = Counter(ts_offsprint);
                dict_cnt_keys = list(dict_cnt.keys())
    
                data = [dict_cnt[each] for each in dict_cnt_keys]
                row = [each[0] for each in dict_cnt_keys]
                col = [each[1] for each in dict_cnt_keys]
    
                output_cnt[:, :, itr_sim] = output_cnt[:, :, itr_sim] + csr_matrix((data, (row, col)),
                                                                                   shape=(n_hzone, d_pred)).toarray()
            '''
        # print(output_cnt.shape, covids_te.shape)
        mdic = {'output_cnt': output_cnt, 'covids_te': covids_te[:, 0:output_cnt.shape[1]]}
        savemat( sim_path_save, mdic)

    for each_proc in range(0, n_proc):
        p = ls_p.pop()
        p.terminate()
