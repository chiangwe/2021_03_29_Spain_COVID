#%% ----- Import packages -----#

import sys
print("This script is running on server :", sys.executable)
#import matlab

import os
os.environ["OMP_NUM_THREADS"]			= "6" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"]		= "6" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"]			= "6" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"]	= "6" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"]		= "6" # export NUMEXPR_NUM_THREADS=1

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
#from set_size import set_size
#import rpy2
import pickle
import time
from platform import python_version
print("python_version: ", python_version())
import nvsmi

import multiprocessing
import queue
import pdb

# ---- Find the host name ----- # 
global myhost
myhost = os.uname()[1]

if ( ( myhost == 'volta') ):
	list_avail_gpu = [ each for each in nvsmi.get_available_gpus()]

	if len(list_avail_gpu) > 0:
		list_avail_gpu[0]
		os.environ['CUDA_VISIBLE_DEVICES'] = list_avail_gpu[0].id
	else:
		print("No Available GPU")

	# Force to use CUP since I don't have access
	# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	if tf.test.gpu_device_name():
		print('GPU found')
	else:
		print("No GPU found")
else:
	print("No GPU found")
	os.environ['CUDA_VISIBLE_DEVICES'] = ''

#from sklearnex import patch_sklearn
#patch_sklearn()

# conda install -c intel scikit-learn
import daal4py.sklearn
daal4py.sklearn.patch_sklearn()

def raw_data_load(path_demo, path_infect, path_death, path_movement):
	
	#----- Read-in
	
	df_demo   = pd.read_csv( path_demo )
	df_infect = pd.read_csv( path_infect )
	df_death  = pd.read_csv( path_death )
	#
	d = shelve.open( path_movement );
	ls_arr_move = d['ls_arr_move'];

	# Scale the movment data with log 
	ls_arr_move = np.log(ls_arr_move+1)
	
	ls_hzone_code_move = d['ls_hzone_code_move'];
	ls_date_move = d['ls_date_move'];
	d.close()
	
	#----- Align the mobility dates
	ls_date_move = [ pd.to_datetime(each).strftime('%Y-%m-%d')	for each in ls_date_move];
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
	
	# Get all date range (predict in every 7 days)
	date_ranges = pd.date_range(start = st_date, end =max_date, freq ='7D')

	return df_infect, df_death, df_demo, ls_arr_move, ls_date_move, ls_hzone_code_move, date_ranges

def data_eng( pred_st, df_infect, df_death, df_demo, ls_arr_move, ls_date_move, ls_hzone_code_move ):

	# Get index of date in the movement date
	itr_d_st = ls_date_move.index(pred_st)
	
	# Get number of health zones 
	n_hzone = len(ls_hzone_code_move);

	# Get number of dates for training and validation
	n_dates_tr = itr_d_st + 1
	n_dates_te = len(ls_date_move) - n_dates_tr
	if n_dates_te >= d_pred_ahead:
		n_dates_te = copy.deepcopy(d_pred_ahead)

	# Tol number of dates 
	n_dates = n_dates_tr + n_dates_te;
	
	# ----- Make covariates  -----#
	COV_name = { 'tvar': ['movement'], 'tstat': df_demo.keys().tolist()[6:] }
	
	n_feat_tvar  = len(COV_name['tvar'])
	n_feat_tstat = len(COV_name['tstat'])

	# Get time-varying features and demographic static features
	COV_tvar  = np.swapaxes(ls_arr_move[:, :, 0:n_dates ], 0, 1);
	COV_tstat = df_demo.iloc[:, 6:].values;
	
	# Fix the zero movement
	COV_tvar = np.log( COV_tvar + 1)
	
	# --- Training set with boundary correction --- #
	
	# time-varying features
	COV_tvar_X = np.reshape( COV_tvar[:, :, 0:n_dates_tr - ds_crt], \
							 [n_hzone * n_hzone * (n_dates_tr - ds_crt), n_feat_tvar],\
					   		 order='F')
	COV_tvar_X_mean, COV_tvar_X_std = \
		np.expand_dims( COV_tvar_X.mean(0), axis=0), np.expand_dims( COV_tvar_X.std(0), axis=0)
	COV_tvar_X = ( COV_tvar_X - COV_tvar_X_mean ) / COV_tvar_X_std;
	
	COV_tvar_Xall = np.reshape( COV_tvar, \
								[n_hzone * n_hzone * (n_dates), n_feat_tvar],\
								order='F')
	COV_tvar_Xall = ( COV_tvar_Xall - COV_tvar_X_mean ) / COV_tvar_X_std;
	
	# demographic static features
	COV_tstat_X = COV_tstat
	COV_tstat_X_mean, COV_tstat_X_std = \
		np.expand_dims( COV_tstat_X.mean(0), axis=0), np.expand_dims( COV_tstat_X.std(0), axis=0)
	COV_tstat_X = ( COV_tstat_X - COV_tstat_X_mean ) / COV_tstat_X_std;

	# --- All for prediction --- #
	
	# time-varying features
	COV_tvar_te = np.reshape( COV_tvar[:, :, n_dates_tr:], \
							[ n_hzone * n_hzone * n_dates_te, n_feat_tvar],\
					   		 order='F')
	COV_tvar_te = ( COV_tvar_te - COV_tvar_X_mean ) / COV_tvar_X_std

	# demographic static features
	COV_tstat_te = COV_tstat_X;
	
	# --- Case data --- #
	
	if case_type == 'confirm':
		covids =	df_infect.iloc[:, 6:].values[:, 0:n_dates_tr]
		covids_te = df_infect.iloc[:, 6:].values[:, n_dates_tr:]
	else:
		covids = df_death.iloc[:, 6:].values[:, 0:n_dates_tr]
		covids_te = df_death.iloc[:, 6:].values[:, n_dates_tr:]

	# Processes Sample weight (case number )
	smpl_wgt = np.tile( covids[:, 0:n_dates_tr - ds_crt], [n_hzone, 1, 1]).flatten(order='F')
	
	return n_dates_tr, n_dates_te, n_dates, n_feat_tvar, n_feat_tstat, n_hzone, \
		   covids, covids_te, COV_name, \
		   COV_tvar_Xall, COV_tvar_X, COV_tvar_te, \
		   COV_tstat_X, COV_tstat_te, smpl_wgt
			

def PR_fit_tvar( q_PR, q_job, q_data, arr_states, proc_id):
	
	while( arr_states[proc_id] != 4 ):
		
		if( ( arr_states[proc_id] == 0 ) ):
			
			#print("arr_states[proc_id]: ", arr_states[proc_id])
			COV_tvar_Xall, COV_tvar_X_t, COV_tstat_X, \
			smpl_wgt, n_hzone, n_dates_tr, ds_crt = q_data.get(block=True)
			
			# Update State
			arr_states[proc_id] = 1;
		elif( ( arr_states[proc_id] == 1 ) & (q_data.empty()==True)  ):
			
			#print("arr_states[proc_id]: ", arr_states[proc_id])
			# Update State
			arr_states[proc_id] = 2;
			
		elif( ( arr_states[proc_id] == 2 ) ):
			
			#print("arr_states[proc_id]: ", arr_states[proc_id])
			y_tr, R0_tvar, R0_tstat = q_data.get(block=True)
			
			# Update State
			arr_states[proc_id] = 3;
			
		elif( (arr_states[proc_id] == 3) & (q_job.empty() != True ) ):
			#print("arr_states[proc_id]: ", arr_states[proc_id])
			try:
				each_t = q_job.get(timeout=2)
			except queue.Empty as error:
				continue
			np.random.seed(each_t)
			
			# Only use +- d_pr days of data to work 
			tr_st = n_hzone * n_hzone * each_t - n_hzone * n_hzone * d_pr;
			tr_ed = n_hzone * n_hzone * each_t + n_hzone * n_hzone * d_pr;
			
			## Adjust 
			if tr_st <= 0:
				tr_st = 0;
			elif tr_st >= (n_hzone * n_hzone * (n_dates_tr-ds_crt-d_pr)):
				tr_st =   (n_hzone * n_hzone * (n_dates_tr-ds_crt-d_pr))

			if tr_ed >= n_hzone * n_hzone * (n_dates_tr-ds_crt):
				tr_ed = n_hzone * n_hzone * (n_dates_tr-ds_crt)
			
			# Model the time varying 
			clf = linear_model.PoissonRegressor()
			
			new_weight_pre = ( COV_tvar_X_t[tr_st:tr_ed] - each_t )/bw;
			new_weight = np.exp( -(new_weight_pre ** 2)/2 ) / ((np.sqrt(2*np.pi))*bw) 
			new_weight = np.multiply( smpl_wgt[tr_st:tr_ed], new_weight)
			
			weight_adjust = np.tile( R0_tstat[:, :, 0:n_dates_tr-ds_crt], [ n_hzone, 1, 1 ]).flatten(order='F')[tr_st:tr_ed]\
						    + np.finfo(float).eps**3;
			
			# Remove weight == 0 
			weight_in = new_weight * weight_adjust + np.finfo(float).eps**3;
			weight_in = weight_in/np.max(weight_in);
			
			y_tr_in = y_tr[tr_st:tr_ed]/weight_adjust;
			COV_tvar_in = COV_tvar_Xall[tr_st:tr_ed, :]
			
			idx_keep = np.where(weight_in!=0)
			
			clf.fit( COV_tvar_in[idx_keep[0], :], y_tr_in[idx_keep], weight_in[idx_keep] )

			theta_est = clf.coef_;
			intercept_est = clf.intercept_;
			coef_tvar = np.hstack((intercept_est, theta_est))

			R0_tvar_est = clf.predict( COV_tvar_Xall[ each_t * n_hzone * n_hzone: (each_t + 1) * n_hzone * n_hzone, :] );
			#q_PR.put( (each_t, coef, R0_est) )
			
			# Model the statistic feature 
			clf = linear_model.PoissonRegressor()
			
			weight_adjust = R0_tvar[ :, :, 0:(n_dates_tr-ds_crt) ].flatten(order='F')[tr_st:tr_ed];
			
			# Remove weight == 0
			weight_in = new_weight * weight_adjust + np.finfo(float).eps**3;
			
			y_tr_in = y_tr[tr_st:tr_ed]/weight_adjust;
			
			hz_j = np.unravel_index( range(0, y_tr.shape[0]), (n_hzone, n_hzone, (n_dates_tr-ds_crt) ), order='F')[1][tr_st:tr_ed];	
			
			data_in = pd.DataFrame( np.vstack((y_tr_in, weight_in, y_tr_in*weight_in ,hz_j)).T ).groupby(by=3).sum().reset_index()
			data_in = data_in.sort_values(by=3)
			
			idx_keep = np.where(data_in[1].values!=0)[0]
			
			y_tr_in = (data_in[2]/data_in[1]).values
			weight_in = data_in[1].values
			weight_in = weight_in/np.max(weight_in);
			
			clf.fit( COV_tstat_X[idx_keep, :], y_tr_in[idx_keep], weight_in[idx_keep]+np.finfo(float).eps**2)
			
			theta_est = clf.coef_;
			intercept_est = clf.intercept_;
			coef_tstat = np.hstack((intercept_est, theta_est))
			
			R0_tstat_est = clf.predict( COV_tstat_X );
			R0_tstat_est = np.tile( np.expand_dims( R0_tstat_est, axis=0), [ n_hzone, 1] ).flatten(order='F')
			
			q_PR.put( (each_t, coef_tvar, coef_tstat, R0_tvar_est, R0_tstat_est ) )
		
		else:
			#print("arr_states[proc_id]: ", arr_states[proc_id])
			time.sleep(0.1)
	
def EM_algm( mdl_path_save, n_dates_tr, n_dates_te, n_dates, n_feat_tvar, n_feat_tstat, n_hzone, \
				 covids, covids_te, COV_name,\
				 COV_tvar_Xall, COV_tvar_X, COV_tvar_te, \
				 COV_tstat_X, COV_tstat_te, smpl_wgt, alpha_shape, beta_scale ):

	# Initial intermediate
	di_m_dj = np.tril(np.expand_dims(np.arange(0, n_dates_tr), 1) - np.expand_dims(np.arange(0, n_dates_tr), 0), -1);

	# Initial Coefficient 
	mus = np.random.uniform(  0.0005,   0.001, [ n_hzone, 1]);
	
	R0_tvar   = np.random.uniform(0.0000005, 0.000001, [ n_hzone, n_hzone, n_dates ]);
	R0_tstat  = np.random.uniform(0.0000005, 0.000001, [       1, n_hzone, n_dates ]);
	R0  = np.multiply( R0_tvar, R0_tstat )
	
	wbl_para = np.expand_dims( np.array([alpha_shape, beta_scale]), axis=1 )
	
	tolcoef_tvar  = np.random.uniform(0.0005, 0.001,[  n_feat_tvar + 1, n_dates])
	tolcoef_tstat = np.random.uniform(0.0005, 0.001,[ n_feat_tstat + 1, n_dates])
	
	# dict for checking the convergence, difference between previous variable and current variables
	dict_delta = {'delta_mu': [], 'delta_tolcoef_tvar': [], 'delta_tolcoef_tstat': [], 'delta_wbl':[]}
	
	# Create Time index  
	COV_tvar_X_t  = np.unravel_index( range(0, COV_tvar_X.shape[0]), \
					( n_hzone, n_hzone, n_dates_tr - ds_crt), order='F')[2];
	
	# List of chunks for multiprocesses
	s_processes = range(0, n_dates);
	# Note we will delete this later
	#s_processes = range(0, 10);
	
	# Initialized all data
	for each_proc in range(0, n_proc):
		q_data.put( (COV_tvar_Xall, COV_tvar_X_t, COV_tstat_X, smpl_wgt, n_hzone, n_dates_tr, ds_crt) )
	
	# ================ EM step for all ==================================================#

	for itr_em in range(0, max_itr):
		# ================================ E step ================================#
		
		# ----- Calculate Lambda -----#

		wbl_val = weibull_min.pdf(di_m_dj, alpha_shape, 0, beta_scale);	
		# print("wbl_val: ", type(wbl_val), wbl_val.shape)
		
		tf_sp_wbl_val = tf.sparse.from_dense(wbl_val, name='tf_sp_wbl_val'); 
		# print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
		
		tf_sp_wbl_val = tf.sparse.expand_dims(tf_sp_wbl_val, axis=0);	
		# print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
		
		tf_sp_wbl_val = tf.sparse.expand_dims(tf_sp_wbl_val, axis=0);	
		# print("tf_sp_wbl_val: ", type(tf_sp_wbl_val), tf_sp_wbl_val.shape)
		
		tf_mus = tf.convert_to_tensor(mus, name='tf_mus');
		tf_R0 = tf.expand_dims(R0[:, :, 0:n_dates_tr ], axis=(2), name='tf_R0');  
		# print("tf_mus: ", type(tf_mus), tf_mus.shape)
		# print("tf_R0: ", type(tf_R0), tf_R0.shape)
		
		#
		tf_covid_dj = tf.cast( covids, tf.float64);	
		# print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
		
		tf_covid_dj = tf.expand_dims(tf_covid_dj, axis=0);  
		# print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
		
		tf_covid_dj = tf.expand_dims(tf_covid_dj, axis=2, name='tf_covid_dj'); 
		# print("tf_covid_dj: ", type(tf_covid_dj), tf_covid_dj.shape)
		
		tf_dj_wbl = tf.sparse.concat(1, [tf_sp_wbl_val] * n_hzone);	
		# print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
		
		tf_dj_wbl = tf.sparse.SparseTensor.__mul__(tf_dj_wbl, tf_covid_dj);  
		# print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)
		
		tf_dj_wbl = tf.sparse.to_dense(tf_dj_wbl, name='tf_dj_wbl');  
		# print("tf_dj_wbl: ", type(tf_dj_wbl), tf_dj_wbl.shape)

		# Get Lambda
		tf_lamb = tf.concat([ \
			tf.math.reduce_sum(
				tf.math.reduce_sum(tf.math.multiply(tf_R0[itr_hzone:itr_hzone + 1, :, :, :], tf_dj_wbl), axis=3),
				axis=1) \
			for itr_hzone in range(0, n_hzone)], axis=0) + tf_mus;

		
		# ----- Calculate P_j -----#

		tf_covid_di = tf.expand_dims(tf.expand_dims(tf.cast( covids, tf.float64), axis=(1)), axis=(3),
									 name='tf_covid_di');
		tf_di_wbl = tf.sparse.to_dense(
			tf.sparse.SparseTensor.__mul__(tf.sparse.concat(0, [tf_sp_wbl_val] * n_hzone), tf_covid_di),
			name='tf_di_wbl')

		new_epi = tf_lamb.numpy()[tf_lamb.numpy()!=0].min() * tf.keras.backend.epsilon()**2;
		tf_lamb = tf.dtypes.cast((tf_lamb==0), dtype=tf.float64) * new_epi + tf_lamb;

		P_j = tf.concat([ \
			tf.math.reduce_sum( \
				tf.math.divide( \
					tf.math.multiply(tf_R0[:, itr_hzone:itr_hzone + 1, :, :], tf_di_wbl), \
					tf.expand_dims(tf.expand_dims(tf_lamb, axis=(1)), axis=(3)) \
					), \
				axis=2) for itr_hzone in range(0, n_hzone)], axis=1);
		
		weight_wbl = tf.math.reduce_sum( \
				tf.math.multiply( tf.expand_dims( tf.squeeze(tf_covid_dj), 1), \
			tf.concat([ \
			tf.math.reduce_sum( \
				tf.math.divide( \
					tf.math.multiply(tf_R0[:, itr_hzone:itr_hzone + 1, :, :], tf_di_wbl), \
					tf.expand_dims(tf.expand_dims(tf_lamb, axis=(1)), axis=(3)) \
					), \
				axis=0) for itr_hzone in range(0, n_hzone)], axis=0) ) , axis=0);
		
		# ================================ M step ================================#
		
		
		# ----- Update mu -----#

		tf_mus = tf.math.multiply(tf.math.divide(tf_mus, tf_lamb), tf.cast( covids, tf.float64))[:, 0:n_dates_tr - ds_crt];
		tf_mus = tf.math.reduce_sum(tf_mus, axis=1) / (n_dates_tr - ds_crt);
		mus_est = tf_mus.numpy()
		
		# ----- Update Theta for Time Varying Data -----#
		# Update Global variables
		y_tr = P_j[:, :, 0:n_dates_tr - ds_crt].numpy().flatten(order='F')
	
		# ------------------------------------- #
		
		for each_proc in range(0, n_proc):
			q_data.put( ( y_tr, R0_tvar, R0_tstat ) )
		
		tolcoef_tvar_est  = np.zeros( ( COV_tvar_X.shape[1] + 1,  n_dates ) )
		tolcoef_tstat_est = np.zeros( ( COV_tstat_X.shape[1] + 1, n_dates ) )
		
		R0_tvar_est  = np.zeros( ( n_hzone*n_hzone*n_dates, ))
		R0_tstat_est = np.zeros( ( n_hzone*n_hzone*n_dates, ))
		
		bw_obj  = np.zeros( ( n_dates, ))
		bw_grad = np.zeros( ( n_dates, ))
		
		# Calculate the time sample weight

		# Put each t in que
		for each_t in s_processes:
			q_job.put(each_t)
		
		for each_q in tqdm(s_processes):
			each_t, coef_tvar, coef_tstat, r0_tvar, r0_tstat = q_PR.get()
			tolcoef_tvar_est[:, each_t]  = coef_tvar
			tolcoef_tstat_est[:, each_t] = coef_tstat
			#
			R0_tvar_est[ each_t * n_hzone * n_hzone: (each_t + 1) * n_hzone * n_hzone, ]  = r0_tvar
			R0_tstat_est[each_t * n_hzone * n_hzone: (each_t + 1) * n_hzone * n_hzone, ] = r0_tstat
		
		R0_est = R0_tvar_est * R0_tstat_est;

		# ----- Estimate Update Alpha Beta ----- #
		if bool_wbl:
			indx = np.tril_indices(n_dates_tr - ds_crt, -1) 
			obs = indx[0] - indx[1]
			weight_wbl = weight_wbl.numpy()[indx[0], indx[1]]
			
			obs_weight = pd.DataFrame( np.vstack((obs, weight_wbl)).T, \
						columns=['obs','weight_wbl']).groupby('obs').sum().reset_index()
			obs_weight['weight_wbl'] = obs_weight['weight_wbl']/obs_weight['weight_wbl'].sum()
			
			#print(np.unique( np.random.choice(obs_weight['obs'], 100000, p=obs_weight['weight_wbl']) ))
			wel_est = weibull_min.fit( np.random.choice(obs_weight['obs'], 100000, p=obs_weight['weight_wbl']), floc=0 )
			
			alpha_shape_est = wel_est[0]
			beta_scale_est = wel_est[2]
		else:
			alpha_shape_est = copy.deepcopy(alpha_shape)
			beta_scale_est  = copy.deepcopy(beta_scale)
		
		# Reshape mus, R0, (alpha_shape, beta_scale)
		R0_est = np.reshape( R0_est, [n_hzone,n_hzone,n_dates],order='F' )
		
		R0_tvar_est = np.reshape( R0_tvar_est, [n_hzone,n_hzone,n_dates],order='F' )
		
		R0_tstat_est = np.reshape( R0_tstat_est, [n_hzone,n_hzone,n_dates],order='F' )[0, :, :]
		R0_tstat_est = np.expand_dims( R0_tstat_est, axis=0)
		
		mus_est = np.expand_dims( mus_est, axis=1 )
		
		wbl_para_est = np.expand_dims( np.array([alpha_shape_est, beta_scale_est]), axis=1 )
		# ================================ Calculate the difference parameter ================================#

		dict_delta['delta_mu'].append(para_diff(mus, mus_est));
		dict_delta['delta_tolcoef_tvar'].append(para_diff( tolcoef_tvar,  tolcoef_tvar_est));
		dict_delta['delta_tolcoef_tstat'].append(para_diff( tolcoef_tstat, tolcoef_tstat_est));
		dict_delta['delta_wbl'].append(para_diff( wbl_para, wbl_para_est));
		
		
		# ================================ Update all parameter ================================#
		mus           = copy.deepcopy(mus_est)
		R0_tvar       = copy.deepcopy(R0_tvar_est)
		R0_tstat      = copy.deepcopy(R0_tstat_est)
		R0            = copy.deepcopy(R0_est)
		tolcoef_tvar  = copy.deepcopy(tolcoef_tvar_est)
		tolcoef_tstat = copy.deepcopy( tolcoef_tstat_est )
		alpha_shape = copy.deepcopy(alpha_shape_est)
		beta_scale = copy.deepcopy(beta_scale_est)
		wbl_para = copy.deepcopy(wbl_para_est)
		#
		print("Itr: ", "{:04d}".format(itr_em), \
			  ", mus_mean: ", "{:6f}".format(mus.mean()), \
			  ", mus_diff: ", "{:6e}".format(dict_delta['delta_mu'][-1]), \
			  ", wbl_diff: ", "{:6e}".format(dict_delta['delta_wbl'][-1]), \
			  ", R0_mean: ", "{:.8f}".format(R0.mean()), \
			  ", delta_tolcoef_tvar: ", "{:.6e}".format(dict_delta['delta_tolcoef_tvar'][-1]), \
			  ", delta_tolcoef_tstat: ", "{:.6e}".format(dict_delta['delta_tolcoef_tstat'][-1]), \
			  ", delta_wbl: ", "{:.6e}".format(dict_delta['delta_wbl'][-1]), \
			  ", alpha_shape: ", "{:.3e}".format(alpha_shape), ", beta_scale: ", "{:.3e}".format(beta_scale))
		#----- Early Stop -------#
		if itr_em > 3:
			if (   np.all( np.array( dict_delta['delta_tolcoef_tvar'][-3:]) < tol         ) \
				 & np.all( np.array(dict_delta['delta_tolcoef_tstat'][-3:]) < tol         ) \
				 & np.all( np.array(           dict_delta['delta_mu'][-3:]) < tol         ) \
				 & np.all( np.array(          dict_delta['delta_wbl'][-3:]) < 10**-1      )):
				
				for proc_id in range(0, n_proc):
					arr_states[proc_id] = 0
				break;
			else:
				pass;
		else:
				pass;

		for proc_id in range(0, n_proc):
			arr_states[proc_id] = 1
	# ================================ Save what I think that is important ================================#
	mdic = { \
		'mus': mus, 'R0': R0, 'R0_tvar':R0_tvar, 'R0_tstat':R0_tstat,\
		'alpha_shape': alpha_shape, 'beta_scale': beta_scale, \
		'COV_name': COV_name, 'tolcoef_tvar': tolcoef_tvar, 'tolcoef_tstat':tolcoef_tstat\
		}
	
	savemat(mdl_path_save, mdic)

#%% ----- Utilities Functions -----#

def para_diff(prev_arr, curr_arr):
	n_ele = np.size( prev_arr )
	diff_out = np.sqrt( np.sum( np.abs( prev_arr.flatten() - curr_arr.flatten() ) ** 2 ) / n_ele )
	return diff_out

def arg_parse():
	
	# Parse the passed in parameters
	parser = argparse.ArgumentParser()
	#
	parser.add_argument("--ds_crt", 		type=int,	help="Days for boundary correction", default=14)
	parser.add_argument("--tol", 			type=float,	help="Tolerance for convergence", default=10**-3)
	parser.add_argument("--max_itr", 		type=int,	help="# of maximum iterations", default=50)
	parser.add_argument("--n_proc", 		type=str,	help="Number of processes", default=7)
	#
	parser.add_argument("--st_date", 		type=str,	help="Prediction start date", default='2020-10-04')
	parser.add_argument("--d_pred_ahead", 	type=int,	help="Number of days ahead for prediction", default=28)
	#
	parser.add_argument("--mdl_name", 		type=str,	help="# of maximum iterations", default='MHPsTsVarCoef')
	parser.add_argument("--case_type", 		type=str,	help="# of maximum iterations", default='confirm')
	parser.add_argument("--bw",			 	type=float, help="bandwidth for the kernel", default=1) #[50 1] 
	parser.add_argument("--d_pr",          	type=float, help="days used for regression", default=30) #[30 60]
	parser.add_argument("--pd_date",		type=str, help="start date for prediction", default=30)
	parser.add_argument("--alpha_shape", 	type=float, help="Shape parameters for wbl", default= 2)
	parser.add_argument("--beta_scale",  	type=float, help="bandwidth for the kernel", default=10)
	args = parser.parse_args()
	
	# Assign to global
	dict_parser = [ ( each, getattr(args, each) ) for each in args.__dict__.keys() ]
	dict_parser = dict(dict_parser)
	globals().update( dict_parser )
	
	global bool_wbl, alpha_shape, beta_scale
	if (( dict_parser['alpha_shape']==0) & (dict_parser['beta_scale']==0)):
		bool_wbl = True;
		# Set initial
		alpha_shape = 2; beta_scale=10; 
	else:
		bool_wbl = False;
	
	# Print out all arguments
	print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
	
	# Fixing the number of the processes 
	myhost = os.uname()[1]
	
	global n_proc
	if ( myhost == 'volta'):
		n_proc = 30
	elif ( myhost == 'gibson.science.iupui.edu' ):
		n_proc = 30
	else:
		n_proc = 5
	
def main():
	
	# Parse the auguments 
	arg_parse()
	
	# Define Input data path
	path_demo   = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Demo.csv'
	path_infect = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Infect.csv'
	path_death  = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Death.csv'
	#
	path_movement = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/move_metrics.pkl'
	#
	
	# Create Queues 
	global q_job, q_PR, q_data, arr_states
	
	q_job = multiprocessing.Queue()
	q_PR = multiprocessing.Queue()
	q_data = multiprocessing.Queue()
	arr_states = multiprocessing.Array('i', [0]*n_proc )
	
	ls_proc = [];
	for each_proc in range(0, n_proc):
		p = multiprocessing.Process(target=PR_fit_tvar, args=( q_PR, q_job, q_data, arr_states, each_proc ))
		p.start()
		ls_proc.append(p)
	
	# Load The raw data 
	df_infect, df_death, df_demo, ls_arr_move, ls_date_move, ls_hzone_code_move, date_ranges = \
		raw_data_load( path_demo, path_infect, path_death, path_movement )
	
	# Loop over each prediction start date 
	each_date = pd.to_datetime( pd_date )
	
	# Over write the data range since we are doing one a a time 
	date_ranges = [each_date]
	
	for each_date in tqdm(date_ranges):
		
		# ----- Set initial parameters -----#
		if bool_wbl == True:
			para_str = 	'type_' + case_type + '_bw_' + str(int(bw)) + \
						'_alpha_' + str(0) + '_beta_' + \
						str(0) + '_predat_';
		else:
			para_str = 	'type_' + case_type + '_bw_' + str(int(bw)) + \
						'_alpha_' + str(int(alpha_shape)) + '_beta_' + \
						str(int(beta_scale)) + '_predat_';
			
		#
		mdl_path_save = './mdl/' + mdl_name + '/' + mdl_name + \
						'_Spain_' + para_str + each_date.strftime("%Y-%m-%d") + '.mat';
		
		# Shift a prediction date forward 
		pred_st = ( each_date - pd.Timedelta(1, unit='D') ).strftime("%Y-%m-%d")
		
		# ------- Check trained ---------#
		print(mdl_path_save)
		if os.path.exists(mdl_path_save):
			continue;
		
		# ------- Feature Engineer ------# 
		n_dates_tr, n_dates_te, n_dates, n_feat_tvar, n_feat_tstat, n_hzone, \
		covids, covids_te, COV_name, \
		COV_tvar_Xall, COV_tvar_X, COV_tvar_te, \
		COV_tstat_X, COV_tstat_te, smpl_wgt = \
		data_eng( pred_st, df_infect, df_death, df_demo, ls_arr_move, ls_date_move, ls_hzone_code_move )
		
		# ------- EM algorithm ------#
		EM_algm( mdl_path_save, n_dates_tr, n_dates_te, n_dates, n_feat_tvar, n_feat_tstat, n_hzone, \
				 covids, covids_te, COV_name, \
				 COV_tvar_Xall, COV_tvar_X, COV_tvar_te, \
				 COV_tstat_X, COV_tstat_te, smpl_wgt, alpha_shape, beta_scale \
				)
		
		# print("Done: ")
		# for proc_id in range(0, n_proc):
		#	arr_states[proc_id] = 0
	
	#for proc_id in range(0, n_proc):
	#	arr_states[proc_id] = 4
	for each_p in ls_proc:
		each_p.terminate()
	
if __name__ == "__main__":
	main()
##
# define coefficient
