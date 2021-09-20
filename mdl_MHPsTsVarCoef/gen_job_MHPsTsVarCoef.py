import pandas as pd
import numpy as np
import shelve
import pdb

#----- Path 
# Define Input data path
path_demo   = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Demo.csv'
path_infect = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Infect.csv'
path_death  = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/Death.csv'
#
path_movement = '/home/chiangwe/PhD2018/2021_03_29_Spain_COVID/Input/move_metrics.pkl'

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
ls_date_move = [ pd.to_datetime(each).strftime('%Y-%m-%d')  for each in ls_date_move];
date_covid = df_infect.keys()[6:]

min_date = max( pd.to_datetime(ls_date_move[0]), pd.to_datetime(date_covid[0]) ).strftime('%Y-%m-%d')
max_date = min( pd.to_datetime(ls_date_move[-1]), pd.to_datetime(date_covid[-1]) ).strftime('%Y-%m-%d')

st_date = '2020-10-04';
date_ranges = pd.date_range(start = st_date, end =max_date, freq ='7D');
date_ranges = [ each.strftime('%Y-%m-%d') for each in date_ranges ]

# --- Define parameters --- #

Method = 'MHPsTsVarCoef'
case_type = [ 'confirm', 'death' ]
ls_bw = [ '5', '10'];
ls_alpha_shape = [ '0' ];
ls_beta_scale = [ '0' ];

for each_type in case_type:
	for each_bw in ls_bw:
		for each_alpha in ls_alpha_shape:
			for each_beta in ls_beta_scale:
				for pd_date in date_ranges:
					
					command = "python " + Method + ".py --case_type " + each_type + \
							  " --pd_date " + pd_date+ " --bw " + each_bw + \
							  " --alpha_shape " + each_alpha + " --beta_scale " + each_beta + " >/dev/null";
					
					path_job = './job/' + Method + '_Spain_type_' + each_type \
							+ '_bw_' + each_bw + '_alpha_' + each_alpha \
							+ '_beta_' + each_beta + '_predat_' + pd_date + '.job'
					
					FilePtr = open(path_job, 'w+');
					FilePtr.write( command + "\n" )
					FilePtr.close()




