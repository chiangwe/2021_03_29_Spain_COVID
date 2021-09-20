import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os
import pdb

each_method = 'UHPOnly'
ls_files = os.listdir( './' + each_method + '/' );

# We only look into estimated wbl distribution
ls_files = [ each for each in ls_files if 'alpha_0_beta_0_' in each ]
ls_files.sort()

for file in ls_files:
	print(file)
	if '.mat' not in file:
		continue;
	else:

		mat_dict = loadmat( './' + each_method + '/' + file );
		#covids_te = mat_dict['covids_te']
		#output_cnt = mat_dict['output_cnt']
		
		if (mat_dict['R0'] > 5).sum() > 0:
			pdb.set_trace()
