import numpy as np
import pandas as pd
import pdb


QuanName = ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00', '0.00-1.00'];
Type     = ['confirm','death'];

df_all = pd.read_csv('./results_tables/agg_results.csv')

ls_method = [ 'UHPOnly','MHPsOnly','MHPsPRs','MHPsTsVarCoef']
ls_method = [ each.rjust(20) for each in ls_method ]

# Formattting 
df_all['methods'] = df_all['methods'].str.pad(20, 'left')
df_all['abs'] = df_all['abs'].apply(lambda x: "{:.2f}".format(x) ).str.pad(10, 'left')
df_all['wis'] = df_all['wis'].apply(lambda x: "{:.2f}".format(x) ).str.pad(10, 'left')
df_all['ndcg'] = df_all['ndcg'].apply(lambda x: "{:.4f}".format(x) ).str.pad(10, 'left')

for each_quan in QuanName:
	#
	for each_type in Type:
		PathName = './results_tables/path_' + each_type + '_' + each_quan + '.csv';
		
		df_save = df_all[ (df_all['quan'] == each_quan) & (df_all['type'] == each_type) ]
		df_save = df_save.pivot(index='methods',columns='week_ahead')[['abs','wis','ndcg']].reindex( ls_method )
		df_save.to_csv(PathName, sep='&')


