import numpy as np
import pandas as pd
import pdb


# Read in raw files 
raw_df = pd.read_csv('./results_tables/raw_results.csv')
raw_df = raw_df[raw_df.apply(lambda x: 'alpha_0_beta_0' in x['para'], axis=1)];

raw_df = raw_df[raw_df.apply(lambda x: 'bw_5' not in x['para'], axis=1)]


# Select best metric
ls_metric = ['abs','wis','ndcg'];
ls_group = ['methods', 'type', 'predat', 'week_ahead', 'quan']

ls_df = [];
for each_metric in ls_metric:
	
	if each_metric != 'ndcg':
		para_index = raw_df.loc[raw_df.groupby( ls_group )[each_metric].idxmin()][ls_group+['para']]
	else:
		para_index = raw_df.loc[raw_df.groupby( ls_group )[each_metric].idxmax()][ls_group+['para']]
	
	date_index = para_index['predat'].apply(lambda x: pd.to_datetime(x)+pd.Timedelta(7, unit='D') )
	bool_keep = date_index <= pd.to_datetime('2021-05-02')
	para_index['predat'] =  date_index.apply( lambda x: x.strftime('%Y-%m-%d') )
	
	para_index = para_index[bool_keep]
	para_index = pd.Index(para_index)
	
	ls_df.append( raw_df.set_index(ls_group)[each_metric] )

df_all = pd.concat(ls_df, axis=1)
# Get mean and std and list
ls_group = ['methods', 'type', 'week_ahead', 'quan']

df_mean = df_all.groupby(ls_group).mean().rename( columns=dict( zip( ls_metric, [ each+'_avg' for each in ls_metric] )))
df_std  = df_all.groupby(ls_group).std().rename( columns=dict( zip( ls_metric, [ each+'_std' for each in ls_metric] )))
df_all = pd.concat([df_mean, df_std], axis=1).reset_index()

df_arr = df_all.groupby(ls_group).apply( lambda x: np.array(x) )

############# Formate ###################
prec_abs = "{:.3f}";
prec_wis = "{:.3f}";
prec_ndcg = "{:.4f}";

QuanName = ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00', '0.00-1.00'];
Type     = ['confirm','death'];
ls_method = [ 'UHPOnly','MHPsOnly','MHPsPRs','MHPsTsVarCoef']
n_methods = len(ls_method)

ls_method = [ each.rjust(20) for each in ls_method ]
multiIndx = pd.MultiIndex.from_product([Type, QuanName, ls_method])


# Formattting 
#df_all['abs_avg']  = df_all['abs_avg'].apply(lambda x: prec_abs.format(x) ).str.pad(10, 'left')
#df_all['wis_avg']  = df_all['wis_avg'].apply(lambda x: prec_wis.format(x) ).str.pad(10, 'left')
#df_all['ndcg_avg'] = df_all['ndcg_avg'].apply(lambda x: prec_ndcg.format(x) ).str.pad(10, 'left')

# Work on mean 
df_mean = df_all.drop(['abs_std','wis_std','ndcg_std'], axis=1)
df_mean['methods'] = df_mean['methods'].str.pad(20, 'left')

df_mean = pd.pivot_table(df_mean, values=['abs_avg','wis_avg'], \
						index=['type','quan','methods'], columns=['week_ahead'])\
						.reset_index().set_index(['type', 'quan', 'methods'])

df_mean = df_mean.loc[multiIndx]

def bold_format(df_series):
	for itr in range(0, df_series.shape[0]-n_methods+1, n_methods ):
		prefix = [''] * n_methods;
		postfix = [''] * n_methods;
		
		min_idx = np.argmin( df_series.iloc[itr:itr+n_methods].values );
		prefix[min_idx] = '\\textbf{'
		postfix[min_idx] = '}'
		
		df_series.iloc[itr:itr+n_methods] = \
		df_series.iloc[itr:itr+n_methods].apply(lambda x: prec_abs.format(x) )

		#.str.pad(10, 'left')
		df_series.iloc[itr:itr+n_methods] = \
		[ pre+strin+post for pre, strin, post in zip( prefix, df_series.iloc[itr:itr+n_methods].tolist(), postfix) ]
		df_series.iloc[itr:itr+n_methods] = df_series.iloc[itr:itr+n_methods].str.pad(20, 'left')
	return df_series

df_mean.apply(lambda x: bold_format(x), axis=0 )

df_mean['abs_avg', 3] = None
df_mean['wis_avg', 3] = None
df_mean = df_mean.sort_index(axis=1)
# Bold the font 

df_mean.to_csv('results_tables/tbl_restuls.csv', sep='&')
pdb.set_trace()



