# %%

# Disable autoCloseBreckets
'''
from notebook.services.config import ConfigManager
c = ConfigManager()
c.update('notebook', {"CodeCell": {"cm_config": {"autoCloseBrackets": False}}})
'''

# %%

import numpy as np
import pandas as pd
import os, sys
import shelve

df_raw = pd.read_csv('../../Raw_Dataset/Castilla/01March_2020_27April_2021_CYL_Combined.csv');
print('num of df_raw rows: ', df_raw.shape[0])

# %%

# ----- Check the keys
print('num of hzone:', df_raw[['hzone_code', 'hzone_name']].drop_duplicates().shape[0]);
# display( df_raw.keys().tolist() )

# Key for demo
ky_region = df_raw.keys()[0: 6].tolist()
ky_demo = df_raw.keys()[6:26].tolist()
ky_infect = [each for each in df_raw.keys().tolist() if 'infect' in each]
ky_death = [each for each in df_raw.keys().tolist() if 'death' in each]

st_data = '2020-03-15';
dates = pd.date_range(start=st_data, periods=len(ky_infect), freq='D');
dates = [each.strftime('%Y-%m-%d') for each in dates];

ls_hzone_code = df_raw['hzone_code']

# %%

# ----- Seprate all dataframe and save it in the ./Input

df_raw[ky_region + ky_demo].to_csv('./Input/Demo.csv', index=None)
df_raw[ky_region + ky_infect].rename(columns=dict(zip(ky_infect, dates)), errors="raise").to_csv('./Input/Infect.csv',
                                                                                                 index=None)
df_raw[ky_region + ky_death].rename(columns=dict(zip(ky_death, dates)), errors="raise").to_csv('./Input/Death.csv',
                                                                                               index=None)

# %%

# ----- Save all movement metrices into pkl

path = '../../Raw_Dataset/Castilla/CastillaYLeon/'
dirs = os.listdir(path);
dirs = [each for each in dirs if '.csv' in each];
dirs.sort()

ls_date_move = [];
ls_arr_move = [];

for each in dirs:
    path_read = path + each;

    ls_date_move.append(each.replace('.csv', ''))
    ls_arr_move.append(pd.read_csv(path_read).iloc[:, 1:].values)

ls_arr_move = np.stack(ls_arr_move).swapaxes(0, 2);

# %%

# ----- There are some csv fils

path = '../../Raw_Dataset/Castilla/CastillaYLeon/replaced_by_new_data/'
dirs = os.listdir(path);
dirs = [each for each in dirs if '.csv' in each];
dirs.sort()

ls_date_move_replace = [];
ls_arr_move_replace = [];

for each in dirs:
    path_read = path + each;

    ls_date_move_replace.append(each.replace('.csv', ''))
    ls_arr_move_replace.append(pd.read_csv(path_read).iloc[:, 1:].values)

ls_arr_move_replace = np.stack(ls_arr_move_replace).swapaxes(0, 2);

ls_hzone_code_move = pd.read_csv(path_read).iloc[:, 0].values.tolist()

# %%

# ----- There are some csv fils in replace one replace into it

ind_replace = [ls_date_move.index(each) for each in ls_date_move_replace];

for itr in range(0, ls_arr_move_replace.shape[2]):
    ls_arr_move[:, :, ind_replace[itr]] = ls_arr_move_replace[:, :, itr]

# %%

# ----- There are some hcode demographic does have

df_demo = pd.read_csv('./Input/Demo.csv')
df_infect = pd.read_csv('./Input/Infect.csv')
df_death = pd.read_csv('./Input/Death.csv')

df_demo = df_demo[df_demo['hzone_code'].isin(ls_hzone_code_move)]
df_infect = df_infect[df_infect['hzone_code'].isin(ls_hzone_code_move)]
df_death = df_death[df_death['hzone_code'].isin(ls_hzone_code_move)]

df_demo.to_csv('./Input/Demo.csv', index=None)
df_infect.to_csv('./Input/Infect.csv', index=None)
df_death.to_csv('./Input/Death.csv', index=None)

# %%

# ----- Save move metrics
d = shelve.open('./Input/move_metrics.pkl');
d['ls_arr_move'] = ls_arr_move;
d['ls_hzone_code_move'] = ls_hzone_code_move;
d['ls_date_move'] = ls_date_move;
d.close()

# %%


