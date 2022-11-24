# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:29:36 2022
Concatenating the results obtained from the fitting algos
@author: ihoxha
"""

import pandas as pd 

#%% basics
datapath='C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/'

subjects=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

#%% and saving
for subject in subjects:
    dataname=f'performance_Wagenmakers_fatigue_{subject}.csv'
    if subject==1:
        results_df=pd.read_csv(datapath+dataname).set_index('Subject').drop('Unnamed: 0',axis=1)
    else:
        cur_df=pd.read_csv(datapath+dataname).set_index('Subject').drop('Unnamed: 0',axis=1)
        results_df=pd.concat([results_df, cur_df])
        
results_df.to_csv(datapath+'performance_Wagenmakers_fatigue_global.csv')

#%% do the same for DDM and nl-DDM
nlDDM=datapath+'fitting_nlDDM_binary_bounded.csv'
param_df=pd.read_csv(nlDDM).set_index('Subject').drop('Unnamed: 0',axis=1)
DDM=datapath+'fitting_ddm_binary_bounded.csv'
ddm_df=pd.read_csv(DDM).set_index('Subject').drop('Unnamed: 0',axis=1)
param_df=pd.concat([param_df, ddm_df], axis=1)
param_df.to_csv(datapath+'fitting_binary_bounded.csv')