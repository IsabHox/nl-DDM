# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:04:08 2022

@author: ihoxha
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

#%% import data
ddm_params={'b':[1,0.2,0.5,2,3,5],
            'v':[1,0,0.1,0.5,0.7, 2],
            'z':[0.5, 0.1, 0.2, 0.4, 0.6, 0.8],
            'sz':[0, 0.01, 0.05, 0.1, 0.2, 0.4],
            'sv':[0, 0.02, 0.05, 0.1, 0.2]}

nlddm_names=['k','z','a/B','x0','sz','nondectime']
nlddm_params=pd.read_csv('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/fitting_nlDDM_sim.csv', sep=';').drop('Unnamed: 0',axis=1)#.set_index('Parameters')
pearson_table=np.zeros((len(nlddm_names), len(ddm_params)))
p_table=np.zeros((len(nlddm_names), len(ddm_params)))
spearman_table=np.zeros((len(nlddm_names), len(ddm_params)))
ps_table=np.zeros((len(nlddm_names), len(ddm_params)))
#%% loop
for (i,p) in enumerate(ddm_params):
    nlddm_var=nlddm_params.loc[(nlddm_params['Parameters'] == 'no_var') | [p in k and len(k)==len(p)+1 for k in nlddm_params['Parameters']]]
    for (j,n) in enumerate(nlddm_names):
        fit_list=nlddm_var[n]
        pearson_table[j,i],p_table[j,i]=pearsonr(ddm_params[p],fit_list)
        spearman_table[j,i],ps_table[j,i]=spearmanr(ddm_params[p],fit_list)
        
#%% plotting
nlddm_names=['k','z','a','x0','sz','Tnd']
ddm_names=['B','v','x0','sz','sv']
plt.figure(figsize=(8,5))
sns.heatmap(pearson_table, annot=True, xticklabels=ddm_names, yticklabels=nlddm_names, vmin=-1, vmax=1, cmap='vlag_r')
plt.yticks(rotation=0) 
plt.xlabel('DDM parameters')
plt.ylabel('nl-DDM parameters')
#plt.title('Correlation coefficients')
plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/simulated_correlations.png', dpi=300)
plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/simulated_correlations.eps')

plt.figure()
sns.heatmap(p_table, annot=True, xticklabels=ddm_names, yticklabels=nlddm_names, vmin=0, vmax=1)
plt.title('p values')

plt.figure()
sns.heatmap(spearman_table, annot=True, xticklabels=ddm_names, yticklabels=nlddm_names, vmin=-1, vmax=1, cmap='vlag_r')
plt.title('Correlation coefficients (Spearman)')

plt.figure()
sns.heatmap(ps_table, annot=True, xticklabels=ddm_names, yticklabels=nlddm_names, vmin=0, vmax=1)
plt.title('p values (Spearman)')