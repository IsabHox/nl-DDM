# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:26:11 2022

@author: ihoxha
"""


import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

#%% import data
fitted_params=nlddm_params=pd.read_csv('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/fitting_binary_bounded_correlations.csv', sep=';').set_index('Subject')

#%% compute correlations and p-values
correlation_matrix=fitted_params.corr()
p_matrix=fitted_params.corr(method=lambda x, y: pearsonr(x, y)[1])- np.eye(*correlation_matrix.shape)
p = p_matrix.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))

#%%plot
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))
annot=correlation_matrix.round(2).astype(str)
annot=annot+p
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, mask=mask, cmap='vlag_r', vmin=-1, vmax=1, annot=annot, fmt='')
# plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/experimental_correlations_heatmaps.png')
plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/experimental_correlations_heatmaps.eps')

plt.figure(figsize=(10,8))
sns.heatmap(p_matrix, mask=mask, cmap='vlag', vmin=0, vmax=1, annot=True)