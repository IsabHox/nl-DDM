# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:22:13 2022
Correlation analysis on simulated data
@author: ihoxha
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

#%% import data and concatenate
list_params=['b','v','sz','x0']
params=pd.DataFrame(columns=['Unnamed: 0', 'B (DDM)', 'v (DDM)', 'x0 (DDM)', 'sz (DDM)', 'k', 'z',
       'a/B', 'x0', 'sz']).set_index('Unnamed: 0')
for p in list_params:
    params=pd.concat([params,pd.read_csv(f'C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/fitting_DDM_simulated_from_PyDDM_vary_{p}.csv').set_index('Unnamed: 0')])

#%% compute correlation matrix
corr = params.corr()


p_matrix=params.corr(method=lambda x, y: pearsonr(x, y)[1])- np.eye(*corr.shape)
p = p_matrix.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))

mask = np.tril(np.ones_like(corr, dtype=bool))
corr=corr.mask(mask)
corr=corr.drop(['B (DDM)','v (DDM)','x0 (DDM)', 'sz (DDM)'], axis=1)
corr=corr.drop(['sz'],axis=0)
p=p.mask(mask)
p=p.drop(['B (DDM)','v (DDM)','x0 (DDM)', 'sz (DDM)'], axis=1)
p=p.drop(['sz'],axis=0)

#%% plot
xlabels=['k (nl-DDM)','z (nl-DDM)','a (nl-DDM)', 'x0 (nl-DDM)', 'sz (nl-DDM)']
ylabels=['B (DDM)','v (nl-DDM)', 'x0 (DDM)', 'sz (DDM)',
         'k (nl-DDM)','z (nl-DDM)','a (nl-DDM)', 'x0 (nl-DDM)']
annot=corr.round(2).astype(str)
annot=annot+p
plt.figure(figsize=(8,5))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=None, cmap='vlag_r', center=0, fmt='', vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot,
            xticklabels=xlabels, yticklabels=ylabels)
plt.yticks(rotation=0)  
       
#%% saving
plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/DDM_simulated_correlations.png', dpi=300)
plt.savefig('C:/Users/ihoxha/Desktop/PhD/nl-DDM/results/Figures/DDM_simulated_correlations.eps')
