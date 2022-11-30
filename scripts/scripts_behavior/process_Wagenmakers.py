# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 07:53:11 2022
behavioral analysis of Wagenmakers' dataset
@author: ihoxha
"""

import pandas as pd
import numpy as np

#%% import data
wgdata=pd.read_csv('../../data/Wagenmakers/SpeedAccData_filtered.csv').set_index('Unnamed: 0')
wgdata=wgdata[wgdata.Subject!=2]

#%% prepare settings for 
result_df=pd.DataFrame(columns=['Subject', 'RT', 'Mean accuracy', 'Word type', 'Instruction', 'Time'])
time_map={0:'Early', 1:'Late'}
word_map={0:'non-existent', 1:'frequent' , 2:'rare', 3:'very rare'}
inst_map={0:'accuracy', 1:'speed'}

#%% and run
subjects=np.unique(wgdata.Subject)

for subject in subjects:
    subdat=wgdata[wgdata.Subject==subject]
    for w in np.unique(subdat.word_type):
        wdat=subdat[subdat.word_type==w]
        for i in np.unique(wdat.Condition):
            idat=wdat[wdat.Condition==i]
            for t in np.unique(idat.Late):
                tdat=idat[idat.Late==t]
                res=[subject,tdat['RT'].mean(), tdat['correct'].mean(),word_map[w],inst_map[i],time_map[t]]
                result_df.loc[len(result_df)]=res
                
#%% save
result_df.to_csv('../../data/Wagenmakers/SpeedAccData_filtered_for_behavioral_analyses.csv', index='Subject')
