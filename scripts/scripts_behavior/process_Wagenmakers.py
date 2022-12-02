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
time_map={0:'Early', 1:'Late'}
word_map={0:'non-existent', 1:'frequent' , 2:'rare', 3:'very rare'}
inst_map={0:'accuracy', 1:'speed'}

#%% and run
subjects=np.unique(wgdata.Subject)
cols=['Subject']
for w in word_map:
    for t in time_map:
        for i in inst_map:
            cols.append(f'RT {word_map[w]}, {time_map[t]}, {inst_map[i]}')
            cols.append(f'Mean accuracy {word_map[w]}, {time_map[t]}, {inst_map[i]}')
# result_df=pd.DataFrame(columns=['Subject', 'RT non-existent', 'RT easy', 'RT rare', 'RT very rare',
#                                 'Mean accuracy non-existent', 'Mean accuracy frequent', 'Mean accuracy rare', 'Mean accuracy very rare', 
#                                 'RT early', 'RT late', 
#                                 'Mean accuracy early', 'Mean accuracy late', 
#                                 'RT accuracy', 'RT speed', 
#                                 'Instruction', 'Time'])
result_df=pd.DataFrame(index=subjects, columns=cols)
for subject in subjects:
    subdat=wgdata[wgdata.Subject==subject]
    res=[subject]
    res.extend(subdat.groupby(['word_type', 'Late','Condition'])['RT'].mean())
    res.extend(subdat.groupby(['word_type', 'Late','Condition'])['correct'].mean())
    # res.extend(subdat.groupby('Late')['RT'].mean())
    # res.extend(subdat.groupby('Late')['correct'].mean())
    # res.extend(subdat.groupby('Condition')['RT'].mean())
    # res.extend(subdat.groupby('Condition')['correct'].mean())
    # for w in np.unique(subdat.word_type):
    #     wdat=subdat[subdat.word_type==w]
    #     for i in np.unique(wdat.Condition):
    #         idat=wdat[wdat.Condition==i]
    #         for t in np.unique(idat.Late):
    #             tdat=idat[idat.Late==t]
    #             res=[subject,tdat['RT'].mean(), tdat['correct'].mean(),word_map[w],inst_map[i],time_map[t]]
    result_df.loc[subject]=res
                
#%% save
result_df.to_csv('../../data/Wagenmakers/ANOVA_SpeedAccData_filtered_for_behavioral_analyses.csv', index='Subject')
