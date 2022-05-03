# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:37:05 2022
Scripts to compute behavioral quantities before processing in JASP
@author: ihoxha
"""

import pandas as pd
import numpy as np

#%% load data (to modify later with Zenodo)
data_file=pd.read_csv(r'D:\PhD\data to share\RT_data.csv')
subjects=np.unique(data_file.Subject)

#%% instantiate where the results will be stores
#can also have them per stimulus type
RT0=[]
RT1=[]
acc0=[]
acc1=[]

#and also the number of "numbers" that appear
count=[]

#we will also need the SR mapping corresponding to the participant
SR_mapping=[]

#%% time to compute!
for subject in subjects:
    # load data
    data_df=data_file[data_file['Subject'] == subject]
    stimuli=data_df['Stimulus'].to_numpy()
    RTs=data_df['RT'].to_numpy()
    responses=data_df['Response'].to_numpy()
    
    side0=np.where(stimuli==0)[0]
    side1=np.where(stimuli==1)[0]
    
    correct=np.where(stimuli==responses)[0]#np.where(responses==0)[0]#
    
    correct0=np.intersect1d(correct,side0)
    correct1=np.intersect1d(correct,side1)
    
    RT0.append(np.mean(RTs[side0]))
    RT1.append(np.mean(RTs[side1]))
    acc0.append(len(correct0)/len(side0))
    acc1.append(len(correct1)/len(side1))
    
    count.append(np.mean(stimuli))

    SR_mapping.append(np.unique(data_df['S-R mapping'])[0])
    
#%% embed all that in a dataframe
behavioral_df=pd.DataFrame()
behavioral_df['Subject']=subjects
behavioral_df['S-R mapping']=SR_mapping
behavioral_df['Face RT']=RT0
behavioral_df['Number+Sound RT']=RT1
behavioral_df['Face accuracy']=acc0
behavioral_df['Number+Sound accuracy']=acc1

behavioral_df=behavioral_df.set_index('Subject')

#%% save that struct
behavioral_df.to_csv('./../../results/behavioral_analysis.csv')