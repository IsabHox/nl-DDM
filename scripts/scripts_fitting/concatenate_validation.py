# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:13:58 2023
concatenate the results
@author: ihoxha
"""

import pandas as pd

#%% loop and save
concatenation=pd.DataFrame(columns=['Subject','k','z1','z2','z3','zNW','a0/BA',
                                    'a1/BS','x0','sz','nondectime','l','m1',
                                    'm2','m3','mNW','BA','BS','x0','sz',
                                    'nondectime',"LogLoss (nlDDM, train)",
                                    "LogLoss (OU, train)","BIC (nlDDM, train)",
                                    "BIC (OU, train)","LogLoss (nlDDM, test)",
                                    "LogLoss (OU, test)","BIC (nlDDM, test)",
                                    "BIC (OU, test)","# train samples",
                                    "# test samples","CV"])
subjects=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] #,12,13,14,15,16,17

for subject in subjects:
    for cv in range (5):
        
        datpath=f'../../results/OU_CV/5crossval_Wagenmakers_noFatigue_{subject}_{cv}.csv'
        dat=pd.read_csv(datpath).drop('Unnamed: 0', axis=1)
        dat['CV']=cv
        if subject==1 and cv==0:
            concatenation=dat
        else:
            concatenation=pd.concat([concatenation, dat])
        
concatenation.to_csv('../../results/OU_CV/5crossval_Wagenmakers_noFatigue_all.csv')

#%% grouping
grouped_cat=concatenation.groupby(['Subject']).mean()
grouped_cat.to_csv('../../results/OU_CV/5crossval_Wagenmakers_noFatigue_grouped.csv')