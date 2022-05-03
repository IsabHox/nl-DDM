# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:31:56 2022
Script to fit Wagenmakers' dataset
@author: ihoxha
"""

import sys
import pandas as pd
#import wget

#%% load data
!wget https://www.ejwagenmakers.com/Code/2008/LexDecData.zip -P ../data/

#%%
subjects=np.unique(filtered_dat['Subject'])
acc_params_nlm=pd.read_csv('D:/PhD/Wagenmakers_fitting.csv').set_index('Subject')
acc_params_ddm=pd.read_csv('D:/PhD/Wagenmakers_fitting_red.csv').set_index('Subject')

# subjects=subjects[4::] #added that to not rerun computations already made
for i in range(1): #len(subjects)
    subject=subjects[i]
    print('Processing subject number {}/{}'.format(subject,np.max(subjects)))
    acc_df=filtered_dat[(filtered_dat.Condition==0) & (filtered_dat.Subject==subject)]
    speed_df=filtered_dat[(filtered_dat.Condition==1) & (filtered_dat.Subject==subject)]
    
    acc_sample = Sample.from_pandas_dataframe(acc_df, rt_column_name="RT", correct_column_name="correct")
    speed_sample=Sample.from_pandas_dataframe(speed_df, rt_column_name="RT", correct_column_name="correct")
    