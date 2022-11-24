# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:38:58 2022

@author: ihoxha
"""

import pandas as pd
import numpy as np
import sys
import zipfile

sys.path.append('./../../src/')
sys.path.append('./../../src/pyddm_extensions/')

from utilities import process_Wagenmakers
from nlDDM import nlddmFatigueEarlyLate, nlddmFatigueIsA
from DDM import ddmWagenmakers
from extras import LossByMeans,BoundsPerCondition, BoundsPerFatigueEarlyLate, BoundsPerConditionFatigueIsA

from ddm import Model, Fittable
from ddm.sample import Sample
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayUniformMixture, OverlayNonDecision, LossRobustLikelihood, Drift, ICPoint
from ddm.functions import fit_adjust_model, get_model_loss
# import ddm.plot

# import matplotlib.pyplot as plt

# import urllib.request
from joblib import Parallel, delayed
#%% Import data
# try:
#     !wget https://www.ejwagenmakers.com/Code/2008/LexDecData.zip -P ../../data/Wagenmakers/
# except:
    
# def download_url(url, save_path):
#     with urllib.request.urlopen(url) as dl_file:
#         with open(save_path, 'wb') as out_file:
#             out_file.write(dl_file.read())
# download_url('https://www.ejwagenmakers.com/Code/2008/LexDecData.zip','../../data/Wagenmakers/LexDecData.zip')
        
# with zipfile.ZipFile('../../data/Wagenmakers/LexDecData.zip','r') as zipObj:
#     myfile=zipfile.ZipFile.extract(zipObj,'SpeedAccData.txt','../../data/Wagenmakers/')
    
column_names=['Subject','Block','Practice','Condition','Stimulus','word_type','response','RT','censor']
wagenmakers_dat=pd.read_csv('../../data/Wagenmakers/SpeedAccData.txt', sep='\s+', header=None, names=column_names)

#%% Preprocess data
filtered_dat = process_Wagenmakers(wagenmakers_dat)

#%% check which subjects didn't perform all the blocks:
gross_subjects=np.unique(filtered_dat['Subject'])
subjects=np.unique(filtered_dat['Subject'])
bad_subjects=[]
for subject in gross_subjects:
    subdat=filtered_dat[(filtered_dat.Subject==subject)]
    nblock=len(np.unique(subdat.Block))
    if nblock!=20:
        print(subject, nblock)
        subjects=np.delete(subjects, np.where(subjects==subject)[0])
        
#%% Then, we can loop across subjects or just pick one subject
# s=0
# subjects=[12]
# for subject in subjects:
def fitting_module(subject):
    # s=np.where(subjects==subject)[0][0]
    subdat=filtered_dat[(filtered_dat.Subject==subject)]
    my_sample=Sample.from_pandas_dataframe(subdat, rt_column_name="RT", correct_column_name="correct")
    
#%% Instanciate models for fitting the accuracy condition
    a0=Fittable(minval = .1, maxval = 5)
    a1=Fittable(minval = .1, maxval = 5)
    non_lin_model_acc=Model(name="my model", drift=nlddmFatigueIsA(k1=Fittable(minval = .1, maxval = 10),
                                        k2=Fittable(minval = .1, maxval = 10),
                                       a0=a0,
                                       a1=a1,
                                       z1=Fittable(minval = -1, maxval=1),#
                                       z2=Fittable(minval = -1, maxval=1),#
                                       z3=Fittable(minval = -1, maxval=1),#
                                       zNW=Fittable(minval = -1, maxval=1)),#
             noise=NoiseConstant(noise=.3),
             bound=BoundsPerConditionFatigueIsA(B0=a0, B1=a1), 
             # IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)), #changed from x0=0
             IC = ICPoint(x0=Fittable(minval=0, maxval=0.1)),
             overlay = OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
                                              # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))
                                              ]),
             dx=0.005,
             dt=0.005,
             T_dur=3.0)
    
    ddm_model_acc=Model(name="my DDM", drift=ddmWagenmakers(vNW=Fittable(minval = .01, maxval = 1), 
                                          v1=Fittable(minval = 0.01, maxval=1),
                                          v2=Fittable(minval = 0.01, maxval=1),
                                          v3=Fittable(minval = 0.01, maxval=1)),
                noise=NoiseConstant(noise=.3),
                bound=BoundsPerFatigueEarlyLate(BA1=Fittable(minval=0.1,maxval=1),
                                       BA2=Fittable(minval=0.1,maxval=1),
                                       BS1=Fittable(minval=0.1,maxval=1),
                                       BS2=Fittable(minval=0.1,maxval=1)),
                # IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
                IC = ICPoint(x0=Fittable(minval=0, maxval=0.1)),
                # overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                overlay = OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
                                                 # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))
                                                 ]), 
                dx=0.005,
                dt=0.005,
                T_dur=3.0)
      
    #%% and on to the fitting                  
    print('Processing subject number {}/{}'.format(subject,np.max(subjects)))
    
    fit_adjust_model(my_sample, ddm_model_acc,
                          fitting_method="differential_evolution",
                          method="implicit",
                          lossfunction=LossRobustLikelihood, verbose=False)
    fit_adjust_model(my_sample, non_lin_model_acc,
                          fitting_method="differential_evolution",
                          method="implicit",
                          lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% Plotting the result
    # plt.figure()
    # ddm.plot.plot_fit_diagnostics(model=non_lin_model_acc, sample=acc_sample)
    # plt.figure()
    # ddm.plot.plot_fit_diagnostics(model=ddm_model_acc, sample=acc_sample)
    
    #%% computing the fitting performance
    sample_size=len(my_sample)
    
    #knowing the number of parameters fitted is needed for the BIC
    nparams_nl=10
    nparams_dm=10
    
    nl_loss=non_lin_model_acc.fitresult.value()
    dm_loss=ddm_model_acc.fitresult.value()
    
    nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
    dm_bic=np.log(sample_size)*nparams_dm+2*dm_loss
    
    nl_prediction_performance=0#get_model_loss(non_lin_model_acc, my_sample, lossfunction=LossByMeans)
    dm_prediction_performance=0#get_model_loss(ddm_model_acc, my_sample, lossfunction=LossByMeans)
    
    #%% and save the results
    # if s==0:
    col_results=['Subject']
    
    col_results.extend(non_lin_model_acc.get_model_parameter_names())
    col_results.extend(ddm_model_acc.get_model_parameter_names())
    
    results_df=pd.DataFrame(columns=col_results)
    
    performance=pd.DataFrame(columns=['Subject', 'LogLoss (nlDDM)',
                                      'LogLoss (DDM)',
                                      'BIC (nlDDM, bounded SP)', 'BIC (DDM)',
                                      'Perf (nlDDM)','Perf (DDM)'])
        
        
    result_dat=[subject]
    
    result_dat.extend([k.default() for k in non_lin_model_acc.get_model_parameters()])
    result_dat.extend([k.default() for k in ddm_model_acc.get_model_parameters()])
    
    results_df.loc[len(results_df.index)]=result_dat
    
    perf_list=[subject,nl_loss,dm_loss,
               nl_bic,dm_bic,
               nl_prediction_performance,dm_prediction_performance]
    performance.loc[len(performance.index)]=perf_list
    # s+=1

#%%
    results_df.to_csv(f'../../results/fitting_Wagenmakers_fatigue_EL_a_{subject}.csv')
    
    performance.to_csv(f'../../results/performance_Wagenmakers_fatigue_EL_a_{subject}.csv')
    return subject
    
Parallel(n_jobs=17)(delayed(fitting_module)(subject) for subject in subjects)