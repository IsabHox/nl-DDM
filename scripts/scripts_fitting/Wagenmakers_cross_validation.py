# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:47:36 2023
Cross validation with OU and nl-DDM fitting on Wagenmakers dataset
@author: ihoxha
"""

#%% imports
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import sys
sys.path.append('./../../src/')
sys.path.append('./../../src/pyddm_extensions/')

from utilities import process_Wagenmakers
from nlDDM import nlddmWagenmakers
from OU import OUWagenmakers
from extras import BoundsPerCondition, ICIntervalRatio

from ddm import Model, Fittable
from ddm.sample import Sample
from ddm.models import NoiseConstant, OverlayNonDecision, LossRobustLikelihood

from ddm.functions import fit_adjust_model, get_model_loss

from sklearn.model_selection import train_test_split

#%% iteration definition
def crossvalidation(subject):
    '''Define a function for looping so that cross validation can run in parallel'''
    
    subdat=filtered_dat[(filtered_dat.Subject==subject)]
    subdat['stratification']=10*subdat['Block']+subdat['word_type']
    
    Xtrain, Xtest,_,_=train_test_split(subdat,subdat['word_type'],test_size=.2, random_state=42,stratify=subdat['stratification'])
    my_sample_train=Sample.from_pandas_dataframe(subdat, rt_column_name="RT", correct_column_name="correct")
    my_sample_test=Sample.from_pandas_dataframe(subdat, rt_column_name="RT", correct_column_name="correct")
    
# Instanciate models for fitting the accuracy condition
    a0=Fittable(minval = .1, maxval = 5)
    a1=Fittable(minval = .1, maxval = 5)
    nl=Model(name="my model", drift=nlddmWagenmakers(k=Fittable(minval = .1, maxval = 10),
                                       a0=a0,
                                       a1=a1,
                                       z1=Fittable(minval = -1, maxval=1),#
                                       z2=Fittable(minval = -1, maxval=1),#
                                       z3=Fittable(minval = -1, maxval=1),#
                                       zNW=Fittable(minval = -1, maxval=1)),#
             noise=NoiseConstant(noise=.3),
             bound=BoundsPerCondition(BA=a0, BS=a1), 
             IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)), #changed from x0=0
             # IC = ICPoint(x0=Fittable(minval=0, maxval=0.1)),
             overlay = OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
             dx=0.005,
             dt=0.005,
             T_dur=3.0)
    
    ou=Model(name="my model", drift=OUWagenmakers(l=Fittable(minval = .1, maxval = 10),
                                       m1=Fittable(minval = .1, maxval=10),#
                                       m2=Fittable(minval = .10, maxval=10),#
                                       m3=Fittable(minval = .10, maxval=10),#
                                       mNW=Fittable(minval = .10, maxval=10)),#
             noise=NoiseConstant(noise=.3),
             bound=BoundsPerCondition(B0=a0, B1=a1), 
             IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)), #changed from x0=0
             # IC = ICPoint(x0=Fittable(minval=0, maxval=0.1)),
             overlay = OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
             dx=0.005,
             dt=0.005,
             T_dur=3.0)
    
    print('Processing subject number {}/{}'.format(subject,np.max(subjects)))
    
    fit_adjust_model(my_sample_train, ou,
                          fitting_method="differential_evolution",
                          method="implicit",
                          lossfunction=LossRobustLikelihood, verbose=False)
    fit_adjust_model(my_sample_train, nl,
                          fitting_method="differential_evolution",
                          method="implicit",
                          lossfunction=LossRobustLikelihood, verbose=False)
    
    sample_size=len(my_sample_train)
    test_sample_size=len(my_sample_test)
    
    nparams_nl=10
    nparams_ou=10
    
    test_nl=get_model_loss(nl, my_sample_test)
    test_ou=get_model_loss(ou, my_sample_test)
    test_nl_bic=np.log(test_sample_size)*nparams_nl+2*test_nl
    test_ou_bic=np.log(test_sample_size)*nparams_ou+2*test_ou
    
    nl_loss=nl.fitresult.value()
    ou_loss=ou.fitresult.value()
    
    nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
    ou_bic=np.log(sample_size)*nparams_ou+2*ou_loss
    
    # if s==0:
    col_results=['Subject']
    
    col_results.extend(nl.get_model_parameter_names())
    col_results.extend(ou.get_model_parameter_names())
    col_results.extend(['LogLoss (nlDDM, train)','LogLoss (OU, train)',
                        'BIC (nlDDM, train)', 'BIC (OU, train)',
                        'LogLoss (nlDDM, test)','LogLoss (OU, test)',
                        'BIC (nlDDM, test)', 'BIC (OU, test)', '# train samples', '# test samples'])
    
    results_df=pd.DataFrame(columns=col_results)
    
    # performance=pd.DataFrame(columns=['Subject', 'LogLoss (nlDDM)',
    #                                   'LogLoss (DDM)',
    #                                   'BIC (nlDDM, bounded SP)', 'BIC (DDM)'])
        
        
    result_dat=[subject]
    perf_list=[nl_loss,ou_loss,nl_bic,ou_bic,
               test_nl, test_ou, test_nl_bic, test_ou_bic, sample_size, test_sample_size]
    
    result_dat.extend([k.default() for k in nl.get_model_parameters()])
    result_dat.extend([k.default() for k in ou.get_model_parameters()])
    result_dat.extend(perf_list)
    results_df.loc[len(results_df.index)]=result_dat
    
    
    # performance.loc[len(performance.index)]=perf_list
    results_df.to_csv(f'../../results/crossval_Wagenmakers_noFatigue_{subject}.csv')
    return subject
    
#%% import and process data
column_names=['Subject','Block','Practice','Condition','Stimulus','word_type','response','RT','censor']
wagenmakers_dat=pd.read_csv('../../data/Wagenmakers/SpeedAccData.txt', sep='\s+', header=None, names=column_names)

filtered_dat = process_Wagenmakers(wagenmakers_dat)

gross_subjects=np.unique(filtered_dat['Subject'])
subjects=np.unique(filtered_dat['Subject'])
bad_subjects=[]
for subject in gross_subjects:
    subdat=filtered_dat[(filtered_dat.Subject==subject)]
    nblock=len(np.unique(subdat.Block))
    if nblock!=20:
        print(subject, nblock)
        subjects=np.delete(subjects, np.where(subjects==subject)[0])
        
#%% running the script
Parallel(n_jobs=6)(delayed(crossvalidation)(subject) for subject in subjects)