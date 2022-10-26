# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:59:55 2022
Fitting my data
@author: ihoxha
"""
#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ddm
from ddm import Model, Fittable
from ddm.sample import Sample
from ddm.models import NoiseConstant, BoundConstant, LossRobustLikelihood, OverlayChain, OverlayNonDecisionUniform, OverlayUniformMixture
from ddm.functions import fit_adjust_model, get_model_loss

import sys
sys.path.append('../../src/')
sys.path.append('../../src/pyddm_extensions/')

from DDM import ddmTwoStimuli
from nlDDM import nlddmTwoStimuli
from OU import OUTwoStimuli
from DWM import DWMTwoStimuli
from extras import ICIntervalRatio, LossByMeans
from utilities import process_binary

import paranoid as pns
pns.settings.Settings.set(enabled=False)

#%% load data (this will be changed after upload of the dataset on Zenodo)
data_file=pd.read_csv('../../data/Binary/RT_data.csv')#r'D:\PhD\data to share\RT_data.csv')
subjects=np.unique(data_file.Subject)

#%% select participant data
for s in range (len(subjects)):
    subject=subjects[s]
    data_raw=data_file[data_file['Subject'] == subject]
    #create the column to assess if the response was correct or not
    data_df=process_binary(data_raw)
    #create the samples readable by PyDDM
    my_samples=Sample.from_pandas_dataframe(data_df, rt_column_name="RT", correct_column_name="correct")

    #%% Here, we create our model
    a=Fittable(minval = .1, maxval = 5)
    my_nlddm=Model(name="non-linear model", drift=nlddmTwoStimuli(k=Fittable(minval = 0.1, maxval = 10),
                                      a=a,
                                      z0=Fittable(minval = -1, maxval=-.5),
                                      z1=Fittable(minval = -1, maxval=-.5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a), 
            #IC = ddm.models.ic.ICUniform(), #change for a smaller window
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
            #overlay=OverlayNonDecisionLR(TndL=Fittable(minval=0.1, maxval=.8),
                                          # TndR=Fittable(minval=0.1, maxval=.8)),# change for a single
            # overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.8),
            #                                   halfwidth=Fittable(minval=0., maxval=0.2)),
            overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                             OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=3.0)

    
    #%% and then we fit it
    fit_adjust_model(my_samples, my_nlddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% then the DDM can be created and fitted 
    
    my_ddm=Model(name="my DDM", drift=ddmTwoStimuli(v0=Fittable(minval = 0.1, maxval = 10),
                                      v1=Fittable(minval = 0.1, maxval=10)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=5)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
            # overlay=OverlayNonDecisionLR(TndL=nlddm_param[tl_ix],
            #                               TndR=nlddm_param[tr_ix]),
            overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                             OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
            dx=0.005,
            dt=0.005,
            T_dur=3.0)
    
    fit_adjust_model(my_samples, my_ddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% We do the same with the OU
    my_ou=Model(name='my OU', drift=OUTwoStimuli(l=Fittable(minval=.1, maxval=10),
                                                 m0=Fittable(minval=.1, maxval=10),
                                                 m1=Fittable(minval=.1, maxval=10)),
                noise=NoiseConstant(noise=.3),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=5)),
                IC=ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
                overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                                 OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
                dx=0.005,
                dt=0.005,
                T_dur=3.0
                )
    
    fit_adjust_model(my_samples, my_ou,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%%
    my_dwm=Model(name='my DWM', drift=DWMTwoStimuli(a=Fittable(minval=.1, maxval=10),
                                                  m0=Fittable(minval=.1, maxval=10),
                                                  m1=Fittable(minval=.1, maxval=10),
                                                  tau=Fittable(minval=.01, maxval=5)),
                noise=NoiseConstant(noise=.3),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=5)),
                IC=ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
                overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                                  OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
                dx=0.005,
                dt=0.005,
                T_dur=3.0
                )
    
    fit_adjust_model(my_samples, my_dwm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% some plotting to enjoy the result
    # plt.figure()
    # ddm.plot.plot_fit_diagnostics(model=my_ou, sample=my_samples)
    # plt.title('nl-DDM fit result for participant {}'.format(subject))
    # plt.figure()
    # ddm.plot.plot_fit_diagnostics(model=my_dwm, sample=my_samples)
    # plt.title('DDM fit result for participant {}'.format(subject))
    
    #%% and on to the computation of the loss
    sample_size=len(my_samples)
    
    #knowing the number of parameters fitted is needed for the BIC
    nparams_nl=9
    nparams_dm=8
    nparams_ou=9
    nparams_dw=10
    
    nl_loss=get_model_loss(my_nlddm, my_samples, lossfunction=LossRobustLikelihood)
    dm_loss=get_model_loss(my_ddm, my_samples, lossfunction=LossRobustLikelihood)
    ou_loss=get_model_loss(my_ou, my_samples, lossfunction=LossRobustLikelihood)
    dw_loss=get_model_loss(my_dwm, my_samples, lossfunction=LossRobustLikelihood)
    
    nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
    dm_bic=np.log(sample_size)*nparams_dm+2*dm_loss
    ou_bic=np.log(sample_size)*nparams_ou+2*ou_loss
    dw_bic=np.log(sample_size)*nparams_dw+2*dw_loss
    
    nl_prediction_performance=get_model_loss(my_nlddm, my_samples, lossfunction=LossByMeans)
    dm_prediction_performance=get_model_loss(my_ddm, my_samples, lossfunction=LossByMeans)
    ou_prediction_performance=get_model_loss(my_ou, my_samples, lossfunction=LossByMeans)
    dw_prediction_performance=get_model_loss(my_dwm, my_samples, lossfunction=LossByMeans)
    
    #%% and save the results
    col_nlDDM=['Subject']
    if s==0:
        col_nlDDM=['Subject']
        col_DDM=['Subject']
        col_OU=['Subject']
        col_DWM=['Subject']
        col_nlDDM.extend(my_nlddm.get_model_parameter_names())
        col_DDM.extend(my_ddm.get_model_parameter_names())
        col_OU.extend(my_ou.get_model_parameter_names())
        col_DWM.extend(my_dwm.get_model_parameter_names())
        nlddm_params=pd.DataFrame(columns=col_nlDDM)
        ddm_params= pd.DataFrame(columns=col_DDM)
        ou_params=pd.DataFrame(columns=col_OU)
        dwm_params=pd.DataFrame(columns=col_DWM)
        performance=pd.DataFrame(columns=['Subject', 'LogLoss (nlDDM)','LogLoss (DDM)',
                                          'LogLoss (OU)','LogLoss (DWM)',
                                          'BIC (nlDDM)','BIC (DDM)','BIC (OU)','BIC (DWM)',
                                          'Perf (nlDDM)','Perf (DDM)','Perf (OU)', 'Perf (DWM)'])
    nlddm_dat, ddm_dat, ou_dat, dwm_dat=[subject], [subject], [subject], [subject]
    nlddm_dat.extend([k.default() for k in my_nlddm.get_model_parameters()])
    ddm_dat.extend([k.default() for k in my_ddm.get_model_parameters()])
    ou_dat.extend([k.default() for k in my_ou.get_model_parameters()])
    dwm_dat.extend([k.default() for k in my_dwm.get_model_parameters()])
    
    nlddm_params.loc[len(nlddm_params.index)]=nlddm_dat
    ddm_params.loc[len(ddm_params.index)]=ddm_dat
    ou_params.loc[len(ou_params.index)]=ou_dat
    dwm_params.loc[len(dwm_params.index)]=dwm_dat
    
    performance.loc[len(performance.index)]=[subject, nl_loss, dm_loss, ou_loss, dw_loss,
                                             nl_bic, dm_bic, ou_bic,dw_bic,
                                              nl_prediction_performance,
                                              dm_prediction_performance,
                                              ou_prediction_performance,
                                              dw_prediction_performance]
    
#%%
nlddm_params.to_csv('../../results/fitting_nlDDM_binary_IC.csv')
ddm_params.to_csv('../../results/fitting_ddm_binary_IC.csv')
ou_params.to_csv('../../results/fitting_ou_binary_IC.csv')
dwm_params.to_csv('../../results/fitting_dwm_binary_IC.csv')

performance.to_csv('../../results/performance_IC.csv')