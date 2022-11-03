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
from ddm.models import NoiseConstant, BoundConstant, LossRobustLikelihood, OverlayChain, OverlayNonDecisionUniform, OverlayUniformMixture, OverlayNonDecision
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
data_file=data_file[data_file['RT']<=2000]
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
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
            #overlay=OverlayNonDecisionLR(TndL=Fittable(minval=0.1, maxval=.8),
                                          # TndR=Fittable(minval=0.1, maxval=.8)),# change for a single
            # overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.8),
            #                                   halfwidth=Fittable(minval=0., maxval=0.2)),
            overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)), #OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                             # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))
                                             #]),
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=2.0)
    # and then we fit it
    fit_adjust_model(my_samples, my_nlddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% Here, we create our model with a uniform distribution
    a2=Fittable(minval = .1, maxval = 5)
    Tnd=data_df['RT'].min()
    nlddm_uniform=Model(name="non-linear model", drift=nlddmTwoStimuli(k=Fittable(minval = 0.1, maxval = 10),
                                      a=a2,
                                      z0=Fittable(minval = -1, maxval=-.5),
                                      z1=Fittable(minval = -1, maxval=-.5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a2), 
            IC = ddm.models.ic.ICUniform(), 
            overlay = OverlayNonDecision(nondectime=Tnd), 
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=2.0)
    
    fit_adjust_model(my_samples, nlddm_uniform,
                  fitting_method="differential_evolution",
                  method="implicit",
                  lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% Here, we create our model with a mixture
    a3=Fittable(minval = .1, maxval = 5)
    nlddm_mix=Model(name="non-linear model", drift=nlddmTwoStimuli(k=Fittable(minval = 0.1, maxval = 10),
                                      a=a3,
                                      z0=Fittable(minval = -1, maxval=-.5),
                                      z1=Fittable(minval = -1, maxval=-.5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a3), 
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
            overlay = OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
                                             OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=2.0)
    
    fit_adjust_model(my_samples, nlddm_mix,
                  fitting_method="differential_evolution",
                  method="implicit",
                  lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% then the DDM can be created and fitted 
    
    my_ddm=Model(name="my DDM", drift=ddmTwoStimuli(v0=Fittable(minval = 0.1, maxval = 5),
                                      v1=Fittable(minval = 0.1, maxval=5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=2)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
            # overlay=OverlayNonDecisionLR(TndL=nlddm_param[tl_ix],
            #                               TndR=nlddm_param[tr_ix]),
            overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),
            # overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
            #                                   OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
            dx=0.005,
            dt=0.005,
            T_dur=2.0)
    
    fit_adjust_model(my_samples, my_ddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% fit the DDM when Tnd is fixed from data
    my_ddm_fixed=Model(name="my DDM", drift=ddmTwoStimuli(v0=Fittable(minval = 0.1, maxval = 5),
                                      v1=Fittable(minval = 0.1, maxval=5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=2)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
            overlay = OverlayNonDecision(nondectime=Tnd),
            dx=0.005,
            dt=0.005,
            T_dur=2.0)
    
    fit_adjust_model(my_samples, my_ddm_fixed,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    #the two other DDMs, that include drift and Tnd variability, will be fitted
    # in fast-dm (bc pyDDM doesn't include precise drift variability fitting)
    
    #%% We do the same with the OU : we'll compare it in B and C
    my_ou=Model(name='my OU', drift=OUTwoStimuli(l=Fittable(minval=.1, maxval=10),
                                                 m0=Fittable(minval=.1, maxval=10),
                                                 m1=Fittable(minval=.1, maxval=10)),
                noise=NoiseConstant(noise=.3),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=5)),
                IC=ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
                overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),#OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                                 #OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))
                                                 #]),
                dx=0.005,
                dt=0.005,
                T_dur=2.0
                )
    
    fit_adjust_model(my_samples, my_ou,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% We do the same with the DWM : we'll compare it in B and C
    my_dwm=Model(name='my DWM', drift=DWMTwoStimuli(a=Fittable(minval=.1, maxval=10),
                                                  m0=Fittable(minval=.1, maxval=10),
                                                  m1=Fittable(minval=.1, maxval=10),
                                                  tau=Fittable(minval=.01, maxval=5)),
                noise=NoiseConstant(noise=.3),
                bound=BoundConstant(B=Fittable(minval=.1, maxval=10)),
                IC=ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
                overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),#OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),
                                                  # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
                dx=0.005,
                dt=0.005,
                T_dur=2.0
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
    nparams_nl=7
    nparams_nu=4
    nparams_nm=8
    
    nparams_dm=8
    nparams_df=5
    
    nparams_ou=7
    nparams_dw=8
    
    nl_loss=get_model_loss(my_nlddm, my_samples, lossfunction=LossRobustLikelihood)
    nu_loss=get_model_loss(nlddm_uniform, my_samples, lossfunction=LossRobustLikelihood)
    nm_loss=get_model_loss(nlddm_mix, my_samples, lossfunction=LossRobustLikelihood)
    dm_loss=get_model_loss(my_ddm, my_samples, lossfunction=LossRobustLikelihood)
    df_loss=get_model_loss(my_ddm_fixed, my_samples, lossfunction=LossRobustLikelihood)
    ou_loss=get_model_loss(my_ou, my_samples, lossfunction=LossRobustLikelihood)
    dw_loss=get_model_loss(my_dwm, my_samples, lossfunction=LossRobustLikelihood)
    # 
    nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
    nu_bic=np.log(sample_size)*nparams_nu+2*nu_loss
    nm_bic=np.log(sample_size)*nparams_nm+2*nm_loss
    dm_bic=np.log(sample_size)*nparams_dm+2*dm_loss
    df_bic=np.log(sample_size)*nparams_df+2*df_loss
    ou_bic=np.log(sample_size)*nparams_ou+2*ou_loss
    dw_bic=np.log(sample_size)*nparams_dw+2*dw_loss
    #
    nl_pp=get_model_loss(my_nlddm, my_samples, lossfunction=LossByMeans)
    nu_pp=get_model_loss(nlddm_uniform, my_samples, lossfunction=LossByMeans)
    nm_pp=get_model_loss(nlddm_mix, my_samples, lossfunction=LossByMeans)
    dm_pp=get_model_loss(my_ddm, my_samples, lossfunction=LossByMeans)
    df_pp=get_model_loss(my_ddm_fixed, my_samples, lossfunction=LossByMeans)
    ou_pp=get_model_loss(my_ou, my_samples, lossfunction=LossByMeans)
    dw_pp=get_model_loss(my_dwm, my_samples, lossfunction=LossByMeans)
    
    perf_list=[subject,nl_loss,nu_loss,nm_loss,dm_loss,df_loss,ou_loss,dw_loss,
               nl_bic,nu_bic,nm_bic,dm_bic,df_bic,ou_bic,dw_bic,
               nl_pp,nu_pp,nm_pp,dm_pp,df_pp,ou_pp,dw_pp]
    
    #%% and save the results
    col_nlDDM=['Subject']
    if s==0:
        col_nlDDM=['Subject']
        col_nu=['Subject']
        col_nm=['Subject']
        col_DDM=['Subject']
        col_df=['Subject']
        col_OU=['Subject']
        col_DWM=['Subject']
        
        col_nlDDM.extend(my_nlddm.get_model_parameter_names())
        col_nu.extend(nlddm_uniform.get_model_parameter_names())
        col_nm.extend(nlddm_mix.get_model_parameter_names())
        col_DDM.extend(my_ddm.get_model_parameter_names())
        col_df.extend(my_ddm_fixed.get_model_parameter_names())
        col_OU.extend(my_ou.get_model_parameter_names())
        col_DWM.extend(my_dwm.get_model_parameter_names())
        
        nlddm_params=pd.DataFrame(columns=col_nlDDM)
        nu_params=pd.DataFrame(columns=col_nu)
        nm_params=pd.DataFrame(columns=col_nm)
        ddm_params= pd.DataFrame(columns=col_DDM)
        df_params=pd.DataFrame(columns=col_df)
        ou_params=pd.DataFrame(columns=col_OU)
        dwm_params=pd.DataFrame(columns=col_DWM)
        
        performance=pd.DataFrame(columns=['Subject', 'LogLoss (nlDDM, bounded SP)', 'LogLoss (nlDDM, uniform SP, Tnd from data)', 'LogLoss (nlDDM, mixture)',
                                          'LogLoss (DDM)', 'LogLoss (DDM, Tnd from data)',
                                          'LogLoss (OU)','LogLoss (DWM)',
                                          'BIC (nlDDM, bounded SP)', 'BIC (nlDDM, uniform SP, Tnd from data)', 'BIC (nlDDM, mixture)','BIC (DDM)', 'BIC (DDM, Tnd from data)',
                                          'BIC (OU)','BIC (DWM)',
                                          'Perf (nlDDM, bounded SP)','Perf (nlDDM, uniform SP, Tnd from data)','Perf (nlDDM, mixture)',
                                          'Perf (DDM)','Perf (DDM, Tnd from data)',
                                          'Perf (OU)', 'Perf (DWM)'])
        
        
    nlddm_dat, nu_dat, nm_dat, ddm_dat, df_dat, ou_dat, dwm_dat=[subject], [subject], [subject], [subject], [subject], [subject], [subject]
    
    nlddm_dat.extend([k.default() for k in my_nlddm.get_model_parameters()])
    nu_dat.extend([k.default() for k in nlddm_uniform.get_model_parameters()])
    nm_dat.extend([k.default() for k in nlddm_mix.get_model_parameters()])
    ddm_dat.extend([k.default() for k in my_ddm.get_model_parameters()])
    df_dat.extend([k.default() for k in my_ddm_fixed.get_model_parameters()])
    ou_dat.extend([k.default() for k in my_ou.get_model_parameters()])
    dwm_dat.extend([k.default() for k in my_dwm.get_model_parameters()])
    
    nlddm_params.loc[len(nlddm_params.index)]=nlddm_dat
    nu_params.loc[len(nu_params.index)]=nu_dat
    nm_params.loc[len(nm_params.index)]=nm_dat
    ddm_params.loc[len(ddm_params.index)]=ddm_dat
    df_params.loc[len(df_params.index)]=df_dat
    ou_params.loc[len(ou_params.index)]=ou_dat
    dwm_params.loc[len(dwm_params.index)]=dwm_dat
    
    performance.loc[len(performance.index)]=perf_list
    
#%%
nlddm_params.to_csv('../../results/fitting_nlDDM_binary_bounded.csv')
nu_params.to_csv('../../results/fitting_nlDDM_binary_uniform.csv')
nm_params.to_csv('../../results/fitting_nlDDM_binary_mixture.csv')
ddm_params.to_csv('../../results/fitting_ddm_binary_bounded.csv')
df_params.to_csv('../../results/fitting_ddm_binary_TndFixed.csv')
ou_params.to_csv('../../results/fitting_ou_binary_bounded.csv')
dwm_params.to_csv('../../results/fitting_dwm_binary_bounded.csv')

performance.to_csv('../../results/performance_multiple_spec.csv')