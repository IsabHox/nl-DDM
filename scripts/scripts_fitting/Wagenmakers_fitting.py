# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:36:39 2022
Fitting Wagenmakers dataset
@author: ihoxha
"""

import pandas as pd
import numpy as np
import sys
import zipfile

sys.path.append('./../../src/')
sys.path.append('./../../src/pyddm_extensions/')

from utilities import process_Wagenmakers
from nlDDM import nlddmWagenmakers
from DDM import ddmWagenmakers
from extras import ICIntervalRatio, LossByMeans

from ddm import Model, Fittable
from ddm.sample import Sample
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayUniformMixture, OverlayNonDecision, OverlayNonDecisionUniform, LossRobustLikelihood, Drift 
from ddm.functions import fit_adjust_model, get_model_loss
import ddm.plot

import matplotlib.pyplot as plt

#%% Import data
!wget https://www.ejwagenmakers.com/Code/2008/LexDecData.zip -P ../../data/Wagenmakers/

with zipfile.ZipFile('../../data/Wagenmakers/LexDecData.zip','r') as zipObj:
    myfile=zipfile.ZipFile.extract(zipObj,'SpeedAccData.txt','../../data/Wagenmakers/')
    
column_names=['Subject','Block','Practice','Condition','Stimulus','word_type','response','RT','censor']
wagenmakers_dat=pd.read_csv('../../data/Wagenmakers/SpeedAccData.txt', sep='\s+', header=None, names=column_names)

#%% Preprocess data
filtered_dat = process_Wagenmakers(wagenmakers_dat)

#%% Then, we can loop across subjects or just pick one subject
subjects=np.unique(filtered_dat['Subject'])
subject=subjects[0]

acc_df=filtered_dat[(filtered_dat.Condition==0) & (filtered_dat.Subject==subject)]
speed_df=filtered_dat[(filtered_dat.Condition==1) & (filtered_dat.Subject==subject)]

acc_sample = Sample.from_pandas_dataframe(acc_df, rt_column_name="RT", correct_column_name="correct")
speed_sample=Sample.from_pandas_dataframe(speed_df, rt_column_name="RT", correct_column_name="correct")

#%% Instanciate models for fitting the accuracy condition
a_nl=Fittable(minval = .1, maxval = 5)

non_lin_model_acc=Model(name="my model", drift=nlddmWagenmakers(k=Fittable(minval = .1, maxval = 10),
                                   a=a_nl,
                                   z1=Fittable(minval = -1, maxval=1),#
                                   z2=Fittable(minval = -1, maxval=1),#
                                   z3=Fittable(minval = -1, maxval=1),#
                                   zNW=Fittable(minval = -1, maxval=1)),#
         noise=NoiseConstant(noise=Fittable(minval=0.1, maxval=3)),
         bound=BoundConstant(B=a_nl), 
         IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)), #changed from x0=0
         # overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)), #changed to uniform
         overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                          OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]) 
         dx=0.005,
         dt=0.005,
         T_dur=3.0)

ddm_model_acc=Model(name="my DDM", drift=ddmWagenmakers(vNW=Fittable(minval = .01, maxval = 1), 
                                      v1=Fittable(minval = 0.01, maxval=1),
                                      v2=Fittable(minval = 0.01, maxval=1),
                                      v3=Fittable(minval = 0.01, maxval=1)),
            noise=NoiseConstant(noise=Fittable(minval = 0.1, maxval = 3)),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=1)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
            # overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
            overlay = OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                             OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]) 
            dx=0.005,
            dt=0.005,
            T_dur=3.0)
  
#%% and on to the fitting                  
print('Processing subject number {}/{}'.format(subject,np.max(subjects)))

fit_adjust_model(acc_sample, ddm_model_acc,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
fit_adjust_model(acc_sample, non_lin_model_acc,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)

#%% Plotting the result
plt.figure()
ddm.plot.plot_fit_diagnostics(model=non_lin_model_acc, sample=acc_sample)
plt.figure()
ddm.plot.plot_fit_diagnostics(model=ddm_model_acc, sample=acc_sample)

#%% computing the fitting performance
sample_size=len(acc_sample)

#knowing the number of parameters fitted is needed for the BIC
nparams_nl=9
nparams_dm=10

nl_loss=get_model_loss(non_lin_model_acc, acc_sample, lossfunction=LossRobustLikelihood)
dm_loss=get_model_loss(ddm_model_acc, acc_sample, lossfunction=LossRobustLikelihood)

nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
dm_bic=np.log(sample_size)*nparams_dm+2*dm_loss

nl_prediction_performance=get_model_loss(non_lin_model_acc, acc_sample, lossfunction=LossByMeans)
dm_prediction_performance=get_model_loss(ddm_model_acc, acc_sample, lossfunction=LossByMeans)


#%% repeat the same process for the speed condition
nl_param_names=non_lin_model_acc.get_model_parameter_names()
dm_param_names=ddm_model_acc.get_model_parameter_names()

nl_acc_params=non_lin_model_acc.get_model_parameters()
dm_acc_params=ddm_model_acc.get_model_parameters()

non_lin_model_speed=Model(name="my model", drift=nlddmWagenmakers(k=nl_acc_params[nl_param_names.index('k')].default(),
                                   a=a_nl,
                                   z1=nl_acc_params[nl_param_names.index('z1')].default(),#
                                   z2=nl_acc_params[nl_param_names.index('z2')].default(),#
                                   z3=nl_acc_params[nl_param_names.index('z3')].default(),#
                                   zNW=nl_acc_params[nl_param_names.index('zNW')].default()),#
         noise=NoiseConstant(noise=nl_acc_params[nl_param_names.index('noise')].default()),
         bound=BoundConstant(B=a_nl), 
         IC = ICIntervalRatio(x0=0, sz=Fittable(minval=0., maxval=1.)),
         overlay=OverlayNonDecision(nondectime=Fittable(minval = 0.1, maxval = 0.8)),

         dx=0.005,
         dt=0.005,
         T_dur=3.0)

ddm_model_speed=Model(name="my DDM", drift=ddmWagenmakers(vNW=dm_acc_params[dm_param_names.index('vNW')].default(), 
                                      v1=dm_acc_params[dm_param_names.index('v1')].default(),
                                      v2=dm_acc_params[dm_param_names.index('v2')].default(),
                                      v3=dm_acc_params[dm_param_names.index('v3')].default()),
            noise=NoiseConstant(noise=dm_acc_params[dm_param_names.index('noise')].default()),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=1)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=dm_acc_params[dm_param_names.index('sz')].default()),
            overlay=OverlayNonDecisionUniform(nondectime=Fittable(minval=0.1, maxval=0.8), halfwidth=dm_acc_params[dm_param_names.index('halfwidth')].default()),
            dx=0.005,
            dt=0.005,
            T_dur=3.0)
                    
print('Processing subject number {}/{}'.format(subject,np.max(subjects)))

fit_adjust_model(speed_sample, ddm_model_speed,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
fit_adjust_model(speed_sample, non_lin_model_speed,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)

plt.figure()
ddm.plot.plot_fit_diagnostics(model=non_lin_model_speed, sample=speed_sample)
plt.figure()
ddm.plot.plot_fit_diagnostics(model=ddm_model_speed, sample=speed_sample)

sample_size=len(speed_sample)

#knowing the number of parameters fitted is needed for the BIC
nparams_nl=9
nparams_dm=10

nl_loss=get_model_loss(non_lin_model_speed, speed_sample, lossfunction=LossRobustLikelihood)
dm_loss=get_model_loss(ddm_model_speed, speed_sample, lossfunction=LossRobustLikelihood)

nl_bic=np.log(sample_size)*nparams_nl+2*nl_loss
dm_bic=np.log(sample_size)*nparams_dm+2*dm_loss

nl_prediction_performance=get_model_loss(non_lin_model_speed, speed_sample, lossfunction=LossByMeans)
dm_prediction_performance=get_model_loss(ddm_model_speed, speed_sample, lossfunction=LossByMeans)