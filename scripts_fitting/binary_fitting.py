# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:59:55 2022
Fitting my data
@author: ihoxha
"""
#%% imports
from os import listdir
import pandas as pd

import ddm
from ddm import Model, Fittable
from ddm.model import NoiseConstant, BoundConstant

import sys
sys.path.append('../')
sys.path.append('../pyddm_extensions/')

from DDM import ddmTwoStimuli
from nlDDM import nlddmTwoStimuli
from extras import OverlayNonDecisionLR, ICIntervalRatio

#%% load data
filepaths=listdir('D:/PhD/EEG data/behavioral/')
already_passed=['add','dat','new']
subjects=[f[0:3] for f in filepaths if f[0:3] not in already_passed]#

#%%
for s in range (len(subjects)):
    subject=subjects[s]
    print(subject)
    data_df=pd.read_csv('D:/PhD/EEG data/behavioral/{}.csv'.format(subject))#+filepaths[s]
    stimuli=data_df['Stimulus'].to_numpy()
    RTs=data_df['RT'].to_numpy()
    responses=data_df['Response'].to_numpy()

        
    #%% Here, we create our model
    a=Fittable(minval = .1, maxval = 5)
    m=Model(name="my ADM", drift=nlddmTwoStimuli(k=Fittable(minval = 0.1, maxval = 10),
                                      a=a,
                                      z0=Fittable(minval = -1, maxval=-.5),
                                      z1=Fittable(minval = -1, maxval=-.5)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a), 
            IC = ddm.models.ic.ICUniform(),
            overlay=OverlayNonDecisionLR(TndL=Fittable(minval=0.1, maxval=.8),
                                          TndR=Fittable(minval=0.1, maxval=.8)),#
            
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=2.0)
    
    
    adm_names=m.get_model_parameter_names()
    adm_param=m.get_model_parameters()
    tl_ix=adm_names.index('TndL')
    tr_ix=adm_names.index('TndR')
    m0=Model(name="my DDM", drift=ddmTwoStimuli(v0=Fittable(minval = 0.1, maxval = 10),
                                      v1=Fittable(minval = 0.1, maxval=10)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=5)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
            overlay=OverlayNonDecisionLR(TndL=adm_param[tl_ix],
                                          TndR=adm_param[tr_ix]),
            
            dx=0.005,
            dt=0.005,
            T_dur=2.0)