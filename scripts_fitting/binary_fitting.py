# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:59:55 2022
Fitting my data
@author: ihoxha
"""
#%% imports
import pandas as pd
import numpy as np

import ddm
from ddm import Model, Fittable
from ddm.model import NoiseConstant, BoundConstant

import sys
sys.path.append('./../')
sys.path.append('./../pyddm_extensions/')

from DDM import ddmTwoStimuli
from nlDDM import nlddmTwoStimuli
from extras import OverlayNonDecisionLR, ICIntervalRatio

#%% load data (this will be changed after upload of the dataset on Zenodo)
data_file=pd.read_csv(r'D:\PhD\data to share\RT_data.csv')
subjects=np.unique(data_file.Subject)

#%% select participant data
for s in range (len(subjects)):
    subject=subjects[s]
    data_df=data_file[data_file['Subject'] == subject]
    #stimuli=data_df['Stimulus'].to_numpy()
    #RTs=data_df['RT'].to_numpy()
    #responses=data_df['Response'].to_numpy()


    #%% Here, we create our models
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
    
    #%% and now, the fitting stage
    