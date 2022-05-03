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
from ddm.models import NoiseConstant, BoundConstant, LossRobustLikelihood
from ddm.functions import fit_adjust_model

import sys
sys.path.append('./../../src/')
sys.path.append('./../../src/pyddm_extensions/')

from DDM import ddmTwoStimuli
from nlDDM import nlddmTwoStimuli
from extras import OverlayNonDecisionLR, ICIntervalRatio
from utilities import process_binary

#%% load data (this will be changed after upload of the dataset on Zenodo)
data_file=pd.read_csv(r'D:\PhD\data to share\RT_data.csv')
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
            IC = ddm.models.ic.ICUniform(),
            overlay=OverlayNonDecisionLR(TndL=Fittable(minval=0.1, maxval=.8),
                                          TndR=Fittable(minval=0.1, maxval=.8)),#
            
            dx=0.005,
            dt=0.005,#again, a as the bound doesn't work
            T_dur=2.0)

    
    #%% and then we fit it
    fit_adjust_model(my_samples, my_nlddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #%% then the DDM can be created and fitted 
    nlddm_names=my_nlddm.get_model_parameter_names()
    nlddm_param=my_nlddm.get_model_parameters()
    tl_ix=nlddm_names.index('TndL')
    tr_ix=nlddm_names.index('TndR')
    
    my_ddm=Model(name="my DDM", drift=ddmTwoStimuli(v0=Fittable(minval = 0.1, maxval = 10),
                                      v1=Fittable(minval = 0.1, maxval=10)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=Fittable(minval=0.1,maxval=5)),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0, maxval=1)),
            overlay=OverlayNonDecisionLR(TndL=nlddm_param[tl_ix],
                                          TndR=nlddm_param[tr_ix]),
            
            dx=0.005,
            dt=0.005,
            T_dur=2.0)
    
    #%% finally, some plotting !
    plt.figure()
    ddm.plot.plot_fit_diagnostics(model=my_nlddm, sample=my_samples)
    plt.title('nl-DDM fit result for participant {}'.format(subject))
    plt.figure()
    ddm.plot.plot_fit_diagnostics(model=my_ddm, sample=my_samples)
    plt.title('DDM fit result for participant {}'.format(subject))