# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:45:12 2022

@author: ihoxha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ddm
from ddm import Model, Fittable
from ddm.sample import Sample
from ddm.models import NoiseConstant, BoundConstant, LossRobustLikelihood, OverlayNonDecision
from ddm.functions import fit_adjust_model

import sys
sys.path.append('../../src/')
sys.path.append('../../src/pyddm_extensions/')

from nlDDM import nlddmDummy
from extras import ICIntervalRatio
from utilities import import_simulated

import paranoid as pns
pns.settings.Settings.set(enabled=False)

#%% 
parameters=['b','st','sv','sz','v','z']

for param in parameters:
    for i in range (1,6):
        datapath=f'../../../simulations/{param}{i}/sim_0.lst'
        data, Tdur=import_simulated(datapath)
        my_samples=Sample.from_pandas_dataframe(data, rt_column_name="RT", correct_column_name="Correct")
        
        a=Fittable(minval = .1, maxval = 5)
        sim_nlddm=Model(name="non-linear model", drift=nlddmDummy(k=Fittable(minval = 0.1, maxval = 10),
                                          a=a,
                                          z=Fittable(minval = -1, maxval=-.5)),
                noise=NoiseConstant(noise=.3),
                bound=BoundConstant(B=a),
                IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
                overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),#OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                                  # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
                dx=0.005,
                dt=0.005,#again, a as the bound doesn't work
                T_dur=Tdur)
        
        fit_adjust_model(my_samples, sim_nlddm,
                          fitting_method="differential_evolution",
                          method="implicit",
                          lossfunction=LossRobustLikelihood, verbose=False)
        
        if param=='b' and i==1:
            col_nlDDM=['Parameters']
            col_nlDDM.extend(sim_nlddm.get_model_parameter_names())
            nlddm_params=pd.DataFrame(columns=col_nlDDM)
        nlddm_dat=[f'{param}{i}']
        nlddm_dat.extend([k.default() for k in sim_nlddm.get_model_parameters()])
        
        nlddm_params.loc[len(nlddm_params.index)]=nlddm_dat

datapath=f'../../../simulations/no_var/sim_0.lst'
data, Tdur=import_simulated(datapath)
my_samples=Sample.from_pandas_dataframe(data, rt_column_name="RT", correct_column_name="Correct")

a=Fittable(minval = .1, maxval = 5)
sim_nlddm=Model(name="non-linear model", drift=nlddmDummy(k=Fittable(minval = 0.1, maxval = 10),
                                  a=a,
                                  z=Fittable(minval = -1, maxval=-.5)),
        noise=NoiseConstant(noise=.3),
        bound=BoundConstant(B=a),
        IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=0., maxval=1.)),
        overlay = OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=0.8)),#OverlayChain(overlays=[OverlayNonDecisionUniform(nondectime=Fittable(minval = 0.1, maxval = 0.8), halfwidth=Fittable(minval=0., maxval=0.2)),
                                          # OverlayUniformMixture(umixturecoef=Fittable(minval=0, maxval=.1))]),
        dx=0.005,
        dt=0.005,#again, a as the bound doesn't work
        T_dur=Tdur)

fit_adjust_model(my_samples, sim_nlddm,
                  fitting_method="differential_evolution",
                  method="implicit",
                  lossfunction=LossRobustLikelihood, verbose=False)
nlddm_dat=[f'{param}{i}']
nlddm_dat.extend([k.default() for k in sim_nlddm.get_model_parameters()])


nlddm_params.to_csv('../../results/fitting_nlDDM_simulated.csv')