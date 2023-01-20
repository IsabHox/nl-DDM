# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:27:00 2023

@author: ihoxha
"""

import pandas as pd
import numpy as np

from ddm import Model, Fittable
from ddm.models import NoiseConstant, BoundConstant, DriftConstant, LossRobustLikelihood, OverlayNonDecision
from ddm.functions import fit_adjust_model

import sys
sys.path.append('../../src/')
sys.path.append('../../src/pyddm_extensions/')

from nlDDM import nlddmDummy
from extras import ICIntervalRatio

import paranoid as pns
pns.settings.Settings.set(enabled=False)

import itertools

from ddm import set_N_cpus #let's see if it works though
set_N_cpus(4) #this is used to speed things up and to parallelize

#%% set up parameters
# bounds=[0.2, 0.5, 1, 2]
# starting_points=[0.1, 0.2, 0.5, 0.8]
# sp_variability=[0, 0.05, 0.1, 0.2]
# drifts=[0, 0.5, 0.7, 1]

b_collection=np.linspace(0.2,5,100)
# k_collection=np.linspace(0.1,10,100)
v_collection=np.linspace(0,10,100)
x0_collection=np.linspace(-1,1,100)
sz_collection=np.linspace(0,1,100)

b=1
v=.2
x0=0
sz=0

#%% and loop
count=0
# for [b,x,sz,v] in itertools.product(bounds, starting_points, sp_variability,drifts):
for x0 in x0_collection:
    count+=1
    #first simulate data#then fit the DDM
    my_ddm = Model(name='Simple model',
              drift=DriftConstant(drift=v),
              noise=NoiseConstant(noise=.3),
              bound=BoundConstant(B=b),
              IC=ICIntervalRatio(x0=x0, sz=sz),
              overlay=OverlayNonDecision(nondectime=.3),
              dx=.001, dt=.001, T_dur=2)
    
    sol = my_ddm.solve()
    my_samples = sol.resample(500, seed=42)
    
    a=Fittable(minval=0.1,maxval=5)
    sim_nlddm=Model(name="non-linear model", drift=nlddmDummy(k=Fittable(minval=.1, maxval=10),
                                      a=a,
                                      z=Fittable(minval=-1,maxval=1)),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a),
            IC = ICIntervalRatio(x0=Fittable(minval=-1, maxval=1), sz=Fittable(minval=-1, maxval=1)),
            overlay = OverlayNonDecision(nondectime=.3),
            dx=0.001,
            dt=0.001,#again, a as the bound doesn't work
            T_dur=2)
    
    
    
    fit_adjust_model(my_samples, sim_nlddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #and then onto the saving
    if count==1:
        col_names=['B (DDM)','v (DDM)','x0 (DDM)','sz (DDM)']
        col_names.extend(sim_nlddm.get_model_parameter_names())
        params_df=pd.DataFrame(columns=col_names)
        
    current_params=[b,v,x0,sz]
    current_params.extend([k.default() for k in sim_nlddm.get_model_parameters()])
    params_df.loc[len(params_df.index)]=current_params
    
params_df.to_csv('../../results/fitting_DDM_simulated_from_PyDDM_vary_x0.csv')