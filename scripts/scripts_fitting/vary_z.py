# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:45:16 2023
Vary z
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

a_collection=np.linspace(0.2,5,100)
k_collection=np.linspace(0.1,10,100)
z_collection=np.linspace(-1,1,100)
x0_collection=np.linspace(-1,1,100)
sz_collection=np.linspace(0,1,100)

a=1
k=5
z=0
x0=0
sz=0

#%% and loop
count=0
# for [b,x,sz,v] in itertools.product(bounds, starting_points, sp_variability,drifts):
for z in z_collection:
    count+=1
    #first simulate data
    a=Fittable(minval = .1, maxval = 5)
    sim_nlddm=Model(name="non-linear model", drift=nlddmDummy(k=k,
                                      a=a,
                                      z=z),
            noise=NoiseConstant(noise=.3),
            bound=BoundConstant(B=a),
            IC = ICIntervalRatio(x0=x0, sz=sz),
            overlay = OverlayNonDecision(nondectime=.3),
            dx=0.001,
            dt=0.001,#again, a as the bound doesn't work
            T_dur=2)
    sol = sim_nlddm.solve()
    my_samples = sol.resample(500, seed=42)
    
    #then fit the DDM
    my_ddm = Model(name='Simple model',
              drift=DriftConstant(drift=Fittable(minval = .01, maxval = 10)),
              noise=NoiseConstant(noise=.3),
              bound=BoundConstant(B=Fittable(minval=0.1,maxval=10)),
              IC=ICIntervalRatio(x0=Fittable(minval=-1,maxval=1), sz=Fittable(minval=-1,maxval=1)),
              overlay=OverlayNonDecision(nondectime=.3),
              dx=.001, dt=.001, T_dur=2)
    

    
    fit_adjust_model(my_samples, my_ddm,
                      fitting_method="differential_evolution",
                      method="implicit",
                      lossfunction=LossRobustLikelihood, verbose=False)
    
    #and then onto the saving
    if count==1:
        col_names=['a (nl-DDM)','k (nl-DDM)','z (nl-DDM)','x0 (nl-DDM)','sz (nl-DDM)']
        col_names.extend(my_ddm.get_model_parameter_names())
        params_df=pd.DataFrame(columns=col_names)
        
    current_params=[a,k,z,x0,sz]
    current_params.extend([k.default() for k in my_ddm.get_model_parameters()])
    params_df.loc[len(params_df.index)]=current_params
    
params_df.to_csv('../../results/fitting_nlDDM_simulated_from_PyDDM_vary_z.csv')