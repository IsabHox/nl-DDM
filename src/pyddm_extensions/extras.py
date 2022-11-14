# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:09:05 2022
Other daughter classes that were needed when fitting the nlDDM and DDM
@author: ihoxha
"""
import numpy as np
from ddm import InitialCondition, Overlay, Solution, LossFunction, Bound

class ICIntervalRatio(InitialCondition):
    name = "Starting point range as a ratio of the distance between bounds."
    required_parameters = ["x0","sz"]
    def get_IC(self, x, dx, conditions):
        x0 = self.x0/2 + .5 #rescale to between 0 and 1 (x0 is in [-1,1])
        shift_i = int((len(x)-1)*x0)
        #create the interval over which we compute the variation
        min_interval_size=min(shift_i,len(x)-shift_i-1) 
        sz=int(self.sz*min_interval_size)
        assert shift_i-sz >= 0 and shift_i+sz < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i-sz:shift_i+sz+1] = 1./(2*sz+1) # Initial condition at x=x0*2*B.
        return pdf
    
    
class OverlayNonDecisionLR(Overlay):
    '''Creates a non-decision time per stimulus response side. 
    This is taken almost identically from PyDDM recipe'''
    name = "Separate non-decision time for left and right sides"
    required_parameters = ["TndL", "TndR"]
    required_conditions = ["side"] # Side coded as 0=Left or 1=Right
    def apply(self, solution):
        # Check parameters and conditions
        assert solution.conditions['side'] in [0, 1], "Invalid side"
        # Unpack solution object
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        # Compute non-decision time
        ndtime = self.TndL if cond['side'] == 0 else self.TndR
        shifts = int(ndtime/m.dt) # truncate
        # Shift the distribution
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        if shifts > 0:
            newcorr[shifts:] = corr[:-shifts]
            newerr[shifts:] = err[:-shifts]
        elif shifts < 0:
            newcorr[:shifts] = corr[-shifts:]
            newerr[:shifts] = err[-shifts:]
        else:
            newcorr = corr
            newerr = err
        return Solution(newcorr, newerr, m, cond, undec)
    
class LossByMeans(LossFunction):
    name = "Mean RT and accuracy"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
    def loss(self, model):
        sols = self.cache_by_conditions(model)
        MSE = 0
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            c = frozenset(comb.items())
            subset = self.sample.subset(**comb)
            MSE += (sols[c].prob_correct() - subset.prob_correct())**2
            if sols[c].prob_correct() > 0:
                MSE += (sols[c].mean_decision_time() - np.mean(list(subset)))**2
        return MSE
    
class BoundsPerFatigue(Bound):
    name='Boundary depending on Megablock and condition'
    required_parameters = ["BA1","BA2","BA3","BA4","BA5",
                           "BS1","BS2","BS3","BS4","BS5"]
    required_conditions = ['Megablock','Condition']
    def get_bound(self, conditions, *args, **kwargs):
        if conditions['Condition']==0:
            if conditions['Megablock']+1==1:
                return self.BA1
            elif conditions['Megablock']+1==2:
                return self.BA2
            elif conditions['Megablock']+1==3:
                return self.BA3
            elif conditions['Megablock']+1==4:
                return self.BA4
            elif conditions['Megablock']+1==5:
                return self.BA5
        elif conditions['Condition']==1:
            if conditions['Megablock']+1==1:
                return self.BS1
            elif conditions['Megablock']+1==2:
                return self.BS2
            elif conditions['Megablock']+1==3:
                return self.BS3
            elif conditions['Megablock']+1==4:
                return self.BS4
            elif conditions['Megablock']+1==5:
                return self.BS5
    
class BoundsPerCondition(Bound):
    name='Boundary depending on Megablock and condition'
    required_parameters = ["BA","BS"]
    required_conditions = ['Condition']
    def get_bound(self, conditions, *args, **kwargs):
        if conditions['Condition']==0:
            return self.BA
        elif conditions['Condition']==1:
            return self.BS