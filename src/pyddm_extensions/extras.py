# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:09:05 2022
Other daughter classes that were needed when fitting the nlDDM and DDM
@author: ihoxha
"""
import numpy as np
from ddm import InitialCondition, Overlay, Solution, LossFunction

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