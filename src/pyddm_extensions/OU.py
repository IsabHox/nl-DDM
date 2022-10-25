# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 08:25:01 2022
Ornstein-Uhlenback process
@author: ihoxha
"""
#%% necessary imports
from ddm.models import Drift

#%% nlDDM classes
'''These classes inherit from PyDDM Drift class'''
class OUTwoStimuli(Drift):
    name="Two stimuli Ornstein-Uhlenbeck"
    required_conditions=["side"] #do the fitting at the same time for the two conditions 0:left, 1:right
    required_parameters=["l","m0","m1"] #  0",do we actually need a and x0? We don't need x0: it's accounted for in IC
    def get_drift(self,x,t,conditions, *args, **kwargs):
        if conditions['side']==1:
            m=self.m1
        else:
            m=self.m0
        return self.l*x+m
