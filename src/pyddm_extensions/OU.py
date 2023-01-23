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

class OUWagenmakers(Drift):
    '''Class needed to fit the nl-DDM drift on Wagenmakers' 2008 dataset, where
    4 classes of words may appear, each generating a different drift term'''
    name="Wagenmakers' nl-DDM"
    required_conditions=["word_type"] #simultaneous fitting of all stimuli
    required_parameters=["l","m1","m2","m3","mNW"] 
    def get_drift(self,x,t,conditions, *args, **kwargs):
        if conditions['word_type']==1:
            m=self.m1
        elif conditions['word_type']==2:
            m=self.m2
        elif conditions['word_type']==3:
            m=self.m3
        else:
            m=self.mNW
        return -self.l*x+m