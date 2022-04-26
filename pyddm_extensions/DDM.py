# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:00:48 2022
Drift classes created to fit the DDM on our datasets
@author: ihoxha
"""

from ddm.models import Drift

class DDM_Wagenmakers(Drift):
    '''Class needed to fit the nl-DDM drift on Wagenmakers' 2008 dataset, where
    4 classes of words may appear, each generating a different drift term'''
    name="Wagenmakers' DDM"
    required_conditions=["word_type"]
    required_parameters=["vNW",'v1','v2', 'v3'] #one drift per word type
    def get_drift(self,t,conditions,*args,**kwargs):
        drift=(self.v1 if conditions['word_type']==1 
               else self.v2 if conditions['word_type']==2 
               else self.v3 if conditions['word_type']==3 
               else self.vNW)
        return drift
    
class DDM_two_stimuli(Drift):
    '''This class can be called when fitting the drift for an experiment with
    two stimulus types'''
    name="Two stimuli DDM"
    required_conditions=["side"]
    required_parameters=["v0",'v1'] #two drifts, one per stimulus
    def get_drift(self,t,conditions,*args,**kwargs):
        drift=(self.v0 if conditions['side']==0 else self.v1)
        return drift