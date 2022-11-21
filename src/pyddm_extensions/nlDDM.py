# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:42:18 2022
Some classes are needed to fit the models using Py-DDM. These classes are
presented here.
@author: ihoxha
"""

#%% necessary imports
from ddm.models import Drift

#%% nlDDM classes
'''These classes inherit from PyDDM Drift class'''
class nlddmWagenmakers(Drift):
    '''Class needed to fit the nl-DDM drift on Wagenmakers' 2008 dataset, where
    4 classes of words may appear, each generating a different drift term'''
    name="Wagenmakers' nl-DDM"
    required_conditions=["word_type"] #simultaneous fitting of all stimuli
    required_parameters=["k","z1","z2","z3","zNW","a"] 
    def get_drift(self,x,t,conditions, *args, **kwargs):
        if conditions['word_type']==1:
            z=self.z1
        elif conditions['word_type']==2:
            z=self.z2
        elif conditions['word_type']==3:
            z=self.z3
        else:
            z=self.zNW
        return -self.k*(x-z*self.a)*(x-self.a)*(x+self.a)

class nlddmFatigue(Drift):
    name="Wagenmakers with fatigue"
    required_conditions=['word_type','Megablock', 'Condition']
    required_parameters=['k1','k2','k3','k4','k5', #5 k for as many megablocks
                         'z1','z2','z3','zNW', #4 z for as many word types
                         'a0', 'a1'] #2 a for as many conditions
    def get_drift(self, x, t, conditions, *args, **kwargs):
        if conditions['word_type']==1:
            z=self.z1
        elif conditions['word_type']==2:
            z=self.z2
        elif conditions['word_type']==3:
            z=self.z3
        else:
            z=self.zNW
        if conditions['Megablock']==0:
            k=self.k1
        elif conditions['Megablock']==1:
            k=self.k2
        elif conditions['Megablock']==2:
            k=self.k3
        elif conditions['Megablock']==3:
            k=self.k4
        elif conditions['Megablock']==4:
            k=self.k5
        if conditions['Condition']==0:
            a=self.a0
        else:
            a=self.a1
        return -k*(x-z*a)*(x-a)*(x+a) 
    
class nlddmFatigueEarlyLate(Drift):
    name="Wagenmakers with fatigue"
    required_conditions=['word_type','Late', 'Condition']
    required_parameters=['k1','k2', #2 k for as many megablocks
                         'z1','z2','z3','zNW', #4 z for as many word types
                         'a0', 'a1'] #2 a for as many conditions
    def get_drift(self, x, t, conditions, *args, **kwargs):
        if conditions['word_type']==1:
            z=self.z1
        elif conditions['word_type']==2:
            z=self.z2
        elif conditions['word_type']==3:
            z=self.z3
        else:
            z=self.zNW
        if conditions['Late']==0:
            k=self.k1
        elif conditions['Late']==1:
            k=self.k2
        if conditions['Condition']==0:
            a=self.a0
        else:
            a=self.a1
        return -k*(x-z*a)*(x-a)*(x+a)  
    
class nlddmTwoStimuli(Drift):
    '''This class can be called when fitting the drift for an experiment with
    two stimulus types'''
    name="Two stimuli nlDDM"
    required_conditions=["side"] #do the fitting at the same time for the two conditions 0:left, 1:right
    required_parameters=["k","z0","z1","a"] #  0",do we actually need a and x0? We don't need x0: it's accounted for in IC
    def get_drift(self,x,t,conditions, *args, **kwargs):
        if conditions['side']==1:
            z=self.z1
        else:
            z=self.z0
        return -self.k*(x-z*self.a)*(x-self.a)*(x+self.a) #np.sin(t*z)*self.a
    
    
class nlddmDummy(Drift):
    '''This class is called when only one stimulus type exists. It's the class
    we used when creating the dummy figure'''
    name="simple nl-DDM"
    required_parameters=["k","z","a"]
    def get_drift(self, x, t, *args,**kwargs):
        return -self.k*(x-self.z*self.a)*(x-self.a)*(x+self.a)

