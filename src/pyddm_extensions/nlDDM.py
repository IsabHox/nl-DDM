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

