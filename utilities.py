# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:29:12 2022
Other utilities needed to describe the nl-ddm
@author: ihoxha
"""
import numpy as np

def potential(x,z,a,k):
    '''the potential function associated with the nl-DDM
    
    Parameters
    -------------
    x (ndarray, 1D)
        Extent over which to compute the potential function
    z (float)
        Unstable fixed point, within the interval [-a,a]
    a (float)
        Position of the stable fixed points
    k (float)
        Speed parameter
        
    Returns
    ------------
    p (ndarray, same dimension as x)
        Potential function at each point of the array x'''
    assert np.abs(z)<=np.abs(a), "z sould be in the interval [-a,a]"
    assert a>0, "a must be strictly positive"
    return k/4*x**4-k*z/3*x**3-k*a**2/2*x**2+k*z*a**2*x


def EulerMaruyama(xfun,nfun,x0,t,a,k,z,s=0):
    '''Euler-Maruyama method for computing trajectories of the nl-DDM
    
    Parameters
    -------------
    xfun (callable)
        Callable computing the differential equation of the model
    nfun (callable)
        Callable computing the influence of the noise
    x0 (float)
        Position of the starting point
    t (nd array, 1D)
        
    '''
    x=np.zeros(len(t))
    dt=t[1]-t[0]
    x[0]=x0
    for i in range (1,t.size):
        Wt=np.random.normal(loc=0.0, scale=np.sqrt(dt))
        x[i]=x[i-1]+xfun(x[i-1],a,k,z)*dt+nfun(x[i-1],s)*Wt
    return x


def noise_fun(x,s):
    '''In the (nl-)DDM, the noise term is just the standard deviation
    Parameters
    ------------
    x (float)
        Transparent, used for integration
    s (float)
        Noise standard deviation
        
    Returns
    ------------
    s (float)
        The standard deviation, unmodified'''
    return s

def nlddm(x,a,k,z):
    '''Differential equation describing the nl-DDM
    
    Parameters
    -------------
    x (float, alt. ndarray)
        Decision state(s)
    a (float)
        Position of the stable fixed points
    k (float)
        Speed parameter
    z (float)
        Unstable fixed point, within the interval [-a,a]
    Returns
    ------------
    drift (float, alt. ndarray)
        drift=-k(x-z)(x-a)(x+a)'''
    assert np.abs(z)<=np.abs(a), "z sould be in the interval [-a,a]"
    assert a>0, "a must be strictly positive"
    return -k*(x-z)*(x-a)*(x+a)


def x_max(a,k,z):
    '''Computes the points where the nl-DDM drift reaches its max and min
    within the decision boundaries
    
    Parameters
    -------------
    a (float)
        Position of the stable fixed points
    k (float)
        Speed parameter
    z (float)
        Unstable fixed point, within the interval [-a,a]
    
    Returns
    ------------
    xmin (float)
        Position of the minimum in decision space
    xmax (float)
        Position of the maximum in decision space'''
    assert np.abs(z)<=np.abs(a), "z sould be in the interval [-a,a]"
    assert a>0, "a must be strictly positive"
    xmin=1/3*(z-np.sqrt(z**2+3*a**2))
    xmax=1/3*(z+np.sqrt(z**2+3*a**2))
    return xmin, xmax


def max_drift(x0s,a,k,z):
    '''For given starting points, computes the maximum drift
    
    Parameters
    -------------
    x0s (1D array-like)
        Iterable of all starting points that should be tested
    a (float)
        Position of the stable fixed points
    k (float)
        Speed parameter
    z (float)
        Unstable fixed point, within the interval [-a,a]
        
    Returns
    ------------
    maximum_drift (list)
        List of the maximum drifts for each of the input starting points'''
    maximum_drift=[]
    xmin,xmax=x_max(a,k,z)
    for x0 in x0s:
        if x0>xmax or x0<xmin:
            drift=-k*(x0+a)*(x0-a)*(x0-a*z)
        elif x0<=z:
            drift=-k*(xmin+a)*(xmin-a)*(xmin-a*z)
        else:
            drift=-k*(xmax+a)*(xmax-a)*(xmax-a*z)
        maximum_drift.append(drift)
    return maximum_drift


def mean_max_drift(a,k,z, nsamples=1000):
    '''Computes n estimate of the mean drift for all possible starting points.
    For each trajectory, the drift is estimated as the maximum drift.
    
    Parameters
    -------------
    a (float)
        Position of the stable fixed points
    k (float)
        Speed parameter
    z (float)
        Unstable fixed point, within the interval [-a,a]
    nsamples (int, optional, default=1000)
        Number of samples used to estimate the mean drift
        
    Returns
    ------------
    mean_drift (float)
        Estimate of the mean nl-DDM drift for the given set of parameters'''
    x0s=np.linspace(-a,a,nsamples)
    max_drifts=max_drift(x0s,a,k,z)
    xmin,xmax=x_max(a,k,z)
    for i in range(len(x0s)):
        if x0s[i]<xmin:
            max_drifts[i]/=(xmin+a)
        elif x0s[i]<z:
            max_drifts[i]/=(z-xmin)
        elif x0s[i]<xmax:
            max_drifts[i]/=(xmax-z)
        else:
            max_drifts[i]/=(a-xmax)
    return np.mean(max_drifts)


def draw_line(v,x0,t, Tnd):
    '''Utility to generate a noiseless DDM trajectory.
    
    Parameters
    --------------
    v (float)
        Value of the drift (ie slope of the trajectory)
    x0 (float)
        Starting point of the trajectory
    t (iterable)
        List of time samples on which to compute the trajectory
    Tnd (float)
        Time from which to start the accumulation
    
    Returns
    ------------
    drawn_line (iterable)
        Iterable with the values of the noiseless trajectory'''
    drawn_line=v*(t-Tnd)+x0
    return drawn_line