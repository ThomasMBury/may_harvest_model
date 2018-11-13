#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:49:48 2018

@author: Thomas Bury

Script to simluate a single transient trajectory of
May's harvesting model with additive noise, and compute EWS.

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import EWS function
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute
from ews_spec import pspec_welch, pspec_metrics


# parameters for simulation
dt = 1
t0 = 0
tmax = 400
numSims = 1

# model dx/dt = de_fun(x,t) + sigma dW(t)
def de_fun(x,r,k,h,s):
    return r*x*(1-x/k)  - h*(x**2/(s**2 + x**2))
    
    
# model parameters
sigma = 0.02 # noise intensity
r = 1
k = 1
s = 0.1
hl = 0.15 # min value of control param
hh = 0.28 # max value of control param
hbif = 0.260437 # bifurcation point (from MMA)
x0 = 0.8197 # intial condition at equilibrium (from MMA)




# initialise arrays
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))
x[0] = x0

# initialise DataFrame for realisations
df_sims = pd.DataFrame([])

# control parameter increases linearly in time
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# time at which bifurcation occurs
tbif = h[h > hbif].index[1]


## implement Euler Maryuyama - creates a DataFrame with columns (sim1, sim2, ...simN) indexed by time

# loop over simulations
for j in range(numSims):
    
    # create brownian increments (s.d. sqrt(dt))
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = len(t))
    
    # loop over time
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i],r,k,h[i],s)*dt + sigma*dW[i]
        # make sure that state variable remains >= 0 
        if x[i+1] < 0:
            x[i+1] = 0
        # store as a series
        series = pd.Series(x, index=t)
        # add to DataFrame of realisations
        df_sims['Sim '+str(j+1)] = series
    

## compute EWS using ews_compute with input time-series up to tbif
        
df_ews = ews_compute(df_sims['Sim 1'],
                     band_width=0.1,
                     upto=tbif,
                     roll_window=0.25, 
                     lag_times=[1],
                     ham_length=20,
                     ews=['var','ac','smax','aic'])


# make plot of EWS
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews.plot(y=['State variable','Smoothing'],ax=axes[0])
df_ews.plot(y='Variance', ax=axes[1])
df_ews.plot(y='Lag-1 AC', ax=axes[1], secondary_y=True)
df_ews.dropna().plot(y='Smax', ax=axes[2])
df_ews.dropna().plot(y=['AIC fold','AIC hopf','AIC null'], ax=axes[3])

# investigate power spectrum at intervals
pspec=pspec_welch(df_ews.loc[100:200,'Residuals'], df_ews.index[1]-df_ews.index[0], ham_length=40, w_cutoff=1)

# put the power spectrum into pspec_metrics
spec_ews = pspec_metrics(pspec, ews=['smax', 'cf', 'aic', 'aic_params'])
# define models to fit
def fit_fold(w,sigma,lam):
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))
        
def fit_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))
        
def fit_null(w,sigma):
    return sigma**2/(2*np.pi)* w**0
w_vals = np.linspace(-max(pspec.index),max(pspec.index),100)


plt.figure(2)
pspec.plot()
plt.plot(w_vals, fit_fold(w_vals, spec_ews['Params fold']['sigma'], spec_ews['Params fold']['lam']))
plt.plot(w_vals, fit_hopf(w_vals, spec_ews['Params hopf']['sigma'], spec_ews['Params hopf']['mu'], spec_ews['Params hopf']['w0']))
plt.plot(w_vals, fit_null(w_vals, spec_ews['Params null']['sigma']))

print(spec_ews)



