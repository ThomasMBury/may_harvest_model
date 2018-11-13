#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:49:48 2018

@author: tb460

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


# parameters for simulation
dt = 1
t0 = 0
tmax = 400
numSims = 3

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
x0 = 0.8197 # intial condition at equilibrium (from MMA)



# initialise arrays
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))
x[0] = x0

# initialise DataFrame for realisations
df_sims = pd.DataFrame([])

# linearly increasing control parameter h
h = np.linspace(hl,hh,len(t))



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
    
# cheeky plot
df_sims.plot()


# feed single realisation into ews_compute
df_ews = ews_compute(df_sims['Sim 1'],
                     band_width=0.1,
                     roll_window=0.25, 
                     lag_times=[1],
                     ham_length=40,
                     ews=['var','ac'])

# plot of smoothing
df_ews[['State variable','Smoothing']].plot()

# plot EWS
#df_ews['Variance'].plot()



