#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:20:32 2018

@author: Thomas Bury

Script to simluate multiple tipping point trajectories of May's harvesting model
using different types of noise, andlayse EWS, and compute the distirubtion of kendall tau values.

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import EWS function
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute



#----------------------------------
# Simulate many (transient) realisations of May's harvesting model
#----------------------------------


# Simulation parameters
dt = 0.1
t0 = 0
tmax = 1000
burn_time = 100 # burn-in period
numSims = 10
seed = 0 # random number generation seed

# parameters to add noise to
noisy_params = ['k']


# Model: dx/dt = de_fun(x,t) + sigma dW(t)
def de_fun(x,r,k,h,s):
    return r*x*(1-x/k)  - h*(x**2/(s**2 + x**2))
    
    
# Model parameters (average)
r = 1 # growth rate
k = 1 # carrying capacity
s = 0.1 # half-saturation constant of harvesting function
hl = 0.15 # initial harvesting rate
hh = 0.28 # final harvesting rate
hbif = 0.260437 # bifurcation point (computed in Mathematica)
x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)

# noise amplitudes
r_amp = 0.1
k_amp = 0.1
s_amp = 0.1
state_add_amp = 0.2 # additive noise to state
state_multi_amp = 0.2 # multiplicative noise proportional to size of state




# initialise DataFrame to store all realisations
df_sims = pd.DataFrame([])

# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
tburn = np.arange(t0,burn_time,dt)
x = np.zeros(len(t))

# Set up control parameter h, that increases linearly in time from hl to hh
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# Time at which bifurcation occurs
tbif = h[h > hbif].index[1]

## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)


# Create noisy parameter values
if 'r' in noisy_params:
    r_burn = r + np.random.normal(loc=0, scale=r_amp, size=[numSims,tburn.size])
    r = r + np.random.normal(loc=0, scale=r_amp, size=[numSims,t.size])
else:
    r_burn = r*np.ones([numSims,tburn.size])
    r = r*np.ones([numSims,t.size])
    
if 's' in noisy_params:
    s_burn = s + np.random.normal(loc=0, scale=s_amp, size=[numSims,tburn.size])
    s = s + np.random.normal(loc=0, scale=s_amp, size=[numSims,t.size])
else:
    s_burn = s*np.ones([numSims,tburn.size])
    s = s*np.ones([numSims,t.size])    
    
if 'k' in noisy_params:
    k_burn = k + np.random.normal(loc=0, scale=k_amp, size=[numSims,tburn.size])
    k = k + np.random.normal(loc=0, scale=k_amp, size=[numSims,t.size])
else:
    k_burn = k*np.ones([numSims,tburn.size])
    k = k*np.ones([numSims,t.size])   

# create additive / multiplicative noise components    
if 'add' in noisy_params:
    add_burn_comps = np.random.normal(loc=0, scale=np.sqrt(dt)*state_add_amp, size=[numSims,tburn.size])
    add_comps = np.random.normal(loc=0, scale=np.sart(dt)*state_add_amp, size=[numSims,t.size])
else:
    add_burn_comps = np.zeros([numSims,tburn.size])
    add_comps = np.zeros([numSims,t.size])  
    
if 'multi' in noisy_params:
    multi_burn_comps = np.random.normal(loc=0, scale=np.sqrt(dt)*state_multi_amp, size=[numSims,tburn.size])
    multi_comps = np.random.normal(loc=0, scale=np.sart(dt)*state_multi_amp, size=[numSims,t.size])
else:
    multi_burn_comps = np.zeros([numSims,tburn.size])
    multi_comps = np.zeros([numSims,t.size])   
    
    

# loop over simulations
for j in range(numSims):
    
    
    # Run burn-in period on x0
    for i in range(len(tburn)-1):
        x0 = x0 + de_fun(x0, r[j,i], k[j,i], h[0], s[j,i])*dt + state_add_amp*add_comps[j,i] + state_multi_amp*multi_comps[j,i]*x0
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i], r[j,i], k[j,i] ,h.iloc[i], s[j,i])*dt +  state_add_amp*add_comps[j,i] + state_multi_amp*multi_comps[j,i]*x[i]
        # make sure that state variable remains >= 0 
        if x[i+1] < 0:
            x[i+1] = 0
            
    # Store data as a Series indexed by time
    series = pd.Series(x, index=t)
    # add Series to DataFrame of realisations
    df_sims['Sim '+str(j+1)] = series







#----------------------
## Execute ews_compute for each realisation
#---------------------

# Sample from time-series at uniform intervals of width dt2
dt2 = 1
df_sims_filt = df_sims[np.remainder(df_sims.index,dt2) == 0]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []

# loop through each trajectory as an input to ews_compute
for i in range(numSims):
    df_temp = ews_compute(df_sims_filt['Sim '+str(i+1)], 
                      roll_window=0.5, 
                      band_width=0.1,
                      lag_times=[1], 
                      ews=['var','ac','sd','cv','skew','kurt','smax','aic'],
                      ham_length=20,                     
                      upto=tbif)
    # include a column in the dataframe for realisation number
    df_temp['Realisation number'] = pd.Series((i+1)*np.ones([len(t)],dtype=int),index=t)
    
    # add DataFrame to list
    appended_ews.append(df_temp)
    
    # print status every 10 realisations
    if np.remainder(i+1,1)==0:
        print('Realisation '+str(i+1)+' complete')


# concatenate EWS DataFrames - use realisation number and time as indices
df_ews = pd.concat(appended_ews).set_index('Realisation number',append=True).reorder_levels([1,0])



#------------------------
# Plots of EWS
#-----------------------

# plot of all variance trajectories
df_ews.loc[:,'Variance'].unstack(level=0).plot(legend=False, title='Variance') # unstack puts index back as a column

# plot of all autocorrelation trajectories
df_ews.loc[:,'Lag-1 AC'].unstack(level=0).plot(legend=False, title='Lag-1 AC') 

# plot of all smax trajectories
df_ews.loc[:,'Smax'].unstack(level=0).dropna().plot(legend=False, title='Smax') # drop Nan values



#---------------------------
## Compute distribution of kendall tau values and make box-whisker plots
#----------------------------

# make the time values their own series and use pd.corr to compute kendall tau correlation
time_series = pd.Series(df_sims_filt.index, index=df_sims_filt.index)

# Find kendall tau correlation coefficient for each EWS over each realisation.
# initialise dataframe
df_ktau = pd.DataFrame(columns=df_ews.columns, index=np.arange(numSims)+1,dtype=float)
# loop over simulations
for j in range(numSims):
    # compute kenall tau for each EWS
    ktau = pd.Series([df_ews.loc[j+1,x].corr(time_series,method='kendall') for x in df_ews.columns],index=df_ews.columns)
    # addå to dataframe
    df_ktau.loc[j+1]= ktau

# kendall tau distribution statistics can be found using
ktau_stats=df_ktau.describe()

df_ktau[['Variance','Lag-1 AC','Smax']].plot(kind='box',ylim=(-1,1))


## Export kendall tau values for plotting in MMA
#df_ktau[['Variance','Lag-1 AC','Smax']].to_csv('data_export/ktau_add_tlong.csv')



