#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:49:48 2018

@author: Thomas Bury

Script to simluate multiple tipping point trajectories of May's harvesting model,
andlayse EWS, and compute the distirubtion of kendall tau values.

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
tmax = 400
tburn = 100 # burn-in period
numSims = 100
seed = 0 # random number generation seed

# Model: dx/dt = de_fun(x,t) + sigma dW(t)
def de_fun(x,r,k,h,s):
    return r*x*(1-x/k)  - h*(x**2/(s**2 + x**2))
    
    
# Model parameters
sigma = 0.02 # noise intensity
r = 1 # growth rate
k = 1 # carrying capacity
s = 0.1 # half-saturation constant of harvesting function
hl = 0.15 # initial harvesting rate
hh = 0.28 # final harvesting rate
hbif = 0.260437 # bifurcation point (computed in Mathematica)
x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)




# initialise DataFrame to store all realisations
df_sims = pd.DataFrame([])

# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))

# Set up control parameter h, that increases linearly in time from hl to hh
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# Time at which bifurcation occurs
tbif = h[h > hbif].index[1]

## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)


# loop over simulations
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun(x0,r,k,h[0],s)*dt + sigma*dW_burn[i]
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i],r,k,h.iloc[i],s)*dt + sigma*dW[i]
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
                      ham_length=40,                     
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

# plot of all Fold AIC trajectories
df_ews.loc[:,'AIC fold'].unstack(level=0).dropna().plot(legend=False, title='AIC fold') # drop Nan values




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

df_ktau[['Variance','Lag-1 AC','Smax']].plot(kind='box',ylim=(0,1))


## Export kendall tau values for plotting in MMA
#df_ktau[['Variance','Lag-1 AC','Smax']].to_csv('data_export/ktau_add_tshort.csv')












