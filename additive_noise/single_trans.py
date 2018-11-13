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

# import EWS function
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute


# parameters for simulation
dt = 1
tmax = 100


# model parameters
a1 = 0.02 # noise intensity
r = 1
s = 0.1
hl = 0.14 # min value of control param
hh = 0.28 # max value of control param
x0 = 0.89 # intial condition at equilibrium (from MMA)









