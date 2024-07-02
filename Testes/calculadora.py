#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:06:45 2023

@author: cesar
"""


import numpy as np
import openpnm as op
import math as m
import random as rnd
G=0.01
b=0.68
print(-0.5*b+0.5*np.arcsin((np.tan(b)+4*G)/(np.tan(b)-4*G)))
print(((np.tan(b)+4*G)/(np.tan(b)-4*G)))