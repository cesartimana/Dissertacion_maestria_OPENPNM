#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:24:24 2023

@author: cesar
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m


beta=np.array([30,30]) #half corner angles b1, b2. Should be less than 45 in sexagecimal
beta=np.append(beta,90-beta[0]-beta[1])
beta=m.radians(beta)
theta=0 #contact angle, in sexagesimal
theta=m.radians(theta)
r=5 #radius of inscribed circle.
G=0 #sdhape factor
Aw=0 #wetting area
Anw=0 #non wetting area

for i in range(size(beta)):
    G=G+1/m.tan(beta[i])

G=G**(-1)/4
print(G)



"""
#beta.append(9)

c=5 #x-point of the center of circle corresponding to AM. 



#creating the corner
x=np.linspace(0,c*1,5,50) #vector of numbers in x axis
L1=x*0
L2=x*m.tan(beta*2)

#creating the circle
d=c*m.tan(beta*2) #distance from (0,0) to c
r=0.6*d #radius of 



#plotting
plt.plot(x,L1)
plt.plot(x,L2)
plt.axis('equal')
plt.show()

"""

