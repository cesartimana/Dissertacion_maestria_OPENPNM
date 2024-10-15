import numpy as np
#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r_sexag = 70
theta_a_sexag = 100
theta_r = np.pi / 180 * theta_r_sexag  #Respecto al agua, en sexagecimal
theta_a = np.pi / 180 * theta_a_sexag #Try 120 to have layers
perm_data = 1*0.986923e-12 #Revisar nota final
p_kPa = 10
pmax_drn = p_kPa * 1000
testName ='C10Z6_V2.pnm'

#Red C10Z6_V2 tiene aprox 1000 mD (aprox 2 x 10-12 m2) de permeabilidad
