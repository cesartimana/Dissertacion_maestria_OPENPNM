import numpy as np
#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r_sexag = 0
theta_a_sexag = 50
theta_r = np.pi / 180 * theta_r_sexag  #Respecto al agua, en sexagecimal
theta_a = np.pi / 180 * theta_a_sexag #Try 120 to have layers
perm_data = 2.5e-12 #MODIFICAR POR CADA RED
p_kPa = 6.5
pmax_drn = p_kPa * 1000
testName ='C10Z6mean.pnm'


