import numpy as np
#Properties extracted from Valvatne - Blunt (2004) Table 1
tension = 30e-3 #N/m
water_visc = 1.05e-3 #Pa/s
oil_visc = 1.39e-3 #Pa/s
theta_r_sexag = 00
theta_a_sexag = 60
theta_r = np.pi / 180 * theta_r_sexag  #Respecto al agua, en sexagecimal
theta_a = np.pi / 180 * theta_a_sexag #Try 120 to have layers
perm_data = 1.746*0.986923e-12 #Average K from Berea Network
p_kPa = 7
pmax_drn = p_kPa * 1000
testName ='Rede_2D_30x30.pnm'
