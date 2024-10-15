import numpy as np
import openpnm as op

# network parameters
seed = 15 #numpy random seed
range_Dp = [0.2,0.7]
range_Lp = [0.2,0.7]
porosity = 0.06023768851950325
permeability = 2.4110897250618825e-12
z_mean = 6
shape = np.array([10, 10, 10])
sp = 5 * 1.0e-4 #e-6 para usar um
sampleName = 'C10Z6meanTun'
elements = ['pore', 'throat']


#Berea tiene 0.196 de porosidad y aprox 1.3 D (Aprox 1 D = 10-12 m2) de permeabilidad
#Berea tiene, de D_eq una media de 60 um, min de 9 um y max de 260 um

"""
Si todas las esferas tuvieran Deq = 0.9 sp, porosidad = 0.381
Si todas las esferas tuvieran Deq = 0.1 sp, porosidad = 0.00052

Recomendacion: Mdificar el rango de semillas para calcular el diametro de los poros
Con ello se calcula porosidad (asumiendo todo el volumen en los poros)
Luego se modifica el espaciamiento entre poros sp. Esto debido a que este valor es como un factor que modifica al diametro (a mayor espaciamiento, mayor diametro) Y ello modifica el area de las gargantas, ,odificando la permeabilidad
"""
