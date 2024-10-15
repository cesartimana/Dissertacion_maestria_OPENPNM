import numpy as np
import openpnm as op

# network parameters
seed = 15 #numpy random seed
range_seed = [0.45,0.7] #for pore and throat
porosity = 0.4
permeability = 100.0e-12
z_mean = 6
shape = np.array([10, 10, 10])
sp = 1.0e-4
sampleName = 'C10Z6'
elements = ['pore', 'throat']


#Functions
def eq_diam_2D(area):
    return np.power(4 * area / np.pi, 0.5)

def calc_beta(shape_factor):
    return op.teste.geometry.half_angle_isosceles(shape_factor)
#Berea tiene 0.196 de porosidad y aprox 1.3 D (Aprox 1 D = 10-12 m2) de permeabilidad

"""
Recomendacion: Mdificar el rango de semillas para calcular el diametro de los poros
Con ello se calcula porosidad (asumiendo todo el volumen en los poros)
Luego se modifica el espaciamiento entre poros sp. Esto debido a que este valor es como un factor que modifica al diametro (a mayor espaciamiento, mayor diametro) Y ello modifica el area de las gargantas, ,odificando la permeabilidad
"""
