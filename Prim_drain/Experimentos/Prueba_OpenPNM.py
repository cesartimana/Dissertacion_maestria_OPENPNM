import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
import time
import heapq as hq

op.visualization.set_mpl_style()
np.random.seed(10)
np.set_printoptions(precision=5)
op.Workspace().settings.loglevel = 40

pn = op.network.Cubic(shape=[15, 15, 15], spacing=1e-6)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()

print(pn)


air = op.phase.Air(network=pn,name='air')
air['pore.surface_tension'] = 0.072
air['pore.contact_angle'] = 180.0
air.add_model_collection(op.models.collections.phase.air)
air.add_model_collection(op.models.collections.physics.basic)
air.regenerate_models()
water = op.phase.Water(network=pn,name='water')
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()


text1 = '''
Prueba de tiempo
-----------------------------
Se intentara saber si el método dpara determinar las gargantas que
tiene un poro es adecuado.
Método 1: El metodo usado por OpenPNM
Metodo 2: Usando neighbor throats
Metodo 3, Realizando la matriz usada en el método 1 a mi estilo
'''

espacio = '''


'''


n = 10 # numero del poro

print(text1)

print('Metodo 1')
t0 = time.time()
im = pn.create_incidence_matrix(fmt='csr')
indices = im.indices
ptr = im.indptr
print('Tiempo para hacer la matriz en s:', time.time() - t0)
t1 = time.time()
t = indices[ptr[n]: ptr[n+1]]
print(t)
print('Tiempo para buscar una garganta:', time.time() - t1)
print(espacio)
print('Metodo 2')
t0 = time.time()
t = pn.find_neighbor_throats(pores=n)
print(t)
print('Tiempo para buscar una garganta:', time.time() - t0)

print(espacio)
print('Metodo 3')
t0 = time.time()
Nt = pn.Nt
Np = pn.Np
conn = pn['throat.conns']
row = conn[:, 0]
row = np.append(row, conn[:, 1])
col = np.arange(Nt)
col = np.append(col, col)
mat = np.concatenate(([row],[col]), axis = 0).T
mat = mat[mat[:, 0].argsort()]
indices = mat[:,1]
ptr = np.array([np. count_nonzero(row == i) for i in np.arange(0,Np)])
print('Tiempo para hacer la matriz en s:', time.time() - t0)

print(espacio)
text2 = '''
Prueba de tiempo
-----------------------------
Se intentara saber si el método organizar la informacion del orden de las gargantas
efectuado por OpenPNM es el más efectivo.
Método 1: El metodo usado por OpenPNM
Metodo 2: Ordenando con herramientas numpy
'''

print(text2)
array1 = [7,9,12,3,8,5]
array1q = [7,9,12,3,8,5]
array2 = [6,4]


print(espacio)
print('Metodo 1')
t0 = time.time()
hq.heapify(array1q)
res = hq.heappop(array1q)
print('Tiempo para ordenar el array y retirar el menor valor:', time.time() - t0)
t1 = time.time()
for i in array2:
    hq.heappush(array1q, i)
print('Tiempo para incluir nuevos valores y ordenar:', time.time() - t1)
print(array1q)
print(espacio)
print('Metodo 2')
t0 = time.time()
array1 = np.sort(np.array(array1))
res = array1[0]
array1= array1[1:]
print('Tiempo para ordenar el array y retirar el menor valor:', time.time() - t0)
t1 = time.time()
array1 = np.sort(np.concatenate((array1, np.array(array2))))
print('Tiempo para incluir nuevos valores y ordenar:', time.time() - t1)
print(array1)
