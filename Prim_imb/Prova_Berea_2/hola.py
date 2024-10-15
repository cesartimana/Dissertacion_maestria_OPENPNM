import numpy as np
"""
PARECE QUE SI:
TENGO UN ARRAY "A" DE F FILAS Y C COLUMNAS.
TENGO UN BOOL ARRAY "B" DE F ELEMENTOS

PUEDO HACER A[B] Y ME VA A DAR COMO RESULTADO LAS FILAS DE "A" DONDE EL ELEMENTO DE "B" ES TRUE
"""

"""
a = np.resize( np.arange(30), (10,3) )
b = np.ones(10, dtype = bool)
b[5:7] = False
b[0] = False
c = (a%2 == 0) #residuo
print(a)
print(b)
print(a[b])
print(True == True)
print(False == False)
print(c)
print(c & np.tile(b,(3,1)).T)

"""

#Test de csr_matrix y connected_components)
#Da igual si es simetrico o no, basta un camino de ida sin vuelta para decir que están unidos
from scipy.sparse import csr_matrix
from scipy.linalg import issymmetric
from scipy.sparse.csgraph import connected_components

graph = np.array([
[0, 1, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 1],
[0, 0, 0, 0, 0]
])
#graph = np.maximum( graph, graph.transpose() )
print(graph)
print(issymmetric(graph))
graph = csr_matrix(graph)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
print(n_components)
print(labels)


A = {'A': 1,
     'B': 2,
     'C': 3,
     'D': 4
    }

A1 = {'X': 100,
      'Y': 101,
      'Z': 102
      }
A['nested dict'] = A1.copy()
print(A['nested dict'] )
A1['X'] = 200
print(A['nested dict'] )

a = 50.549165
print('hola ' + str(a))
a = np.inf
print('hola ' + str(a))
