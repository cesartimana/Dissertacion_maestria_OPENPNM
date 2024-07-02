import numpy as np
"""
PARECE QUE SI:
TENGO UN ARRAY "A" DE F FILAS Y C COLUMNAS.
TENGO UN BOOL ARRAY "B" DE F ELEMENTOS

PUEDO HACER A[B] Y ME VA A DAR COMO RESULTADO LAS FILAS DE "A" DONDE EL ELEMENTO DE "B" ES TRUE
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

