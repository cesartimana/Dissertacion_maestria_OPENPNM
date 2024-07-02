#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:15:51 2023

@author: cesar
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# def make_ax(grid=False):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.grid(grid)
#     return ax
# 
# filled = np.array([
#     [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
#     [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
#     [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
# ])
# 
# ax = make_ax(True)
# ax.voxels(filled, edgecolors='gray', shade=False)
# plt.show()
# =============================================================================


# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxelarray = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxelarray.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

plt.show()