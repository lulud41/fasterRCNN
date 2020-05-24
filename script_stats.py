#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("bbox_galbe.csv",delimiter=",",dtype=np.int32)

bbox =  data[:,2:]
print(bbox.shape)

minimum = np.min(bbox,axis=0)
min_H = minimum[0]
min_W = minimum[1]

maximum = np.max(bbox,axis=0)
max_H = maximum[0]
max_W = maximum[1]

H_axis = np.arange(0,max_H+1)
W_axis = np.arange(0, max_W+1)

H_axis, W_axis = np.meshgrid(W_axis,H_axis)

unique_values, counts = np.unique(bbox,axis=0,return_counts=True)

Z = np.zeros(np.shape(H_axis))
#Z[:] = np.nan

Z[unique_values[:,0],unique_values[:,1]] = counts



fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(H_axis, W_axis, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()