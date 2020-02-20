#!/usr/bin/env python

"""
Author(s):  D. Knowles
Date:       19 Feb 2020
Desc:       toy data for AA272 sensor fusion project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# create time vector
time = np.arange(401)

# create square
x_truth = np.concatenate((np.arange(101), 100*np.ones(99), np.flip(np.arange(101)), np.zeros(100))).reshape((-1,1))
y_truth = np.concatenate((np.zeros(100), np.arange(100), 100*np.ones(101), np.flip(np.arange(100)))).reshape((-1,1))
z_truth = np.zeros((401,1))

pos_truth = np.hstack((x_truth,y_truth,z_truth))

# plt.figure()
# plt.plot(time,x_truth)
# plt.plot(time,y_truth)
#
# plt.figure()
# plt.plot(x_truth,y_truth)
# plt.show()

pos_sat = np.array([[ 15000000., -25000000., -7500000.],
                    [-20000000.,   4000000., 15000000.],
                    [  8000000.,  15000000., -1000000.],
                    [  2000000., -40000000., -1500000.],
                    [ -1200000., -10000000., 20000000.],
                    [   800000., -25000000.,  6000000.]])

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plt.xlabel('x')
# for ii in range(6):
#     ax.plot([0.0,pos_sat[ii,0]], [0.0,pos_sat[ii,1]], [0.0,pos_sat[ii,2]])
# plt.show()

pr = []

for ii in range(401):
    for jj in range(6):
        pr.append(np.linalg.norm(pos_sat[jj,:]-pos_truth[ii]) + np.random.normal(0.0,8.0))

sv = np.arange(1,7).reshape(-1,1)

# save to file
df1 = pd.DataFrame()
df1['seconds of week [s]'] = np.repeat(time,6)
df1['SV'] = np.tile(sv,(401,1))
df1['pr [m]'] = pr
df1['sat x ECEF [m]'] = np.tile(pos_sat[:,0].reshape(-1,1),(401,1))# + np.random.normal(1E2,1.,(401*6,1))
df1['sat y ECEF [m]'] = np.tile(pos_sat[:,1].reshape(-1,1),(401,1))# + np.random.normal(-1E2,1.,(401*6,1))
df1['sat z ECEF [m]'] = np.tile(pos_sat[:,2].reshape(-1,1),(401,1))# + np.random.normal(-1E2,1.,(401*6,1))
df1.to_csv('./data/sat_toy.csv')


# time_gps = gpstime.gpsfromutc(time_utc)
