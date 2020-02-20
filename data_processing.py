#!/usr/bin/env python

import pandas as pd
import math
import numpy as np
from lib import gpstime
import datetime
import matplotlib.pyplot as plt
import pymap3d
from mpl_toolkits.mplot3d import Axes3D

# import warnings
# warnings.simplefilter(action='ignore', category='DtypeWarning')

# read csv file into a dataframe
df = pd.read_csv("./data/FLY084.csv")[1:-4]


time_utc = df['GPS:dateTimeStamp'].to_numpy()
N = time_utc.shape[0]
# gps time: (gpsWeek, gpsSOW, gpsDay, gpsSOD)
time_gps = np.zeros((N,4))
for ii in range(N):
    utc = time_utc[ii][:-1]
    year = int(utc[0:4])
    month = int(utc[5:7])
    day = int(utc[8:10])
    hour = int(utc[11:13])
    min = int(utc[14:16])
    sec = int(utc[17:19])
    time_gps[ii,:] = gpstime.gpsFromUTC(year,month,day,hour,min,sec)
    # print(time_gps[ii,1])


# sec_offset = np.tile(np.array([0.0,0.2,0.4,0.6,0.8]),(1,int(N/5))).T
tt_time = time_gps[0,1]
tt = 0
types = dict()
while tt < N-1:
    tt_num = 0
    while time_gps[tt,1] == tt_time:
        tt_num +=1
        tt += 1
    if tt_num in types:
        types[tt_num] += 1
    else:
        types[tt_num] = 1
    # print(time_gps[tt-tt_num:tt,1].shape)
    # print(np.linspace(0.0,1.0-(1.0/tt_num),tt_num).reshape(tt_num,1).shape)
    time_gps[tt-tt_num:tt,1] += np.linspace(0.0,1.0-(1.0/tt_num),tt_num)
    tt_time = time_gps[tt,1]
print(types)

# plt.figure()
# plt.plot(np.arange(N),time_gps[:,1])
# for ii in range(N):
#     print(ii,time_gps[ii,1])



# import velocities
vel_n = df['IMU_ATTI(0):velN[meters/Sec]'].to_numpy()
vel_e = df['IMU_ATTI(0):velE[meters/Sec]'].to_numpy()
vel_d = df['IMU_ATTI(0):velD[meters/Sec]'].to_numpy()
vel_h = df['IMU_ATTI(0):velH[meters/Sec]'].to_numpy()
vel_comp = df['IMU_ATTI(0):velComposite[meters/Sec]'].to_numpy()
print(vel_n.shape)

# import gps lat, lon, h
gps_lat = df['GPS(0):Lat[degrees]'].to_numpy()
gps_lon = df['GPS(0):Long[degrees]'].to_numpy()
baro_h = df['IMU_ATTI(0):barometer:Raw[meters]'].to_numpy()
baro_h -= baro_h[0]

vel_ecef = np.zeros((N,3))

for ii in range(N):
    vel_ecef[ii] = pymap3d.ned2ecef(n=vel_n[ii], e=vel_e[ii], d=vel_d[ii], lat0=gps_lat[0], lon0=gps_lon[0], h0=baro_h[0],deg=True)
vel_ecef -= vel_ecef[0]
vel_ecef_comp = np.linalg.norm(vel_ecef,axis=1)


# parse flight 1 data
f1_initial = np.argwhere(time_gps[:,1] == 594416.0).item(0)
f1_final = np.argwhere(time_gps[:,1] == 594561.0).item(0)
print(f1_initial,f1_final)

# save to file
df1 = pd.DataFrame()
df1['seconds of week [s]'] = time_gps[f1_initial:f1_final,1]
df1['ECEF_vel_x'] = vel_ecef[f1_initial:f1_final,0]
df1['ECEF_vel_y'] = vel_ecef[f1_initial:f1_final,1]
df1['ECEF_vel_z'] = vel_ecef[f1_initial:f1_final,2]
df1['IMU_ATTI(0):velN[meters/Sec]'] = vel_n[f1_initial:f1_final]
df1['IMU_ATTI(0):velE[meters/Sec]'] = vel_e[f1_initial:f1_final]
df1['IMU_ATTI(0):velD[meters/Sec]'] = vel_d[f1_initial:f1_final]
df1['IMU_ATTI(0):velH[meters/Sec]'] = vel_h[f1_initial:f1_final]
df1['IMU_ATTI(0):velComposite[meters/Sec]'] = vel_comp[f1_initial:f1_final]
df1['GPS(0):Lat[degrees]'] = gps_lat[f1_initial:f1_final]
df1['GPS(0):Long[degrees]'] = gps_lon[f1_initial:f1_final]
df1['Normalized barometer:Raw[meters]'] = baro_h[f1_initial:f1_final]
df1.to_csv('./data/dji_data_flight_1.csv')

fig, ax = plt.subplots()
ax.ticklabel_format(useOffset=False)
plt.plot(gps_lon[f1_initial:f1_final],gps_lat[f1_initial:f1_final])
plt.title("DJI Proprietary Position Solution")
plt.show()

# parse flight 2 data
f2_initial = np.argwhere(time_gps[:,1] == 594595.0).item(0)
f2_final = np.argwhere(time_gps[:,1] == 594790.0).item(0)
print(f2_initial,f2_final)

# save to file
df2 = pd.DataFrame()
df2['seconds of week [s]'] = time_gps[f2_initial:f2_final,1]
df2['ECEF_vel_x'] = vel_ecef[f2_initial:f2_final,0]
df2['ECEF_vel_y'] = vel_ecef[f2_initial:f2_final,1]
df2['ECEF_vel_z'] = vel_ecef[f2_initial:f2_final,2]
df2['IMU_ATTI(0):velN[meters/Sec]'] = vel_n[f2_initial:f2_final]
df2['IMU_ATTI(0):velE[meters/Sec]'] = vel_e[f2_initial:f2_final]
df2['IMU_ATTI(0):velD[meters/Sec]'] = vel_d[f2_initial:f2_final]
df2['IMU_ATTI(0):velH[meters/Sec]'] = vel_h[f2_initial:f2_final]
df2['IMU_ATTI(0):velComposite[meters/Sec]'] = vel_comp[f2_initial:f2_final]
df2['GPS(0):Lat[degrees]'] = gps_lat[f2_initial:f2_final]
df2['GPS(0):Long[degrees]'] = gps_lon[f2_initial:f2_final]
df2['Normalized barometer:Raw[meters]'] = baro_h[f2_initial:f2_final]
df2.to_csv('./data/dji_data_flight_2.csv')

# parse flight 3 data
f3_initial = np.argwhere(time_gps[:,1] == 594853.0).item(0)
f3_final = N-1
print(f3_initial,f3_final)

# save to file
df3 = pd.DataFrame()
df3['seconds of week [s]'] = time_gps[f3_initial:f3_final,1]
df3['ECEF_vel_x'] = vel_ecef[f3_initial:f3_final,0]
df3['ECEF_vel_y'] = vel_ecef[f3_initial:f3_final,1]
df3['ECEF_vel_z'] = vel_ecef[f3_initial:f3_final,2]
df3['IMU_ATTI(0):velN[meters/Sec]'] = vel_n[f3_initial:f3_final]
df3['IMU_ATTI(0):velE[meters/Sec]'] = vel_e[f3_initial:f3_final]
df3['IMU_ATTI(0):velD[meters/Sec]'] = vel_d[f3_initial:f3_final]
df3['IMU_ATTI(0):velH[meters/Sec]'] = vel_h[f3_initial:f3_final]
df3['IMU_ATTI(0):velComposite[meters/Sec]'] = vel_comp[f3_initial:f3_final]
df3['GPS(0):Lat[degrees]'] = gps_lat[f3_initial:f3_final]
df3['GPS(0):Long[degrees]'] = gps_lon[f3_initial:f3_final]
df3['Normalized barometer:Raw[meters]'] = baro_h[f3_initial:f3_final]
df3.to_csv('./data/dji_data_flight_3.csv')

plt.figure()
plt.plot(time_gps[:,1],vel_n,label='N')
plt.plot(time_gps[:,1],vel_e,label='E')
plt.plot(time_gps[:,1],vel_d,label='D')
# plt.plot(time_gps[:,1],vel_h,label='H')
plt.legend()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(gps_lon, gps_lat, baro_h, label='ned')
ax.legend()

fig = plt.figure()
plt.plot(time_gps[:,1],vel_ecef[:,0],label='x')
plt.plot(time_gps[:,1],vel_ecef[:,1],label='y')
plt.plot(time_gps[:,1],vel_ecef[:,2],label='z')
plt.legend()

plt.figure()
plt.plot(time_gps[:,1],vel_comp)
plt.plot(time_gps[:,1],vel_ecef_comp)


plt.figure()
plt.plot(time_gps[f1_initial:f1_final,1],vel_comp[f1_initial:f1_final])

plt.figure()
plt.plot(time_gps[f2_initial:f2_final,1],vel_comp[f2_initial:f2_final])

plt.figure()
plt.plot(time_gps[f3_initial:f3_final,1],vel_comp[f3_initial:f3_final])

plt.show()





# time_gps = gpstime.gpsfromutc(time_utc)
