#!/usr/bin/env python
"""
Author(s):  D. Knowles
Date:       14 Feb 2020
Desc:       AA272 sensor fusion project
"""

import numpy as np
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
import math
import progress.bar

class EKF():
    def __init__(self,sat_file,odom_file=None):
        # read in data files as dataframe
        self.sat_df = pd.read_csv(sat_file, index_col=0)
        self.odom_file = odom_file
        if odom_file != None:
            self.odom_df = pd.read_csv(odom_file, index_col=0)

            # initial lat and lon from dji data
            lat0 = np.mean(self.odom_df['GPS(0):Lat[degrees]'][0])
            lon0 = np.mean(self.odom_df['GPS(0):Long[degrees]'][0])
            h0 = 0.0

            # convert lat lon to ecef frame
            self.lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
            self.ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            x0, y0, z0 = pyproj.transform(self.lla, self.ecef, lon0, lat0, h0 , radians=False)

            # concatenate possible time steps from each data file
            self.times = np.concatenate((self.odom_df['seconds of week [s]'].to_numpy(),self.sat_df['seconds of week [s]'].to_numpy()))

            # initial and final time values
            # self.ti = min(self.odom_df['seconds of week [s]'].min(),self.sat_df['seconds of week [s]'].min())
            # self.tf = max(self.odom_df['seconds of week [s]'].max(),self.sat_df['seconds of week [s]'].max())

            self.initialized_odom = False

        else:
            # set initial positions
            x0 = 0.
            y0 = 0.
            z0 = 0.

            # initialize times
            self.times = self.sat_df['seconds of week [s]'].to_numpy()

        # sort timesteps and force unique
        self.times = np.sort(np.unique(self.times))

        # initialize state vector [ x, y, z ]
        self.mu = np.array([[x0,y0,z0]]).T
        self.mu_n = self.mu.shape[0]
        self.mu_history = self.mu.copy()

        # initialize covariance matrix
        self.P = np.eye(self.mu_n)
        self.P_history = [np.trace(self.P)]

        if odom_file != None and 'pr [m]' in self.sat_df.columns:
            # only use the best satellites
            self.check_data(self.mu,lat0,lon0)


    def ECEF_2_ENU(self,x_ECEF,xref,lat0,lon0):
        """
        input(s)
            x_ECEF: 3 X N array
        """
        x_ECEF_ref = xref

        x_REF = np.repeat(x_ECEF_ref,x_ECEF.shape[1],axis=1)
        theta_lat = np.radians(lat0)
        theta_long = np.radians(lon0)
        T_enu = np.array([[-np.sin(theta_long),
                           np.cos(theta_long),
                           0.],
                          [-np.sin(theta_lat)*np.cos(theta_long),
                          -np.sin(theta_lat)*np.sin(theta_long),
                          np.cos(theta_lat)],
                          [np.cos(theta_lat)*np.cos(theta_long),
                          np.cos(theta_lat)*np.sin(theta_long),
                          np.sin(theta_lat)]])
        x_ENU = np.dot(T_enu,(x_ECEF-x_REF))
        return x_ENU

    def check_data(self,xref,lat0,lon0):
        # ans = self.ECEF_2_ENU(self.mu,self.mu,lat0,lon0)
        SVs = np.sort(np.unique(self.sat_df['SV']))
        for sv in SVs:
            sv_subset = self.sat_df[self.sat_df['SV'] == sv]
            sv_x = sv_subset['sat x ECEF [m]'].to_numpy().reshape((1,-1))
            sv_y = sv_subset['sat y ECEF [m]'].to_numpy().reshape((1,-1))
            sv_z = sv_subset['sat z ECEF [m]'].to_numpy().reshape((1,-1))
            sv_time = sv_subset['seconds of week [s]'].to_numpy()
            sv_xyz = np.vstack((sv_x,sv_y,sv_z))
            sv_ENU = self.ECEF_2_ENU(sv_xyz,self.mu,lat0,lon0)


            elev_angles = np.degrees(np.arctan2(sv_ENU[2,:],np.sqrt(sv_ENU[0,:]**2 + sv_ENU[1,:]**2)))
            # if sv in [7,30,28,9,8,5]:
            # if True:
                # plt.plot(sv_time,elev_angles,label=sv)
                # plt.ylabel('Elevation Angle [degrees]')
                # plt.legend()
                # plt.title("Elevation Angle vs. Time")
        # plt.legend()
        # plt.show()

        is7 = self.sat_df['SV'] == 7
        is30 = self.sat_df['SV'] == 30
        is28 = self.sat_df['SV'] == 28
        is9 = self.sat_df['SV'] == 9
        is8 = self.sat_df['SV'] == 8
        is5 = self.sat_df['SV'] == 5

        self.sat_df = self.sat_df[is7 | is30 | is28 | is9 | is8 | is5]



    def predict_imu(self,odom,dt):
        """
            Desc: ekf predict imu step
            Input(s):
                odom:   odometry [vel_x, vel_y, vel_z] [3 x 1]
                dt:     time step difference
            Output(s):
                none
        """
        # build state transition model matrix
        F = np.eye(self.mu_n)

        # build odom transition matrix
        B = np.eye(self.mu_n) * dt

        # update predicted state
        self.mu = F.dot(self.mu) + B.dot(odom)

        # build process noise matrix
        Q_cov = 0.5
        Q = np.eye(self.mu_n) * Q_cov

        # propagate covariance matrix
        self.P = F.dot(self.P).dot(F.T) + Q

    def predict_simple(self):
        """
            Desc: ekf simple predict step
            Input(s):
                dt:     time step difference
            Output(s):
                none
        """
        # build state transition model matrix
        F = np.eye(self.mu_n)

        # update predicted state
        self.mu = F.dot(self.mu)

        # build process noise matrix
        Q_cov = 0.5
        Q = np.eye(self.mu_n) * Q_cov
        # Q = np.ones((3,3)) * Q_cov

        # propagate covariance matrix
        self.P = F.dot(self.P).dot(F.T) + Q

    def update_gnss_raw(self,mes,sat_x,sat_y,sat_z):
        """
            Desc: ekf update gnss step
            Input(s):
                mes:    psuedorange measurements [N x 1]
                sat_pos satellite position [N x 3]
            Output(s):
                none
        """
        num_sats = mes.shape[0]
        zt = mes
        H = np.zeros((num_sats,3))
        h = np.zeros((num_sats,1))
        for ii in range(num_sats):
            dist = np.sqrt((sat_x[ii]-self.mu[0])**2 + (sat_y[ii]-self.mu[1])**2 + (sat_z[ii]-self.mu[2])**2)
            H[ii,0] = (self.mu[0]-sat_x[ii])/dist
            H[ii,1] = (self.mu[1]-sat_y[ii])/dist
            H[ii,2] = (self.mu[2]-sat_z[ii])/dist
            h[ii] = dist
        yt = zt - h

        R_cov = 8.**2
        R = np.eye(num_sats)*R_cov

        Kt = self.P.dot(H.T).dot(np.linalg.inv(R + H.dot(self.P).dot(H.T)))

        self.mu = self.mu.reshape((3,1)) + Kt.dot(yt)
        self.P = (np.eye(self.mu_n)-Kt.dot(H)).dot(self.P).dot((np.eye(self.mu_n)-Kt.dot(H)).T) + Kt.dot(R).dot(Kt.T)

        yt = zt - H.dot(self.mu)

        # Kt = self.P.dot(H)

    def update_gnss(self,lat,lon,alt):
        """
            Desc: ekf update gnss step
            Input(s):
                lat:
                lon:
                alt:
            Output(s):
                none
        """
        # convert lat lon to ecef frame
        self.lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        xi, yi, zi = pyproj.transform(self.lla, self.ecef, lon, lat, alt , radians=False)
        zt = np.array([[xi,yi,zi]]).T

        H = np.eye(3)
        yt = zt - H.dot(self.mu)

        R_cov = 10.**2
        R = np.eye(3)*R_cov

        Kt = self.P.dot(H.T).dot(np.linalg.inv(R + H.dot(self.P).dot(H.T)))

        self.mu = self.mu.reshape((3,1)) + Kt.dot(yt)
        self.P = (np.eye(self.mu_n)-Kt.dot(H)).dot(self.P).dot((np.eye(self.mu_n)-Kt.dot(H)).T) + Kt.dot(R).dot(Kt.T)

        yt = zt - H.dot(self.mu)

        # Kt = self.P.dot(H)


    def run(self):
        """
            Desc: run ekf
            Input(s):
                none
            Output(s):
                none
        """
        t_odom_prev = 0.0 # initialize previous odom time

        # setup progress bar
        print("running kalman filter, please wait...")
        bar = progress.bar.IncrementalBar('Progress:', max=len(self.times))


        for tt, timestep in enumerate(self.times):
            # predict step for odometry
            if self.odom_df['seconds of week [s]'].isin([timestep]).any():
                dt_odom = timestep - t_odom_prev
                t_odom_prev = timestep
                if not self.initialized_odom:
                    self.initialized_odom = True
                    bar.next()
                    continue
                odom_timestep = self.odom_df[self.odom_df['seconds of week [s]'] == timestep]
                odom_vel_x = odom_timestep['ECEF_vel_x'].values[0]
                odom_vel_y = odom_timestep['ECEF_vel_y'].values[0]
                odom_vel_z = odom_timestep['ECEF_vel_z'].values[0]
                self.predict_imu(np.array([[odom_vel_x,odom_vel_y,odom_vel_z]]).T,dt_odom)
            # update gnss step
            if self.sat_df['seconds of week [s]'].isin([timestep]).any():
                sat_timestep = self.sat_df[self.sat_df['seconds of week [s]'] == timestep]
                if 'pr [m]' in self.sat_df.columns:
                    pranges = sat_timestep['pr [m]'].to_numpy().reshape(-1,1)
                    sat_x = sat_timestep['sat x ECEF [m]'].to_numpy().reshape(-1,1)
                    sat_y = sat_timestep['sat y ECEF [m]'].to_numpy().reshape(-1,1)
                    sat_z = sat_timestep['sat z ECEF [m]'].to_numpy().reshape(-1,1)
                    self.update_gnss_raw(pranges,sat_x,sat_y,sat_z)
                else:
                    lat_t = sat_timestep['Latitude'].to_numpy()[0]
                    lon_t = sat_timestep['Longitude'].to_numpy()[0]
                    alt_t = sat_timestep['Altitude'].to_numpy()[0]
                    self.update_gnss(lat_t,lon_t,alt_t)
            # add values to history
            self.mu_history = np.hstack((self.mu_history,self.mu))
            self.P_history.append(np.trace(self.P))
            bar.next() # progress bar


        bar.finish() # end progress bar
        # if finish with different num of items, spoof it.
        if len(self.times) + 1 == self.mu_history.shape[1]:
            self.mu_history = self.mu_history[:,:-1]
            self.P_history = self.P_history[:-1]

    def plot(self):
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False)
        plt.subplot(131)
        plt.plot(self.times,self.mu_history[0,:])
        plt.title("X vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("X [m]")

        plt.subplot(132)
        plt.plot(self.times,self.mu_history[1,:])
        plt.title("Y vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Y [m]")

        plt.subplot(133)
        plt.plot(self.times,self.mu_history[2,:])
        plt.title("Z vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Z [m]")

        # covariance plot
        plt.figure()
        plt.title("Trace of Covariance Matrix vs. Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Trace")
        plt.plot(self.times,self.P_history)

        # trajectory plot
        lla_traj = np.zeros((len(self.times),3))
        lon, lat, alt = pyproj.transform(self.ecef, self.lla, self.mu_history[0,:], self.mu_history[1,:], self.mu_history[2,:], radians=False)
        lla_traj[:,0] = lat
        lla_traj[:,1] = lon
        lla_traj[:,2] = alt
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False)
        plt.plot(lla_traj[:,1],lla_traj[:,0],label="Our Position Solution")

        if self.odom_file != None:
            lat_truth = self.odom_df['GPS(0):Lat[degrees]'].to_numpy()
            lon_truth = self.odom_df['GPS(0):Long[degrees]'].to_numpy()
            plt.plot(lon_truth,lat_truth,'g',label="DJI's Position Solution")
        plt.legend()
        plt.xlim([-122.1759,-122.1754])
        plt.ylim([37.42620,37.42660])
        plt.title("Trajectory")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.show()

if __name__ == '__main__':
    ekf = EKF('./data/android_fix_1.csv','./data/dji_data_flight_1.csv')
    ekf.run()
    ekf.plot()
