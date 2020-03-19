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
from mpl_toolkits.mplot3d import Axes3D

class EKF():
    def __init__(self,sat_file,odom_file=None):
        # read in data files as dataframe
        self.sat_df = pd.read_csv(sat_file, index_col=0)
        self.odom_file = odom_file
        if odom_file != None:
            self.odom_df = pd.read_csv(odom_file, index_col=0)

            # initial lat and lon from dji data
            self.lat0 = np.mean(self.odom_df['GPS(0):Lat[degrees]'][0])
            self.lon0 = np.mean(self.odom_df['GPS(0):Long[degrees]'][0])
            self.h0 = 0.0

            # convert lat lon to ecef frame
            self.lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
            self.ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            x0, y0, z0 = pyproj.transform(self.lla, self.ecef, self.lon0, self.lat0, self.h0 , radians=False)

            # concatenate possible time steps from each data file
            self.times = self.sat_df['seconds of week [s]'].to_numpy()

            # sort timesteps and force unique
            self.times = np.sort(np.unique(self.times))

            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx

            # indexes for comparing later
            self.sat_indexes = []
            for ii in range(len(self.odom_df['seconds of week [s]'].to_numpy())):
                ix = find_nearest(self.times,self.odom_df['seconds of week [s]'].values[ii])
                self.sat_indexes.append(ix)

        else:
            # set initial positions
            x0 = 0.
            y0 = 0.
            z0 = 0.

            self.lat0 = y0
            self.lon0 = x0
            self.h0 = z0

            # initialize times
            self.times = self.sat_df['seconds of week [s]'].to_numpy()



        # initialize state vector [ x, y, z ]
        self.mu = np.array([[x0,y0,z0,0.0]]).T
        self.mu_n = self.mu.shape[0]


        # initialize covariance matrix
        self.P = np.eye(self.mu_n)*10E2
        self.P_history = [np.trace(self.P)]

        self.wls_results = np.zeros((7,len(self.times)))

        # setup progress bar
        print("running weighted least squares, please wait...")
        bar = progress.bar.IncrementalBar('Progress:', max=len(self.times))

        for tt in range(len(self.times)):
            x_calc = [self.mu.item(0), self.mu.item(1), self.mu.item(2)]
            x_calc = [0.,0.,0.]
            bu_calc = 0.0
            input = self.sat_df[self.sat_df['seconds of week [s]'] == self.times[tt]]
            for ii in range(20):
                x_calc, bu_calc = self.least_squares(x_calc,bu_calc,input)
            self.wls_results[:3,tt] = x_calc
            self.wls_results[3,tt] = bu_calc
            bar.next()
        bar.finish()

        wls_lon, wls_lat, wls_alt = pyproj.transform(self.ecef, self.lla, self.wls_results[0,:], self.wls_results[1,:], self.wls_results[2,:], radians=False)
        self.wls_results[4,:] = wls_lat
        self.wls_results[5,:] = wls_lon
        self.wls_results[6,:] = wls_alt

        # plot all of the weighted least squares results
        if False:
            plt.figure()
            plt.subplot(231)
            plt.title("ECEF X vs Time")
            plt.plot(self.times,self.wls_results[0,:])

            plt.subplot(232)
            plt.title("ECEF Y vs Time")
            plt.plot(self.times,self.wls_results[1,:])

            plt.subplot(233)
            plt.title("ECEF Z vs Time")
            plt.plot(self.times,self.wls_results[2,:])

            plt.subplot(234)
            plt.title("Bu vs Time")
            plt.plot(self.times,self.wls_results[3,:])


            plt.subplot(235)
            plt.title("Lat/Lon vs Time")
            plt.plot(self.wls_results[5,:],self.wls_results[4,:])

            plt.subplot(236)
            plt.title("Altitude vs Time")
            plt.plot(self.times,self.wls_results[6,:])
            plt.show()

        self.mu[3][0] = bu_calc
        self.mu_history = self.mu.copy()

        # save to file
        df_wls = pd.DataFrame()
        df_wls['latitude'] = wls_lat
        df_wls['longitude'] = wls_lon
        df_wls['elevation'] = wls_alt
        df_wls.to_csv('./data/wls_calculated_trajectory.csv',index=False)

    def ECEF_2_ENU(self,x_ECEF,xref,lat0,lon0):
        """
        input(s)
            x_ECEF: 3 X N array in ECEF
            xref: reference [x,y,z] location
            lat0: latitude reference
            lon0: longitude reference
        output(s):
            x_ENU: 3 X N array in east north up
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

    def least_squares(self,x_0,bu,sat_df):
        """
        input(s)
            x:  [3 x 1] state estimate
            bu: clock bias
            sat_df: satellite data frame
        output(s)
            x:  [3 x 1] new state estimate
            bu: new clock bias
        """
        numSats = len(sat_df)
        dist = np.zeros((numSats,1))

        G = np.zeros((numSats,4))
        W = np.eye(numSats)
        for ii in range(numSats):
            x_s = sat_df['sat x ECEF [m]'].to_numpy()[ii]
            y_s = sat_df['sat y ECEF [m]'].to_numpy()[ii]
            z_s = sat_df['sat z ECEF [m]'].to_numpy()[ii]

            dist[ii] = np.sqrt((x_s-x_0[0])**2 + \
                               (y_s-x_0[1])**2 + \
                               (z_s-x_0[2])**2)
            G[ii,:] = [-(x_s - x_0[0])/dist[ii],
                       -(y_s - x_0[1])/dist[ii],
                       -(z_s - x_0[2])/dist[ii],
                       1.0]
            W[ii,ii] *= 1./sat_df['Pr_sigma'].to_numpy()[ii]

        c = 299792458.0

        rho_0 = dist + bu - sat_df['idk wtf this is'].to_numpy()[ii] * c # adjusting for relativity??
        rho_dif = sat_df['pr [m]'].to_numpy().reshape(-1,1) - rho_0
        # delta = np.linalg.inv(G.T.dot(G)).dot(G.T).dot(rho_dif)
        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_dif)
        x_new = x_0 + delta[0:3,0]
        bu_new = bu + delta[3,0]
        return x_new,bu_new

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

        # propagate covariance matrix
        self.P = F.dot(self.P).dot(F.T) + Q

    def update_gnss(self,mes,sat_x,sat_y,sat_z,sigmas,time_correction):
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
        H = np.zeros((num_sats,self.mu_n))
        h = np.zeros((num_sats,1))
        R = np.eye(num_sats)
        for ii in range(num_sats):
            dist = np.sqrt((sat_x[ii]-self.mu[0])**2 + (sat_y[ii]-self.mu[1])**2 + (sat_z[ii]-self.mu[2])**2)
            H[ii,0] = (self.mu[0]-sat_x[ii])/dist
            H[ii,1] = (self.mu[1]-sat_y[ii])/dist
            H[ii,2] = (self.mu[2]-sat_z[ii])/dist
            H[ii,3] = 1.0
            c = 299792458.0
            h[ii] = dist + self.mu[3] - time_correction[ii] * c # adjusting for relativity??
            R[ii,ii] *= sigmas[ii]**2
        yt = zt - h

        Kt = self.P.dot(H.T).dot(np.linalg.inv(R + H.dot(self.P).dot(H.T)))

        self.mu = self.mu.reshape((-1,1)) + Kt.dot(yt)
        self.P = (np.eye(self.mu_n)-Kt.dot(H)).dot(self.P).dot((np.eye(self.mu_n)-Kt.dot(H)).T) + Kt.dot(R).dot(Kt.T)

        yt = zt - H.dot(self.mu)


    def run(self):
        """
            Desc: run ekf
            Input(s):
                none
            Output(s):
                none
        """
        # setup progress bar
        print("running kalman filter, please wait...")
        bar = progress.bar.IncrementalBar('Progress:', max=len(self.times))

        for tt, timestep in enumerate(self.times):
            # update gnss step
            if self.sat_df['seconds of week [s]'].isin([timestep]).any():
                sat_timestep = self.sat_df[self.sat_df['seconds of week [s]'] == timestep]
                pranges = sat_timestep['pr [m]'].to_numpy().reshape(-1,1)
                sat_x = sat_timestep['sat x ECEF [m]'].to_numpy().reshape(-1,1)
                sat_y = sat_timestep['sat y ECEF [m]'].to_numpy().reshape(-1,1)
                sat_z = sat_timestep['sat z ECEF [m]'].to_numpy().reshape(-1,1)
                sigmas = sat_timestep['Pr_sigma'].to_numpy().reshape(-1,1)
                time_correction = sat_timestep['idk wtf this is'].to_numpy().reshape(-1,1)
                self.predict_simple()
                self.update_gnss(pranges,sat_x,sat_y,sat_z,sigmas,time_correction)

            # add values to history
            self.mu_history = np.hstack((self.mu_history,self.mu))
            self.P_history.append(np.trace(self.P))
            bar.next() # progress bar

        bar.finish() # end progress bar
        if len(self.times) + 1 == self.mu_history.shape[1]:
            self.mu_history = self.mu_history[:,:-1]
            self.P_history = self.P_history[:-1]

    def plot(self):
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False)
        plt.subplot(141)
        plt.plot(self.times,self.mu_history[0,:])
        plt.title("X vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("X [m]")

        plt.subplot(142)
        plt.plot(self.times,self.mu_history[1,:])
        plt.title("Y vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Y [m]")

        plt.subplot(143)
        plt.plot(self.times,self.mu_history[2,:])
        plt.title("Z vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Z [m]")

        plt.subplot(144)
        plt.plot(self.times,self.mu_history[3,:])
        plt.title("Time Bias vs Time")
        plt.xlabel("Time [hrs]")
        plt.ylabel("Time Bias [m]")

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
        plt.plot(lla_traj[:,1],lla_traj[:,0],'b',label='Our Position Solution')

        plt.title("Trajectory")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        if self.odom_file != None:
            lat_truth = self.odom_df['GPS(0):Lat[degrees]'].to_numpy()
            lon_truth = self.odom_df['GPS(0):Long[degrees]'].to_numpy()
            latf = self.odom_df['GPS(0):Lat[degrees]'].values[-1]
            lonf = self.odom_df['GPS(0):Long[degrees]'].values[-1]
            lat0 = self.odom_df['GPS(0):Lat[degrees]'][0]
            lon0 = self.odom_df['GPS(0):Long[degrees]'][0]
            plt.plot(lon0,lat0,'go')
            plt.plot(lonf,latf,'ro')
            plt.plot(lon_truth,lat_truth,'g',label="DJI's Position Solution")
        plt.legend()
        plt.xlim([-122.1759,-122.1754])
        plt.ylim([37.42620,37.42660])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(lla_traj[:,1], lla_traj[:,0], lla_traj[:,2], label='our solution')
        if self.odom_file != None:
            lat_truth = self.odom_df['GPS(0):Lat[degrees]'].to_numpy()
            lon_truth = self.odom_df['GPS(0):Long[degrees]'].to_numpy()
            h_truth = self.odom_df['GPS(0):heightMSL[meters]'].to_numpy()
            latf = self.odom_df['GPS(0):Lat[degrees]'].values[-1]
            lonf = self.odom_df['GPS(0):Long[degrees]'].values[-1]
            hf = self.odom_df['GPS(0):heightMSL[meters]'].values[-1]
            lat0 = self.odom_df['GPS(0):Lat[degrees]'][0]
            lon0 = self.odom_df['GPS(0):Long[degrees]'][0]
            h0 = self.odom_df['GPS(0):heightMSL[meters]'][0]
            plt.plot([lon0],[lat0],[h0],'go')
            plt.plot([lonf],[latf],[hf],'ro')
            plt.plot(lon_truth,lat_truth,h_truth,'g',label="DJI's Position Solution")

        ax.legend()
        ax.set_xlim([-122.1759,-122.1754])
        ax.set_ylim([37.42620,37.42660])
        ax.set_zlim([0.,60.])
        ax.view_init(elev=10., azim=20.)

        if self.odom_file != None:
            steps = np.arange(len(lat_truth))
            lat_error = np.abs(lat_truth-lla_traj[:,0][self.sat_indexes])
            lon_error = np.abs(lon_truth-lla_traj[:,1][self.sat_indexes])
            h_error = np.abs(h_truth-lla_traj[:,2][self.sat_indexes])
            print("lat error avg: ",np.mean(lat_error))
            print("lon error avg: ",np.mean(lon_error))
            print("h error avg: ",np.mean(h_error))
            plt.figure()
            plt.subplot(131)
            plt.title("Latitude Error [degrees latitude]")
            plt.ylabel("Latitude Error [degrees latitude]")
            plt.xlabel("Time Step")
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.plot(steps,lat_error)
            plt.subplot(132)
            plt.xlabel("Time Step")
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.title("Longitude Error [degrees longitude]")
            plt.ylabel("Longitude Error [degrees longitude]")
            plt.plot(steps,lon_error)
            plt.subplot(133)
            plt.xlabel("Time Step")
            plt.title("Altitude Error [m]")
            plt.ylabel("Altitude Error [m]")
            plt.plot(steps,h_error)

        # save to file
        df_traj = pd.DataFrame()
        df_traj['latitude'] = lla_traj[:,0]
        df_traj['longitude'] = lla_traj[:,1]
        df_traj['elevation'] = lla_traj[:,2]
        df_traj.to_csv('./data/calculated_trajectory.csv',index=False)

        plt.show()

if __name__ == '__main__':
    ekf = EKF('./data/sat_data_v2_flight_1.csv','./data/dji_data_flight_1.csv')
    ekf.run()
    ekf.plot()
