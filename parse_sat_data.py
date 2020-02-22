from lib.log_reader import MakeCsv, MakeCsvFix
import pandas as pd
import math
import numpy as np
from lib import gpstime
import time


# parse txt data into csv and output path
out_path = MakeCsv('./data/flight1.txt')

# read csv file into a dataframe
df = pd.read_csv(out_path)
print(df.shape)
# df = df[:30] # crop to the first 30

TimeNanos = df.TimeNanos.to_numpy()
FullBiasNanos = df.FullBiasNanos.to_numpy()
BiasNanos = df.BiasNanos.to_numpy()
TimeOffsetNanos = df.TimeOffsetNanos.to_numpy()
ReceivedSvTimeNanos = df.ReceivedSvTimeNanos.to_numpy()

ns_per_week = 604800E9
c = 299792458.

N_w = np.floor(-FullBiasNanos/ns_per_week)
t_Rx_hw = TimeNanos+TimeOffsetNanos
b_hw = FullBiasNanos + BiasNanos
t_Rx_GPS = t_Rx_hw - b_hw
t_Rx_w = t_Rx_GPS - N_w * ns_per_week
p_ns = t_Rx_w - ReceivedSvTimeNanos
p_ms = p_ns / 1.E6
print("p_ms",p_ms)
p_s = p_ms / 1000.
p_m = c * p_s
print("p_m",p_m)

# result = pd.DataFrame{'week'N_w}

results = pd.DataFrame({'SVid':df.Svid.to_numpy(),'ContellationType': df.ConstellationType.to_numpy(),'TDoA_[ms]':np.round(p_ms,decimals=3),'Pseudoranges_[m]':p_m.astype(dtype='int')})
results.to_csv('./data/python_data.csv')

# parse txt data into csv and output path
fix_out_path = MakeCsvFix('./data/flight1.txt')

fix_df = pd.read_csv(fix_out_path)

time_utc = fix_df['(UTC)TimeInMs'].to_numpy()
N = time_utc.shape[0]
# gps time: (gpsWeek, gpsSOW, gpsDay, gpsSOD)
time_gps = np.zeros((N,4))
for ii in range(N):
    dt_utc = time.gmtime(time_utc[ii]/1000.)
    year = dt_utc.tm_year
    month = dt_utc.tm_mon
    day = dt_utc.tm_mday
    hour = dt_utc.tm_hour
    min = dt_utc.tm_min
    sec = dt_utc.tm_sec
    print(year,month,day,hour,min,sec)
    time_gps[ii,:] = gpstime.gpsFromUTC(year,month,day,hour,min,sec)
    # print(time_gps[ii,1])

fix_df['seconds of week [s]'] = time_gps[:,1]

fix_df.to_csv('./data/android_fix_1.csv')
