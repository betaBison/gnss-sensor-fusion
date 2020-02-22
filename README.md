# gnss-sensor-fusion
Extended Kalman Filter (EKF) for position estimation using raw GNSS signals, IMU data, ...

## Setup
This repo was built using Python3. A number of packages are needed to run all files including at least:  
`pip3 install pandas numpy scipy pymap3d pyproj progress`

## Run
The full gnss sensor fusion can be run with:  
`python3 gnss_fusion_ekf.py`  
Change the filepaths at the end of the file to specify odometry and satellite data files

A simple 3D example can be run with:  
`python3 gnss_only_ekf_toy`
