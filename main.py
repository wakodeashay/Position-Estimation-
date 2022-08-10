## Importing Libraries
import numpy as np
import math
import random
import pandas as pd
# Libraries for generating gaussian multivariate random number
from numpy.random import randn
from scipy.stats import multivariate_normal
# Library for matrix multiplication
from numpy import linalg as LA
# Bagpy Library for extracting data from rosbag
import bagpy
from bagpy import bagreader
# Matplot library for plotting
import matplotlib.pyplot as plt


## Reading data from rosbag
# handle = bagreader('... address of file in local machine ...')
data = bagreader('/home/ashay/option_2/option_2/assignment_filter_state_estimation.bag')
# printing available topics
print(data.topic_table)
"""
Topics                   Types                                   Message       Count   Frequency
0  /assignement/ground_truth/odom                         nav_msgs/Odometry    12077   22.723625
1         /assignement/sensor/imu                           sensor_msgs/Imu    40827  100.881604
2        /assignement/sensor/pose   geometry_msgs/PoseWithCovarianceStamped      492    0.998954
3    /assignement/sensor/velocity  geometry_msgs/TwistWithCovarianceStamped     3752    9.755806
"""

# Topic = /assignement/ground_truth/odom
ground_truth = data.message_by_topic(topic='/assignement/ground_truth/odom')
g_t = pd.read_csv(ground_truth)
g_t_columns = list(g_t.head(0))
# Fields in ground truth csv file
"""
['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 
'header.frame_id', 'child_frame_id', 'pose.pose.position.x', 
'pose.pose.position.y', 'pose.pose.position.z', 'pose.pose.orientation.x',
'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w', 
'pose.covariance', 'twist.twist.linear.x', 'twist.twist.linear.y', 'twist.twist.linear.z', 
'twist.twist.angular.x', 'twist.twist.angular.y', 'twist.twist.angular.z', 'twist.covariance']
"""

# Topic = /assignement/sensor/imu
sensor_imu = data.message_by_topic(topic='/assignement/sensor/imu')
s_i = pd.read_csv(sensor_imu)
s_i_columns = list(s_i.head(0))
#print(s_i.get(''))
# Fields in imu sensor data csv file
"""
['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs',
'header.frame_id', 'orientation.x', 'orientation.y', 'orientation.z',
'orientation.w', 'orientation_covariance_0', 'orientation_covariance_1',
'orientation_covariance_2', 'orientation_covariance_3', 
'orientation_covariance_4', 'orientation_covariance_5', 'orientation_covariance_6', 
'orientation_covariance_7', 'orientation_covariance_8', 'angular_velocity.x', 
'angular_velocity.y', 'angular_velocity.z', 'angular_velocity_covariance_0',
'angular_velocity_covariance_1', 'angular_velocity_covariance_2',
'angular_velocity_covariance_3', 'angular_velocity_covariance_4',
'angular_velocity_covariance_5', 'angular_velocity_covariance_6',
'angular_velocity_covariance_7', 'angular_velocity_covariance_8',
'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
'linear_acceleration_covariance_0', 'linear_acceleration_covariance_1',
'linear_acceleration_covariance_2', 'linear_acceleration_covariance_3',
'linear_acceleration_covariance_4', 'linear_acceleration_covariance_5',
'linear_acceleration_covariance_6', 'linear_acceleration_covariance_7',
'linear_acceleration_covariance_8']
"""

# Topic = /assignement/sensor/pose
sensor_pose = data.message_by_topic(topic='/assignement/sensor/pose')
s_p = pd.read_csv(sensor_pose)
s_p_columns = list(s_p.head(0))                                             
# Fields in position sensor data csv file
"""
['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 
'pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z', 
'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 
'pose.pose.orientation.w', 'pose.covariance']
"""

# Topic /assignement/sensor/velocity
sensor_velocity = data.message_by_topic(topic='/assignement/sensor/velocity')
s_v = pd.read_csv(sensor_velocity)
s_v_columns = list(s_v.head(0))
# Fields in Odometry sensor data csv file
"""
['Time', 'header.seq', 'header.stamp.secs', 'header.stamp.nsecs', 'header.frame_id', 
'twist.twist.linear.x', 'twist.twist.linear.y', 'twist.twist.linear.z', 
'twist.twist.angular.x', 'twist.twist.angular.y', 'twist.twist.angular.z', 'twist.covariance']
"""

# Translational Velocity 
# Count - 3752; frequency - 9.755
v_x = np.array(s_v.get('twist.twist.linear.x'))
v_y = np.array(s_v.get('twist.twist.linear.y'))
v = np.power((np.add(np.power(v_x,2),np.power(v_y,2))),0.5)
f_v = 9.755

# Angular Velocity
# Count - 40827; frequency - 100.88
omega = np.array(s_i.get('angular_velocity.z'))
f_o = 100.88

# Postion measuremnts from sensor
# Count - 492; frequency - 0.998 
x_s = np.array(s_p.get('pose.pose.position.x'))
y_s = np.array(s_p.get('pose.pose.position.y'))


## Exteded kalman Filter
# number of states
n = 3
# number of inputs
p = 2
# number of measurements
m = 2
# Sampling time
ts = 0.1
# Frequency
f = 1/ts
# Total iteration, restricted by the number of measurement, for this case N = 492
N = len(x_s)
# Matrices
phi = np.array([[1,0,0],[0,1,0],[0,0,1]])
cmat = np.array([[1,0,0],[0,1,0]])
# Covariance matrix of measurement 
Qmat = np.diag((800,0.1))
# Covariance matrix of states
Rmat = np.diag((80,200,0.1))
# Predicted covariance matrix of estimated states
Ppred = np.eye(n)
# gamma_u function
def gamma_u(v,omega,alpha,ts):
    g_u = np.array([[(-math.sin(alpha) + math.sin(alpha+omega*ts))/omega,-v*(math.sin(alpha)+math.sin(alpha+omega*ts))/omega**2 +v*ts*math.cos(alpha+omega*ts)/omega],
    [(-math.cos(alpha) + math.cos(alpha+omega*ts))/omega,v*(math.cos(alpha)+math.cos(alpha+omega*ts))/omega**2 +v*ts*math.sin(alpha+omega*ts)/omega],
    [0,ts]])
    return g_u
## Predicted and estimated states
# Function for predicted states
def pred_states(x_n,y_n,alpha_n,v,omega,ts):
    p_s = np.array([[x_n + v*(-math.sin(alpha_n)+math.sin(alpha_n+omega*ts))/omega],
    [y_n - v*(math.cos(alpha_n)+math.cos(alpha_n+omega*ts))/omega],
    [alpha_n+omega*ts]])
    return p_s 

xhat_est = np.zeros((n,N))
# Quarternion will of the form ct+(st)j, other elements would be zero
ct = s_i.get('orientation.x')
st = s_i.get('orientation.w')
xhat_est[:,0] = np.array([x_s[0],y_s[0],math.asin(2*ct[0]*st[0])])
xhat_pred = np.zeros((n,N))
# Estimate covariance matrix of states
Pest = 10**(2)*np.eye(n)
# error between estimated and real measurement
e = np.zeros((m,N))
for k in range(0,N-1):
    xhat_pred[:,k+1:k+2] = pred_states(xhat_est[0,k],xhat_est[1,k],xhat_est[2,k],v[math.floor(k*f_v/f)],omega[math.floor(k*f_o/f)],ts)
    Ppred = np.matmul(np.matmul(phi,Pest),np.transpose(phi)) + Rmat
    inversemat = LA.inv(np.matmul(np.matmul(cmat,Ppred),np.transpose(cmat))+Qmat)
    L = np.matmul(np.matmul(Ppred,np.transpose(cmat)),inversemat)
    e[:,k+1:k+2] = np.array([[x_s[k+1]],[y_s[k+1]]]) - xhat_pred[0:2,k+1:k+2]
    xhat_est[:,k+1:k+2] = xhat_pred[:,k+1:k+2]+np.dot(L,e[:,k+1:k+2])
    Pest = np.dot(np.eye(n)-np.dot(L,cmat),Ppred)
## velocity Estimation
v_est_x = np.array([])
v_est_y = np.array([])
for i in range(0,N-1):
    v_est_x = np.append(v_est_x,[(xhat_est[0,i+1]-xhat_est[0,i])*10])
    v_est_y = np.append(v_est_y,[(xhat_est[1,i+1]-xhat_est[1,i])*10])

## Plotting graphs
plt.figure(1)
plt.plot(s_p.get('header.stamp.secs'), xhat_est[0,:],g_t.get('header.stamp.secs'), g_t.get('pose.pose.position.x'),s_p.get('header.stamp.secs'), s_p.get('pose.pose.position.x'))
plt.legend(['Estimate X','Ground Truth X', 'Measured X'])
plt.ylabel('x-coordinte')
plt.xlabel('time stamps in sec')
plt.show()

plt.figure(2)
plt.plot(s_p.get('header.stamp.secs'), xhat_est[1,:],g_t.get('header.stamp.secs'), g_t.get('pose.pose.position.y'),s_p.get('header.stamp.secs'), s_p.get('pose.pose.position.y'))
plt.legend(['Estimate Y','Ground Truth Y', 'Measured Y'])
plt.ylabel('y-coordinte')
plt.xlabel('time stamps in sec')
plt.show()

plt.figure(3)
l = s_p.get('header.stamp.secs')
plt.plot(s_v.get('header.stamp.secs'), s_v.get('twist.twist.linear.x'),l[0:491],v_est_x)
plt.legend(['Estimated velocity in X', 'Measured velocity in X'])
plt.ylabel('Velocity in X')
plt.xlabel('time stamps in sec')
plt.show()

plt.figure(4)
plt.plot(s_v.get('header.stamp.secs'),s_v.get('twist.twist.linear.y'),l[0:491],v_est_y)
plt.legend(['Estimated velocity in Y', 'Measured velocity in Y'])
plt.ylabel('Velocity in Y')
plt.xlabel('time stamps in sec')
plt.show()