import bagpy
import pandas as pd
from bagpy import bagreader
import matplotlib.pyplot as plt

data = bagreader('/home/ashay/option_2/option_2/assignment_filter_state_estimation.bag')
#print(data.topic_table)
#odom = data.odometry_data()
#data.
#od = pd.read_csv(odom[0])
#print(list(od.head(0)))
#imu = data.vel_data()
ground_truth = data.message_by_topic(topic='/assignement/ground_truth/odom')
g_t = pd.read_csv(ground_truth)
g_t_columns = list(g_t.head(0))
print(g_t_columns)
print('  ')

sensor_imu = data.message_by_topic(topic='/assignement/sensor/imu')
s_i = pd.read_csv(sensor_imu)
s_i_columns = list(s_i.head(0))
print(s_i_columns)
#print(s_i.get('header.stamp.secs'))
print('  ')


sensor_pose = data.message_by_topic(topic='/assignement/sensor/pose')
s_p = pd.read_csv(sensor_pose)
s_p_columns = list(s_p.head(0))                                             
print(s_p_columns)
#print(s_p.get("header.stamp.secs")) 
print('  ')

sensor_velocity = data.message_by_topic(topic='/assignement/sensor/velocity')
s_v = pd.read_csv(sensor_velocity)
s_v_columns = list(s_v.head(0))
print(s_v_columns)
#print(s_v.get('header.stamp.secs'))
print('  ')

#print(s_v.get('orientation.x'))
plt.plot(s_i.get('header.stamp.secs'), s_i.get('angular_velocity.z'))
#plt.plot(s_p.get('pose.pose.position.x'), s_p.get('pose.pose.position.y'))
#plt.plot(s_p.get('header.stamp.secs'),s_p.get('pose.pose.position.z'),label='z-sensor ')
plt.legend()
plt.show()