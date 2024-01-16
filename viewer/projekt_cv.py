'''
This script's objective is to capture and save single images, their respective poses and a sparse pointcloud of the captured scene.
First, the calibration data is saved to a file. 
For each press of the space button, a PV image and a depth image are captured.
These are then saved to directory together with their respective poses.
Furthermore, for each image, a pointcloud is calculated from the depth image. 
These get added together to form one large pointcloud of the captured scene. 

TODO:
- convert depth image to pointcloud (open3d_pointcloud.py)
- add all temporary pointclouds together

@Author Theodor Kapler
@Author Michael Fessenbecker
@Author Felix Birkelbach
'''

import open3d as o3d
from pynput import keyboard
import numpy as np
import multiprocessing as mp
import cv2
import hl2ss_imshow
import hl2ss_3dcv
import hl2ss
import hl2ss_mp
import time

# =============================================================================
# Initialize Keyboard Listener
# =============================================================================
class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press)
        self.space_pressed = False
        self.esc_pressed = False

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = True
        elif key == keyboard.Key.esc:
            self.esc_pressed = True

# =============================================================================
# Set Settings
# =============================================================================
# General Settings
# -----------------------------------------------------------------------------
host = '192.168.178.45' # Hololens address

ports = [
    hl2ss.StreamPort.PERSONAL_VIDEO,
    hl2ss.StreamPort.RM_DEPTH_LONGTHROW
    ]

buffer_elements = 60 # Maximum number of frames in buffer

# Settings for Personal Video
# -----------------------------------------------------------------------------
mode_pv = hl2ss.StreamMode.MODE_1 # Operating mode
port_pv = hl2ss.StreamPort.PERSONAL_VIDEO
chunksize_pv = hl2ss.ChunkSize.PERSONAL_VIDEO
width_pv = 1920
height_pv = 1080
framerate_pv = 30
profile_pv = hl2ss.VideoProfile.H265_MAIN
bitrate_pv = 5*1024*1024
decoded_format_pv = 'bgr24'
calibration_path_pv = '../calibration_pv'
poses_path_pv = '../poses_pv'
images_path_pv = '../images_pv'

# Settings for Depth image
# -----------------------------------------------------------------------------
mode_depth = hl2ss.StreamMode.MODE_1 # Operating mode
port_depth = hl2ss.StreamPort.RM_DEPTH_LONGTHROW
chunksize_depth = hl2ss.ChunkSize.RM_DEPTH_LONGTHROW
filter_depth = hl2ss.PngFilterMode.Paeth
max_depth = 3.0

# =============================================================================
# Calibrations and Producer, etc..
# Order of appearnace is important for the code!
# =============================================================================
# Start Subsystem for PV
# -----------------------------------------------------------------------------
hl2ss.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

# Compute and show PV Calibration data
# -----------------------------------------------------------------------------
calibration_pv = hl2ss.download_calibration_pv(host, port_pv, width_pv, height_pv, framerate_pv)
print('Calibration')
print(calibration_pv.focal_length)
print(calibration_pv.principal_point)
print(calibration_pv.radial_distortion)
print(calibration_pv.tangential_distortion)
print(calibration_pv.projection)
print(calibration_pv.intrinsics)

# Create Producer, Manager and Sinks for multi Sensor streaming
# -----------------------------------------------------------------------------
producer = hl2ss_mp.producer()
producer = hl2ss_mp.producer()
producer.configure_rm_depth_longthrow(True, host, port_depth, chunksize_depth, mode_depth, filter_depth)
producer.configure_pv(True, host, port_pv, chunksize_pv, mode_pv, width_pv, height_pv, framerate_pv, profile_pv, bitrate_pv, decoded_format_pv)

for port in ports:
    producer.initialize(port, buffer_elements)
    producer.start(port)

manager = mp.Manager()
consumer = hl2ss_mp.consumer()
sinks = {}

for port in ports:
    sinks[port] = consumer.create_sink(producer, port, manager, None)
    sinks[port].get_attach_response()

# Compute Depth Calibration Data for calculating pointcloud
# -----------------------------------------------------------------------------
calibration_depth = hl2ss_3dcv.get_calibration_rm(host, port_depth, '../calibration')
xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(calibration_depth.uv2xy, calibration_depth.scale)

# =============================================================================
# Capturing
# =============================================================================
listener = MyListener()
listener.start()

# Capture Loop
# ----------------------------------------------------------------------------- 
print('Ready to capture') 

xyz = np.empty((0, 3))

while True:
    if listener.esc_pressed:
        break

    if listener.space_pressed:
        _, data_pv = sinks[port_pv].get_most_recent_frame()
        _, data_depth = sinks[port_depth].get_most_recent_frame()
          
        if (data_pv is not None) and (data_depth is not None):
            print('Pose at time {ts}'.format(ts=data_pv.timestamp))
            print(data_pv.pose)
            print('Focal length')
            print(data_pv.payload.focal_length)
            print('Principal point')
            print(data_pv.payload.principal_point)
            cv2.imshow('Video', data_pv.payload.image)
            cv2.waitKey(1)
            
            depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, scale)
            xyz_tmp = hl2ss_3dcv.rm_depth_to_points(depth, xy1)
            xyz_tmp = hl2ss_3dcv.block_to_list(xyz_tmp)
            xyz_tmp = xyz_tmp[(xyz_tmp[:, 2] > 0) & (xyz_tmp[:, 2] < max_depth), :] 
            print(xyz_tmp.shape)
            xyz = np.vstack((xyz, xyz_tmp))
            print(xyz.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.visualization.draw_geometries([pcd])
            
        listener.space_pressed = False
# =============================================================================
# Closing Stuff
# =============================================================================       
for port in ports:
    sinks[port].detach()

for port in ports:
    producer.stop(port)

hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
