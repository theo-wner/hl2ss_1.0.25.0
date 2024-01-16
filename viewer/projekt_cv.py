'''
This script's objective is to capture and save single images, their respective poses and a sparse pointcloud of the captured scene.
First, the calibration data is saved to a file. 
For each press of the space button, a PV image and a depth image are captured.
These are then saved to directory together with their respective poses.
Furthermore, for each image, a pointcloud is calculated from the depth image. 
These get added together to form one large pointcloud of the captured scene. 

TODO:
- Mask out Points further away than 3m
- Save pointcloud to file
- Save poses to file
- Save images to file
- Save calibration data to file
- Add command line arguments

@Author Theodor Kapler
@Author Michael Fessenbecker
@Author Felix Birkelbach
'''

from pynput import keyboard
import multiprocessing as mp
import numpy as np
import open3d as o3d
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import time

# =============================================================================
# Initialize Keyboard Listener
# =============================================================================
class MyListener:
    def __init__(self):
        self.space_pressed = False
        self.esc_pressed = False
        self.listener = keyboard.Listener(on_press=self.on_press, suppress=False)
        self.listener.start()

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
host = '172.21.98.21' # Hololens address
buffer_length = 10 # Buffer length in seconds

# Settings for Personal Video
# -----------------------------------------------------------------------------
mode_pv = hl2ss.StreamMode.MODE_1 # Operating mode
port_pv = hl2ss.StreamPort.PERSONAL_VIDEO
width_pv = 1920
height_pv = 1080
framerate_pv = 30

# Settings for Depth image
# -----------------------------------------------------------------------------
mode_depth = hl2ss.StreamMode.MODE_1 # Operating mode
port_depth = hl2ss.StreamPort.RM_DEPTH_LONGTHROW
width_depth = hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH
height_depth = hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT
framerate_depth = hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS

# =============================================================================
# Main function
# =============================================================================
if __name__ == '__main__':

    # Start Subsystem for PV
    # -----------------------------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, port_pv)

    # Start PV and RM Depth Long Throw streams
    # -----------------------------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(port_pv, hl2ss_lnm.rx_pv(host, port_pv, width=width_pv, height=height_pv, framerate=framerate_pv, decoded_format='rgb24'))
    producer.configure(port_depth, hl2ss_lnm.rx_rm_depth_longthrow(host, port_depth))
    producer.initialize(port_pv, framerate_pv * buffer_length)
    producer.initialize(port_depth, framerate_depth * buffer_length)
    producer.start(port_pv)
    producer.start(port_depth)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, port_pv, manager, None)
    sink_depth = consumer.create_sink(producer, port_depth, manager, ...)

    sink_pv.get_attach_response()
    sink_depth.get_attach_response()

    # Get RM Depth Long Throw calibration
    # Calibration data will be downloaded if it's not in the calibration folder
    # -----------------------------------------------------------------------------
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, port_depth, '../calibration')

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, width_depth, height_depth)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
 
    # Create Open3D visualizer
    # -----------------------------------------------------------------------------
    o3d_lt_intrinsics = o3d.camera.PinholeCameraIntrinsic(width_depth, height_depth, calibration_lt.intrinsics[0, 0], calibration_lt.intrinsics[1, 1], calibration_lt.intrinsics[2, 0], calibration_lt.intrinsics[2, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    first_pcd = True

    # Initialize Keyboard Listener
    # -----------------------------------------------------------------------------
    listener = MyListener()

    # =============================================================================
    # Main Loop
    # =============================================================================
    while True:
        vis.poll_events() # Wait for space button to be pressed
        vis.update_renderer()

        if listener.esc_pressed: # Exit program
            break

        if listener.space_pressed:
            # Get most recent PV and RM Depth Long Throw frames
            # -----------------------------------------------------------------------------
            _, data_lt = sink_depth.get_most_recent_frame()
            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                continue

            _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue

            # Undistort and normalize depth image
            # -----------------------------------------------------------------------------
            depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)

            # Calculate pointcloud from depth image
            # -----------------------------------------------------------------------------
            lt_points         = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
            lt_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
            world_points      = hl2ss_3dcv.transform(lt_points, lt_to_world)
            world_points = world_points.reshape(-1, 3)

            # Extend pointcloud
            # -----------------------------------------------------------------------------
            world_points = o3d.utility.Vector3dVector(world_points)
            pcd.points.extend(world_points)

            if (first_pcd):
                vis.add_geometry(pcd)
                first_pcd = False
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
                
            listener.space_pressed = False

    # =============================================================================
    # Close Stuff
    # =============================================================================
    sink_pv.detach()
    sink_depth.detach()

    producer.stop(port_pv)
    producer.stop(port_depth)

    hl2ss_lnm.stop_subsystem_pv(host, port_pv)
