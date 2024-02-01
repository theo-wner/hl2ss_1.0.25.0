'''
This script's objective is to capture and save single images, their respective poses and a sparse pointcloud of the captured scene.
First, the calibration data is saved to a file. 
For each press of the space button, a PV image and a depth image are captured.
These are then saved to directory together with their respective poses.
Furthermore, for each image, a pointcloud is calculated from the depth image. 
These get added together to form one large pointcloud of the captured scene. 

Press space to capture an image
Press escape to exit the program and show the final pointcloud
Press escape again to close the program

TODO:   
- Add command line arguments

@Author Theodor Kapler
@Author Michael Fessenbecker
@Author Felix Birkelbach
'''

from pynput import keyboard
from pynput import mouse
import multiprocessing as mp
import numpy as np
import open3d as o3d
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import shutil

# =============================================================================
# Initialize Keyboard Listener
# =============================================================================
class MyListener:
    def __init__(self):
        self.space_pressed = False
        self.esc_pressed = False
        self.left_pressed = False
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press, suppress=False)
        self.mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def on_key_press(self, key):
        if key == keyboard.Key.space:
            self.space_pressed = True
        elif key == keyboard.Key.esc:
            self.esc_pressed = True

    def on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.left_pressed = True

# =============================================================================
# Clear directory
# =============================================================================
# Remove folder data with all its subfolders and files
if shutil.os.path.exists('../data'):
    shutil.rmtree('../data')

# Create folder data and subfolder images
if not shutil.os.path.exists('../data'):
    shutil.os.mkdir('../data')
    shutil.os.mkdir('../data/images')

# =============================================================================
# Set Settings
# =============================================================================
# General Settings
# -----------------------------------------------------------------------------
host = '172.21.98.21' # Hololens address
buffer_length = 10 # Buffer length in seconds
voxel_size = 0.01 # Voxel Downsampling Parameter
max_depth = 3.0 # Maximum depth in meters

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
    calibration_depth = hl2ss_3dcv.get_calibration_rm(host, port_depth, '../calibration')

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_depth.intrinsics, width_depth, height_depth)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_depth.scale)

    # Create Open3D visualizer
    # -----------------------------------------------------------------------------
    o3d_lt_intrinsics = o3d.camera.PinholeCameraIntrinsic(width_depth, height_depth, calibration_depth.intrinsics[0, 0], calibration_depth.intrinsics[1, 1], calibration_depth.intrinsics[2, 0], calibration_depth.intrinsics[2, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    first_pcd = True

    # Initialize Keyboard Listener
    # -----------------------------------------------------------------------------
    listener = MyListener()

    # Open File to save poses
    # -----------------------------------------------------------------------------
    f = open('../data/poses.txt', 'w')

    # =============================================================================
    # Main Loop
    # =============================================================================
    while True:
        vis.poll_events() # Wait for space button to be pressed
        vis.update_renderer()

        if listener.esc_pressed: # Exit program, wait until user presses escape button again
            listener.esc_pressed = False
            while True:
                vis.poll_events()
                vis.update_renderer()
                if listener.esc_pressed:
                    # Generate normals for pointcloud
                    # -----------------------------------------------------------------------------
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    
                    # Save pointcloud to file
                    # -----------------------------------------------------------------------------
                    o3d.io.write_point_cloud('../data/points3d.ply', pcd, write_ascii=True, )
                    break
            break

        if listener.space_pressed:
            # Get most recent PV and RM Depth Long Throw frames
            # -----------------------------------------------------------------------------
            _, data_depth = sink_depth.get_most_recent_frame()
            if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
                continue

            _, data_pv = sink_pv.get_nearest(data_depth.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue
            
            # Show PV image
            # -----------------------------------------------------------------------------
            image_rgb = data_pv.payload.image
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image', image_bgr)
            cv2.waitKey(1)

            # Save PV image
            # -----------------------------------------------------------------------------
            img_name = f'image_{data_pv.timestamp}.png'
            cv2.imwrite(f'../data/images/{img_name}', image_bgr)

            # Print PV Pose and Intrinsics
            # -----------------------------------------------------------------------------
            print(f'Pose PV at time {data_pv.timestamp}')
            print(data_pv.pose)
            print(f'Focal length: {data_pv.payload.focal_length}')
            print(f'Principal point: {data_pv.payload.principal_point}')

            # Save PV Pose and Intrinsics into the same txt File
            # -----------------------------------------------------------------------------
            f.write(f'Pose PV at time {data_pv.timestamp}\n')
            f.write(str(data_pv.pose))
            f.write(f'\nFocal length: {data_pv.payload.focal_length}\n')
            f.write(f'Principal point: {data_pv.payload.principal_point}\n\n')

            # Undistort and normalize depth image
            # -----------------------------------------------------------------------------
            depth = hl2ss_3dcv.rm_depth_undistort(data_depth.payload.depth, calibration_depth.undistort_map)
            depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)

            # Mask out points further away than 3m
            # -----------------------------------------------------------------------------
            depth[depth > max_depth] = 0

            # Calculate pointcloud from depth image
            # -----------------------------------------------------------------------------
            depth_points         = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
            depth_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_depth.extrinsics) @ hl2ss_3dcv.reference_to_world(data_depth.pose)
            world_points      = hl2ss_3dcv.transform(depth_points, depth_to_world)
            world_points = world_points.reshape(-1, 3)

            # Extend old pointcloud
            # -----------------------------------------------------------------------------
            world_points = o3d.utility.Vector3dVector(world_points)
            pcd.points.extend(world_points)

            # Downsampling der Punktwolke
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)

            # Set color and size of pointcloud
            # -----------------------------------------------------------------------------
            pcd_downsampled.colors = o3d.utility.Vector3dVector([[.5, .5, .5]]*len(pcd_downsampled.points)) # Set color of pointcloud
            vis.get_render_option().point_size = 1 # Set size of points in pointcloud

            pcd.points = pcd_downsampled.points
            pcd.colors = pcd_downsampled.colors
            
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
    f.close()

    sink_pv.detach()
    sink_depth.detach()

    producer.stop(port_pv)
    producer.stop(port_depth)

    hl2ss_lnm.stop_subsystem_pv(host, port_pv)
