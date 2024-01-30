import re
import json
import math
import numpy as np

def read_pose_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def parse_poses(content):
    pose_list = []
    pose_pattern = re.compile(r'Pose PV at time (\d+)(.*?)\[\[(.*?)\]\]', re.DOTALL)

    matches = pose_pattern.findall(content)
    for match in matches:
        timestamp = int(match[0])
        matrix_content = match[2].strip()
        matrix_lines = matrix_content.split('\n')

        # Extract numbers between square brackets and split on whitespace
        matrix = [list(map(float, re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line))) for line in matrix_lines]

        # Create numpy array
        matrix_np = np.array(matrix)

        # Define the rotation matrix (Rotation by x axis)
        rotation = [[1, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]]
        
        # Multiply the rotation matrix with the pose matrix
        matrix = rotation @ matrix_np

        # Rearange to a list of lists
        matrix = matrix_np.tolist()

        pose_list.append({
            "timestamp": timestamp,
            "transform_matrix": matrix
        })

    return pose_list

def create_json_output(poses, w, h, fl_x, fl_y, cx, cy, angle_x, angle_y):
    json_output = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": 0.,
        "k2": 0.,
        "p1": 0.,
        "p2": 0.,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": [
            {
                "file_path": f"./images/image_{pose['timestamp']}",
                "transform_matrix": [list(row) for row in zip(*pose["transform_matrix"])]  # Transpose the matrix
            }
            for pose in poses
        ]
    }

    return json_output

def save_json_output(output, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(output, json_file, indent=2)

# Parameters
w = 1920
h = 1080
fl_x = 1458.7411
fl_y = 1460.2667
cx = 941.4839
cy = 501.81952
angle_x = math.atan(w / (fl_x * 2)) * 2
angle_y = math.atan(h / (fl_y * 2)) * 2

if __name__ == "__main__":
    pose_file_path = "./data/poses.txt"
    json_output_path_train = "./data/transforms_train.json"
    json_output_path_test = "./data/transforms_test.json"

    pose_content = read_pose_file(pose_file_path)
    parsed_poses = parse_poses(pose_content)
    json_output = create_json_output(parsed_poses, w, h, fl_x, fl_y, cx, cy, angle_x, angle_y)
    save_json_output(json_output, json_output_path_train)
    save_json_output(json_output, json_output_path_test)
