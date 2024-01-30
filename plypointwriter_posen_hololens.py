import numpy as np
import os
from pathlib import Path
from typing import List
import glob
import json



class PlyPointWriter:

    _POINT_COUNT_PLACEHOLDER = "<POINT COUNT>"

    class _AXES:
        X = np.array([1.0, 0.0, 0.0])
        Y = np.array([0.0, 1.0, 0.0])
        Z = np.array([0.0, 0.0, 1.0])

    class COLORS:
        RED = np.array([255, 0, 0])
        GREEN = np.array([0, 255, 0])
        BLUE = np.array([0, 0, 255])

    def __init__(
        self,
        file_path: str,
        use_color: bool = False,
        additional_property_identifiers : List[str] = None
    ):
        self._file_path = file_path
        self._use_color = use_color
        self._additional_property_identifiers = additional_property_identifiers

        self._temp_file_path = _get_file_path_with_postfix(
            file_path,
            "_temp"
        )

    def __enter__(
        self
    ):
        self.open()

        return self

    def __exit__(
        self, 
        type, 
        value, 
        traceback
    ):
        if value is not None:
            raise value

        self.close()

    def open(
        self
    ):
        self._point_count = 0
        self._temp_file = open(self._temp_file_path, "w")

        self._temp_file.write("ply\n")
        self._temp_file.write("format ascii 1.0\n")
        self._temp_file.write(f"element vertex {self._POINT_COUNT_PLACEHOLDER}\n")
        self._temp_file.write("property float x\n")
        self._temp_file.write("property float y\n")
        self._temp_file.write("property float z\n")

        if self._use_color:
            self._temp_file.write("property uchar red\n")
            self._temp_file.write("property uchar green\n")
            self._temp_file.write("property uchar blue\n")

        if self._additional_property_identifiers is not None:
            for additional_property_identifier in self._additional_property_identifiers:
                self._temp_file.write(f"property float {additional_property_identifier}\n")

        self._temp_file.write("element face 0\n")
        self._temp_file.write("property list uchar int vertex_indices\n")
        self._temp_file.write("end_header\n")
    
    def write_point(
        self,
        point: np.array,
        color: np.array = None,
        additional_properties : List[float] = None
    ):
        self._point_count += 1

        self._temp_file.write(f"{point[0]} {point[1]} {point[2]}")

        if self._use_color and color is not None:
            self._temp_file.write(f" {color[0]} {color[1]} {color[2]}")

        if self._additional_property_identifiers is not None \
                and additional_properties is not None:
            
            for additional_property in additional_properties:
                self._temp_file.write(f" {additional_property}")

        self._temp_file.write("\n")

    def write_points(
        self,
        points: np.array,
        color: np.array = None
    ):
        for i in range(points.shape[0]):

            self.write_point(
                points[i, :],
                color
            )

    def write_pose(
        self,
        pose: np.array,
        color: np.array = None,
        additional_properties : List[float] = None,
        axis_length: float = 1.0,
        point_distance: float = 0.001
    ):
        self._write_pose_axis(
            pose,
            self._AXES.X,
            self.COLORS.RED \
                if color is None \
                else color,
            additional_properties,
            axis_length,
            point_distance
        )

        self._write_pose_axis(
            pose,
            self._AXES.Y,
            self.COLORS.GREEN \
                if color is None \
                else color,
            additional_properties,
            axis_length,
            point_distance
        )

        self._write_pose_axis(
            pose,
            self._AXES.Z,
            self.COLORS.BLUE \
                if color is None \
                else color,
            additional_properties,
            axis_length,
            point_distance
        )

    def write_ray(
        self,
        ray_origin : np.array,
        ray_direction : np.array,
        additional_properties : List[float] = None,
        ray_length: float = 1.0,
        point_distance: float = 0.001,
        color: np.array = None
    ):
        for d in np.arange(
                0.0, 
                ray_length, 
                point_distance):
            
            self.write_point(
                ray_origin + d * ray_direction,
                color,
                additional_properties
            )

    def close(
        self
    ):
        self._temp_file.close()

        with open(self._temp_file_path, "r") as temp_file:
            with open(self._file_path, "w") as file:

                while True:
                    
                    line = temp_file.readline()

                    if len(line) == 0:
                        break

                    if self._POINT_COUNT_PLACEHOLDER in line:

                        line = line.replace(
                            self._POINT_COUNT_PLACEHOLDER,
                            str(self._point_count)
                        )
                    
                    file.write(line)
        
        os.remove(self._temp_file_path)
    
    def _write_pose_axis(
        self,
        pose: np.array,
        axis: np.array,
        color: np.array,
        additional_properties,
        axis_length: float,
        point_distance: float
    ):
        for d in np.arange(
                0.0, 
                axis_length, 
                point_distance):
            
            self.write_point(
                _pose_multiply(
                    pose,
                    d * axis
                ),
                color,
                additional_properties
            )

def _get_file_path_with_postfix(
    file_path: str,
    postfix: str
):
    path = Path(file_path)
    return f"{path.parent}/{path.stem}{postfix}{path.suffix}"

def _pose_multiply(
    pose: np.array,
    point: np.array
):
    return np.array([
        pose[0, 0] * point[0] + pose[0, 1] * point[1] + pose[0, 2] * point[2] + pose[0, 3],
        pose[1, 0] * point[0] + pose[1, 1] * point[1] + pose[1, 2] * point[2] + pose[1, 3],
        pose[2, 0] * point[0] + pose[2, 1] * point[1] + pose[2, 2] * point[2] + pose[2, 3]
    ])


if __name__ == "__main__":
    
    pose_list = []

    # Read poses from json file ./data/transforms_train.json
    with open('./data/transforms_train.json') as f:
        data = json.load(f)

    frames = data['frames']
    num_frames = len(frames)

    for i in range(num_frames):
        pose_list.append(np.array(frames[i]['transform_matrix']))

    with PlyPointWriter(
        file_path= './data/poses.ply',
        use_color = True
    ) as writer:
       for pose in pose_list:
           writer.write_pose(pose)


















    