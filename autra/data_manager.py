import os
import json
import numpy as np

from glob import glob
from tqdm import tqdm

from autra.matric_util import rotation_vector_from_quaternion


TOPIC2CAMERANAME = {
    "/autra/sensor/lidar/hesai64" : "lidar",
    "/apollo/sensor/camera/upmiddle_left_30h/image/compressed" : "camera_upmiddle_left",
#    "/apollo/sensor/camera/upmiddle_middle_120h/image/compressed" : "camera_upmiddle_middle",
    "/apollo/sensor/camera/upmiddle_right_60h/image/compressed" : "camera_upmiddle_right",
#    "/apollo/sensor/camera/left_front_120h/image/compressed" : "camera_left_front",
    "/apollo/sensor/camera/left_backward_120h/image/compressed" : "camera_left_backward",
#    "/apollo/sensor/camera/right_front_120h/image/compressed" : "camera_right_front",
    "/apollo/sensor/camera/right_backward_120h/image/compressed" : "camera_right_backward"
}

SIZE = {
    "camera_upmiddle_right" : (2880, 1860),
    "camera_upmiddle_middle" : (1920, 1536),
    "camera_upmiddle_left" : (3840, 2160),
    "camera_left_front" : (1920, 1536),
    "camera_left_backward": (2880, 1860),
    "camera_right_front" : (1920, 1536),
    "camera_right_backward" : (2880, 1860)
}

class DataManager:

    def __init__(self, root, vehicles=["Robin", "James", "ACrush"]):
        self._root = root
        self._vehicles = set(vehicles)
        self._all_files = glob(f"{self._root}/**/msg_meta.json", recursive=True)

    def __call__(self):
        for data_info_path in tqdm(self._all_files):
            current_path = os.path.dirname(data_info_path)
            data_info = json.load(open(data_info_path, "r"))
            if "vehicle_id" not in data_info:
                print(f"there is no key named vehicle_id in {data_info_path}")
                continue
            vehicle = data_info['vehicle_id']
            if vehicle not in self._vehicles:
                continue
            if not data_info.get("align", False):
                continue
            sample = dict()
            for sensor in data_info["sync_meta"]:
                topic = sensor['topic']
                if topic not in TOPIC2CAMERANAME:
                    continue
                camera_name = TOPIC2CAMERANAME[topic]
                sensor_data_path = os.sep.join([current_path, camera_name, sensor['relative_path']])
                assert os.path.isfile(sensor_data_path), sensor_data_path
                sensor_info = {"file" : sensor_data_path}
                sample[camera_name] = sensor_info
                sample["record_name"] = sensor['record_name']
                sensor_info['pose_quat'] = sensor['pose_with_exposure_latency']["value"]
                if camera_name == 'lidar':
                    continue
                sensor_info['intrinsics'] = sensor["intrinsics"]
                extrinsics = np.asarray(sensor['extrinsics']['value'])
                sensor_info['rotate_vectors'] = rotation_vector_from_quaternion(extrinsics[-4:])
                sensor_info['translation'] = extrinsics[:3]
                sensor_info['width'] = SIZE[camera_name][0]
                sensor_info['height'] = SIZE[camera_name][1]
            yield sample

    def __len__(self):
        return len(self._all_files)