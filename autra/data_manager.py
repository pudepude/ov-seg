import os
import json
import numpy as np

from glob import glob
from tqdm import tqdm

from autra.matric_util import rotation_vector_from_quaternion


TOPIC2CAMERANAME = {
    "/autra/sensor/lidar/hesai64" : "lidar",
    "/apollo/sensor/camera/upmiddle_left_30h/image/compressed" : "camera_upmiddle_left",
    "/apollo/sensor/camera/upmiddle_middle_120h/image/compressed" : "camera_upmiddle_middle",
    "/apollo/sensor/camera/upmiddle_right_60h/image/compressed" : "camera_upmiddle_right",
    "/apollo/sensor/camera/left_front_120h/image/compressed" : "camera_left_front",
    "/apollo/sensor/camera/left_backward_120h/image/compressed" : "camera_left_backward",
    "/apollo/sensor/camera/right_front_120h/image/compressed" : "camera_right_front",
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

ROI = {
    "camera_upmiddle_right" : (100, 100, 2780, 1760),
    "camera_upmiddle_middle" : (160, 230, 1750, 900),
    "camera_upmiddle_left" : (0, 0, 3840, 2160),
    "camera_left_front" : (160, 230, 1750, 1300),
    "camera_left_backward": (800, 100, 2780, 1760),
    "camera_right_front" : (160, 230, 1750, 1300),
    "camera_right_backward" : (100, 100, 1990, 1760)
}

CAMERA_PRIORITY_KEYS = [
    "camera_upmiddle_middle",
    "camera_left_front",
    "camera_right_front",
    "camera_upmiddle_right",
    "camera_left_backward",
    "camera_right_backward",
    "camera_upmiddle_left"
]

class DataManager:

    def __init__(self, root, lidar_result_path, vehicles=["Robin", "James", "ACrush"]):
        self._root = root
        self._lidar_result_path = lidar_result_path
        self._vehicles = set(vehicles)
        self._all_files = glob(f"{self._root}/**/msg_meta.json", recursive=True)
        # self._all_files = ['/mnt/cfs/agi/data/pretrain/sun/package_data/Robin-20230517_164224/00001/1684313544023-Robin/msg_meta.json',
        #                    '/mnt/cfs/agi/data/pretrain/sun/package_data/Robin-20230517_164224/00002/1684313544522-Robin/msg_meta.json',
        #                    '/mnt/cfs/agi/data/pretrain/sun/package_data/Robin-20230517_164224/00003/1684313545021-Robin/msg_meta.json',
        #                    '/mnt/cfs/agi/data/pretrain/sun/package_data/Robin-20230517_164224/00004/1684313545521-Robin/msg_meta.json',
        #                    '/mnt/cfs/agi/data/pretrain/sun/package_data/Robin-20230517_164224/00005/1684313546022-Robin/msg_meta.json']

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
                if "record_name" not in sample:
                    sample["record_name"] = sensor['record_name']
                    for word in current_path.split(os.sep):
                        if len(word) != len(sensor['record_name']): continue
                        if word.replace('-', '_') == sensor['record_name'].replace('-', '_'):
                            sample["record_name"] = word
                            break

                sensor_info['pose_quat'] = sensor['pose_with_exposure_latency']["value"]
                if camera_name == 'lidar':
                    sensor_info['origin_file'] = sensor_info['file']
                    sensor_info['file'] = os.sep.join([self._lidar_result_path, sample['record_name'], "lidar", sensor['relative_path']])
                    assert os.path.isfile(sensor_info['file']), sensor_info['file']
                    sensor_info['label'] = os.sep.join(
                        [self._lidar_result_path, sample['record_name'], "label", sensor['relative_path'][:-len(".pcd")] + ".json"])
                    assert os.path.isfile(sensor_info['file']), sensor_info['file']
                    assert os.path.isfile(sensor_info['label']), sensor_info['label']
                    continue
                sensor_info['intrinsics'] = sensor["intrinsics"]
                extrinsics = np.asarray(sensor['extrinsics']['value'])
                sensor_info['rotate_vectors'] = rotation_vector_from_quaternion(extrinsics[-4:])
                sensor_info['translation'] = extrinsics[:3]
                sensor_info['width'] = SIZE[camera_name][0]
                sensor_info['height'] = SIZE[camera_name][1]
                sensor_info['roi'] = ROI[camera_name]
            yield sample

    def __len__(self):
        return len(self._all_files)