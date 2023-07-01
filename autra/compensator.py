from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bisect import bisect
import numpy as np


class Compensator():
    def __init__(self, location_path, is_original_record=False):
        measurement_times = []
        rotations = []
        translations = []
        if is_original_record:
            assert False, "Not Supported"
        else:
            for line in open(location_path):
                arr = line.split(",")
                if len(arr) < 3: continue
                _, measurement_time, x, y, z, qw, qx, qy, qz =  map(float, arr[:9])
                orientation = np.array((qx, qy, qz, qw), dtype=np.float64)
                if np.sum(np.abs(orientation)) == 0:
                    continue
                translation_vec = np.array([x, y, z], dtype=np.float64)
                measurement_times.append(measurement_time)
                rotations.append(orientation)
                translations.append(translation_vec)

        measurement_times = (np.asarray(measurement_times) * 1e9).astype(np.uint64)
        rotations = np.asarray(rotations)
        translations = np.asarray(translations)
        idx = np.argsort(measurement_times)
        self.measurement_times, self.rotations, self.translations = measurement_times[idx], rotations[idx], translations[idx]
        self.rotations = R.from_quat(self.rotations)
        self.slerp = Slerp(self.measurement_times, self.rotations)

    def is_valid(self, t):
        return t >= self.measurement_times[0] and t <= self.measurement_times[-1]

    def trans(self, time_ns):
        if time_ns > self.measurement_times[-1] or time_ns < self.measurement_times[0]:
            raise KeyError(f"look up value {time_ns} is out of range [{self.measurement_times[0]}, {self.measurement_times[-1]}]")
        elif time_ns == self.measurement_times[-1]:
            return self.translations[-1]
        elif time_ns == self.measurement_times[0]:
            return self.translations[0]

        right = bisect(self.measurement_times, time_ns)
        assert right > 0
        left = right - 1
        time_ratio = (time_ns - self.measurement_times[left]) / (self.measurement_times[right] - self.measurement_times[left])
        target_position = (self.translations[right] - self.translations[left]) * time_ratio + self.translations[left]
        return target_position


    def compensate(self, points, ref_time_ns, target_time_ns):
        '''
        compensate lidar points from ref_time_ns to target_time_ns
        :param points: [N , 3]
        :param ref_time_ns:
        :param target_time_ns:
        :return: if points is not None a numpy array [N, 3] after compensate for points . else return a tuple of (quaternion, translation)
        '''
        if not self.is_valid(ref_time_ns) or not self.is_valid(target_time_ns):
            return None
        ref_rots = self.slerp([ref_time_ns])
        target_rot_inverse = self.slerp(target_time_ns).inv()
        rot = target_rot_inverse * ref_rots

        ref_trans = self.trans(ref_time_ns)
        target_trans = self.trans(target_time_ns)

        trans = target_rot_inverse.apply(ref_trans - target_trans)
        if points is None:
            return rot, trans
        if len(points) == 0:
            return []
        return rot.apply(points) + trans

    @classmethod
    def compensate_by_pose(cls, points, target_pose, ref_pose):
        x, y, z, qx, qy, qz, qw = ref_pose
        ref_trans = np.array([x, y, z], dtype=np.float64)
        ref_rots = R.from_quat(np.array([qx, qy, qz, qw]))
        x, y, z, qx, qy, qz, qw = target_pose
        target_rot_inverse = R.from_quat(np.array([qx, qy, qz, qw])).inv()
        rot = target_rot_inverse * ref_rots
        target_trans = np.array([x, y, z], dtype=np.float64)

        trans = target_rot_inverse.apply(ref_trans - target_trans)
        if points is None:
            return rot, trans
        if len(points) == 0:
            return []
        return rot.apply(points) + trans

    @classmethod
    def step2(cls, points, compensate):
        rot, trans = compensate
        if len(points) == 0:
            return []
        return rot.apply(points) + trans