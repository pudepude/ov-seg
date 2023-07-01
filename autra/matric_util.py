import numpy as np
#旋转向量 到 旋转矩阵

def rotation_matrix_from_vector(rvector, translation=None):
    x, y, z = rvector[:, 0]
    theta = np.sqrt(x * x + y * y + z * z)
    r = rvector / theta
    first = np.cos(theta) * np.eye(3)
    second = (1 - np.cos(theta)) * np.dot(r, r.T)
    third = np.sin(theta) * np.array([
        [0, -r[2, 0], r[1, 0]],
        [r[2, 0], 0, -r[0, 0]],
        [-r[1, 0], r[0, 0], 0]
    ])
    final = first + second + third
    if translation is None: return final
    res = np.zeros((4, 4))
    res[:3, :3] = final
    res[:3, 3] = translation.squeeze()
    res[3, 3] = 1
    return res


# 旋转向量 到 四元式
def to_quaternion(rvector):
    x, y, z = rvector[:, 0]
    theta = np.sqrt(x * x + y * y + z * z)
    nx = x / theta
    ny = y / theta
    nz = z / theta
    half_theta = theta / 2
    cos_half_theta = np.cos(half_theta)
    sin_half_theta = np.sin(half_theta)
    return np.asarray([[nx * sin_half_theta],
    [ny * sin_half_theta],
    [nz * sin_half_theta],
    [cos_half_theta]])

#四元式到旋转矩阵
def rotation_matrix_from_quaternion(qvector):
    qx, qy, qz, qw = qvector[:, 0]
    R = np.asarray([
    [1- 2 * qy * qy - 2 * qz * qz, 2 * qx * qy + 2 * qw * qz, 2 * qx * qz - 2 * qw * qy],
    [2 * qx * qy - 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz + 2 * qw * qx],
    [2 * qx * qz + 2 * qw * qy, 2 * qy * qz - 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy]
    ])
    return R.T

def rotation_vector_from_quaternion(qvector):
    qx, qy, qz, qw = qvector.squeeze()
    half_theta = np.arccos(qw)
    sin_half_theta = np.sin(half_theta)
    nx = qx / sin_half_theta
    ny = qy / sin_half_theta
    nz = qz / sin_half_theta
    theta = half_theta * 2
    return nx * theta, ny * theta, nz * theta

def project(points, new_camera_matrix, rotate_vectors, translation):
    camera_xyz = (rotation_matrix_from_vector(np.array(rotate_vectors)) @ points.T).T + np.array(
        translation).reshape(1, 3)
    project_ps = (np.array(new_camera_matrix) @ camera_xyz.T).T
    project_ps = project_ps[:, :2] / project_ps[:, 2:3]
    return camera_xyz, project_ps

def camera_yaw_from_imu_yaw(yaw, rotate_vectors):
    # xy yaw in imu
    point = np.array([np.cos(yaw), np.sin(yaw), 0])
    npoint = rotation_matrix_from_vector(np.array(rotate_vectors)) @ point
    # xz yaw in camera
    npoint[1] = 0
    npoint = npoint / (1e-6 + np.sqrt(npoint[0] ** 2 + npoint[2] ** 2))
    cos_yaw, sin_yaw = npoint[0], npoint[2]
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw, sin_yaw, cos_yaw


def imu_yaw_from_camera_yaw(yaw, rotate_vectors):
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    point = np.array([cos_yaw, 0, sin_yaw])
    npoint = np.linalg.inv(rotation_matrix_from_vector(np.array(rotate_vectors))) @ point
    npoint[2] = 0
    npoint = npoint / (1e-6 + np.sqrt(npoint[0] ** 2 + npoint[1] ** 2))
    cos_yaw, sin_yaw = npoint[0], npoint[1]
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw, sin_yaw, cos_yaw

# rvec = np.asarray([[ 1.60843736],
#  [-0.03440913],
#  [ 0.00975872]])
#
# quaternion = to_quaternion(rvec)
# qx, qy, qz, qw = quaternion[:, 0]
# print(qx, qy, qz, qw)
# mat_from_qua = rotation_matrix_from_quaternion(quaternion)
# mat_from_rvec = rotation_matrix(rvec)
#
# print((mat_from_qua, mat_from_rvec))