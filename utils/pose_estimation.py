import math

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation


def rotation_matrix_to_rotvec(R):
    """将旋转矩阵转换为旋转向量"""
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-10:
        return np.zeros(3)
    else:
        A = (R - R.T) / (2 * np.sin(theta))
        return theta * np.array([A[2,1], A[0,2], A[1,0]])


def rotvec_to_rotation_matrix(phi):
    """将旋转向量转换为旋转矩阵"""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3)
    else:
        n = phi / theta
        K = skew(n)
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def lie_algebra_to_pose(xi):
    """将6维李代数转换为4x4位姿矩阵"""
    rho = xi[:3]
    phi = xi[3:]

    # 旋转部分
    R = rotvec_to_rotation_matrix(phi)

    # 平移部分
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        t = rho
    else:
        J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * np.outer(phi, phi) / theta ** 2 + (
                    1 - np.cos(theta)) / theta ** 2 * skew(phi)
        t = J @ rho

    # 构建位姿矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def skew(v):
    """将3维向量转换为反对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def pose_to_lie_algebra(T):
    """将4x4位姿矩阵转换为6维李代数"""
    R = T[:3, :3]
    t = T[:3, 3]

    # 旋转部分
    phi = rotation_matrix_to_rotvec(R)

    # 平移部分
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        rho = t
    else:
        J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * np.outer(phi, phi) / theta ** 2 + (
                    1 - np.cos(theta)) / theta ** 2 * skew(phi)
        rho = np.linalg.solve(J, t)

    return np.concatenate([rho, phi])


def average_poses(poses):
    """
    使用李群李代数方法计算多个位姿矩阵的平均位姿

    参数:
        poses: N个4x4位姿矩阵组成的列表或数组，形状为(N,4,4)

    返回:
        4x4平均位姿矩阵
    """
    N = len(poses)

    # 步骤1: 将所有位姿转换为李代数表示
    lie_algebras = np.zeros((N, 6))
    for i, T in enumerate(poses):
        lie_algebras[i] = pose_to_lie_algebra(T)

    # 步骤2: 计算李代数的算术平均
    avg_lie_algebra = np.mean(lie_algebras, axis=0)

    # 步骤3: 将平均李代数映射回李群
    avg_pose = lie_algebra_to_pose(avg_lie_algebra)

    return avg_pose


def get_trans_error(src_pts, dest_pts, trans_T):
    src_pts_trans = trans_pts(src_pts, trans_T)
    pts_trans_errors = np.mean(np.sum((src_pts_trans - dest_pts) ** 2, axis=1) ** 0.5)
    return pts_trans_errors


def trans_pts(src_pts, trans_T):
    src_pts_t = np.ones((src_pts.shape[0], 4))
    src_pts_t[:, :3] = src_pts
    src_pts_tt = trans_T @ src_pts_t.T
    src_pts_tt = src_pts_tt.T
    return src_pts_tt[:, :3]


def get_T_from_rt_vec(rvec, tvec, degrees=False):
    rvec_p = rvec
    if degrees:
        rvec_p = rvec * np.pi / 180
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec_p)[0]
    # r = Rotation.from_rotvec(rvec.ravel(), degrees=False)
    # r_mat1 = r.as_matrix()

    T[:3, 3] = tvec.ravel()
    return T


def get_rt_vec_from_T(T, degrees=False):
    # r = Rotation.from_matrix(T[:3, :3])
    # rvec1 = r.as_rotvec(degrees=False)
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    tvec = T[:3, 3].reshape((3, 1))

    if degrees:
        rvec = rvec * 180 / np.pi
    return rvec, tvec


def rotation_matrix_to_rpy_xyz(rotation_matrix, degrees=True):
    """
    将旋转矩阵转换为 XYZ 顺序的 RPY 角
    :param rotation_matrix: 3x3 旋转矩阵
    :return: RPY 角（弧度） (roll, pitch, yaw)
    """
    R = rotation_matrix
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
         pitch = np.arctan2(-R[2, 0], sy)
         roll = np.arctan2(-R[1, 2], R[1, 1])
         yaw = 0

    if degrees:
        roll = roll * 180 / np.pi
        pitch = pitch * 180 / np.pi
        yaw = yaw * 180 / np.pi
    return np.array([roll, pitch, yaw])


def rpy_to_rotation_matrix_xyz(angle_rpy, degrees=True):
    """
    将 XYZ 顺序的 RPY 角转换为旋转矩阵
    :param roll: 翻滚角（弧度）
    :param pitch: 俯仰角（弧度）
    :param yaw: 偏航角（弧度）
    :return: 3x3 旋转矩阵
    """
    # 按 X-Y-Z 顺序旋转
    roll, pitch, yaw = angle_rpy
    if degrees:
        roll = roll * np.pi / 180
        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180
    roll_matrix = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
    pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

    rotation_matrix = np.dot(yaw_matrix, np.dot(pitch_matrix, roll_matrix))
    return rotation_matrix


def get_rt_zyx_from_T(T, degrees=False):
    rvec = get_zyx_from_rotmatix(T[:3, :3], degrees=degrees).reshape((3, 1))
    tvec = T[:3, 3].reshape((3, 1))
    return rvec, tvec


def get_zyx_from_rotmatix(R, degrees=False):
    """
    将旋转矩阵转换为ZYX顺序的RPY欧拉角（yaw, pitch, roll）。

    参数:
        R (np.ndarray): 3x3的旋转矩阵。

    返回:
        np.ndarray: 三个元素的数组，依次为绕Z轴、Y轴、X轴的旋转角度（弧度）。
    """
    beta = np.arcsin(-R[2, 0])
    cos_beta = np.cos(beta)

    # 处理奇异情况（万向节锁）
    if np.abs(cos_beta) < 1e-8:
        # 固定alpha为0，根据beta的符号计算gamma
        alpha = 0.0
        if beta > 0:  # beta = π/2
            gamma = np.arctan2(R[1, 2], R[1, 1])
        else:  # beta = -π/2
            gamma = np.arctan2(-R[1, 2], R[1, 1])
    else:
        # 非奇异情况，正常计算gamma和alpha
        gamma = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
        alpha = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)

    if degrees:
        gamma = gamma * 180 / np.pi
        beta = beta * 180 / np.pi
        alpha = alpha * 180 / np.pi

    return np.array([gamma, beta, alpha])

def get_T_from_rt_rpy_zyx(rpy_zyx, t_vec, degrees=False):
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation_matrix_ZYX(rpy_zyx, degrees=degrees)

    # T[:3, :3] = Rotation.from_euler("ZYX", rpy_zyx, degrees=degrees).as_matrix()
    T[:3, 3] = t_vec.ravel()
    return T


def get_T_from_rt_rpy_rzryrx(rpy_rzryrx, t_vec, degrees=False):
    T = np.eye(4)
    if degrees:
        rpy_rzryrx_proc = rpy_rzryrx[[2, 1, 0]].copy() * np.pi / 180.0
    else:
        rpy_rzryrx_proc = rpy_rzryrx[[2, 1, 0]].copy()
    T[:3, :3] = cv2.Rodrigues(rpy_rzryrx_proc)[0]
    T[:3, 3] = t_vec.ravel()
    return T


def rpy_to_rotation_matrix_ZYX(rpy, degrees=False):
    """
    将ZYX顺序的RPY欧拉角转换为旋转矩阵。

    参数:
        rpy (list或np.ndarray): 三个元素的数组，依次为绕Z轴、Y轴、X轴的旋转角度（弧度）。

    返回:
        np.ndarray: 3x3的旋转矩阵。
    """
    gamma, beta, alpha = rpy[0], rpy[1], rpy[2]
    if degrees:
        gamma = gamma * np.pi / 180
        beta = beta * np.pi / 180
        alpha = alpha * np.pi / 180
    # 绕Z轴的旋转矩阵
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # 绕Y轴的旋转矩阵
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # 绕X轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    # 组合旋转矩阵：R = Rz * Ry * Rx
    rotation_matrix = Rz @ Ry @ Rx
    return rotation_matrix


def rpy_to_rotation_matrix(angle_ZYX, degrees=True):
    """
    将RPY角转换为旋转矩阵
    :param roll: 翻滚角（rad）
    :param pitch: 俯仰角（rad）
    :param yaw: 偏航角（rad）
    :return: 3x3旋转矩阵
    """
    # 旋转矩阵的顺序是 Z-Y-X
    roll, pitch, yaw = angle_ZYX
    if degrees:
        roll = roll * np.pi / 180
        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    rotation_matrix = np.array([
        [cos_yaw * cos_roll - sin_yaw * sin_roll * sin_pitch,
         cos_yaw * sin_roll + sin_yaw * cos_roll * sin_pitch,
         cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll],
        [-sin_yaw * cos_roll - cos_yaw * sin_roll * sin_pitch,
         -sin_yaw * sin_roll + cos_yaw * cos_roll * sin_pitch,
         sin_yaw * cos_pitch * sin_roll + cos_yaw * cos_roll],
        [sin_pitch * cos_roll, sin_pitch * sin_roll, -sin_pitch * sin_roll]
    ])

    return rotation_matrix


def rotation_matrix_to_rpy_ds(rotation_matrix, degrees=True):
    """
    将旋转矩阵转换为XYZ顺序的RPY角（Roll, Pitch, Yaw）
    :param R: 3x3旋转矩阵（列表或数组）
    :return: 元组 (roll, pitch, yaw) 单位为弧度
    """
    R = rotation_matrix
    beta = math.asin(-R[2][0])  # 计算pitch（绕Y轴）

    # 处理万向节锁（Gimbal Lock）情况
    if abs(R[2][0]) >= 0.9999:
        beta = math.copysign(math.pi / 2, -R[2][0])
        # 设定gamma为0，计算alpha
        if beta > 0:  # beta为pi/2
            alpha = math.atan2(R[1][2], R[1][1])
        else:  # beta为-pi/2
            alpha = math.atan2(-R[1][2], R[1][1])
        gamma = 0.0
    else:
        cos_beta = math.cos(beta)
        alpha = math.atan2(R[1][0] / cos_beta, R[0][0] / cos_beta)  # yaw（绕Z轴）
        gamma = math.atan2(R[2][1] / cos_beta, R[2][2] / cos_beta)  # roll（绕X轴）

    if degrees:
        gamma = gamma * 180 / np.pi
        beta = beta * 180 / np.pi
        alpha = alpha * 180 / np.pi

    return np.array([gamma, beta, alpha])  # 返回顺序：roll(X), pitch(Y), yaw(Z)


def get_T_from_rt_rpy(rvec, tvec, degrees=False, rpy_seq='XYZ'):
    '''

    :param rvec:
    :param tvec:
    :param degrees:
    :param rpy_seq: rpy对应xyz轴转角顺序，LH--ZYX，DB--XYZ
    :return:
    '''
    T = np.eye(4)
    # r = Rotation.from_euler(rpy_seq, rvec.ravel(), degrees=degrees)
    # r1 = rpy_to_rotation_matrix(rvec.ravel()[[2, 1, 0]], degrees=degrees)
    # r = rpy_to_rotation_matrix_xyz(rvec.ravel(), degrees=degrees)
    # r_vec11 = r.as_euler(rpy_seq, degrees=degrees).reshape((3, 1))
    # T[:3, :3] = r.as_matrix()
    T[:3, :3] = rpy_to_rotation_matrix_xyz(rvec.ravel(), degrees=degrees)

    T[:3, 3] = tvec.ravel()
    return T


def get_rt_rpy_from_T(T, degrees=False, rpy_seq='XYZ'):
    # r = Rotation.from_matrix(T[:3, :3])
    # rvec1 = r.as_euler(rpy_seq, degrees=degrees).reshape((3, 1))

    rvec = rotation_matrix_to_rpy_xyz(T[:3, :3], degrees=degrees).reshape((3, 1))
    tvec = T[:3, 3].reshape((3, 1))
    return rvec, tvec


def get_trvec_rpy_from_T(T, degrees=False, rpy_seq='XYZ'):
    # r = Rotation.from_matrix(T[:3, :3])
    # rvec1 = r.as_euler(rpy_seq, degrees=degrees).reshape((3, 1))

    rvec = rotation_matrix_to_rpy_xyz(T[:3, :3], degrees=degrees).reshape((3, 1))
    tvec = T[:3, 3].reshape((3, 1))
    tr_vec = np.row_stack([tvec, rvec]).ravel()
    return tr_vec


def get_rt_rpy_zyx_from_T(T, degrees=False):
    # r = Rotation.from_matrix(T[:3, :3])
    # rvec1 = r.as_euler(rpy_seq, degrees=degrees).reshape((3, 1))

    rvec = rotation_matrix_to_rpy_xyz(T[:3, :3], degrees=degrees).reshape((3, 1))
    tvec = T[:3, 3].reshape((3, 1))
    return rvec, tvec

def rpy_to_vec(rpy, degrees=False, rpy_seq='XYZ'):
    # r = Rotation.from_euler(rpy_seq, rpy.ravel(), degrees=degrees)
    # rvec = r.as_rotvec(degrees=degrees).reshape(rpy.shape)

    rot = rpy_to_rotation_matrix_xyz(rpy.ravel(), degrees=degrees)
    rvec = cv2.Rodrigues(rot)[0]
    return rvec


def vec_to_rpy(vec, degrees=False, rpy_seq='XYZ'):
    # r = Rotation.from_rotvec(vec.ravel(), degrees=degrees)
    # rpy = r.as_euler(rpy_seq, degrees=degrees).reshape(vec.shape)

    r = cv2.Rodrigues(vec.ravel())[0]
    rpy = rotation_matrix_to_rpy_xyz(r, degrees=degrees).reshape(vec.shape)
    return rpy


def get_rigid_transform_o3d(src_pts, target_pts, ransac_flag=False, ransac_threshold=0.3):
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts.copy())

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts.copy())
    corres_vec = np.arange(src_pts.shape[0])
    corres_arr = np.row_stack([corres_vec, corres_vec]).T
    corr = o3d.utility.Vector2iVector(corres_arr)

    if ransac_flag:
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(src_pcd, target_pcd, corr,
                                                                                        ransac_threshold,
                                                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                            False),
                                                                                        ransac_n=3, criteria=
                                                                                        o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                            max_iteration=1000,
                                                                                            confidence=0.999))
        trans_T = result.transformation
    else:
        trans_T = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(src_pcd,
                                                                                                           target_pcd,
                                                                                                           corr)
    return trans_T


def get_rigid_transform_o3d_plane(src_pts, target_pts, max_distance=0.1, init_T=np.eye(4)):
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts.copy())

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts.copy())
    corres_vec = np.arange(src_pts.shape[0])
    corres_arr = np.row_stack([corres_vec, corres_vec]).T
    corr = o3d.utility.Vector2iVector(corres_arr)

    reg_p2plane = o3d.pipelines.registration.registration_icp(
        src_pcd, target_pcd, max_distance,  # 最大对应距离
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return reg_p2plane


def get_rigid_transform_cv2(src_pts, target_pts, ransac_flag=False, ransac_threshold=1.0):
    if ransac_flag:
        _, base2cam_T, inlier_mask = cv2.estimateAffine3D(src_pts, target_pts,
                                                          ransacThreshold=ransac_threshold,
                                                          confidence=0.99)
    else:
        _, base2cam_T, inlier_mask = cv2.estimateAffine3D(src_pts, target_pts, ransacThreshold=1e10,
                                                          confidence=1.0)

    if base2cam_T is None:
        trans_T = get_rigid_transform_o3d(src_pts, target_pts, ransac_flag)
    else:
        trans_T = np.eye(4)
        trans_T[:3, :] = base2cam_T
        if np.max(np.abs(base2cam_T[:, 2])) < 1e-8:
            trans_T[2, 2] = 1.0
        if np.max(np.abs(base2cam_T[2, :])) < 1e-8:
            trans_T[2, 2] = 1.0
            # trans_T[:3, :3] = trans_T[:3, :3].T

    return trans_T
