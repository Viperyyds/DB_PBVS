import numpy as np
import cv2

def two_view_triangulate(src_pixs, tar_pixs, src_K, src_dist, src_R, src_t, tar_K, tar_dist, tar_R, tar_t):
    pts_l_ = cv2.undistortPoints(np.expand_dims(src_pixs, axis=1), src_K, src_dist, None, src_K)
    pts_r_ = cv2.undistortPoints(np.expand_dims(tar_pixs, axis=1), tar_K, tar_dist, None, tar_K)

    M_r = np.hstack((tar_R, tar_t))
    M_l = np.hstack((src_R, src_t))

    P_l = np.dot(src_K, M_l)
    P_r = np.dot(tar_K, M_r)
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts_l_, pts_r_)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T
    return point_3d


def get_reproj_error(pixs1, pixs2, K1, dist1, R1, t1, K2, dist2, R2, t2):
    pts_3d = two_view_triangulate(pixs1, pixs2, K1, dist1, R1, t1, K2, dist2, R2, t2)
    proj1 = cv2.projectPoints(pts_3d, cv2.Rodrigues(R1)[0], t1, K1, dist1)[0].reshape(-1, 2)
    proj2 = cv2.projectPoints(pts_3d, cv2.Rodrigues(R2)[0], t2, K2, dist2)[0].reshape(-1, 2)
    # reproject_error = np.row_stack([proj1 - pixs1, proj2 - pixs2])
    # mean_error = np.mean(np.sum(reproject_error ** 2, axis=1) ** 0.5)
    reproject_error = (np.sum((proj1 - pixs1) ** 2, axis=1) ** 0.5 + np.sum((proj2 - pixs2) ** 2, axis=1) ** 0.5) / 2.0
    return reproject_error, pts_3d


def get_reproj_error_stereo(pixs1, pixs2, stereo_params):
    K1 = stereo_params['M1']
    dist1 = stereo_params['dist1']
    R1 = np.eye(3, dtype=np.float32)
    t1 = np.zeros((3, 1), dtype=np.float32)

    K2 = stereo_params['M2']
    dist2 = stereo_params['dist2']
    R2 = stereo_params['R']
    t2 = stereo_params['T'].reshape(3, 1)

    pts_3d = two_view_triangulate(pixs1, pixs2, K1, dist1, R1, t1, K2, dist2, R2, t2)
    proj1 = cv2.projectPoints(pts_3d, cv2.Rodrigues(R1)[0], t1, K1, dist1)[0].reshape(-1, 2)
    proj2 = cv2.projectPoints(pts_3d, cv2.Rodrigues(R2)[0], t2, K2, dist2)[0].reshape(-1, 2)
    # reproject_error = np.row_stack([proj1 - pixs1, proj2 - pixs2])
    reproject_error = (np.sum((proj1 - pixs1) ** 2, axis=1) ** 0.5 + np.sum((proj2 - pixs2) ** 2, axis=1) ** 0.5) / 2.0
    # mean_error = np.mean(np.sum(reproject_error ** 2, axis=1) ** 0.5)
    mean_error = np.mean(reproject_error)
    return reproject_error, pts_3d


def get_stereo_rectified_reproj_error(pts_l, pts_r, Pl, Pr):
    ptsl = pts_l.astype(np.float32).T  # 形状转为 (2, N)
    ptsr = pts_r.astype(np.float32).T

    points_4d = cv2.triangulatePoints(Pl, Pr, ptsl, ptsr)
    points_3d = points_4d[:3] / points_4d[3]  # 齐次坐标转非齐次 (3, N)

    points_homog = np.vstack((points_3d, np.ones(points_3d.shape[1])))

    # 投影到左相机
    proj_l = Pl @ points_homog
    proj_l = proj_l[:2] / proj_l[2]  # 归一化坐标 (2, N)

    # 投影到右相机
    proj_r = Pr @ points_homog
    proj_r = proj_r[:2] / proj_r[2]  # 归一化坐标 (2, N)
    reproject_error = (np.sum((proj_l - ptsl) ** 2, axis=0) ** 0.5 + np.sum((proj_r - ptsr) ** 2,
                                                                            axis=0) ** 0.5) / 2.0
    return reproject_error, points_3d.T


def rectify_stereo_sparse_points(points_left, points_right, stereo_params, PR1, PR2, PP1, PP2):
    K1 = stereo_params['M1']
    dist1 = stereo_params['dist1']
    R1 = np.eye(3, dtype=np.float32)
    t1 = np.zeros((3, 1), dtype=np.float32)

    K2 = stereo_params['M2']
    dist2 = stereo_params['dist2']
    R2 = stereo_params['R']
    t2 = stereo_params['T'].reshape(3, 1)

    # 数据预处理
    ptsL = points_left.astype(np.float32)
    ptsR = points_right.astype(np.float32)

    # 去畸变处理
    ptsL_undist = cv2.undistortPoints(
        ptsL, K1, dist1,
        None, PR1, PP1
    )

    ptsR_undist = cv2.undistortPoints(
        ptsR, K2, dist2,
        None, PR2, PP2
    )

    # 重投影到校正坐标系
    rect_left = cv2.convertPointsToHomogeneous(ptsL_undist).squeeze()
    rect_right = cv2.convertPointsToHomogeneous(ptsR_undist).squeeze()

    return rect_left[:, :2], rect_right[:, :2]