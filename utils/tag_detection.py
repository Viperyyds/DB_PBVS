import copy
import time

import numpy as np
import cv2
# from PyCBD.pipelines import CBDPipeline
# from PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
from pyapriltags import Detector, Detection
from scipy.optimize import least_squares

from .pose_estimation import get_rigid_transform_o3d, get_T_from_rt_vec, get_rt_vec_from_T, trans_pts, \
    get_rigid_transform_cv2
from sklearn.metrics import pairwise_distances

from .stereo_triangulation import two_view_triangulate
from .tag_visualiser import draw_conner_box


class Aurco_Results:
    def __init__(self, tag_id, corners, center):
        self.tag_id = tag_id
        self.corners = corners
        self.center = center


def get_tag_results_dict(tags):
    tags_dict = {}
    for tag in tags:
        tags_dict[tag.tag_id] = tag

    return tags_dict


def get_tag_corners_3d_pts(tag_width, tag_height=None):
    if tag_height is None:
        tag_height = tag_width
    ob_pt1 = [-tag_width / 2, tag_height / 2, 0.0]
    ob_pt2 = [tag_width / 2, tag_height / 2, 0.0]
    ob_pt3 = [tag_width / 2, -tag_height / 2, 0.0]
    ob_pt4 = [-tag_width / 2, -tag_height / 2, 0.0]
    ob_pt5 = [0.0, 0.0, 0.0]
    obj_pts = [ob_pt1] + [ob_pt2] + [ob_pt3] + [ob_pt4] + [ob_pt5]
    obj_pts = np.array(obj_pts)
    return obj_pts


def get_closest_unique_match(src_pts, dest_pts, n_jobs=4, max_treshold=None):
    dist_matrix = pairwise_distances(src_pts, dest_pts, metric='euclidean', n_jobs=n_jobs)
    match_dest_ids = []
    MAX_V = 1e8
    if max_treshold is not None:
        MAX_V = max_treshold

    while True:
        cur_min_v = np.min(dist_matrix)
        if cur_min_v >= MAX_V:
            break
        min_ids = np.where(dist_matrix == cur_min_v)
        min_row = min_ids[0][0]
        min_col = min_ids[1][0]
        dist_matrix[min_row, :] = 1e8
        dist_matrix[:, min_col] = 1e8
        match_dest_ids.append([min_row, min_col])

    match_dest_ids = np.array(match_dest_ids)
    sorted_ids = np.argsort(match_dest_ids[:, 0])
    match_dest_ids = match_dest_ids[sorted_ids, :]
    return match_dest_ids


def get_accurate_corners(apriltag_corners, det_corners, max_treshold=None):
    match_dest_ids = get_closest_unique_match(apriltag_corners, det_corners, n_jobs=4, max_treshold=max_treshold)
    acc_corners = apriltag_corners.copy()
    acc_corners[match_dest_ids[:, 0], :] = det_corners[match_dest_ids[:, 1], :]
    return acc_corners


def perspective_normalization(img, src_pts, output_size=200, exte_size=30):
    """ 透视校正至标准视图 """
    dst_pts = np.array([[exte_size, exte_size], [exte_size, output_size + exte_size],
                        [output_size + exte_size, output_size + exte_size], [output_size + exte_size, exte_size]],
                       dtype=np.float32)

    # M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    # M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts.astype(np.float32))
    M = cv2.findHomography(src_pts, dst_pts)[0]
    M_inv = cv2.findHomography(dst_pts, src_pts.astype(np.float32))[0]
    warped = cv2.warpPerspective(img, M, (output_size + exte_size * 2, output_size + exte_size * 2),
                                 flags=cv2.INTER_CUBIC)

    # warped = cv2.warpAffine(img, M, (output_size + exte_size * 2, output_size + exte_size * 2))

    return warped, M_inv


def homo_trans(src_pts, homo):
    src_pts_p = np.ones((src_pts.shape[0], 3))
    src_pts_p[:, :2] = src_pts.copy()
    dest_pts_p = homo @ src_pts_p.T
    dest_pts = dest_pts_p / dest_pts_p[2, :]
    return dest_pts[:2, :].T


def get_rectified_patch(gray_img, patch_rect, calib_params):
    min_x, min_y, max_x, max_y = patch_rect

    scale_patch = gray_img[min_y:max_y, min_x:max_x]
    if calib_params is not None:
        rh, rw = max_y - min_x, max_x - min_x
        stt = time.time()
        K, dist_params, map_x, map_y = calib_params

        x = np.arange(min_x, max_x, dtype=np.float32)
        y = np.arange(min_y, max_y, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1).reshape(-1, 1, 2)

        # points = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]).reshape(-1, 1, 2).astype(np.float32)
        # 计算对应的原图坐标（通过畸变模型）
        undistorted_points = cv2.undistortPoints(points, K, dist_params, None, K)
        # 转换为映射表
        xmap = undistorted_points[:, 0, 0].reshape(rh, rw).astype(np.float32)
        ymap = undistorted_points[:, 0, 1].reshape(rh, rw).astype(np.float32)

        map_rx = xmap
        map_ry = ymap
        # undistorted_points = undistorted_points[:, 0, :].astype(np.int32)
        # min_rx, min_ry = np.min(undistorted_points, axis=0)
        # max_rx, max_ry = np.max(undistorted_points, axis=0)

        # min_rx, min_ry = min_x, min_y
        # max_rx, max_ry = max_x, max_y

        # map_rx = map_x[min_ry:max_ry, min_rx:max_rx]
        # map_ry = map_y[min_ry:max_ry, min_rx:max_rx]

        scale_patch = cv2.remap(
            gray_img, map_rx, map_ry,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT
        )
        patch_rect = [undistorted_points[0, 0, 0], undistorted_points[0, 0, 1], undistorted_points[-1, 0, 0],
                      undistorted_points[-1, 0, 0]]
        # patch_rect = [min_rx, min_ry, max_rx, max_ry]
        ctt = time.time() - stt
    return scale_patch, patch_rect


def edge_enhance_laplacian(img, ksize=3, alpha=0.7):
    """
    使用拉普拉斯算子增强边缘
    :param ksize: 滤波器大小 (推荐1或3)
    :param alpha: 锐化强度 (0.3-1.5)
    """
    # 应用高斯模糊去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 计算拉普拉斯二阶导数
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    # 转换为8位并叠加
    sharp = img - alpha * laplacian
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


def _sigmoid_contrast(img, alpha, beta):
    """S形曲线增强 - 增强中间调的同时保留极端值"""
    if img.dtype == np.uint8:
        enhanced = img.astype(np.float32) / 255.0
    else:
        enhanced = img.copy()
    # Sigmoid函数调整
    sigmoid = 1 / (1 + np.exp(-alpha * (enhanced - beta)))
    # 归一化到0-1
    sigmoid = (sigmoid - np.min(sigmoid)) / (np.max(sigmoid) - np.min(sigmoid))

    return (sigmoid * 255).astype(np.uint8)


def adaptive_gamma_correction(img, adaptive_gamma_range=[0.5, 1.5]):
    """自适应Gamma校正"""
    # 计算局部对比度图
    window_size = max(15, min(img.shape) // 20)
    local_std = cv2.boxFilter(img.astype(np.float32) ** 2, -1, (window_size, window_size))
    local_std = np.sqrt(local_std - cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size)) ** 2)

    # 归一化标准差
    local_std_norm = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-10)

    # 基于局部对比度计算Gamma值
    min_gamma, max_gamma = adaptive_gamma_range
    gamma_map = min_gamma + local_std_norm * (max_gamma - min_gamma)

    # 应用Gamma校正
    gamma_img = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            gamma_img[y, x] = np.power(img[y, x] / 255.0, gamma_map[y, x]) * 255

    return np.clip(gamma_img, 0, 255).astype(np.uint8)
    
def get_db_tag_grid_local_pts3d(tag_size=31.2, tag_sgap=4.5, tag_ids_lst=[18, 19, 20, 21]):
    tag_ids_arr = np.array(tag_ids_lst).reshape(2, 2)
    tag_grid_obj_dict = {}
    tag_corners_temp = get_tag_corners_3d_pts(tag_size, tag_size) - np.array([tag_size / 2.0, tag_size / 2.0, 0.0]).reshape(1, 3)
    for irow in range(tag_ids_arr.shape[0]):
        for ic in range(tag_ids_arr.shape[1]):
            sl_x = tag_size*ic + ic*tag_sgap
            sl_y = tag_size*irow + irow*tag_sgap
            tag_corners = tag_corners_temp.copy()
            tag_corners[:, 0] -= sl_x
            tag_corners[:, 1] -= sl_y

            tag_id = tag_ids_arr[irow, ic]
            tag_grid_obj_dict[tag_id] = tag_corners.copy()

    return tag_grid_obj_dict


def apriltag_two_stage_en_cv_rej_init(arucoDict, at_detector_params, at_detector_re, src_gray, proc_ratio, exter_size=7, det_tags=[],
                                  tag_size=55):
    resize_ratio = 1.0 / proc_ratio
    if proc_ratio != 1.0:
        proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        proc_gray = src_gray.copy()

    gamma = 3  # 伽马值 >1 降低亮度，<1 提高亮度
    lookup_table = ((np.arange(0, 256) / 255.0) ** gamma) * 255
    lookup_table = lookup_table.astype(np.uint8)
    proc_gray_f = proc_gray

    # apriltag_results = at_detector.detect(proc_gray_f, estimate_tag_pose=False, tag_size=tag_size)
    (corners, tg_ids, rejected) = cv2.aruco.detectMarkers(proc_gray_f, arucoDict, parameters=at_detector_params)

    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    # pre_corners = np.vstack([corners, rejected])
    pre_corners = corners
    if tg_ids is None:
        return refined_results
    for idx in range(len(pre_corners)):
        corner = pre_corners[idx][0, ...]

        min_x, min_y = np.min(corner, axis=0)
        max_x, max_y = np.max(corner, axis=0)

        min_x = int(max(0, min_x * proc_ratio - exter_size))
        min_y = int(max(0, min_y * proc_ratio - exter_size))
        max_x = int(min(image_size[0] - 1, max_x * proc_ratio + exter_size))
        max_y = int(min(image_size[1] - 1, max_y * proc_ratio + exter_size))

        scale_patch = src_gray[min_y:max_y, min_x:max_x]
        min_max_diff = np.max(scale_patch) - np.min(scale_patch)
        # if min_max_diff < 160:
        #     continue
        # cv2.imwrite(f'scale_patch_{idx}.png', scale_patch)

        scale_patch = _sigmoid_contrast(scale_patch, alpha=5.0, beta=0.9)

        patch_results = at_detector_re.detect(scale_patch, estimate_tag_pose=False, tag_size=tag_size)

        # # 3. 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (3, 3)  # 搜索窗口大小
        zero_zone = (-1, -1)  # 禁用死区

        if len(patch_results) > 0:
            corner_patch = patch_results[0]

            # patch_corners = corner_patch.corners.copy()
            # sub_p_corners = cv2.cornerSubPix(scale_patch, np.ascontiguousarray(patch_corners[:, np.newaxis, :].astype(np.float32)),
            #                                  win_size, (-1, -1), criteria)

            corner_patch.center[0] += min_x
            corner_patch.center[1] += min_y

            # acc_corners = get_accurate_corners(corner_patch.corners, harris_corners[:, 0, :], max_treshold=None)
            # corner_patch.corners[:, 0] = acc_corners[:, 0]
            # corner_patch.corners[:, 1] = acc_corners[:, 1]

            corner_patch.corners[:, 0] = corner_patch.corners[:, 0] + min_x
            corner_patch.corners[:, 1] = corner_patch.corners[:, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0] + min_x
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0]
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1]
            # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)

            refined_results.append(corner_patch)

    return refined_results


def apriltag_two_stage_en_cv_init(arucoDict, at_detector_params, at_detector_re, src_gray, proc_ratio, exter_size=7, det_tags=[],
                                  tag_size=55):
    resize_ratio = 1.0 / proc_ratio
    if proc_ratio != 1.0:
        proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        proc_gray = src_gray.copy()

    gamma = 3  # 伽马值 >1 降低亮度，<1 提高亮度
    lookup_table = ((np.arange(0, 256) / 255.0) ** gamma) * 255
    lookup_table = lookup_table.astype(np.uint8)
    proc_gray_f = proc_gray

    # apriltag_results = at_detector.detect(proc_gray_f, estimate_tag_pose=False, tag_size=tag_size)
    (corners, tg_ids, rejected) = cv2.aruco.detectMarkers(proc_gray_f, arucoDict, parameters=at_detector_params)

    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    if tg_ids is None:
        return refined_results
    for idx in range(len(tg_ids)):
        tag_id = tg_ids.ravel()[idx]
        if det_tags is not None and len(det_tags) > 0:
            if tag_id not in det_tags:
                continue
        corner = corners[idx][0, ...]

        min_x, min_y = np.min(corner, axis=0)
        max_x, max_y = np.max(corner, axis=0)

        min_x = int(max(0, min_x * proc_ratio - exter_size))
        min_y = int(max(0, min_y * proc_ratio - exter_size))
        max_x = int(min(image_size[0] - 1, max_x * proc_ratio + exter_size))
        max_y = int(min(image_size[1] - 1, max_y * proc_ratio + exter_size))

        scale_patch = src_gray[min_y:max_y, min_x:max_x]

        scale_patch = _sigmoid_contrast(scale_patch, alpha=5.0, beta=0.9)

        patch_results = at_detector_re.detect(scale_patch, estimate_tag_pose=False, tag_size=tag_size)

        # # 3. 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (3, 3)  # 搜索窗口大小
        zero_zone = (-1, -1)  # 禁用死区

        if len(patch_results) > 0:
            corner_patch = patch_results[0]

            # patch_corners = corner_patch.corners.copy()
            # sub_p_corners = cv2.cornerSubPix(scale_patch, np.ascontiguousarray(patch_corners[:, np.newaxis, :].astype(np.float32)),
            #                                  win_size, (-1, -1), criteria)

            corner_patch.center[0] += min_x
            corner_patch.center[1] += min_y

            # acc_corners = get_accurate_corners(corner_patch.corners, harris_corners[:, 0, :], max_treshold=None)
            # corner_patch.corners[:, 0] = acc_corners[:, 0]
            # corner_patch.corners[:, 1] = acc_corners[:, 1]

            corner_patch.corners[:, 0] = corner_patch.corners[:, 0] + min_x
            corner_patch.corners[:, 1] = corner_patch.corners[:, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0] + min_x
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0]
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1]
            # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)

            refined_results.append(corner_patch)

    return refined_results


def apriltag_two_stage_en_cv(arucoDict, at_detector_params, src_gray, proc_ratio, exter_size=7, det_tags=[],
                             at_detector_params_re=None):
    if at_detector_params_re is None:
        at_detector_params_re = at_detector_params
    resize_ratio = 1.0 / proc_ratio
    if proc_ratio != 1.0:
        proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        proc_gray = src_gray.copy()

    gamma = 3  # 伽马值 >1 降低亮度，<1 提高亮度
    lookup_table = ((np.arange(0, 256) / 255.0) ** gamma) * 255
    lookup_table = lookup_table.astype(np.uint8)
    proc_gray_f = proc_gray

    # apriltag_results = at_detector.detect(proc_gray_f, estimate_tag_pose=False, tag_size=tag_size)
    (corners, tg_ids, rejected) = cv2.aruco.detectMarkers(proc_gray_f, arucoDict, parameters=at_detector_params)

    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    for idx in range(len(tg_ids)):
        tag_id = tg_ids.ravel()[idx]
        if det_tags is not None and len(det_tags) > 0:
            if tag_id not in det_tags:
                continue
        corner = corners[idx][0, ...]

        corner_patch = Detection(tag_family=0, tag_id=tag_id, hamming=0, decision_margin=0,
                                 homography=np.eye(3), center=np.mean(corner, axis=0), corners=corner.copy())

        refined_results.append(corner_patch)

    return refined_results


def apriltag_two_stage_en(at_detector, src_gray, proc_ratio, exter_size=7, tag_size=0.05, det_tags=[],
                          at_detector_re=None):
    if at_detector_re is None:
        at_detector_re = at_detector
    resize_ratio = 1.0 / proc_ratio
    if proc_ratio != 1.0:
        proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        proc_gray = src_gray.copy()

    # # # img_l_g = proc_gray
    gamma = 3  # 伽马值 >1 降低亮度，<1 提高亮度
    gamma = 0.5
    # # lookup_table = np.array([((i / 255.0) ** gamma) * 255
    # #                          for i in np.arange(0, 256)]).astype("uint8")
    #
    # lookup_table = ((np.arange(0, 256) / 255.0) ** gamma) * 255
    # lookup_table = lookup_table.astype(np.uint8)
    #
    # #
    # img_l_g_f = cv2.LUT(proc_gray, lookup_table)
    #
    # proc_gray_f = img_l_g_f

    # proc_gray_f = _sigmoid_contrast(proc_gray, alpha=5.0, beta=0.8)
    # proc_gray_f = _sigmoid_contrast(proc_gray, alpha=3.0, beta=0.2)
    # cv2.imwrite("proc_gray111.png", proc_gray)
    # cv2.imwrite("img_l_g_re111.png", proc_gray_f)
    proc_gray_f = proc_gray

    apriltag_results = at_detector.detect(proc_gray_f, estimate_tag_pose=False, tag_size=tag_size)
    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    # show_img = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)

    for result in apriltag_results:
        tag_id = result.tag_id
        if det_tags is not None and len(det_tags) > 0:
            if tag_id not in det_tags:
                continue
        tag_family = result.tag_family
        center = result.center
        corner = result.corners

        min_x, min_y = np.min(corner, axis=0)
        max_x, max_y = np.max(corner, axis=0)

        min_x = int(max(0, min_x * proc_ratio - exter_size))
        min_y = int(max(0, min_y * proc_ratio - exter_size))
        max_x = int(min(image_size[0] - 1, max_x * proc_ratio + exter_size))
        max_y = int(min(image_size[1] - 1, max_y * proc_ratio + exter_size))

        # cv2.imwrite("scale_patch_raw.png", src_gray[min_y:max_y, min_x:max_x])
        # scale_patch, rectified_rect = get_rectified_patch(src_gray, [min_x, min_y, max_x, max_y], calib_params)
        scale_patch = src_gray[min_y:max_y, min_x:max_x]
        # cv2.imwrite("scale_patch_rectified.png", scale_patch)
        # scale_patch = cv2.GaussianBlur(scale_patch, (3, 3), 3)
        # cv2.imwrite("proc_gray.png", scale_patch)
        # scale_patch = _sigmoid_contrast(scale_patch, alpha=5.0, beta=0.9)

        # scale_patch = _sigmoid_contrast(scale_patch, alpha=6.0, beta=0.9)

        # scale_patch = adaptive_gamma_correction(scale_patch, adaptive_gamma_range=[0.5, 1.5])
        # cv2.imwrite("img_l_g_re.png", scale_patch)

        # min_x, min_y, max_x, max_y = rectified_rect
        # scale_patch_eq_img = cv2.equalizeHist(scale_patch)
        # cv2.imwrite("scale_patch.png", scale_patch)
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
        # scale_patch = clahe.apply(scale_patch)
        # cv2.imwrite("scale_patch_eq_img.png", scale_patch)
        # enhanced = global_normalization(scale_patch, target_mean=150, target_std=100)
        # img_l_g = scale_patch
        # img_l_g = cv2.LUT(scale_patch, lookup_table)
        # # #
        # scale_patch = img_l_g

        # contrast_lg = scale_patch.std()
        # print(f'str time cost: {np.round(ctt*1000, 1)}')
        #
        # cv2.imwrite("edge_sharped.png", edge_sharped)
        # scale_patch = edge_sharped
        # scale_patch = adaptive_log_mapping(scale_patch, base=10, L_max=100)
        # cv2.imwrite("scale_patch_log_mapping.png", scale_patch)
        # harris_corners = robust_harris_corner(scale_patch, useHarrisDetector=False, qualityLevel=0.01, min_distance=20)
        # fast_corners = multi_scale_fast(scale_patch, scales=[0.5, 1.0, 2.0])

        # detect_scale = 1
        # scale_patch = cv2.resize(scale_patch, None, fx=detect_scale, fy=detect_scale,
        #                          interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('scale_patch.png', scale_patch)
        # scale_patch = cv2.GaussianBlur(scale_patch, (3, 3), 3)
        #
        # scale_patch = cv2.bilateralFilter(scale_patch, 7, 11, 11)
        # scale_patch = cv2.GaussianBlur(scale_patch, (3, 3), 3)
        # # # #
        # scale_blur = cv2.GaussianBlur(scale_patch, (0, 0), 11)
        # scale_patch = cv2.addWeighted(scale_patch, 1.5, scale_blur, -0.5, 0.0)

        # cv2.imwrite('scale_patch_filtered.png', scale_patch)

        # scale_patch_w, H = perspective_normalization(scale_patch, corner - np.array([min_x, min_y]), output_size=300)

        # cv2.imwrite('scale_patch_w.png', scale_patch_w)

        patch_results = at_detector_re.detect(scale_patch, estimate_tag_pose=False, tag_size=tag_size)

        # max_corners = 100  # 最大角点数量
        # quality_level = 0.01  # 质量等级（最低可接受的角点质量）
        # min_distance = 10  # 角点之间的最小欧氏距离
        #
        # corners = cv2.goodFeaturesToTrack(
        #     scale_patch, max_corners, quality_level, min_distance
        # )
        #
        # # 3. 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)  # 搜索窗口大小
        zero_zone = (-1, -1)  # 禁用死区
        #
        # # corners需为浮点型
        # corners_subpix = cv2.cornerSubPix(
        #     scale_patch, np.float32(corners), win_size, zero_zone, criteria
        # )
        #
        # det_corners = corners[:, 0, :]
        # show_img = cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)
        # for corner in corners_subpix:
        #     x, y = corner.ravel()
        #     cv2.drawMarker(show_img, position=(int(x), int(y)), color=(0, 255, 0), markerSize=3,
        #                    markerType=cv2.MARKER_CROSS, thickness=1)
        #
        # for corner in patch_results[0].corners:
        #     x, y = corner.ravel()
        #     cv2.drawMarker(show_img, position=(int(x), int(y)), color=(0, 0, 255), markerSize=3,
        #                    markerType=cv2.MARKER_CROSS, thickness=1)
        # # patch_results = rescaled_results(patch_results, detect_scale)
        # cv2.imwrite("show_img_accurate_corners.png", show_img)

        if len(patch_results) > 0:
            corner_patch = patch_results[0]

            # center_p = corner_patch.center.reshape(-1, 2)
            # corners_p = corner_patch.corners
            # center_ph = homo_trans(center_p, H)
            # corners_ph = homo_trans(corners_p, H)
            #
            # corner_patch.center[0] = center_ph[0, 0]
            # corner_patch.center[1] = center_ph[0, 1]
            #
            # corner_patch.corners[:, 0] = corners_ph[:, 0]
            # corner_patch.corners[:, 1] = corners_ph[:, 1]

            # dist_mat = distance_matrix(corner_patch.corners, corners_re[:, 0, :])
            # m_ids = np.argmin(dist_mat, axis=1)
            # corner_patch.corners = corners_re[m_ids, 0, :]

            # patch_corners = corner_patch.corners.copy()
            # sub_p_corners = cv2.cornerSubPix(scale_patch, np.ascontiguousarray(patch_corners[:, np.newaxis, :].astype(np.float32)),
            #                                  win_size, (-1, -1), criteria)

            corner_patch.center[0] += min_x
            corner_patch.center[1] += min_y

            # acc_corners = get_accurate_corners(corner_patch.corners, harris_corners[:, 0, :], max_treshold=None)
            # corner_patch.corners[:, 0] = acc_corners[:, 0]
            # corner_patch.corners[:, 1] = acc_corners[:, 1]

            corner_patch.corners[:, 0] = corner_patch.corners[:, 0] + min_x
            corner_patch.corners[:, 1] = corner_patch.corners[:, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0] + min_x
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1] + min_y
            #
            # corner_patch.center[0] = np.mean(corner_patch.corners[:, 0])
            # corner_patch.center[1] = np.mean(corner_patch.corners[:, 1])

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0]
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1]
            # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)

            refined_results.append(corner_patch)
        # else:
        #     corner_patch = result
        #     corner_patch.center[0] = corner_patch.center[0] * proc_ratio
        #     corner_patch.center[1] = corner_patch.center[1] * proc_ratio
        #
        #     # if criteria is not None:
        #     #     corner_patch = cv2.cornerSubPix(gray, corner_patch, (5, 5), (-1, -1), criteria)
        #     corner_patch.corners[:, 0] = corner_patch.corners[:, 0] * proc_ratio
        #     corner_patch.corners[:, 1] = corner_patch.corners[:, 1] * proc_ratio
        #     # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)
        #     refined_results.append(corner_patch)

    #         for pix in corner_patch.corners:
    #             show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
    #
    # cv2.imwrite("show_img.png", show_img)
    return refined_results


def apriltag_two_stage(at_detector, src_gray, proc_ratio, exter_size=7, tag_size=0.05, det_tags=[],
                       at_detector_re=None):
    if at_detector_re is None:
        at_detector_re = at_detector
    resize_ratio = 1.0 / proc_ratio
    if proc_ratio != 1.0:
        proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        proc_gray = src_gray.copy()

    proc_gray_f = proc_gray

    apriltag_results = at_detector.detect(proc_gray_f, estimate_tag_pose=False, tag_size=tag_size)
    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    # show_img = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)

    for result in apriltag_results:
        tag_id = result.tag_id
        if det_tags is not None and len(det_tags) > 0:
            if tag_id not in det_tags:
                continue
        tag_family = result.tag_family
        center = result.center
        corner = result.corners

        min_x, min_y = np.min(corner, axis=0)
        max_x, max_y = np.max(corner, axis=0)

        min_x = int(max(0, min_x * proc_ratio - exter_size))
        min_y = int(max(0, min_y * proc_ratio - exter_size))
        max_x = int(min(image_size[0] - 1, max_x * proc_ratio + exter_size))
        max_y = int(min(image_size[1] - 1, max_y * proc_ratio + exter_size))

        # cv2.imwrite("scale_patch_raw.png", src_gray[min_y:max_y, min_x:max_x])
        # scale_patch, rectified_rect = get_rectified_patch(src_gray, [min_x, min_y, max_x, max_y], calib_params)
        scale_patch = src_gray[min_y:max_y, min_x:max_x]
        # cv2.imwrite("scale_patch_rectified.png", scale_patch)

        # min_x, min_y, max_x, max_y = rectified_rect
        # scale_patch_eq_img = cv2.equalizeHist(scale_patch)
        # cv2.imwrite("scale_patch_eq_img.png", scale_patch_eq_img)
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
        # enhanced = clahe.apply(scale_patch)

        # enhanced = global_normalization(scale_patch, target_mean=150, target_std=100)
        # cv2.imwrite("enhanced.png", enhanced)
        # stt = time.time()
        # contrast = scale_patch.std()
        # img_l_g = scale_patch
        # img_l_g = cv2.LUT(img_l_g, lookup_table)
        #
        # # scale_patch = img_l_g
        # contrast_lg = scale_patch.std()
        # print(f'str time cost: {np.round(ctt*1000, 1)}')
        #
        # cv2.imwrite("edge_sharped.png", edge_sharped)
        # scale_patch = edge_sharped
        # scale_patch = adaptive_log_mapping(scale_patch, base=10, L_max=100)
        # cv2.imwrite("scale_patch_log_mapping.png", scale_patch)
        # harris_corners = robust_harris_corner(scale_patch, useHarrisDetector=False, qualityLevel=0.01, min_distance=20)
        # fast_corners = multi_scale_fast(scale_patch, scales=[0.5, 1.0, 2.0])

        #
        # low_th = (1 - 0.95) * 255
        # high_th = (1 - 0.90) * 255
        #
        # img_l_g_re = img_l_g.copy()
        # img_l_g_re[img_l_g > high_th] = 255
        # img_l_g_re[img_l_g < low_th] = 0

        # detect_scale = 1
        # scale_patch = cv2.resize(scale_patch, None, fx=detect_scale, fy=detect_scale,
        #                          interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('scale_patch.png', scale_patch)
        # scale_patch = cv2.GaussianBlur(scale_patch, (3, 3), 3)
        #
        # scale_patch = cv2.bilateralFilter(scale_patch, 7, 11, 11)
        # scale_patch = cv2.GaussianBlur(scale_patch, (3, 3), 3)
        # # #
        # scale_blur = cv2.GaussianBlur(scale_patch, (0, 0), 11)
        # scale_patch = cv2.addWeighted(scale_patch, 1.5, scale_blur, -0.5, 0.0)

        # cv2.imwrite('scale_patch_filtered.png', scale_patch)

        # scale_patch_w, H = perspective_normalization(scale_patch, corner - np.array([min_x, min_y]), output_size=300)

        # cv2.imwrite('scale_patch_w.png', scale_patch_w)

        patch_results = at_detector_re.detect(scale_patch, estimate_tag_pose=False, tag_size=tag_size)

        # max_corners = 100  # 最大角点数量
        # quality_level = 0.01  # 质量等级（最低可接受的角点质量）
        # min_distance = 10  # 角点之间的最小欧氏距离
        #
        # corners = cv2.goodFeaturesToTrack(
        #     scale_patch, max_corners, quality_level, min_distance
        # )
        #
        # # 3. 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (3, 3)  # 搜索窗口大小
        # zero_zone = (-1, -1)  # 禁用死区
        #
        # # corners需为浮点型
        # corners_subpix = cv2.cornerSubPix(
        #     scale_patch, np.float32(corners), win_size, zero_zone, criteria
        # )
        #
        # det_corners = corners[:, 0, :]
        # show_img = cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)
        # for corner in corners_subpix:
        #     x, y = corner.ravel()
        #     cv2.drawMarker(show_img, position=(int(x), int(y)), color=(0, 255, 0), markerSize=3,
        #                    markerType=cv2.MARKER_CROSS, thickness=1)
        #
        # for corner in patch_results[0].corners:
        #     x, y = corner.ravel()
        #     cv2.drawMarker(show_img, position=(int(x), int(y)), color=(0, 0, 255), markerSize=3,
        #                    markerType=cv2.MARKER_CROSS, thickness=1)
        # # patch_results = rescaled_results(patch_results, detect_scale)
        # cv2.imwrite("show_img_accurate_corners.png", show_img)

        if len(patch_results) > 0:
            corner_patch = patch_results[0]

            # center_p = corner_patch.center.reshape(-1, 2)
            # corners_p = corner_patch.corners
            # center_ph = homo_trans(center_p, H)
            # corners_ph = homo_trans(corners_p, H)
            #
            # corner_patch.center[0] = center_ph[0, 0]
            # corner_patch.center[1] = center_ph[0, 1]
            #
            # corner_patch.corners[:, 0] = corners_ph[:, 0]
            # corner_patch.corners[:, 1] = corners_ph[:, 1]

            # dist_mat = distance_matrix(corner_patch.corners, corners_re[:, 0, :])
            # m_ids = np.argmin(dist_mat, axis=1)
            # corner_patch.corners = corners_re[m_ids, 0, :]

            # patch_corners = corner_patch.corners
            # sub_p_corners = cv2.cornerSubPix(scale_patch, np.ascontiguousarray(patch_corners[:, np.newaxis, :].astype(np.float32)),
            #                                  win_size, (-1, -1), criteria)

            corner_patch.center[0] += min_x
            corner_patch.center[1] += min_y

            # acc_corners = get_accurate_corners(corner_patch.corners, harris_corners[:, 0, :], max_treshold=None)
            # corner_patch.corners[:, 0] = acc_corners[:, 0]
            # corner_patch.corners[:, 1] = acc_corners[:, 1]

            corner_patch.corners[:, 0] = corner_patch.corners[:, 0] + min_x
            corner_patch.corners[:, 1] = corner_patch.corners[:, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0] + min_x
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1] + min_y

            # corner_patch.corners[:, 0] = sub_p_corners[:, 0, 0]
            # corner_patch.corners[:, 1] = sub_p_corners[:, 0, 1]
            # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)

            refined_results.append(corner_patch)
        # else:
        #     corner_patch = result
        #     corner_patch.center[0] = corner_patch.center[0] * proc_ratio
        #     corner_patch.center[1] = corner_patch.center[1] * proc_ratio
        #
        #     # if criteria is not None:
        #     #     corner_patch = cv2.cornerSubPix(gray, corner_patch, (5, 5), (-1, -1), criteria)
        #     corner_patch.corners[:, 0] = corner_patch.corners[:, 0] * proc_ratio
        #     corner_patch.corners[:, 1] = corner_patch.corners[:, 1] * proc_ratio
        #     # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)
        #     refined_results.append(corner_patch)

    #         for pix in corner_patch.corners:
    #             show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
    #
    # cv2.imwrite("show_img.png", show_img)
    return refined_results


def cv2_apriltag_two_stage(ARUCO_DICT, ARUCO_PARAMETERS, src_gray, proc_ratio, exter_size=7):
    resize_ratio = 1.0 / proc_ratio
    proc_gray = cv2.resize(src_gray, dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
    corners, ids, _ = cv2.aruco.detectMarkers(proc_gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    image_size = [src_gray.shape[1], src_gray.shape[0]]
    refined_results = []

    if ids is not None and len(ids) > 0:
        for i, corner in zip(ids, corners):
            min_x, min_y = np.min(corner[0, :, :], axis=0)
            max_x, max_y = np.max(corner[0, :, :], axis=0)

            min_x = int(max(0, min_x * proc_ratio - exter_size))
            min_y = int(max(0, min_y * proc_ratio - exter_size))
            max_x = int(min(image_size[0] - 1, max_x * proc_ratio + exter_size))
            max_y = int(min(image_size[1] - 1, max_y * proc_ratio + exter_size))

            scale_patch = src_gray[min_y:max_y, min_x:max_x]
            scale_patch = cv2.bilateralFilter(scale_patch, 7, 17, 17)
            scale_blur = cv2.GaussianBlur(scale_patch, (0, 0), 11)
            scale_patch = cv2.addWeighted(scale_patch, 1.5, scale_blur, -0.5, 0.0)

            corners_patch, id_patch, _ = cv2.aruco.detectMarkers(scale_patch, ARUCO_DICT,
                                                                 parameters=ARUCO_PARAMETERS)
            if len(corners_patch) > 0:
                corner_patch = corners_patch[0]
                corner_patch = corner_patch[0, :, :]
                # corner_patch.corners = corner_patch.corners.astype(np.int32).astype(np.float32)
                corner_patch[:, 0] = corner_patch[:, 0] + min_x
                corner_patch[:, 1] = corner_patch[:, 1] + min_y
                center_patch = np.mean(corner_patch, axis=0)
                refined_results.append(Aurco_Results(id_patch[0][0], corner_patch, center_patch))

            else:
                corner_patch = corner[0, ...]
                corner_patch[:, 0] = corner_patch[:, 0] * proc_ratio
                corner_patch[:, 1] = corner_patch[:, 1] * proc_ratio
                center_patch = np.mean(corner_patch, axis=0)
                refined_results.append(Aurco_Results(i[0], corner_patch, center_patch))

    return refined_results


def flip_bit(original_code: int, bit_position: int) -> int:
    # 创建位掩码 (将 1 左移指定位数)
    mask = 1 << bit_position
    # 通过异或操作翻转比特
    return original_code ^ mask


class DebugDetector(Detector):
    def detect(self, gray, return_quad_metadata=True):
        return super().detect(gray, return_quad_metadata=True)


def bundle_adjustment_residuals(params, obj_pts, left_img_pts, right_img_pts, stereo_cam_calib):
    """
    BA优化目标函数
    :param params: [left_rvec(3), left_tvec(3), right_rvec(3), right_tvec(3)]
    :param obj_pts: 世界坐标系下的3D点 (Nx3)
    :param left_img_pts: 左视图观测点 (Nx2)
    :param right_img_pts: 右视图观测点 (Nx2)
    :return: 残差向量
    """
    # 解析参数
    left_rvec = params[0:3]
    left_tvec = params[3:6]

    K0 = stereo_cam_calib['M1']
    dist0 = stereo_cam_calib['dist1']

    K1 = stereo_cam_calib['M2']
    dist1 = stereo_cam_calib['dist2']

    EXT_R1 = stereo_cam_calib['R']
    EXT_t1 = stereo_cam_calib['T'].reshape(3, 1)
    EXT_R_T = np.eye(4)
    EXT_R_T[:3, :3] = EXT_R1
    EXT_R_T[:3, 3] = EXT_t1.ravel()
    # 计算投影矩阵
    left_proj = get_T_from_rt_vec(left_rvec, left_tvec)
    right_proj = EXT_R_T @ left_proj

    # 计算重投影误差
    proj1 = cv2.projectPoints(obj_pts, cv2.Rodrigues(left_proj[:3, :3])[0], left_proj[:3, 3], K0, dist0)[0].reshape(-1,
                                                                                                                    2)
    proj2 = cv2.projectPoints(obj_pts, cv2.Rodrigues(right_proj[:3, :3])[0], right_proj[:3, 3], K1, dist1)[0].reshape(
        -1, 2)
    reproject_error = np.row_stack([proj1 - left_img_pts, proj2 - right_img_pts])
    mean_error = np.mean(np.sum(reproject_error ** 2, axis=1) ** 0.5)

    residuals = reproject_error.flatten()
    return residuals


def bundle_adjustment_residuals2(params, obj_pts, left_img_pts, right_img_pts, stereo_cam_calib):
    """
    BA优化目标函数
    :param params: [left_rvec(3), left_tvec(3), right_rvec(3), right_tvec(3)]
    :param obj_pts: 世界坐标系下的3D点 (Nx3)
    :param left_img_pts: 左视图观测点 (Nx2)
    :param right_img_pts: 右视图观测点 (Nx2)
    :return: 残差向量
    """
    # 解析参数
    left_rvec = params[0:3]
    left_tvec = params[3:6]
    pts_3d_est = params[6:21].reshape(5, 3)

    K0 = stereo_cam_calib['M1']
    dist0 = stereo_cam_calib['dist1']

    K1 = stereo_cam_calib['M2']
    dist1 = stereo_cam_calib['dist2']

    EXT_R1 = stereo_cam_calib['R']
    EXT_t1 = stereo_cam_calib['T'].reshape(3, 1)
    EXT_R_T = np.eye(4)
    EXT_R_T[:3, :3] = EXT_R1
    EXT_R_T[:3, 3] = EXT_t1.ravel()
    # 计算投影矩阵
    left_proj = get_T_from_rt_vec(left_rvec, left_tvec)
    right_proj = EXT_R_T @ left_proj

    true_trans_pts = trans_pts(obj_pts, left_proj)
    # 计算重投影误差
    proj1 = cv2.projectPoints(obj_pts, cv2.Rodrigues(left_proj[:3, :3])[0], left_proj[:3, 3], K0, dist0)[0].reshape(-1,
                                                                                                                    2)
    proj2 = cv2.projectPoints(obj_pts, cv2.Rodrigues(right_proj[:3, :3])[0], right_proj[:3, 3], K1, dist1)[0].reshape(
        -1, 2)
    reproject_error = np.column_stack([proj1 - left_img_pts, proj2 - right_img_pts, (pts_3d_est - true_trans_pts) * 10])
    # reproject_error = reproject_error[:-1, :]
    # reproject_error = (pts_3d_est - obj_pts).ravel()
    mean_error = np.mean(np.sum(reproject_error ** 2, axis=1) ** 0.5)

    residuals = reproject_error.flatten()
    return residuals


def bundle_adjustment_residuals1(params, obj_pts, stereo_cam_calib):
    """
    BA优化目标函数
    :param params: [left_rvec(3), left_tvec(3), right_rvec(3), right_tvec(3)]
    :param obj_pts: 世界坐标系下的3D点 (Nx3)
    :param left_img_pts: 左视图观测点 (Nx2)
    :param right_img_pts: 右视图观测点 (Nx2)
    :return: 残差向量
    """
    # 解析参数
    left_rvec = params[0:3]
    left_tvec = params[3:6]

    left_img_pts = params[6:16].reshape(5, 2)
    right_img_pts = params[16:26].reshape(5, 2)

    K0 = stereo_cam_calib['M1']
    dist0 = stereo_cam_calib['dist1']

    K1 = stereo_cam_calib['M2']
    dist1 = stereo_cam_calib['dist2']

    EXT_R1 = stereo_cam_calib['R']
    EXT_t1 = stereo_cam_calib['T'].reshape(3, 1)
    EXT_R_T = np.eye(4)
    EXT_R_T[:3, :3] = EXT_R1
    EXT_R_T[:3, 3] = EXT_t1.ravel()
    # 计算投影矩阵
    left_proj = get_T_from_rt_vec(left_rvec, left_tvec)
    right_proj = EXT_R_T @ left_proj

    pts_3d = two_view_triangulate(left_img_pts, right_img_pts, K0, dist0, np.eye(3), np.zeros((3, 1)), K1, dist1,
                                  EXT_R1, EXT_t1)
    true_trans_pts = trans_pts(obj_pts, left_proj)
    # 计算重投影误差
    proj1 = cv2.projectPoints(obj_pts, cv2.Rodrigues(left_proj[:3, :3])[0], left_proj[:3, 3], K0, dist0)[0].reshape(-1,
                                                                                                                    2)
    proj2 = cv2.projectPoints(obj_pts, cv2.Rodrigues(right_proj[:3, :3])[0], right_proj[:3, 3], K1, dist1)[0].reshape(
        -1, 2)
    reproject_error = np.column_stack([proj1 - left_img_pts, proj2 - right_img_pts, (pts_3d - true_trans_pts) * 10])
    mean_error = np.mean(np.sum(reproject_error ** 2, axis=1) ** 0.5)

    residuals = reproject_error.flatten()
    return residuals


def optimize_tag_pose(obj_pts, tag_3d_pts, left_pts, right_pts, init_rvec, init_tvec, stereo_calib_params):
    """执行光束法平差优化"""

    # 执行非线性优化
    # result = least_squares(
    #     bundle_adjustment_residuals,
    #     np.concatenate([init_rvec.ravel(), init_tvec.ravel()]),
    #     args=(obj_pts, left_pts, right_pts, stereo_calib_params),
    #     method='lm',
    #     ftol=1e-6,
    #     xtol=1e-6,
    #     max_nfev=100
    # )
    error_0 = bundle_adjustment_residuals2(np.concatenate([init_rvec.ravel(), init_tvec.ravel(), tag_3d_pts.ravel()]),
                                           obj_pts, left_pts, right_pts, stereo_calib_params)
    error_0_re = error_0.reshape(-1, 7)[:, :4] ** 2
    error_0_re_m = np.mean(
        np.row_stack([(error_0_re[:, 0] + error_0_re[:, 1]) ** 0.5, (error_0_re[:, 2] + error_0_re[:, 3]) ** 0.5]))
    result = least_squares(
        bundle_adjustment_residuals2,
        np.concatenate([init_rvec.ravel(), init_tvec.ravel(), tag_3d_pts.ravel()]),
        args=(obj_pts, left_pts, right_pts, stereo_calib_params),
        method='lm',
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=100
    )

    error_0 = bundle_adjustment_residuals2(result.x,
                                           obj_pts, left_pts, right_pts, stereo_calib_params)
    error_0_re = error_0.reshape(-1, 7)[:, :4] ** 2
    error_1_re_m = np.mean(
        np.row_stack([(error_0_re[:, 0] + error_0_re[:, 1]) ** 0.5, (error_0_re[:, 2] + error_0_re[:, 3]) ** 0.5]))

    # result = least_squares(
    #     bundle_adjustment_residuals1,
    #     np.concatenate([init_rvec.ravel(), init_tvec.ravel(), left_pts.ravel(), right_pts.ravel()]),
    #     args=(obj_pts, stereo_calib_params),
    #     method='trf',
    #     ftol=1e-6,
    #     xtol=1e-6,
    #     max_nfev=100
    # )

    return result.x[:3], result.x[3:6], result.x[6:21].reshape(5, 3)


def optimize_tag_pose_pixs(obj_pts, tag_3d_pts, left_pts, right_pts, init_rvec, init_tvec, stereo_calib_params):
    """执行光束法平差优化"""

    # 执行非线性优化
    # result = least_squares(
    #     bundle_adjustment_residuals,
    #     np.concatenate([init_rvec.ravel(), init_tvec.ravel()]),
    #     args=(obj_pts, left_pts, right_pts, stereo_calib_params),
    #     method='lm',
    #     ftol=1e-6,
    #     xtol=1e-6,
    #     max_nfev=100
    # )
    error_0 = bundle_adjustment_residuals1(
        np.concatenate([init_rvec.ravel(), init_tvec.ravel(), left_pts.ravel(), right_pts.ravel()]),
        obj_pts, stereo_calib_params)
    error_0_re = error_0.reshape(-1, 7)[:, :4] ** 2
    error_0_re_m = np.mean(
        np.row_stack([(error_0_re[:, 0] + error_0_re[:, 1]) ** 0.5, (error_0_re[:, 2] + error_0_re[:, 3]) ** 0.5]))
    # result = least_squares(
    #     bundle_adjustment_residuals2,
    #     np.concatenate([init_rvec.ravel(), init_tvec.ravel(), tag_3d_pts.ravel()]),
    #     args=(obj_pts, left_pts, right_pts, stereo_calib_params),
    #     method='lm',
    #     ftol=1e-6,
    #     xtol=1e-6,
    #     max_nfev=100
    # )

    result = least_squares(
        bundle_adjustment_residuals1,
        np.concatenate([init_rvec.ravel(), init_tvec.ravel(), left_pts.ravel(), right_pts.ravel()]),
        args=(obj_pts, stereo_calib_params),
        method='trf',
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=100
    )

    error_0 = bundle_adjustment_residuals1(result.x,
                                           obj_pts, stereo_calib_params)
    error_0_re = error_0.reshape(-1, 7)[:, :4] ** 2
    error_1_re_m = np.mean(
        np.row_stack([(error_0_re[:, 0] + error_0_re[:, 1]) ** 0.5, (error_0_re[:, 2] + error_0_re[:, 3]) ** 0.5]))
    left_opt_pts = result.x[6:16].reshape(5, 2)
    right_opt_pts = result.x[16:26].reshape(5, 2)
    return result.x[:3], result.x[3:6], left_opt_pts, right_opt_pts


def get_tag_board_pose(tag_poses_dict, tag_board_params, by_pnp=False, cam_params=None, tag_id_lst=None):
    board_cam_det_pts = []
    board_ref_pts = []
    tag_corners_arr = []
    f_det_tags = []
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst
    if cam_params is not None:
        K0, dist0 = cam_params
    else:
        K0, dist0 = np.eye(3), np.zeros(5)

    nn = 5
    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            if len(tag_poses_dict[d_tag_id]) == 3:
                d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
            else:
                d_tag_3d_corners, d_tag_pos, tag_corners, _ = tag_poses_dict[d_tag_id]
            nn = d_tag_3d_corners.shape[0]
            b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
            # b_3d_pts = np.array(b_3d_pts)
            board_cam_det_pts.append(d_tag_3d_corners)
            board_ref_pts.append(np.array(b_3d_pts).copy())
            tag_corners_arr.append(tag_corners)
            f_det_tags.append(d_tag_id)

    board_pose = None
    if len(board_cam_det_pts) > 0:
        board_cam_det_pts = np.row_stack(board_cam_det_pts)
        board_ref_pts = np.row_stack(board_ref_pts)
        tag_corners_arr = np.row_stack(tag_corners_arr)
        if by_pnp:
            _, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(board_ref_pts),
                                         np.ascontiguousarray(tag_corners_arr[:, :2]), K0, dist0)
            board_pose_init = get_T_from_rt_vec(rvec, np.zeros(3))
            initial_trans_pts = trans_pts(board_ref_pts, board_pose_init)
            t = np.mean(board_cam_det_pts - initial_trans_pts, axis=0)
            board_pose = get_T_from_rt_vec(rvec, t)
        else:
            th = 1
            board_pose = get_rigid_transform_o3d(board_ref_pts, board_cam_det_pts)
            # board_pose = get_rigid_transform_o3d(board_ref_pts, board_cam_det_pts, ransac_flag=True, ransac_threshold=th)
            # board_pose = get_rigid_transform_cv2(board_ref_pts, board_cam_det_pts, ransac_flag=False)
            initial_trans_pts = trans_pts(board_ref_pts, board_pose)
            # pts_trans_errors = np.mean(np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5)
            pts_trans_errors = np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5

            # opt_rvec, opt_tvec, pts3d_opt = optimize_tag_pose(tag_3d_ref_pts, tag_3d_pts, tag_corners0_e,
            #                                                   tag_corners1_e, init_rvec.ravel(), init_tvec.ravel(),
            #                                                   stereo_calb_params)

            valid_projs = pts_trans_errors < th
            # valid_tags = valid_projs.reshape(-1, nn)
            # valid_tags_f = np.sum(valid_tags, axis=1) < nn
            # valid_tags[valid_tags_f, :] = False
            # valid_projs = valid_tags.flatten()
            if np.sum(valid_projs) < 3:
                board_pose = get_rigid_transform_o3d(board_ref_pts, board_cam_det_pts)
            else:
                board_pose = get_rigid_transform_o3d(board_ref_pts[valid_projs, :], board_cam_det_pts[valid_projs, :])

    # print(f"flower det tags: {f_det_tags}")
    return board_pose


def get_tag_board_pose_trans_T(tag_poses_dict, tag_board_params, tag_id_lst=None):
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst

    f_pose_rtvec_lst = []
    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
            f_pose_T = d_tag_pos @ tag_board_params[d_tag_id][1]
            frvec, ftvec = get_rt_vec_from_T(f_pose_T, degrees=True)
            f_pose_rtvec_lst.append(np.row_stack([frvec, ftvec]).ravel())
    f_pose_rtvec_lst = np.array(f_pose_rtvec_lst)
    f_pose_rtvec_mean = np.mean(f_pose_rtvec_lst, axis=0)
    board_pose = get_T_from_rt_vec(f_pose_rtvec_mean[:3], f_pose_rtvec_mean[3:], degrees=True)
    return board_pose


def get_tag_board_pose_stereo_opt(tag_poses_dict, tag_board_params, pose_optimizer, tag_id_lst=None):
    board_cam_det_pts = []
    board_ref_pts = []
    tag_corners_arr = []
    f_det_tags = []
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst

    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
            nn = d_tag_3d_corners.shape[0]
            b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
            # b_3d_pts = np.array(b_3d_pts)
            board_cam_det_pts.append(d_tag_3d_corners)
            board_ref_pts.append(np.array(b_3d_pts).copy())
            tag_corners_arr.append(tag_corners)
            f_det_tags.append(d_tag_id)

    board_pose = None
    if len(board_cam_det_pts) > 0:
        board_cam_det_pts = np.row_stack(board_cam_det_pts)
        board_ref_pts = np.row_stack(board_ref_pts)
        tag_corners_arr = np.row_stack(tag_corners_arr)

        _, rvec0, tvec0 = cv2.solvePnP(np.ascontiguousarray(board_ref_pts),
                                       np.ascontiguousarray(tag_corners_arr[:, :2]), pose_optimizer.K1,
                                       pose_optimizer.dist1)
        initial_pose = (rvec0.ravel(), tvec0.ravel())
        optimized_pose, final_cost, precision_report = pose_optimizer.optimize_pose(
            board_ref_pts, tag_corners_arr[:, :2], tag_corners_arr[:, 2:], initial_pose, 1
        )
        board_pose = get_T_from_rt_vec(optimized_pose[:3], optimized_pose[3:])

    return board_pose


def residuals(p, objp, corners_l, corners_r, K1, dist1, K2, dist2, R_stereo, t_stereo):
    """代价函数：计算总重投影误差"""
    rvec_L = p[:3]
    tvec_L = p[3:]
    R_L, _ = cv2.Rodrigues(rvec_L)

    # 计算棋盘格在右相机坐标系下的位姿
    R_R = R_stereo @ R_L
    tvec_R = (R_stereo @ tvec_L.reshape(3, 1) + t_stereo.reshape(3, 1)).flatten()
    rvec_R, _ = cv2.Rodrigues(R_R)

    # 在左右相机中重投影
    img_pts_l_proj, _ = cv2.projectPoints(objp, rvec_L, tvec_L, K1, dist1)
    img_pts_r_proj, _ = cv2.projectPoints(objp, rvec_R, tvec_R, K2, dist2)

    # 计算误差
    error_l = (img_pts_l_proj - corners_l)
    error_r = (img_pts_r_proj - corners_r)
    error_l_p = np.sum(error_l[:, 0, :] ** 2, axis=1) ** 0.5
    error_r_p = np.sum(error_r[:, 0, :] ** 2, axis=1) ** 0.5
    # 返回拼接后的总误差向量
    return np.concatenate((error_l_p, error_r_p))


def find_chessboard_pose_stereo(corners_l_subpix, corners_r_subpix, objp, K1, dist1, K2, dist2, R_stereo, t_stereo):
    """
    使用已标定的双目相机精确求解棋盘格的位姿。

    Args:
        img_left, img_right (np.ndarray): 左右相机拍摄的图像。
        K1, dist1: 左相机的内参矩阵和畸变系数。
        K2, dist2: 右相机的内参矩阵和畸变系数。
        R_stereo, t_stereo: 右相机相对于左相机的旋转矩阵和平移向量。
        pattern_size (tuple): 棋盘格内部角点的数量 (e.g., (9, 6))。
        square_size (float): 棋盘格方块的边长 (米)。

    Returns:
        tuple: (rvec_final, tvec_final, R_mat_final)
            - rvec_final: 最终优化后的旋转向量 (棋盘格->左相机)。
            - tvec_final: 最终优化后的平移向量 (棋盘格->左相机)。
            - R_mat_final: 最终优化后的旋转矩阵 (棋盘格->左相机)。
        Returns None if corners are not found in both images.
    """
    # 4. 使用左相机进行初始位姿估计
    ret, rvec_initial, tvec_initial = cv2.solvePnP(objp, corners_l_subpix, K1, dist1, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ret:
        print("Error: Initial PnP solve failed.")
        return None

    # 5. 立体联合优化
    # 优化变量是6自由度位姿向量 [rvec, tvec]
    p_initial = np.concatenate((rvec_initial.flatten(), tvec_initial.flatten()))

    # 使用非线性最小二乘法求解
    opt_result = least_squares(
        fun=residuals,
        x0=p_initial,
        args=(objp, corners_l_subpix, corners_r_subpix, K1, dist1, K2, dist2, R_stereo, t_stereo),
        method='lm',  # Levenberg-Marquardt is robust
        verbose=2
    )

    p_final = opt_result.x
    rvec_final = p_final[:3]
    tvec_final = p_final[3:]
    R_mat_final, _ = cv2.Rodrigues(rvec_final)
    res = residuals(p_final, objp, corners_l_subpix, corners_r_subpix, K1, dist1, K2, dist2, R_stereo, t_stereo)

    return rvec_final, tvec_final, res


def get_tag_board_pose_repj(tag_poses_dict, tag_board_params, cam_params=None, tag_id_lst=None):
    board_cam_det_pts = []
    board_ref_pts = []
    f_det_tags = []
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst
    if cam_params is not None:
        K0, dist0, K1, dist1, stereo_R, stereo_t = cam_params
    else:
        K0, dist0 = np.eye(3), np.zeros(5)

    corners_l_subpix = []
    corners_r_subpix = []
    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            d_tag_3d_corners, d_tag_pos, cam0_corners, cam1_corners = tag_poses_dict[d_tag_id]
            nn = d_tag_3d_corners.shape[0]
            b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
            # b_3d_pts = np.array(b_3d_pts)
            board_cam_det_pts.append(d_tag_3d_corners)
            board_ref_pts.append(b_3d_pts)
            corners_l_subpix.append(cam0_corners)
            corners_r_subpix.append(cam1_corners)
            f_det_tags.append(d_tag_id)

    board_pose = None
    if len(board_cam_det_pts) > 0:
        board_cam_det_pts = np.row_stack(board_cam_det_pts)
        board_ref_pts = np.row_stack(board_ref_pts)
        corners_l_subpix = np.row_stack(corners_l_subpix)
        corners_r_subpix = np.row_stack(corners_r_subpix)

        rvec_final, tvec_final, res = find_chessboard_pose_stereo(corners_l_subpix[:, np.newaxis, :],
                                                                  corners_r_subpix[:, np.newaxis, :],
                                                                  board_ref_pts, K0, dist0, K1, dist1,
                                                                  stereo_R, stereo_t)

        board_pose = get_T_from_rt_vec(rvec_final, tvec_final)
        proj1 = cv2.projectPoints(board_ref_pts, rvec_final, tvec_final, K0, dist0)[
            0].reshape(-1, 2)

        error_pix = np.mean(np.sum((proj1 - corners_l_subpix) ** 2, axis=1) ** 0.5)

    # print(f"flower det tags: {f_det_tags}")
    return board_pose


def get_tag_board_pose_opt(tag_poses_dict, tag_board_params, stereo_calb_params, by_pnp=False, cam_params=None,
                           tag_id_lst=None):
    board_cam_det_pts = []
    board_ref_pts = []
    tag_corners_arr = []
    f_det_tags = []
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst
    if cam_params is not None:
        K0, dist0 = cam_params
    else:
        K0, dist0 = np.eye(3), np.zeros(5)

    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
            b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
            # b_3d_pts = np.array(b_3d_pts)
            board_cam_det_pts.append(d_tag_3d_corners)
            board_ref_pts.append(b_3d_pts)
            tag_corners_arr.append(tag_corners)
            f_det_tags.append(d_tag_id)

    board_pose = None
    if len(board_cam_det_pts) > 0:
        board_cam_det_pts = np.row_stack(board_cam_det_pts)
        board_ref_pts = np.row_stack(board_ref_pts)
        tag_corners_arr = np.row_stack(tag_corners_arr)
        if by_pnp:
            _, rvec, tvec = cv2.solvePnP(board_ref_pts, tag_corners_arr, K0, dist0)
            board_pose_init = get_T_from_rt_vec(rvec, np.zeros(3))
            initial_trans_pts = trans_pts(board_ref_pts, board_pose_init)
            t = np.mean(board_cam_det_pts - initial_trans_pts, axis=0)
            board_pose = get_T_from_rt_vec(rvec, t)
        else:
            board_pose = get_rigid_transform_o3d(board_ref_pts, board_cam_det_pts)
            # board_pose = get_rigid_transform_cv2(board_ref_pts, board_cam_det_pts, ransac_flag=False)
            initial_trans_pts = trans_pts(board_ref_pts, board_pose)
            # pts_trans_errors = np.mean(np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5)
            pts_trans_errors = np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5

            # opt_rvec, opt_tvec, pts3d_opt = optimize_tag_pose(tag_3d_ref_pts, tag_3d_pts, tag_corners0_e,
            #                                                   tag_corners1_e, init_rvec.ravel(), init_tvec.ravel(),
            #                                                   stereo_calb_params)

            # valid_projs = pts_trans_errors < 3
            # if np.sum(valid_projs) > 3:
            #     board_pose = get_rigid_transform_o3d(board_ref_pts[valid_projs, :], board_cam_det_pts[valid_projs, :])

        tag_rvec1, tag_tvec1 = get_rt_vec_from_T(board_pose)
        proj1 = cv2.projectPoints(board_ref_pts, tag_rvec1, tag_tvec1, K0, dist0)[
            0].reshape(-1, 2)

        error_pix = np.mean(np.sum((proj1 - tag_corners_arr) ** 2, axis=1) ** 0.5)

    # print(f"flower det tags: {f_det_tags}")
    return board_pose


def get_tag_board_pose1(tag_poses_dict, tag_board_params, by_pnp=False, cam_params=None, tag_id_lst=None):
    board_cam_det_pts = []
    board_ref_pts = []
    tag_corners_arr = []
    f_det_tags = []
    if tag_id_lst is None or len(tag_id_lst) == 0:
        proc_tag_ids = list(tag_poses_dict.keys())
    else:
        proc_tag_ids = tag_id_lst
    if cam_params is not None:
        K0, dist0 = cam_params
    else:
        K0, dist0 = np.eye(3), np.zeros(5)

    min_error = 1e5
    board_pose1 = None
    for d_tag_id in proc_tag_ids:
        if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
            d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
            b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
            board_cam_det_pts.append(d_tag_3d_corners)
            board_ref_pts.append(b_3d_pts)
            tag_corners_arr.append(tag_corners)
            f_det_tags.append(d_tag_id)
            if np.linalg.cond(np.array(tag2b_T)) > 1e10:
                b2tag_T = np.linalg.pinv(np.array(tag2b_T))
            else:
                b2tag_T = np.linalg.inv(np.array(tag2b_T))

            # b2tag_T = tag2b_T
            b_2cam_pose_T = d_tag_pos @ b2tag_T
            proj_error = np.mean(
                np.sum((d_tag_3d_corners - trans_pts(np.array(b_3d_pts), b_2cam_pose_T)) ** 2, axis=1) ** 0.5)
            if proj_error < min_error:
                min_error = proj_error
                board_pose1 = b_2cam_pose_T
    # board_pose = None
    # if len(board_cam_det_pts) > 0:
    #     board_cam_det_pts = np.row_stack(board_cam_det_pts)
    #     board_ref_pts = np.row_stack(board_ref_pts)
    #     tag_corners_arr = np.row_stack(tag_corners_arr)
    #     if by_pnp:
    #         _, rvec, tvec = cv2.solvePnP(board_ref_pts, tag_corners_arr, K0, dist0)
    #         board_pose_init = get_T_from_rt_vec(rvec, np.zeros(3))
    #         initial_trans_pts = trans_pts(board_ref_pts, board_pose_init)
    #         t = np.mean(board_cam_det_pts - initial_trans_pts, axis=0)
    #         board_pose = get_T_from_rt_vec(rvec, t)
    #     else:
    #         board_pose = get_rigid_transform_o3d(board_ref_pts, board_cam_det_pts)
    #         # board_pose = get_rigid_transform_cv2(board_ref_pts, board_cam_det_pts, ransac_flag=False)
    #         initial_trans_pts = trans_pts(board_ref_pts, board_pose)
    #         # pts_trans_errors = np.mean(np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5)
    #         pts_trans_errors = np.sum((initial_trans_pts - board_cam_det_pts) ** 2, axis=1) ** 0.5
    #         # valid_projs = pts_trans_errors < 1
    #         # if np.sum(valid_projs) > 3:
    #         #     board_pose = get_rigid_transform_o3d(board_ref_pts[valid_projs, :], board_cam_det_pts[valid_projs, :])
    #
    #     tag_rvec1, tag_tvec1 = get_rt_vec_from_T(board_pose)
    #     proj1 = cv2.projectPoints(board_ref_pts, tag_rvec1, tag_tvec1, K0, dist0)[
    #         0].reshape(-1, 2)
    #
    #     error_pix = np.mean(np.sum((proj1 - tag_corners_arr) ** 2, axis=1) ** 0.5)
    #
    # for d_tag_id in proc_tag_ids:
    #     if d_tag_id in tag_board_params and d_tag_id in tag_poses_dict:
    #         d_tag_3d_corners, d_tag_pos, tag_corners = tag_poses_dict[d_tag_id]
    #         b_3d_pts, tag2b_T = tag_board_params[d_tag_id]
    #
    #         if np.linalg.cond(np.array(tag2b_T)) > 1e10:
    #             b2tag_T = np.linalg.pinv(np.array(tag2b_T))
    #         else:
    #             b2tag_T = np.linalg.inv(np.array(tag2b_T))
    #         b_2cam_pose_T = d_tag_pos@b2tag_T
    #         proj_error = np.mean(np.sum((d_tag_3d_corners - trans_pts(np.array(b_3d_pts), b_2cam_pose_T))**2, axis=1)**0.5)
    #         if proj_error < min_error:
    #             min_error = proj_error
    #             board_pose1 = b_2cam_pose_T
    # print(f"flower det tags: {f_det_tags}")
    return board_pose1


def validate_detection(corners, H):
    """验证单应性矩阵精度"""
    # 理想角点坐标（标签坐标系）
    tag_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    # 重投影计算
    projected = cv2.perspectiveTransform(tag_corners.reshape(-1, 1, 2), H)

    # 计算误差
    error = np.linalg.norm(corners - projected.squeeze(), axis=1).mean()
    return error < 2.0  # 阈值可调


def gradient_refinement(image, corners, radius=5):
    """
    基于梯度方向的角点修正
    :param image: 灰度图像
    :param corners: 初始角点
    :param radius: 搜索半径
    :return: 优化后的角点
    """
    refined = []
    for (x1, y1) in corners:
        # 提取局部ROI
        x, y = int(x1), int(y1)
        roi = image[y - radius:y + radius + 1, x - radius:x + radius + 1]

        new_corners = cv2.goodFeaturesToTrack(roi, 1, 0.01, 10)[:, 0, :]
        new_corners = new_corners + np.array([x - radius, y - radius])
        diff = np.sum((new_corners - np.array([x1, y1])) ** 2, axis=1) ** 0.5
        nn_idx = np.argmin(diff)
        refined.append(new_corners[nn_idx, :])
        # # 计算梯度
        # dx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        # dy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        #
        # # 寻找梯度最大点
        # mag = np.sqrt(dx ** 2 + dy ** 2)
        # max_idx = np.argmax(mag)
        # dy, dx = np.unravel_index(max_idx, roi.shape)
        #
        # # 更新坐标
        # refined.append((x + dx - radius, y + dy - radius))

    return np.array(refined)


def refine_corners_subpix(image, corners, win_size=5, max_iters=50):
    """
    亚像素级角点修正
    :param image: 输入灰度图像 (uint8)
    :param corners: 初始角点坐标 (Nx2 numpy数组)
    :param win_size: 搜索窗口大小 (建议5-15)
    :param max_iters: 最大迭代次数
    :return: 优化后的角点坐标
    """
    # 设置迭代终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                max_iters, 0.001)

    corners_s = np.ascontiguousarray(corners).reshape(-1, 1, 2).astype(np.float32)
    # 执行亚像素优化
    corners_s = cv2.cornerSubPix(image, corners_s, (win_size, win_size), (-1, -1), criteria)

    return corners_s.squeeze()


def expand_quadrilateral(pts, expand_pixels=5):
    """
    四边形顶点外扩
    :param pts: 原始四边形顶点，形状为(4,2)的np数组
    :param expand_pixels: 外扩像素数
    :return: 扩展后的四边形顶点
    """
    expanded = []
    for i in range(4):
        # 获取相邻顶点索引
        prev_idx = (i - 1) % 4
        next_idx = (i + 1) % 4

        # 计算相邻边向量
        vec_prev = pts[i] - pts[prev_idx]
        vec_next = pts[next_idx] - pts[i]

        # 计算单位法向量（外扩方向）
        norm_prev = np.array([-vec_prev[1], vec_prev[0]])
        norm_prev = norm_prev / (np.linalg.norm(norm_prev) + 1e-5)

        norm_next = np.array([-vec_next[1], vec_next[0]])
        norm_next = norm_next / (np.linalg.norm(norm_next) + 1e-5)

        # 计算顶点外扩方向
        move_dir = (norm_prev + norm_next) * 0.5
        move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-5)

        # 外扩顶点
        new_pt = pts[i] + move_dir * expand_pixels
        expanded.append(new_pt)

    return np.array(expanded).astype(np.int32)


def extract_quad_roi(img, quad_pts):
    """
    提取四边形区域像素
    :param img: 输入图像 (BGR或灰度)
    :param quad_pts: 四边形顶点坐标，形状为(4,2)
    :return: 提取的ROI图像和掩模
    """
    # 生成掩模
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad_pts], 255)

    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(quad_pts)

    # 裁剪ROI区域
    roi = img[y:y + h, x:x + w]
    mask_roi = mask[y:y + h, x:x + w]

    # 应用掩模
    result = cv2.bitwise_and(roi, roi, mask=mask_roi)
    return result, mask


def non_max_suppression(mag, dir_rad):
    """
    非极大值抑制细化边缘
    :param mag: 梯度幅值图
    :param dir_rad: 梯度方向图(弧度)
    """
    h, w = mag.shape
    mag_suppressed = np.zeros_like(mag)

    # 将方向量化到4个主要区间
    dir_deg = np.degrees(dir_rad) % 180
    dir_quant = np.zeros_like(dir_deg, dtype=np.uint8)
    dir_quant[(0 <= dir_deg) & (dir_deg < 22.5)] = 0  # 水平
    dir_quant[(157.5 <= dir_deg) & (dir_deg <= 180)] = 0  # 水平
    dir_quant[(22.5 <= dir_deg) & (dir_deg < 67.5)] = 1  # 45度
    dir_quant[(67.5 <= dir_deg) & (dir_deg < 112.5)] = 2  # 垂直
    dir_quant[(112.5 <= dir_deg) & (dir_deg < 157.5)] = 3  # 135度

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current_mag = mag[y, x]
            if current_mag == 0:
                continue

            # 根据方向选择比较位置
            if dir_quant[y, x] == 0:  # 水平方向
                neighbors = [mag[y, x - 1], mag[y, x + 1]]
            elif dir_quant[y, x] == 1:  # 45度
                neighbors = [mag[y - 1, x + 1], mag[y + 1, x - 1]]
            elif dir_quant[y, x] == 2:  # 垂直方向
                neighbors = [mag[y - 1, x], mag[y + 1, x]]
            else:  # 135度
                neighbors = [mag[y - 1, x - 1], mag[y + 1, x + 1]]

            # 保留局部极大值
            if current_mag >= max(neighbors):
                mag_suppressed[y, x] = current_mag

    return mag_suppressed


def detect_tag_edges(image, original_corners):
    """
    检测属于AprilTag的直线边缘
    :param image: 灰度图像
    :param original_corners: 原始检测角点（4x2数组）
    :return: 筛选后的边缘直线列表
    """
    # 使用LSD直线检测

    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    # magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # magnitude_img = (magnitude - np.min(magnitude))*255 / (np.max(magnitude) - np.min(magnitude))
    #
    # direction = np.arctan2(sobel_y, sobel_x)
    # direction_img = (direction - np.min(direction)) * 255 / (np.max(direction) - np.min(direction))
    # cv2.imwrite('direction_img.png', direction_img.astype(np.uint8))
    # cv2.imwrite('magnitude_img.png', magnitude_img.astype(np.uint8))
    #
    # magnitude1 = non_max_suppression(magnitude, direction)
    # magnitude_img1 = (magnitude1 - np.min(magnitude1)) * 255 / (np.max(magnitude1) - np.min(magnitude1))
    # cv2.imwrite('magnitude_img1.png', magnitude_img1.astype(np.uint8))
    #
    # _, binary = cv2.threshold(magnitude_img1, 30, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('binary.png', binary.astype(np.uint8))
    # hough_params = {
    #     'rho': 1,  # 距离分辨率(像素)
    #     'theta': 1 / 1024,  # 角度分辨率(弧度)
    #     'threshold': 10,  # 累加器阈值
    #     'minLineLength': 3,  # 最小线段长度
    #     'maxLineGap': 3  # 最大允许间隙
    # }
    # lines = cv2.HoughLinesP(binary.astype(np.uint8), ** hough_params)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, scale=1, sigma_scale=0.3, quant=1.0, ang_th=20, n_bins=1024)
    lines, _, _, _ = lsd.detect(image.astype(np.uint8))
    if lines is None:
        return []
    # _, edge_mask = cv2.threshold(magnitude_img.astype(np.uint8), 100, 200, cv2.THRESH_BINARY)

    # # 步骤4：线段检测与筛选
    # lines1 = cv2.ximgproc.createFastLineDetector(
    #     length_threshold=10,
    #     do_merge=False,
    #     distance_threshold=1
    # ).detect(binary.astype(np.uint8))

    # fld = cv2.ximgproc.createFastLineDetector(
    #     length_threshold=15,  # 线段最小长度
    #     do_merge=True  # 是否合并相邻线段
    # )
    # # 执行检测结果
    # lines = fld.detect(image)

    # edges = cv2.Canny(image, 50, 200, apertureSize=3)
    # cv2.imwrite('edge_mask.png', edges)

    # img0 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for dline in lines:
    #     x0 = int(round(dline[0][0]))
    #     y0 = int(round(dline[0][1]))
    #     x1 = int(round(dline[0][2]))
    #     y1 = int(round(dline[0][3]))
    #     cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #
    # # 显示并保存结果
    # cv2.imwrite('test3_r_lines.png', img0)

    # 计算原始四边形的边向量方向
    edge_vectors = []
    for i in range(4):
        vec = original_corners[(i + 1) % 4] - original_corners[i]
        edge_vectors.append(vec / (np.linalg.norm(vec) + 1e-5))

    # 筛选与原始边方向相近的直线
    valid_lines = []
    angle_threshold = np.deg2rad(15)  # 15度容差
    for line in lines:
        x1, y1, x2, y2 = line[0]
        vec = np.array([x2 - x1, y2 - y1])
        vec_norm = vec / (np.linalg.norm(vec) + 1e-5)

        # 计算与最近边的角度差
        min_angle = np.pi
        for edge_vec in edge_vectors:
            cos_sim = np.dot(vec_norm, edge_vec)
            angle = np.arccos(np.clip(cos_sim, -1, 1))
            min_angle = min(min_angle, angle)

        if min_angle < angle_threshold:
            valid_lines.append(line)

    # img0 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for dline in valid_lines:
    #     x0 = int(round(dline[0][0]))
    #     y0 = int(round(dline[0][1]))
    #     x1 = int(round(dline[0][2]))
    #     y1 = int(round(dline[0][3]))
    #     cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #
    # # 显示并保存结果
    # cv2.imwrite('test3_r.png', img0)
    return valid_lines


def line_intersection(line1, line2):
    """
    计算两条直线的交点
    :param line1: ((x1,y1), (x2,y2))
    :param line2: ((x3,y3), (x4,y4))
    :return: 交点坐标 (x,y) 或 None
    """
    line1 = line1.reshape(2, 2)
    line2 = line2.reshape(2, 2)
    # 转换为参数方程
    a1 = line1[1][1] - line1[0][1]
    b1 = line1[0][0] - line1[1][0]
    c1 = a1 * line1[0][0] + b1 * line1[0][1]

    a2 = line2[1][1] - line2[0][1]
    b2 = line2[0][0] - line2[1][0]
    c2 = a2 * line2[0][0] + b2 * line2[0][1]

    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-6:  # 平行线
        return None

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return (x, y)


def find_edge_intersections(lines, original_corners, max_dist=5, img=None):
    """
    寻找边缘交点作为修正角点
    :return: 修正后的角点坐标 (4x2数组)
    """
    # 构建四边索引
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0)  # 相邻边索引
    ]

    corner_fake_mask = np.zeros(4)
    refined_corners = original_corners.copy()
    # img0 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(4):
        # 获取当前边和相邻边的候选直线
        current_edge_lines = [line for line in lines if is_adjacent(line, original_corners, i, max_dist)]

        # 寻找最佳交点
        best_pt = original_corners[i]
        min_dist = np.inf
        for lid1 in range(len(current_edge_lines) - 1):
            line1 = current_edge_lines[lid1]
            for lid2 in range(1, len(current_edge_lines)):
                line2 = current_edge_lines[lid2]
                pt = line_intersection(line1[0], line2[0])
                if pt is not None:
                    dist = np.linalg.norm(pt - original_corners[i])
                    if dist < min_dist and dist < max_dist:  # 最大允许偏移20像素
                        best_pt = pt
                        min_dist = dist
                        corner_fake_mask[i] = 1
        # confidence = 0.5
        # best_pt = refined_corners[i] * confidence + np.array(best_pt) * (1 - confidence)
        refined_corners[i] = best_pt

        # for dline in current_edge_lines:
        #     x0 = int(round(dline[0][0]))
        #     y0 = int(round(dline[0][1]))
        #     x1 = int(round(dline[0][2]))
        #     y1 = int(round(dline[0][3]))
        #     cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

        # pix = original_corners[i, :]
        # img0[int(pix[1]), int(pix[0]), :] = np.array([255, 0, 0])
        # # cv2.drawMarker(img0, position=(int(pix[0]), int(pix[1])), color=(255, 0, 0), markerSize=3,
        # #                markerType=cv2.MARKER_CROSS, thickness=1)
        # # cv2.circle(img0, (int(pix[0]), int(pix[1])), 1, (255, 0, 0), -1)
        #
        # pix = best_pt
        # img0[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
        # # cv2.drawMarker(img0, position=(int(pix[0]), int(pix[1])), color=(0, 0, 255), markerSize=3,
        # #                markerType=cv2.MARKER_CROSS, thickness=1)
        # # cv2.circle(img0, (int(pix[0]), int(pix[1])), 1, (0, 0, 255), -1)

        # 显示并保存结果
    # cv2.imwrite('test3_r.png', img0)
    # print('finished')

    return refined_corners, corner_fake_mask


def is_adjacent(line, corners, edge_idx, max_dist=5):
    """
    判断直线是否属于指定边
    """
    pt1 = np.array([line[0][0], line[0][1]])
    pt2 = np.array([line[0][2], line[0][3]])
    start_corner = corners[edge_idx]

    # 计算端点距离
    d1 = min(np.linalg.norm(pt1 - start_corner),
             np.linalg.norm(pt2 - start_corner))
    return d1 < max_dist  # 端点靠近边起点和终点


def fuse_corners(original, edge_refined, confidence=0.7):
    """
    融合原始检测点与边缘修正点
    :param confidence: 0-1, 对原始点的信任度
    """
    return original * confidence + edge_refined * (1 - confidence)


def refine_flower_tags(img_raw, det_tags_dict, true_flower_results_dict):
    matched_corners = []
    for tag_id in det_tags_dict:
        if tag_id in true_flower_results_dict:
            matched_corners.append(
                np.column_stack([det_tags_dict[tag_id].corners, true_flower_results_dict[tag_id].corners]))

    matched_corners = np.row_stack(matched_corners)
    homo, _ = cv2.findHomography(matched_corners[:, 2:], matched_corners[:, :2])

    # whb = np.array([[0, 0], [0, true_flower_img.shape[0]], [true_flower_img.shape[1], true_flower_img.shape[0]],
    #                 [true_flower_img.shape[1], 0]])
    # whbT = homo_trans(whb, homo)
    # warp_w, warp_h = np.max(whbT.astype(np.int32), axis=0) + 1
    # warped_truth = cv2.warpPerspective(true_flower_img, homo, dsize=(warp_w, warp_h), flags=cv2.INTER_CUBIC)
    # exter_size = 5
    proj_result_dict = {}

    gamma = 3  # 伽马值 >1 降低亮度，<1 提高亮度
    # lookup_table = np.array([((i / 255.0) ** gamma) * 255
    #                          for i in np.arange(0, 256)]).astype("uint8")

    lookup_table = ((np.arange(0, 256) / 255.0) ** gamma) * 255
    lookup_table = lookup_table.astype(np.uint8)

    #
    # img_l_g = cv2.LUT(img_l_g, lookup_table)

    for tag_id in true_flower_results_dict:
        if tag_id in det_tags_dict:
            continue
        corners = true_flower_results_dict[tag_id].corners
        center = true_flower_results_dict[tag_id].center

        center_t = homo_trans(center.reshape(1, -1), homo)[0, :]
        corners_t = homo_trans(corners, homo)

        x, y, w, h = cv2.boundingRect(corners_t.astype(np.int32))
        roi_expand = 3  # 扩展检测区域
        sy_c = max(0, y - roi_expand)
        sx_c = max(0, x - roi_expand)
        ey_c = min(img_raw.shape[0], y + h + roi_expand)
        ex_c = min(img_raw.shape[1], x + w + roi_expand)

        roi = img_raw[sy_c:ey_c, sx_c:ex_c]
        cv2.imwrite('roi.png', roi)
        img_l_g = roi
        img_l_g = cv2.LUT(img_l_g, lookup_table)
        roi = img_l_g

        lines = detect_tag_edges(roi, corners_t - np.array([sx_c, sy_c]))
        if len(lines) < 2:
            continue
        edge_refined, corner_edge_mask = find_edge_intersections(lines, corners_t - np.array([sx_c, sy_c]), max_dist=5,
                                                                 img=roi)

        # refined_corners = fuse_corners(corners_t, edge_refined + np.array([sx_c, sy_c]), confidence=0.5)
        refined_corners = edge_refined + np.array([sx_c, sy_c])
        # for pix in refined_corners:
        #     show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
        #     # cv2.drawMarker(show_img, position=(int(pix[0]), int(pix[1])), color=(0, 0, 255), markerSize=3,
        #     #                markerType=cv2.MARKER_CROSS, thickness=1)
        #
        # for pix in corners_t:
        #     # cv2.drawMarker(show_img, position=(int(pix[0]), int(pix[1])), color=(0, 255, 0), markerSize=3,
        #     #                markerType=cv2.MARKER_CROSS, thickness=1)
        #     show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 255, 0])

        # draw_conner_box(show_img, refine_corners, tag_id=tag_id, color=[0, 255, 255])

        refined_corners_f = np.column_stack([refined_corners, corner_edge_mask.reshape(-1, 1)])
        proj_result_dict[tag_id] = Detection(corners=refined_corners_f, center=np.mean(refined_corners, axis=0),
                                             tag_id=tag_id,
                                             tag_family="reproj", hamming=0, decision_margin=0, homography=np.eye(3))

        det_tags_dict[tag_id] = proj_result_dict[tag_id]

    return det_tags_dict


if __name__ == '__main__':
    at_detector_re1 = Detector(families='tag25h9',
                               nthreads=4,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=True,
                               decode_sharpening=0.0,
                               debug=False
                               )
    # test_img = cv2.imread(r"D:\projects\duban_visual_servo\utils\saved_imgs_20250319141421\030.png", 0)
    test_img = cv2.imread(r"D:\projects\pic_examples\007_bright.bmp", 0)
    # test_img_l = test_img[:, :4096]
    test_img_l = test_img

    det_results = at_detector_re1.detect(test_img_l, estimate_tag_pose=False, tag_size=55)

    det_results_dict = get_tag_results_dict(det_results)

    show_img = cv2.cvtColor(test_img_l, cv2.COLOR_GRAY2BGR)
    for tag_id in det_results_dict:
        corners = det_results_dict[tag_id].corners
        center = det_results_dict[tag_id].center
        for pix in corners:
            show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])

    cv2.imwrite("show_img_tag_det000.png", show_img)

    true_flower_img = cv2.imread(r'D:\ruben_flower_20250324\1.png', 0)

    true_results = at_detector_re1.detect(true_flower_img, estimate_tag_pose=False, tag_size=55)

    show_img = cv2.cvtColor(test_img_l, cv2.COLOR_GRAY2BGR)
    for tag_id in det_results_dict:
        corners = det_results_dict[tag_id].corners
        center = det_results_dict[tag_id].center
        for pix in corners:
            show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])

    cv2.imwrite("show_img_tag_det001.png", show_img)

    # detector_cbc = CBDPipeline(expand=True, predict=False)
    # checkerboard_detector = CheckerboardDetector()
    # board_uv_l, detected_board_xy_l, cor_uv_l = checkerboard_detector.detect_checkerboard(test_img_l)
    # show_img = cv2.cvtColor(test_img_l, cv2.COLOR_GRAY2BGR)

    # for pix in board_uv_l:
    #     show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
    # cv2.imwrite("show_img_tag_det22.png", show_img)
    true_results_dict = get_tag_results_dict(true_results)

    det_results_r = refine_flower_tags(test_img_l, det_results_dict, true_results_dict)

    show_img = cv2.cvtColor(test_img_l, cv2.COLOR_GRAY2BGR)
    for tag_id in det_results_r:
        corners = det_results_r[tag_id].corners
        center = det_results_r[tag_id].center
        for pix in corners:
            show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])

    cv2.imwrite("show_img_tag_det11.png", show_img)
    show_img = cv2.cvtColor(test_img_l, cv2.COLOR_GRAY2BGR)
    matched_corners = []
    for tag_id in det_results_dict:
        if tag_id in true_results_dict:
            matched_corners.append(
                np.column_stack([det_results_dict[tag_id].corners, true_results_dict[tag_id].corners]))

    true_corners = np.row_stack([true_results_dict[tag_id].corners for tag_id in true_results_dict])
    matched_corners = np.row_stack(matched_corners)

    homo, inliers = cv2.findHomography(matched_corners[:, 2:], matched_corners[:, :2])

    true_corners_t = homo_trans(true_corners, homo)
    # for pix in true_corners_t:
    #     cv2.drawMarker(show_img, position=(int(pix[0]), int(pix[1])), color=(0, 0, 255), markerSize=5,
    #                    markerType=cv2.MARKER_CROSS, thickness=2)
    # cv2.imwrite("show_img_tag_det_true_proj.png", show_img)

    whb = np.array([[0, 0], [0, true_flower_img.shape[0]], [true_flower_img.shape[1], true_flower_img.shape[0]],
                    [true_flower_img.shape[1], 0]])
    whbT = homo_trans(whb, homo)
    warp_w, warp_h = np.max(whbT.astype(np.int32), axis=0) + 1
    warped_truth = cv2.warpPerspective(true_flower_img, homo, dsize=(warp_w, warp_h), flags=cv2.INTER_CUBIC)
    exter_size = 5
    proj_result_dict = {}

    for tag_id in true_results_dict:
        # if tag_id in det_results_dict:
        #     continue
        corners = true_results_dict[tag_id].corners
        center = true_results_dict[tag_id].center

        center_t = homo_trans(center.reshape(1, -1), homo)[0, :]
        corners_t = homo_trans(corners, homo)

        x, y, w, h = cv2.boundingRect(corners_t.astype(np.int32))
        roi_expand = 3  # 扩展检测区域
        sy_c = max(0, y - roi_expand)
        sx_c = max(0, x - roi_expand)
        ey_c = min(test_img_l.shape[0], y + h + roi_expand)
        ex_c = min(test_img_l.shape[1], x + w + roi_expand)

        roi = test_img_l[sy_c:ey_c, sx_c:ex_c]
        lines = detect_tag_edges(roi, corners_t - np.array([sx_c, sy_c]))
        edge_refined = find_edge_intersections(lines, corners_t - np.array([sx_c, sy_c]), max_dist=5, img=roi)

        refined_corners = fuse_corners(corners_t, edge_refined[0] + np.array([sx_c, sy_c]), confidence=0.5)

        re_det_c = gradient_refinement(test_img_l, corners_t, radius=10)
        # re_det_c = refine_corners_subpix(test_img_l, corners_t, win_size=3, max_iters=50)
        search_size = 20
        refine_corners = []
        for det_c in corners_t:

            template = cv2.getRectSubPix(warped_truth, (search_size * 2, search_size * 2), det_c)
            s_x = int(max(det_c[0] - search_size * 3, 0))
            s_y = int(max(det_c[1] - search_size * 3, 0))
            e_x = int(min(det_c[0] + search_size * 3, test_img_l.shape[1] - 1))
            e_y = int(min(det_c[1] + search_size * 3, test_img_l.shape[0] - 1))
            search_roi = test_img_l[s_y:e_y + 1, s_x:e_x + 1]

            res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.8:  # 置信度阈值
                dx = max_loc[0] + det_c[0]
                dy = max_loc[1] + det_c[1]

                # dx = max_loc[0] + s_x
                # dy = max_loc[1] + s_y

                # 亚像素级优化
                term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                # cv2.cornerSubPix(test_img_l,
                #                  np.array([[[dx, dy]]], dtype=np.float32),
                #                  (3, 3), (-1, -1), term_crit)
                refine_corners.append([dx, dy])
            else:
                refine_corners.append([det_c[0], det_c[1]])
        refine_corners = np.array(refine_corners)

        diff = refine_corners - corners_t
        refine_corners = corners_t
        tag_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H, _ = cv2.findHomography(tag_corners, refine_corners)
        # 重投影计算
        projected = homo_trans(tag_corners, H)

        # 计算误差
        error = np.linalg.norm(refine_corners - projected, axis=1).mean()

        for pix in refined_corners:
            show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 0, 255])
            # cv2.drawMarker(show_img, position=(int(pix[0]), int(pix[1])), color=(0, 0, 255), markerSize=3,
            #                markerType=cv2.MARKER_CROSS, thickness=1)

        # for pix in corners_t:
        #     # cv2.drawMarker(show_img, position=(int(pix[0]), int(pix[1])), color=(0, 255, 0), markerSize=3,
        #     #                markerType=cv2.MARKER_CROSS, thickness=1)
        #     show_img[int(pix[1]), int(pix[0]), :] = np.array([0, 255, 0])

        # draw_conner_box(show_img, refine_corners, tag_id=tag_id, color=[0, 255, 255])
        proj_result_dict[tag_id] = Detection(corners=refine_corners, center=np.mean(refine_corners, axis=0),
                                             tag_id=tag_id,
                                             tag_family="", hamming=0, decision_margin=0, homography=np.eye(3))

    cv2.imwrite("show_img_tag_det.png", show_img)

    print('finished')
