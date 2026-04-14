import numpy as np
import cv2
import asyncio
import time
from multiprocessing import Queue, Pipe
from SCameraProcess import SCameraProcess
from Core.RobotCore import RobotCore
from utils.pose_estimation import get_T_from_rt_vec, get_rt_vec_from_T
from queue import Empty, Full
from AXYBCalibrate.lmi_axyb import LMI_AXYB

def get_most_supported_mean(pose_lst, ratio=0.8):
    pose_arr = np.array(pose_lst)
    pose_arr_mean = np.mean(pose_arr, axis=0)

    mean_diff = np.sum((pose_arr[:, 3:] - pose_arr_mean[3:]) ** 2, axis=1) ** 0.5
    sort_ids = np.argsort(mean_diff)
    supported_num = max(1, int(len(mean_diff) * ratio))

    diff_tr_vecs_mean = np.mean(pose_arr[sort_ids[:supported_num], :], axis=0)
    return diff_tr_vecs_mean

def get_current_flower_cam_pose(image_data_queue, sequ_len=20, sensus_ratio=0.6):
    flower_cam_pose_T_f = None
    flower_poses_lst = []
    while True:
        st1 = time.time()
        if image_data_queue.empty():
            continue
        try:
            output_data = image_data_queue.get_nowait()
        except Empty as e:
            continue
        if output_data is None:
            continue

        (flower_cam_pose, tag_poses_3d_pts, cap_time_str) = output_data

        if flower_cam_pose is None:
            print('hand_pose_loss')
            break

        if cap_time_str == -1:
            print('camera process break!')
            break

        if flower_cam_pose is not None:
            f_rvec, f_tvec = get_rt_vec_from_T(flower_cam_pose)
            flower_poses_lst.append(np.vstack([f_rvec, f_tvec]).ravel())

            mean_db_tag_pose_cam = get_most_supported_mean(flower_poses_lst, sensus_ratio)

            f_cam_pose_vec = mean_db_tag_pose_cam
            flower_cam_pose_T_f = get_T_from_rt_vec(f_cam_pose_vec[:3], f_cam_pose_vec[3:])

            # print(f'db_tag_cam_pose_vec = {np.round(f_cam_pose_vec, 2)}')
            if len(flower_poses_lst) >= sequ_len:
                break
            else:
                continue
    return flower_cam_pose_T_f


# --- 1. 正运动学解算模块 ---
def get_forward_kinematics(joint_angles):
    """
    根据 MDH 参数计算正运动学
    joint_angles: 长度为 6 的关节角列表 (单位: 弧度)
    """
    # MDH参数表: [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
    duban_mdh_params = [
        [0, 0, 0.3085, 0],
        [-np.pi / 2, 0, 0, -np.pi / 2],
        [0, 0.3, 0, np.pi / 2],
        [np.pi / 2, 0, 0.6865, 0],
        [-np.pi / 2, 0, 0, 0],
        [np.pi / 2, 0, 0.2649, 0],
    ]

    T_base_ee = np.eye(4)

    for i in range(6):
        alpha = duban_mdh_params[i][0]
        a = duban_mdh_params[i][1]
        d = duban_mdh_params[i][2]
        theta_offset = duban_mdh_params[i][3]

        # 当前关节的总旋转角
        theta_star = joint_angles[i] + theta_offset

        # 计算 MDH 单步变换矩阵
        ct = np.cos(theta_star)
        st = np.sin(theta_star)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T_i = np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

        # 连乘得到最终位姿
        T_base_ee = T_base_ee @ T_i

    return T_base_ee

async def collect_calibration_data(robot_core, image_queue, num_poses=10):
    """
    通过移动机器人并记录位姿对 (bMe, cMo)
    """
    bMe_list = []
    cMo_list = []
    print(f"开始采集标定数据，建议采集 {num_poses} 组以上不同姿态的数据。")
    print("注意：每组姿态之间，机器人末端应既有平移也有显著的角度变化。")
    count = 0
    while count < num_poses:
        input(f"[Pose{count + 1} / {num_poses}] 请手动移动机器人到一个新位姿，按回车采集数据...")
        # A. 获取视觉位姿 cMo
        # 增加采样长度以提高精度
        cMo = get_current_flower_cam_pose(image_queue, sequ_len=20)
        if cMo is None:
            print("错误：无法识别靶标，请重新调整姿态！")
            continue
        # 将cMo的位置部分单位转化为m(原始单位是mm)
        cMo[:3, 3] = cMo[:3, 3] / 1000.0
        # B. 获取机器人关节角并计算 bMe
        q = await robot_core.getCurrentJointAngles()
        bMe = get_forward_kinematics(q)
        bMe_list.append(bMe)
        cMo_list.append(cMo)
        print(f"数据记录成功！(关节角: {np.rad2deg(q).round(2)})")
        count += 1
    return bMe_list, cMo_list


def calculate_error(bMe_list, cMo_list, eMo, bMc):
    """误差分析：以相机实际测量的 cMo 为基准"""
    err_t = []
    err_r = []
    for bMe, cMo_meas in zip(bMe_list, cMo_list):
        # 预测值: cMo = inv(bMc) * bMe * eMo
        cMo_est = np.linalg.inv(bMc) @ bMe @ eMo

        # 平移误差 (mm)
        dt = np.linalg.norm(cMo_est[:3, 3] - cMo_meas[:3, 3]) * 1000
        err_t.append(dt)

        # 旋转误差 (deg)
        R_diff = cMo_est[:3, :3] @ cMo_meas[:3, :3].T
        angle, _ = cv2.Rodrigues(R_diff)
        dr = np.linalg.norm(angle) * 180 / np.pi
        err_r.append(dr)

    return np.mean(err_t), np.std(err_t), np.mean(err_r), np.std(err_r)


def check_data_consistency(bMe_list, cMo_list):
    print("" + " >> > 开始进行数据一致性校验(相对旋转角应几乎相等) << < ")
    for i in range(1, len(bMe_list)):
        # 机器人末端两次动作之间的相对运动
        R_bMe1 = bMe_list[i - 1][:3, :3]
        R_bMe2 = bMe_list[i][:3, :3]
        rel_R_robot = R_bMe1.T @ R_bMe2
        angle_robot, _ = cv2.Rodrigues(rel_R_robot)
        deg_robot = np.linalg.norm(angle_robot) * 180 / np.pi

        # 相机观测到靶标的两次相对运动
        R_cMo1 = cMo_list[i - 1][:3, :3]
        R_cMo2 = cMo_list[i][:3, :3]
        # 注意眼在手外 (Eye-to-Hand) 的相对旋转计算方式
        rel_R_cam = R_cMo1 @ R_cMo2.T
        angle_cam, _ = cv2.Rodrigues(rel_R_cam)
        deg_cam = np.linalg.norm(angle_cam) * 180 / np.pi

        diff = abs(deg_robot - deg_cam)
        print(f"动作 {i}: 机器人转了 {deg_robot:6.2f}° | 视觉看到转了 {deg_cam:6.2f}° | 差值: {diff:5.2f}°")

    print(">>> 校验结束。如果差值普遍大于 3~5度，说明数据本身是错的，换什么算法都没用！ <<<")


def orthonormalize_transform(T):
    """Project rotation part of a 4x4 transform to SO(3) using SVD."""
    T_fix = T.copy()
    R = T_fix[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    T_fix[:3, :3] = R_ortho
    return T_fix


def refine_translation_after_rotation_fix(bMe_list, cMo_list, eMo, bMc):
    """Re-estimate translations with fixed rotations using least squares."""
    R_x = eMo[:3, :3]
    R_y = bMc[:3, :3]
    lhs_blocks = []
    rhs_blocks = []

    for bMe, cMo in zip(bMe_list, cMo_list):
        R_a = bMe[:3, :3]
        t_a = bMe[:3, 3]
        t_b = cMo[:3, 3]

        # From bMe * X = Y * cMo:
        # R_a * t_x - t_y = R_y * t_b - t_a
        lhs_blocks.append(np.hstack([R_a, -np.eye(3)]))
        rhs_blocks.append((R_y @ t_b - t_a).reshape(3, 1))

    lhs = np.vstack(lhs_blocks)
    rhs = np.vstack(rhs_blocks)
    sol, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

    t_x = sol[:3, 0]
    t_y = sol[3:, 0]

    eMo_fix = eMo.copy()
    bMc_fix = bMc.copy()
    eMo_fix[:3, 3] = t_x
    bMc_fix[:3, 3] = t_y
    return eMo_fix, bMc_fix

async def main_calibration():
    # --- 初始化资源 ---
    flower_tag_board_params = np.load('flower_tag_cali_params_board_2025032418.npy', allow_pickle=True).item()
    stereo_calb_params = np.load('baser4096_camera_stereo_matlab_20250617.npy', allow_pickle=True).item()
    db_tag_ids = np.arange(7, 25)

    image_data_queue = Queue(1)
    out_pipe, in_pipe = Pipe(True)
    s_cam_process = SCameraProcess(camera_calib_params=stereo_calb_params,
                                   flower_tag_board_params=flower_tag_board_params, db_tag_id_lst=db_tag_ids,
                                   cam_exp_time=10000, pipe=[out_pipe, in_pipe],
                                   result_queue=image_data_queue)
    s_cam_process.start()
    robot_core = RobotCore('192.168.1.253')
    try:
        # 1. 采集数据
        bMe_list, cMo_list = await collect_calibration_data(robot_core, image_data_queue, num_poses=10)
        # 将采集的数据保存
        bMe_arr = np.array(bMe_list)
        cMo_arr = np.array(cMo_list)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_data_filename = f'./EyehandCaliData/calib_raw_data_{timestamp}.npz'
        np.savez(raw_data_filename, bMe_list=bMe_arr, cMo_list=cMo_arr)
        print(f"[数据保存]原始位姿数据已成功保存至: {raw_data_filename}")
        print("[数据保存] 你可以使用该文件进行离线算法验证！")

        # 2. 数据一致性校验
        check_data_consistency(bMe_list, cMo_list)
        # 3. 求解标定结果
        # 读取A矩阵
        A = np.zeros((len(bMe_list),4,4))
        for i in range(len(bMe_list)):
            A[i] = bMe_list[i]
        # 读取B矩阵
        B = np.zeros((len(bMe_list),4,4))
        for i in range(len(bMe_list)):
            B[i] = cMo_list[i]
        eMo, bMc = LMI_AXYB(A, B)
        eMo = orthonormalize_transform(eMo)
        bMc = orthonormalize_transform(bMc)
        eMo, bMc = refine_translation_after_rotation_fix(bMe_list, cMo_list, eMo, bMc)
        # 4. 误差分析
        err_t_mean, err_t_std, err_r_mean, err_r_std = calculate_error(bMe_list, cMo_list, eMo, bMc)
        print(f"\n标定结果 eMo:\n{eMo}")
        print(f"标定结果 bMc:\n{bMc}")
        print(f"\n标定结果分析:")
        print(f"平均位置误差: {err_t_mean:.2f} ± {err_t_std:.2f} mm")
        print(f"平均旋转误差: {err_r_mean:.2f} ± {err_r_std:.2f} deg")
        
        # 5. 保存
        np.save('./EyehandCaliData/flower_eMo_calib.npy', eMo)
        np.save('./EyehandCaliData/robot_bMc_calib.npy', bMc)
        print("标定结果已成功保存。")
    finally:
        s_cam_process.terminate()
        s_cam_process.join()

if __name__ == "__main__":
    asyncio.run(main_calibration())