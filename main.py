import numpy as np
from multiprocessing import Process, Pipe, Queue

from utils.pose_estimation import get_rigid_transform_o3d, get_T_from_rt_vec, trans_pts, get_rt_vec_from_T, \
    get_T_from_rt_rpy, get_rt_rpy_from_T, vec_to_rpy, get_trans_error, get_trvec_rpy_from_T
from utils.utils import get_time_acc
import cv2
import time
from queue import Empty, Full
from Core.RobotCore import *
from visp.core import HomogeneousMatrix, PoseVector
from DBRobotPBVS import dbrobot_pbvs_control
from SCameraProcess import SCameraProcess


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

            print(f'db_tag_cam_pose_vec = {np.round(f_cam_pose_vec, 2)}')
            if len(flower_poses_lst) >= sequ_len:
                break
            else:
                continue
    return flower_cam_pose_T_f


async def start_servo_task(e_M_o_np, cdMo_np, image_queue, robot_core):
    """
    封装异步调用的中间层
    :param e_M_o_np: 靶标在机器人末端法兰下的位姿 (Numpy 4x4)
    :param cdMo_np: 靶标在相机坐标系下的期望位姿 (Numpy 4x4)
    """
    print("正在启动视觉伺服核心循环...")
    exit_reason = "not_started"
    experiment_logger = None
    try:
        robot_core.pause_background_status_updates()
        # 调用之前写的 PBVS 核心函数
        exit_reason, experiment_logger = await dbrobot_pbvs_control(
            e_M_o_np=e_M_o_np,
            cdMo_np=cdMo_np,
            image_queue=image_queue,
            duban_robot_con=robot_core,
            opt_adaptive_gain=True,
            opt_plot=True,
            opt_task_sequencing=False
        )
    except Exception as e:
        print(f"视觉伺服运行中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 使用 try-except 保护，确保即使机器人连接断开，也能退出异步环境回到 main 清理进程
        try:
            if exit_reason == "converged":
                print("视觉伺服正常收敛，已发送 MoveS 软保持，不发送全局停止指令。")
            elif exit_reason == "user_stop":
                print("视觉伺服已由用户停止，跳过重复全局停止指令。")
            else:
                print(f"视觉伺服退出原因: {exit_reason}，正在尝试停止机器人...")
                await asyncio.wait_for(robot_core.RobotStop(), timeout=2.0)
        except Exception as stop_err:
            print(f"安全停机指令发送失败 (可能机器人连接已断开): {stop_err}")
        print("视觉伺服流程结束")

    if experiment_logger is not None:
        print("Experiment finished. Drawing recorded PBVS curves.")
        experiment_logger.plot()


if __name__ == '__main__':
    # 靶标板标定的参数
    flower_tag_board_params = np.load('flower_tag_cali_params_board_2025032418.npy', allow_pickle=True).item()
    # 双目相机标定参数
    stereo_calb_params = np.load('baser4096_camera_stereo_matlab_20250617.npy', allow_pickle=True).item()
    db_tag_ids = np.arange(7, 25)
    # 开启相机进程
    image_data_queue = Queue(1)
    out_pipe, in_pipe = Pipe(True)
    s_cam_process = SCameraProcess(camera_calib_params=stereo_calb_params,
                                   flower_tag_board_params=flower_tag_board_params, db_tag_id_lst=db_tag_ids,
                                   cam_exp_time=10000, pipe=[out_pipe, in_pipe],
                                   result_queue=image_data_queue)  # 实例化进程对象
    s_cam_process.start()

    robot_core = RobotCore('192.168.1.253')
    # --- 步骤 A: 运动到初始位姿 (Start Pose) ---
    input(">>> 请手动或通过程序控制机器人运动到 [初始位姿]，完成后按回车确认...")
    print("已确认初始位姿")
    # --- 步骤 B: 运动到期望位姿 (Desired Pose) 并记录 ---
    input(">>> 请手动或通过程序控制机器人运动到 [目标位姿(期望位置)]，完成后按回车确认...")
    print("正在采集期望位姿数据，请保持机器人静止...")
    T_target_cam = get_current_flower_cam_pose(image_data_queue, sequ_len=20)
    if T_target_cam is not None:
        T_target_cam[:3,3] = T_target_cam[:3,3] / 1000.0  # mm → m
        print("成功记录期望位姿 (相机坐标系):")
        print(T_target_cam)
    else:
        print("错误：无法识别靶标，请检查视野！")
        exit(-1)

    # --- 步骤 C: 回到初始位姿 ---
    input(">>> 请将机器人运动回 [初始位姿] 以准备开始实验，完成后按回车确认...")
    print("机器人已就绪")

    e_M_o_np = np.load('./EyehandCaliData/flower_eMo_calib.npy')
    # --- 步骤 D: 执行 PBVS 视觉伺服 ---
    input(">>> [警告] 即将开始 PBVS 自动控制实验，请确保急停开关可用，按回车开始...")
    try:
        # 使用 asyncio 运行异步控制任务
        asyncio.run(start_servo_task(
            e_M_o_np=e_M_o_np,
            cdMo_np=T_target_cam,  # 你采集到的期望位姿
            image_queue=image_data_queue,
            robot_core=robot_core
        ))
    except KeyboardInterrupt:
        print("[用户强制停止]")
    finally:
        # 彻底关闭后台进程
        print("清理系统资源...")
        try:
            robot_core.stop()
        except Exception as robot_stop_err:
            print(f"停止机器人通信服务失败: {robot_stop_err}")
        s_cam_process.terminate()
        s_cam_process.join()
        out_pipe.close()
        in_pipe.close()
        image_data_queue.close()
        print("程序已安全退出")

