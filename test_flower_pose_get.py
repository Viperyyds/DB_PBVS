from SCameraProcess import SCameraProcess
from Core.RobotCore import RobotCore
import numpy as np
from multiprocessing import Queue, Pipe
from utils.pose_estimation import get_T_from_rt_vec, get_rt_vec_from_T
import time
from queue import Empty
import msvcrt
import asyncio

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


async def test_loop(robot_core, image_data_queue):
    """测试循环:按c键采集数据,按q键退出"""
    print("" + " = "*50)
    print("测试模式启动")
    print("按 'c' 键：采集并打印靶标位姿和机器人关节角")
    print("按 'q' 键：退出程序")
    print("=" * 50 + "")

    while True:
        # Windows 非阻塞键盘检测
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'c':
                print("[采集中...]")

                # 获取靶标位姿
                try:
                    cMo = get_current_flower_cam_pose(image_data_queue, sequ_len=20)
                    if cMo is not None:
                        print("✓ 靶标位姿 (cMo):")
                        print(cMo)
                        print()
                    else:
                        print("✗ 无法获取靶标位姿")
                except Exception as e:
                    print(f"✗ 获取靶标位姿失败: {e}\n")

                # 获取机器人关节角
                try:
                    joint_angles = await robot_core.getCurrentJointAngles()
                    print("✓ 机器人关节角: ")
                    print(joint_angles)
                    print()
                except Exception as e:
                    print(f"✗ 获取关节角失败: {e}")
                    print("-" * 50 + "")

            elif key == 'q':
                print("退出测试模式...")
                break
        # 避免CPU占用过高
        await asyncio.sleep(0.05)

async def main():
    # --- 初始化资源 ---
    flower_tag_board_params = np.load('flower_tag_cali_params_board_2025032418.npy', allow_pickle=True).item()
    stereo_calib_params = np.load('baser4096_camera_stereo_matlab_20250617.npy', allow_pickle=True).item()
    db_tag_ids = np.arange(7, 25)

    image_data_queue = Queue(1)
    out_pipe, in_pipe = Pipe(True)
    s_cam_process = SCameraProcess(camera_calib_params=stereo_calib_params,
                                   flower_tag_board_params=flower_tag_board_params, db_tag_id_lst=db_tag_ids,
                                   cam_exp_time=10000, pipe=[out_pipe, in_pipe],
                                   result_queue=image_data_queue)
    s_cam_process.start()
    robot_core = RobotCore('192.168.1.253')
    try:
        await test_loop(robot_core, image_data_queue)
    finally:
        print("正在清理资源...")
        s_cam_process.terminate()
        s_cam_process.join()
        print("资源清理完成")

if __name__ == '__main__':
    asyncio.run(main())
