from visp.core import HomogeneousMatrix
from visp.visual_features import FeatureTranslation
from visp.visual_features import FeatureThetaU
from visp.vs import Servo
from visp.core import measureTimeMs
from get_flower_pose_ema_visp import get_flower_pose_ema_visp, reset_flower_pose_ema
from visp.core import Matrix
from robot import RobotMDH
import numpy as np
from visp.core import Math
import msvcrt
import math
import time
import matplotlib.pyplot as plt
import asyncio

class PBVSExperimentLogger:
    def __init__(self):
        self.time_ms = []
        self.feature_errors = []
        self.joint_velocities_deg_s = []
        self.joint_angles_deg = []

        self.feature_error_legends = [
            "error_feat_tx", "error_feat_ty", "error_feat_tz",
            "error_feat_theta_ux", "error_feat_theta_uy", "error_feat_theta_uz"
        ]
        self.joint_legends = ["q1", "q2", "q3", "q4", "q5", "q6"]

    @staticmethod
    def _to_float_list(values, size):
        data = np.asarray(values, dtype=float).reshape(-1)
        return [float(data[i]) for i in range(size)]

    def append(self, time_ms, feature_error, joint_velocity_rad_s, joint_angle_rad):
        self.time_ms.append(float(time_ms))
        self.feature_errors.append(self._to_float_list(feature_error, 6))
        self.joint_velocities_deg_s.append(self._to_float_list(np.rad2deg(joint_velocity_rad_s), 6))
        self.joint_angles_deg.append(self._to_float_list(np.rad2deg(joint_angle_rad), 6))

    def plot(self):
        if not self.time_ms:
            print("No PBVS data was recorded. Skip offline plotting.")
            return

        time_s = np.asarray(self.time_ms, dtype=float) / 1000.0
        feature_errors = np.asarray(self.feature_errors, dtype=float)
        joint_velocities_deg_s = np.asarray(self.joint_velocities_deg_s, dtype=float)
        joint_angles_deg = np.asarray(self.joint_angles_deg, dtype=float)

        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        fig.canvas.manager.set_window_title("PBVS experiment curves")

        plot_groups = [
            (axs[0], feature_errors, self.feature_error_legends, "Visual features error", "error"),
            (axs[1], joint_velocities_deg_s, self.joint_legends, "Joint velocities", "deg/s"),
            (axs[2], joint_angles_deg, self.joint_legends, "Joint angles", "deg"),
        ]

        for ax, data, legends, title, ylabel in plot_groups:
            for i, legend in enumerate(legends):
                ax.plot(time_s, data[:, i], label=legend)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc='upper right', fontsize='small')

        axs[-1].set_xlabel("time (s)")
        fig.tight_layout(pad=3.0)
        plt.show()


async def dbrobot_pbvs_control(e_M_o_np, cdMo_np, image_queue, duban_robot_con, opt_adaptive_gain, opt_plot, opt_task_sequencing):
    '''
    :param e_M_o_np: 标签在机器人末端法兰坐标系下的表示 --np.array
    :param cdMo: 期望位姿 --np.array
    :param image_queue: 相机获取到的图像序列
    :param duban_robot_con: 堵板机器人的python API接口
    :param opt_adaptive_gain: 动态增益启动 bool
    :param opt_plot 绘图是否启用 bool
    :para opt_task_sequencing 是否使用任务序列 bool
    :return:
    '''
    # 定义PBVS收敛阈值
    convergence_threshold_t = 0.5  # mm
    convergence_threshold_tu = 0.1  # deg
    # 定义堵板机器人
    duban_mdh_params = [
        [0, 0, 0.3085, 0],
        [-np.pi / 2, 0, 0, -np.pi / 2],
        [0, 0.3, 0, np.pi / 2],
        [np.pi / 2, 0, 0.6865, 0],
        [-np.pi / 2, 0, 0, 0],
        [np.pi / 2, 0, 0.2649, 0],
    ]
    robot = RobotMDH(duban_mdh_params)
    # numpy → vpHomogeneousMatrix
    cMo = HomogeneousMatrix()
    cdMo = HomogeneousMatrix(cdMo_np)
    e_M_o = HomogeneousMatrix(e_M_o_np)
    oMo = HomogeneousMatrix()
    cdMc = cdMo * cMo.inverse()

    # 创建特征点
    # 当前特征
    t = FeatureTranslation(FeatureTranslation.cdMc)
    tu = FeatureThetaU(FeatureThetaU.cdRc)
    t.buildFrom(cdMc)
    tu.buildFrom(cdMc)

    # 期望特征点
    td = FeatureTranslation(FeatureTranslation.cdMc)
    tud = FeatureThetaU(FeatureThetaU.cdRc)

    # 设置眼在手外视觉伺服类型
    task = Servo()
    task.addFeature(t, td)
    task.addFeature(tu, tud)
    task.setServo(Servo.EYETOHAND_L_cVe_eJe)
    task.setInteractionMatrixType(Servo.CURRENT)

    # Record all samples during control and draw once after the experiment.
    experiment_logger = PBVSExperimentLogger() if opt_plot else None
    reset_flower_pose_ema()

    final_quit = False
    has_converged = False
    exit_reason = "unknown"
    # send_velocities = True
    servo_started = False

    dt_sim = 0.08  # 期望控制周期，仅作为积分周期的默认值
    vision_alpha = 0.6  # PBVS 闭环内不宜过度平滑，否则末端会因滤波滞后产生残余误差
    idle_sleep = 0.005
    min_dt_control = 0.04
    max_dt_control = 0.12
    command_horizon = 0.10
    min_joint_step_rad = 2e-4
    max_vel_far = 0.20
    max_vel_mid = 0.10
    max_vel_near = 0.035
    min_vel_far = 0.018
    min_vel_mid = 0.010
    min_vel_near = 0.004
    max_acc = 1.0
    strict_converged_samples = 3
    fine_converged_samples = 8
    fine_threshold_t = 0.8  # mm，用于噪声区软收敛保护
    fine_threshold_tu = 0.2  # deg
    t_virtual = 0.0  # 仅保留兼容旧逻辑，不再用于绘图时间
    servo_start_time = time.perf_counter()
    last_command_time = None
    last_target_lost_hold_time = None
    prev_q_dot_rad_s = np.zeros(6, dtype=float)
    prev_error_norm = None
    strict_converged_count = 0
    fine_converged_count = 0
    # 视觉伺服主循环
    first_time = True
    while not has_converged:
        # --- 键盘退出检测逻辑 ---
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # 27 是 Esc 的 ASCII 码，b'q' 是字符 q
            if key == b'q' or key == b'\x1b':
                print(" 检测到退出指令，正在停止机器人... ")
                await duban_robot_con.RobotStop()
                exit_reason = "user_stop"
                break  # 跳出 while 循环
        # 记录循环开始的时间
        loop_start = time.perf_counter()

        t_sim_ms = (loop_start - servo_start_time) * 1000.0
        # 1. 先取视觉。没有新视觉帧时不更新控制目标，避免用旧误差重复积分。
        t0 = time.perf_counter()
        cMo_raw, has_new_vision, vision_stamp = get_flower_pose_ema_visp(
            image_queue, vision_alpha, return_metadata=True
        )
        t1 = time.perf_counter()

        if not has_new_vision:
            await asyncio.sleep(idle_sleep)
            continue

        if cMo_raw is None:
            print("Target Lost! Skip PBVS update and hold once if needed.")
            now = time.perf_counter()
            if last_target_lost_hold_time is None or now - last_target_lost_hold_time > 0.5:
                q_hold_rad = await duban_robot_con.getCurrentJointAngles()
                await duban_robot_con.MoveS(q_hold_rad.tolist())
                last_target_lost_hold_time = time.perf_counter()
            await asyncio.sleep(idle_sleep)
            continue

        # 2. 有新视觉样本时再获取当前机器人关节位置 (单位: rad)
        t2 = time.perf_counter()
        q_current_rad = await duban_robot_con.getCurrentJointAngles()
        t3 = time.perf_counter()
        cMo = cMo_raw
        last_target_lost_hold_time = None

        if first_time:
            # Introduce security wrt tag positioning to avoid PI rotation
            v_oMo = [HomogeneousMatrix(), HomogeneousMatrix()]
            v_cdMc = [HomogeneousMatrix(), HomogeneousMatrix()]
            v_oMo[1].buildFrom(0, 0, 0, 0, 0, math.pi)
            for i in range(2):
                v_cdMc[i] = cdMo * v_oMo[i] * cMo.inverse()

            theta0 = abs(v_cdMc[0].getThetaUVector().getTheta())
            theta1 = abs(v_cdMc[1].getThetaUVector().getTheta())

            if theta0 < theta1:
                oMo = v_oMo[0]
            else:
                print("Desired frame modified to avoid PI rotation of the camera")
                oMo = v_oMo[1]
        # 更新视觉特征
        cdMc = cdMo * oMo * cMo.inverse()
        t.buildFrom(cdMc)
        tu.buildFrom(cdMc)
        # 计算机器人末端在相机坐标系下的位姿
        c_M_e = HomogeneousMatrix()
        c_M_e = cMo * e_M_o.inverse()
        task.set_cVe(c_M_e)

        # 获取机器人雅可比矩阵
        e_J_e = robot.get_eJe(q_current_rad)
        task.set_eJe(e_J_e)

        task.computeControlLaw()
        J1_np = np.array(task.getTaskJacobian())
        error_np = np.array(task.getError())

        # 计算当前的特征误差；如果已经收敛，先软保持当前位置，不再下发新的 PBVS 目标
        cd_t_c = cdMc.getTranslationVector()
        cd_tu_c = cdMc.getThetaUVector()

        error_tr = (cd_t_c.sumSquare() ** 0.5) * 1000.0
        error_tu = Math.deg(cd_tu_c.sumSquare() ** 0.5)
        print(f"平移误差：{error_tr} mm;角度误差{error_tu} deg")

        if error_tr < convergence_threshold_t and error_tu < convergence_threshold_tu:
            strict_converged_count += 1
        else:
            strict_converged_count = 0

        if strict_converged_count >= strict_converged_samples:
            has_converged = True
            exit_reason = "converged"
            print("Servo task has converged on consecutive fresh vision samples; holding current joint position with MoveS")
            q_hold_rad = await duban_robot_con.getCurrentJointAngles()
            await duban_robot_con.MoveS(q_hold_rad.tolist())
            if opt_plot:
                experiment_logger.append(float(t_sim_ms), error_np, np.zeros(6), q_hold_rad)
            break

        # 5. DLS 核心算法
        # 设定阻尼因子 (可以根据需要微调，一般在 0.01 到 0.1 之间)
        lambda_dls = 0.02
        J1_T = J1_np.T
        J1_J1T_damped = np.dot(J1_np, J1_T) + (lambda_dls ** 2) * np.eye(J1_np.shape[0])
        J_pseudo_dls = np.dot(
            J1_T,
            np.linalg.solve(J1_J1T_damped, np.eye(J1_J1T_damped.shape[0])),
        )
        # 自适应动态增益：保持中误差推进力，但避免过零后因 MoveS 延迟形成振荡。
        x = np.linalg.norm(error_np, ord=np.inf)
        lambda_min = 0.35
        lambda_max = 0.95
        error_scale = 0.015
        dynamic_gain = lambda_min + (lambda_max - lambda_min) * (1.0 - np.exp(-x / error_scale))

        # 将 opt_adaptive_gain 开关作用到实际控制律增益上
        if opt_adaptive_gain:
            selected_gain = dynamic_gain
        else:
            selected_gain = 0.6

        error_norm = float(np.linalg.norm(error_np))
        error_is_increasing = prev_error_norm is not None and error_norm > prev_error_norm * 1.03
        if error_is_increasing:
            selected_gain *= 0.55

        q_dot_rad_s = -selected_gain * np.dot(J_pseudo_dls, error_np).flatten()

        now = time.perf_counter()
        if last_command_time is None:
            dt_control = dt_sim
        else:
            dt_control = min(max(now - last_command_time, min_dt_control), max_dt_control)

        near_target = error_tr < 2.0 and error_tu < 0.3
        mid_target = error_tr < 8.0 and error_tu < 0.8
        if near_target:
            max_vel = max_vel_near
            min_vel = min_vel_near
        elif mid_target:
            max_vel = max_vel_mid
            min_vel = min_vel_mid
        else:
            max_vel = max_vel_far
            min_vel = min_vel_far
        current_max = np.max(np.abs(q_dot_rad_s))
        if current_max > max_vel:
            # 等比例缩放，保持运动方向不变
            q_dot_rad_s = (q_dot_rad_s / current_max) * max_vel
            current_max = max_vel

        # MoveS 对极小目标步长不敏感；误差仍明显时给一个很小的速度下限。
        # 若误差开始增大，则取消下限，只保留阻尼降增益，避免再次振荡。
        if (
            not error_is_increasing
            and current_max > 1e-6
            and current_max < min_vel
            and (error_tr > 1.2 or error_tu > 0.18)
        ):
            q_dot_rad_s = (q_dot_rad_s / current_max) * min_vel

        max_delta_vel = max_acc * dt_control
        q_dot_rad_s = np.clip(
            q_dot_rad_s,
            prev_q_dot_rad_s - max_delta_vel,
            prev_q_dot_rad_s + max_delta_vel,
        )

        fine_ok = error_tr < fine_threshold_t and error_tu < fine_threshold_tu
        if fine_ok and np.max(np.abs(q_dot_rad_s)) < 0.015:
            fine_converged_count += 1
        else:
            fine_converged_count = 0

        if fine_converged_count >= fine_converged_samples:
            has_converged = True
            exit_reason = "converged"
            print("Servo task has reached the practical visual-noise band; holding current joint position with MoveS")
            q_hold_rad = await duban_robot_con.getCurrentJointAngles()
            await duban_robot_con.MoveS(q_hold_rad.tolist())
            if opt_plot:
                experiment_logger.append(float(t_sim_ms), error_np, np.zeros(6), q_hold_rad)
            break

        # 向机器人发送关节角度指令
        dt_command = min(max(dt_control, dt_sim), command_horizon)
        q_step_rad = q_dot_rad_s * dt_command
        if fine_ok and np.max(np.abs(q_step_rad)) < min_joint_step_rad:
            prev_q_dot_rad_s = np.zeros(6, dtype=float)
            prev_error_norm = error_norm
            if opt_plot:
                experiment_logger.append(float(t_sim_ms), error_np, np.zeros(6), q_current_rad)
            await asyncio.sleep(idle_sleep)
            continue

        q_next_rad_np = q_current_rad + q_step_rad
        q_next_rad = q_next_rad_np.tolist()
        t4 = time.perf_counter()
        await duban_robot_con.MoveS(q_next_rad)
        t5 = time.perf_counter()
        last_command_time = t5
        prev_q_dot_rad_s = q_dot_rad_s
        prev_error_norm = error_norm

        print(
            f"获取角度: {(t3-t2)*1000:.1f} | 视觉: {(t1-t0)*1000:.1f} | "
            f"计算: {(t4-t3)*1000:.1f} | MoveS: {(t5-t4)*1000:.1f} | "
            f"dt: {dt_control*1000:.1f} | cmd_dt: {dt_command*1000:.1f} | gain: {selected_gain:.2f} | "
            f"step: {np.rad2deg(np.max(np.abs(q_step_rad))):.3f} deg"
        )
        # 记录关节角速度、关节角和特征误差，实验结束后统一绘图
        if opt_plot:
            experiment_logger.append(float(t_sim_ms), error_np, q_dot_rad_s, q_current_rad)
        if first_time:
            first_time = False
        t_virtual += dt_sim  # 控制时间步进
        # 动态补偿时间
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt_sim - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # 如果处理时间超过了 dt_sim，直接进入下一轮，不睡眠
            print(f"Warning: Control loop is running behind schedule by {-sleep_time:.3f} seconds!")
            pass
    return exit_reason, experiment_logger
