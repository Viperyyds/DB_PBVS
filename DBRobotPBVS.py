from visp.core import HomogeneousMatrix
from visp.visual_features import FeatureTranslation
from visp.visual_features import FeatureThetaU
from visp.vs import Servo
from visp.core import measureTimeMs
from get_flower_pose_ema_visp import get_flower_pose_ema_visp
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
    convergence_threshold_t = 0.3  # mm
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

    final_quit = False
    has_converged = False
    # send_velocities = True
    servo_started = False

    dt_sim = 0.15  # 控制系统运行周期(这里需要与视觉更新位姿的时间相匹配)
    t_virtual = 0.0  # 运行总时间
    # 视觉伺服主循环
    first_time = True
    while not has_converged:
        # --- 键盘退出检测逻辑 ---
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # 27 是 Esc 的 ASCII 码，b'q' 是字符 q
            if key == b'q' or key == b'\x1b':
                print(" 检测到退出指令，正在停止机器人... ")
                # 紧急安全停止：发送当前位置
                q_stop = await duban_robot_con.getCurrentJointAngles()
                await duban_robot_con.MoveJ(q_stop.tolist())
                break  # 跳出 while 循环
        # 记录循环开始的时间
        loop_start = time.perf_counter()

        t_sim_ms = t_virtual * 1000.0
        # 1. 获取当前机器人关节位置 (单位: rad)
        t0 = time.perf_counter()
        q_current_rad = await duban_robot_con.getCurrentJointAngles()
        t1 = time.perf_counter()
        # 获取物体在相机坐标系下的位姿
        cMo_raw = get_flower_pose_ema_visp(image_queue, 0.3)
        t2 = time.perf_counter()
        if cMo_raw is None:
            print("Target Lost! Holding current position.")
            # 策略：向机器人发送当前的关节位置，使其原地不动
            await duban_robot_con.MoveS(q_current_rad.tolist())

            # 目标丢失时同样进行周期补偿，保持控制循环节拍一致
            elapsed_lost = time.perf_counter() - loop_start
            sleep_time_lost = dt_sim - elapsed_lost
            if sleep_time_lost > 0:
                await asyncio.sleep(sleep_time_lost)
            else:
                print(f"Warning: Control loop (target lost) is running behind schedule by {-sleep_time_lost:.3f} seconds!")

            t_virtual += dt_sim
            continue  # 跳过后续计算，进入下一轮感知
        cMo = cMo_raw

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

        # 5. DLS 核心算法
        # 设定阻尼因子 (可以根据需要微调，一般在 0.01 到 0.1 之间)
        lambda_dls = 0.02
        J1_T = J1_np.T
        J1_J1T_damped = np.dot(J1_np, J1_T) + (lambda_dls ** 2) * np.eye(J1_np.shape[0])
        J_pseudo_dls = np.dot(J1_T, np.linalg.inv(J1_J1T_damped))
        # 恒定增益
        gain = 5.0
        # 自适应动态增益
        x = np.linalg.norm(error_np, ord=np.inf)
        lambda_0 = 4.0  # λ(0): 误差为 0 时的增益 (近处的高增益，用于突破最后的误差壁垒)
        lambda_inf = 2.5  # λ(∞): 误差极大时的增益 (远处的低增益，用于保证初始运动平缓)
        lambda_prime_0 = 30.0  # λ'(0): x=0 处的斜率绝对值 (控制增益从高到底的下降速率，值越大降得越快)

        # 3. 根据官方公式计算动态增益 λ(x)
        # 公式: λ(x) = (λ_0 - λ_inf) * exp( - (λ'_0 / (λ_0 - λ_inf)) * x ) + λ_inf
        if abs(lambda_0 - lambda_inf) > 1e-6:  # 防止除以 0
            exponent = - (lambda_prime_0 / (lambda_0 - lambda_inf)) * x
            dynamic_gain = (lambda_0 - lambda_inf) * np.exp(exponent) + lambda_inf
        else:
            dynamic_gain = lambda_0

        # 将 opt_adaptive_gain 开关作用到实际控制律增益上
        if opt_adaptive_gain:
            selected_gain = dynamic_gain
        else:
            selected_gain = gain

        q_dot_rad_s = -selected_gain * np.dot(J_pseudo_dls, error_np).flatten()

        max_vel = 0.5
        current_max = np.max(np.abs(q_dot_rad_s))
        if current_max > max_vel:
            # 等比例缩放，保持运动方向不变
            q_dot_rad_s = (q_dot_rad_s / current_max) * max_vel

        # 向机器人发送关节角度指令
        q_next_rad = (q_current_rad + q_dot_rad_s * dt_sim).tolist()
        t3 = time.perf_counter()
        await duban_robot_con.MoveS(q_next_rad)
        t4 = time.perf_counter()

        print(f"获取角度: {(t1-t0)*1000:.1f} | 视觉: {(t2-t1)*1000:.1f} | 计算: {(t3-t2)*1000:.1f} | MoveS: {(t4-t3)*1000:.1f}")
        # 记录关节角速度、关节角和特征误差，实验结束后统一绘图
        if opt_plot:
            experiment_logger.append(float(t_sim_ms), error_np, q_dot_rad_s, q_current_rad)

        # 计算当前的特征误差
        cd_t_c = cdMc.getTranslationVector()
        cd_tu_c = cdMc.getThetaUVector()

        error_tr = (cd_t_c.sumSquare() ** 0.5) * 1000.0
        error_tu = Math.deg(cd_tu_c.sumSquare() ** 0.5)
        print(f"平移误差：{error_tr} mm;角度误差{error_tu} deg")
        # 收敛判断
        if (not has_converged and error_tr < convergence_threshold_t and
                error_tu < convergence_threshold_tu):
            has_converged = True
            print("Servo task has converged")
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
        
    if opt_plot:
        print("Experiment finished. Drawing recorded PBVS curves.")
        experiment_logger.plot()
