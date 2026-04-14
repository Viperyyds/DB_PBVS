'''
测试用例：MoveL 直线运动（基于 JSON 文件的轨迹点执行）
说明：
- 该测试用例通过读取 JSON 文件中的轨迹点数据，逐步发送 MoveL 指令给机器人。
- 用户可以在 JSON 文件中定义多组轨迹点，机器人将依次执行这些指令。
- 该测试用例适用于需要批量处理直线运动的场景。
'''
import asyncio
import json
from time import sleep

from PythonWorkFlow.Core.RobotCore import RobotCore
from PythonWorkFlow.Core.Basic import ControlMode


import math
import numpy as np

def main():
    """主函数：组织测试流程，明确初始化→测试→清理的逻辑"""
    # 1. 定义连接配置（单独提取，便于修改和维护）
    CODESYS_CONFIG = {
        "target_ip": "192.168.232.155",
        # "target_ip": "192.168.1.2",
        "port": 502, 
        "unit_id": 1 
    }

    # 2. 直接声明并初始化RobotCore实例
    print("开始初始化RobotCore实例...")
    try:
        # 显式传入配置参数，明确初始化依赖
        rc = RobotCore(target_ip=CODESYS_CONFIG["target_ip"])
        print("RobotCore实例创建成功")
    except Exception as e:
        print(f"RobotCore初始化失败：{e}")
        return  # 初始化失败直接退出

    # 3. 验证连接状态（初始化后立即检查）
    if not rc.connected:
        print("❌ 与Codesys Modbus服务器连接失败，请检查：")
        print(f"  - IP: {CODESYS_CONFIG['target_ip']}:{CODESYS_CONFIG['port']}")
        print("  - Codesys Modbus从站是否启用")
        print("  - 网络是否通畅")
        return

    print("✅ 与Codesys Modbus服务器连接成功！")

    # 4. 执行异步测试逻辑
    try:
        asyncio.run(run_test_cases(rc))  # 分离测试用例
    except KeyboardInterrupt:
        print('\n收到 KeyboardInterrupt，正在中止测试并清理资源...')
        try:
            rc.stop()
        except Exception as e:
            print(f'清理时发生错误：{e}')
        raise
    finally:
        # 5. 确保资源清理（无论测试成功与否）
        print("开始清理资源...")
        try:
            rc.stop()
        except Exception as e:
            print(f"清理时发生错误：{e}")
        print("资源已清理，测试结束")


async def run_test_cases(rc: RobotCore):
    """测试用例集合"""
    
    # 测试0：初始化流程验证
    await test_init_tasks(rc)
    
    # 测试1：机器人使能
    await test_robot_enable(rc)

    # 测试2：设置关节运动模式
    await test_set_control_mode(rc)

    # 测试3：关节运动（MoveL）
    await test_move_L(rc)
    
async def test_init_tasks(rc: RobotCore):
    """测试初始化流程（init_tasks）的完整执行结果"""
    print("\n测试初始化流程（init_tasks）...")
    
    # 1. 确认机器人已连接（初始化的前提）
    if not rc.connected:
        print("❌ 机器人未连接，无法执行初始化流程")
        return
    
    # 2. 等待初始化任务执行完成（根据实际情况调整等待时间，确保异步任务跑完）
    print("等待初始化任务执行...（3秒）")
    await asyncio.sleep(3.0)  # 给足时间让init_tasks中的所有步骤执行
    
    # 3. 读取当前机器人状态，验证初始化结果
    try:
        status_json = await rc.getRobotStatus()
        status_dict = json.loads(status_json)
    except Exception as e:
        print(f"❌ 读取机器人状态失败（初始化验证失败）：{e}")
        raise
    
    # 4. 验证每个初始化步骤
    errors = []
    
    # 4.1 验证使能状态
    if not status_dict.get("isEnabled", False):
        errors.append("机器人未成功使能（RobotEnable失败）")

    # 4.2 验证控制模式是否为Calibration
    expected_mode = ControlMode.Calibration.value
    actual_mode = status_dict.get("mode", 0)
    if actual_mode != expected_mode:
        errors.append(f"控制模式未设置为Calibration（期望：{expected_mode}，实际：{actual_mode}）")

    # 5. 输出验证结果
    if not errors:
        print("✅ 初始化流程（init_tasks）所有步骤执行成功！")
    else:
        print(f"❌ 初始化流程存在以下问题：")
        for err in errors:
            print(f" - {err}")
        raise Exception("初始化流程验证失败")
    
# codesys中Power_On 置为 True
async def test_robot_enable(rc: RobotCore):
    """测试机器人使能"""
    print("\n1. 调用 RobotEnable（向Codesys发送使能指令）...")
    try:
        await rc.RobotEnable()
        await asyncio.sleep(0.5) 

        # 读取状态验证
        status_json = await rc.getRobotStatus()
        status_dict = json.loads(status_json)
        if status_dict["isEnabled"]:
            print("✅ 机器人使能成功（Codesys反馈已使能）")
        else:
            print("⚠️ 机器人使能指令已发送，但Codesys反馈未使能")
    except Exception as e:
        print(f"❌ 调用RobotEnable失败：{e}")
        raise  # 抛出异常，终止后续测试


async def test_set_control_mode(rc: RobotCore):
    """测试设置控制模式为关节运动"""
    print("\n2. 调用SetControlMode...")
    try:
        await rc.SetControlMode(ControlMode.MoveLinear)
        print("✅ 控制模式设置完成")
    except Exception as e:
        print(f"❌ 设置控制模式失败：{e}")
        raise


async def test_move_L(rc: RobotCore):
    """测试按JSON文件中的轨迹点执行运动和延时（RPY角度→旋转矢量弧度）"""
    print("\n3. 开始执行JSON文件中的轨迹指令...")
    
    # 1. 读取JSON文件（请确保路径正确）
    json_file_path = "Communication/Tests/waypointsA.json"
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            robot_data = json.load(f)
        commands = robot_data.get("robot_commands", [])
        if not commands:
            print("⚠️ JSON文件中未找到轨迹指令数据")
            return
        print(f"✅ 成功读取JSON文件，共{len(commands)}条指令")
    except Exception as e:
        print(f"❌ 读取JSON文件失败：{e}")
        raise

    # 2. 定义位姿误差阈值和超时参数（根据机器人精度调整）
    pos_threshold = 0.001  # 位置误差阈值（毫米）
    rot_threshold = 0.01   # 姿态误差阈值（弧度）
    timeout = 30           # 最大等待时间（秒）


    # 3. 按顺序执行每条指令
    for idx, cmd in enumerate(commands, 1):
        cmd_type = cmd.get("type")
        print(f"\n----- 执行第{idx}条指令（类型：{cmd_type}）-----")

        if cmd_type == "move":
            # 提取xyz和RPY角度（JSON中是角度制）
            xyz_rpy = cmd.get("xyz_rpy", {})
            x = xyz_rpy.get("x", 0.0)
            y = xyz_rpy.get("y", 0.0)
            z = xyz_rpy.get("z", 0.0)
            # RPY角度：[roll, pitch, yaw]（对应rx, ry, rz）
            rpy_deg = [
                xyz_rpy.get("rx", 0.0),
                xyz_rpy.get("ry", 0.0),
                xyz_rpy.get("rz", 0.0)
            ]
            location = cmd.get("location", "未知位置")
            print(f"原始姿态（RPY角度）：{[round(p, 3) for p in rpy_deg]}°")

            # 4. 核心转换：RPY角度→旋转矢量（弧度）
            try:
                rotvec_rad = rpy_deg_to_rotvec_rad(rpy_deg)
                print(f"转换后姿态（旋转矢量弧度）：{[round(p, 6) for p in rotvec_rad]}")
            except Exception as e:
                print(f"❌ 姿态转换失败：{e}")
                raise

            # 5. 组成目标位姿（xyz + 旋转矢量）：[x, y, z, rx, ry, rz]
            target_pose = [x, y, z] + rotvec_rad
            print(f"执行直线运动（{location}），目标位姿：{[round(p, 3) for p in target_pose]}")
            
            #预检查：当前位姿是否已达标（避免发送相同指令）
            try:
                # 读取当前实际位姿和运动模式
                pre_status_json = await rc.getRobotStatus()
                pre_status_dict = json.loads(pre_status_json)
                pre_tcp_pose = pre_status_dict.get("pose", {})
                pre_actual_pos = [pre_tcp_pose["x"], pre_tcp_pose["y"], pre_tcp_pose["z"]]
                pre_actual_rot = [pre_tcp_pose["roll"], pre_tcp_pose["pitch"], pre_tcp_pose["yaw"]]
                # 计算与目标位姿的误差
                pre_pos_error = sum((a - t)**2 for a, t in zip(pre_actual_pos, target_pose[:3])) **0.5
                pre_rot_error = sum((a - t)** 2 for a, t in zip(pre_actual_rot, target_pose[3:])) **0.5
                # 读取当前运动模式（假设getRobotStatus返回"controlMode"字段）
            except Exception as e:
                print(f"⚠️ 预检查失败，继续执行：{e}")
                pre_pos_error = float('inf')  # 强制不跳过指令

            # 若当前位姿已达标，直接跳过运动指令
            if pre_pos_error < pos_threshold and pre_rot_error < rot_threshold:
                print("✅ 当前位姿已满足目标，无需执行运动")
                continue

            # 6. 发送MoveL指令
            try:
                await rc.MoveL(target_pose)
                print("✅ MoveL指令已发送至机器人")
                
                # 7. 等待运动完成：对比目标位姿与实际TcpPose
                print(f"等待机器人到达目标位姿（位置阈值：{pos_threshold}mm，姿态阈值：{rot_threshold}rad）...")
                start_time = asyncio.get_event_loop().time()  # 记录开始时间
                reached = False

                while not reached:
                    # 检查超时
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    if elapsed_time > timeout:
                        raise TimeoutError(f"等待超时（{timeout}秒），未到达目标位姿")

                    # 读取机器人状态，提取实际TcpPose、关节位置、moving标志
                    try:
                        status_json = await rc.getRobotStatus()
                        status_dict = json.loads(status_json)

                        # print(f"\n【调试】完整status_dict结构：{list(status_dict.keys())}")  # 只打印键名，避免冗余
                        # print(f"\n【调试】打印完整status_dict内容：")
                        # print(status_dict)

                        # 从TcpPose中提取实际位置和姿态（根据getRobotStatus的结构）
                        tcp_pose = status_dict.get("pose", {})
                        actual_pos = [
                            tcp_pose["x"],  # 实际X坐标
                            tcp_pose["y"],  # 实际Y坐标
                            tcp_pose["z"]   # 实际Z坐标
                        ]
                        actual_rot = [
                            tcp_pose["roll"],   # 实际旋转矢量rx
                            tcp_pose["pitch"],  # 实际旋转矢量ry
                            tcp_pose["yaw"]     # 实际旋转矢量rz
                        ]
                        actual_pose = actual_pos + actual_rot

                        # 读当前关节角度
                        actual_joints = [
                            status_dict["jointPositions"]["J1"],
                            status_dict["jointPositions"]["J2"],
                            status_dict["jointPositions"]["J3"],
                            status_dict["jointPositions"]["J4"],
                            status_dict["jointPositions"]["J5"],
                            status_dict["jointPositions"]["J6"]
                        ]

                        # 读取moving标志
                        moving = [status_dict["isMoving"]]
                        print(f"机器人移动状态：{'正在移动' if moving[0] else '静止中'}")

                        # 将弧度转为角度制
                        actual_joints = [math.degrees(r) for r in actual_joints]
                        print(
                            f"当前Tcp位姿：{[round(p, 6) for p in actual_pose]} | 当前关节位置：{[round(p, 6) for p in actual_joints]}  ",
                            end='\r'
                        )


                    except Exception as e:
                        print(f"⚠️ 获取实际Tcp位姿失败，重试：{e}")
                        await asyncio.sleep(0.2)  # 重试间隔
                        continue

                    # 计算位置误差（欧氏距离）
                    pos_error = sum(
                        (actual - target)**2 for actual, target in zip(actual_pos, target_pose[:3])
                    )** 0.5

                    # 计算姿态误差（旋转矢量差的模长）
                    rot_error = sum(
                        (actual - target)**2 for actual, target in zip(actual_rot, target_pose[3:])
                    )** 0.5

                    # 判断是否满足阈值
                    if pos_error < pos_threshold and rot_error < rot_threshold:
                        reached = True
                        print("\n✅ 已到达目标位姿，继续执行下一条指令")
                    else:
                        # 未达标，短延时后再次检测
                        await asyncio.sleep(0.2)

            except Exception as e:
                print(f"\n❌ 运动指令执行失败：{e}")
                raise

        elif cmd_type == "sleep":
            # 延时处理（毫秒转秒）
            duration_ms = cmd.get("duration_ms", 0)
            duration_sec = duration_ms / 1000.0
            print(f"执行延时：{duration_sec}秒（{duration_ms}毫秒）")
            try:
                await asyncio.sleep(duration_sec)  # 异步延时
                print("✅ 延时完成")
            except Exception as e:
                print(f"❌ 延时执行失败：{e}")
                raise

        else:
            print(f"⚠️ 忽略未知指令类型：{cmd_type}")

    print("\n----- 所有轨迹指令执行完毕 -----")
    
# 调试用
async def read_joint_positions(rc: RobotCore):
    """辅助函数：通过read_直接读取当前关节位置"""
    positions = []
    for i in range(6):
        pos = await rc._service.read_real("Status.Joint_Actual_Position", i + 1)
        positions.append(pos)
    print(f"读取的关节位置：{positions}")
    return positions


def deg_to_rad(degrees):
    """角度转弧度"""
    return [math.radians(d) for d in degrees]

def rpy_to_rotation_matrix(rpy_rad):
    """
    RPY（弧度）转旋转矩阵（Z-Y-X顺序：先绕Z轴Yaw，再绕Y轴Pitch，最后绕X轴Roll）
    rpy_rad: [roll, pitch, yaw] 弧度制
    返回：3x3旋转矩阵
    """
    roll, pitch, yaw = rpy_rad
    
    # 绕X轴旋转（Roll）
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # 绕Y轴旋转（Pitch）
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # 绕Z轴旋转（Yaw）
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 旋转矩阵：Z-Y-X顺序（先Yaw，再Pitch，最后Roll）
    R = Rz @ Ry @ Rx  # 矩阵乘法顺序不可颠倒
    return R

def rotation_matrix_to_rotvec(R):
    """
    旋转矩阵转旋转矢量（轴角表示），修复θ=π的特殊情况
    R: 3x3旋转矩阵
    返回：旋转矢量 [x, y, z]（单位：弧度）
    """
    trace = np.trace(R)
    theta = math.acos((trace - 1) / 2.0)
    
    # 处理θ接近0的情况
    if theta < 1e-6:
        return [0.0, 0.0, 0.0]
    
    # 处理θ接近π的特殊情况（sinθ≈0）
    if abs(theta - math.pi) < 1e-6:
        # 利用旋转矩阵性质：R + I 的列向量平行于旋转轴
        R_plus_I = R + np.eye(3)  # R + 单位矩阵
        # 选择非零列向量作为旋转轴方向（避免全零）
        if np.linalg.norm(R_plus_I[:, 0]) > 1e-6:
            u = R_plus_I[:, 0]  # 取第0列
        elif np.linalg.norm(R_plus_I[:, 1]) > 1e-6:
            u = R_plus_I[:, 1]  # 取第1列
        else:
            u = R_plus_I[:, 2]  # 取第2列
        u = u / np.linalg.norm(u)  # 单位化旋转轴
    else:
        # 常规情况：通过矩阵元素差值计算旋转轴
        ux = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
        uy = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
        uz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
        u = np.array([ux, uy, uz])
        u = u / np.linalg.norm(u)  # 确保单位化
    
    # 旋转矢量 = 旋转轴 × 旋转角
    rotvec = [u[0] * theta, u[1] * theta, u[2] * theta]
    return rotvec

def rpy_deg_to_rotvec_rad(rpy_deg):
    """
    总转换函数：RPY角度制→旋转矢量弧度制
    rpy_deg: [roll_deg, pitch_deg, yaw_deg] 角度制
    返回：旋转矢量 [x, y, z] 弧度制
    """
    # 1. 角度转弧度
    rpy_rad = deg_to_rad(rpy_deg)
    
    # 2. RPY弧度转旋转矩阵
    R = rpy_to_rotation_matrix(rpy_rad)
    
    # 3. 旋转矩阵转旋转矢量（弧度）
    rotvec = rotation_matrix_to_rotvec(R)
    
    return rotvec


if __name__ == '__main__':
    main()
