'''
测试用例：MoveL 直线运动（基于 JSON 文件的轨迹点执行）
说明：
- 该测试用例通过读取 JSON 文件中的轨迹点数据，逐步发送 MoveL 指令给机器人。
- 用户可以在 JSON 文件中定义多组轨迹点，机器人将依次执行这些指令。
- 该测试用例适用于需要批量处理直线运动的场景。
'''
import asyncio
import json
import math
import sys  
from time import sleep, time
from typing import Dict, List

import numpy as np
from PythonWorkFlow.Core.RobotCore import RobotCore
from PythonWorkFlow.Core.Basic import ControlMode 


# ==============================================================================
# 几何转换函数 (RPY角度 -> 旋转矢量弧度)
# ==============================================================================

def deg_to_rad(degrees: List[float]) -> List[float]:
    """角度转弧度"""
    return [math.radians(d) for d in degrees]

def rpy_to_rotation_matrix(rpy_rad: List[float]) -> np.ndarray:
    """RPY（弧度）转旋转矩阵（Z-Y-X顺序）"""
    roll, pitch, yaw = rpy_rad
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return R

def rotation_matrix_to_rotvec(R: np.ndarray) -> List[float]:
    """旋转矩阵转旋转矢量（轴角表示）"""
    trace = np.trace(R)
    trace = np.clip(trace, -1.0, 3.0) 
    
    theta = math.acos((trace - 1) / 2.0)
    
    if theta < 1e-6:
        return [0.0, 0.0, 0.0]
    
    if abs(theta - math.pi) < 1e-6:
        R_plus_I = R + np.eye(3) 
        norms = [np.linalg.norm(R_plus_I[:, i]) for i in range(3)]
        u = R_plus_I[:, np.argmax(norms)]
        u = u / np.linalg.norm(u)
    else:
        sin_theta = 2 * math.sin(theta)
        ux = (R[2, 1] - R[1, 2]) / sin_theta
        uy = (R[0, 2] - R[2, 0]) / sin_theta
        uz = (R[1, 0] - R[0, 1]) / sin_theta
        u = np.array([ux, uy, uz])
        u = u / np.linalg.norm(u) 
    
    rotvec = [u[0] * theta, u[1] * theta, u[2] * theta]
    return rotvec

def rpy_deg_to_rotvec_rad(rpy_deg: List[float]) -> List[float]:
    """总转换函数：RPY角度制→旋转矢量弧度制"""
    rpy_rad = deg_to_rad(rpy_deg)
    R = rpy_to_rotation_matrix(rpy_rad)
    rotvec = rotation_matrix_to_rotvec(R)
    return rotvec


# ==============================================================================
# 异步测试用例
# ==============================================================================

async def test_init_tasks(rc: RobotCore):
    """测试初始化流程（init_tasks）的完整执行结果"""
    print("\n测试初始化流程（init_tasks）...")
    # ... (代码不变)
    if not rc.connected:
        print("❌ 机器人未连接，无法执行初始化流程")
        return
    await asyncio.sleep(0.5) 
    try:
        status_json = await rc.getRobotStatus()
        status_dict = json.loads(status_json)
    except Exception as e:
        print(f"❌ 读取机器人状态失败（初始化验证失败）：{e}")
        raise
    errors = []
    if not status_dict.get("isEnabled", False):
        errors.append("机器人未成功使能（RobotEnable失败）")
    expected_mode = ControlMode.Calibration.value
    actual_mode = status_dict.get("mode", 0)
    if actual_mode != expected_mode:
        errors.append(f"控制模式未设置为Calibration（期望：{expected_mode}，实际：{actual_mode}）")
    if not errors:
        print("✅ 初始化流程（init_tasks）所有步骤执行成功！")
    else:
        print("❌ 初始化流程存在以下问题：")
        for err in errors:
            print(f" - {err}")
        raise Exception("初始化流程验证失败")
    
async def test_robot_enable(rc: RobotCore):
    """测试机器人使能"""
    print("\n1. 调用 RobotEnable（向Codesys发送使能指令）...")
    try:
        await rc.RobotEnable()
        await asyncio.sleep(0.5) 
        status_json = await rc.getRobotStatus()
        status_dict = json.loads(status_json)
        if status_dict.get("isEnabled", False):
            print("✅ 机器人使能成功（Codesys反馈已使能）")
        else:
            print("⚠️ 机器人使能指令已发送，但Codesys反馈未使能")
    except Exception as e:
        print(f"❌ 调用RobotEnable失败：{e}")
        raise 

async def test_move_L(rc: RobotCore):
    """测试按JSON文件中的轨迹点执行运动和延时"""
    print("\n3. 开始执行JSON文件中的轨迹指令...")
    
    json_file_path = "Communication/Tests/waypointsB.json"
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

    # 运动参数
    pos_threshold = 0.05      # 位置误差阈值（毫米）
    rot_threshold = 0.01      # 姿态误差阈值（弧度）
    timeout = 60              # 最大等待时间（秒）
    max_print_length = 0      # 用于优化打印刷新的最大长度

    # 3. 按顺序执行每条指令
    for idx, cmd in enumerate(commands, 1):
        cmd_type = cmd.get("type")
        print(f"\n----- 执行第{idx}条指令（类型：{cmd_type}）-----")

        if cmd_type == "move":
            xyz_rpy = cmd.get("xyz_rpy", {})
            x = xyz_rpy.get("x", 0.0)
            y = xyz_rpy.get("y", 0.0)
            z = xyz_rpy.get("z", 0.0)
            rpy_deg = [xyz_rpy.get("rx", 0.0), xyz_rpy.get("ry", 0.0), xyz_rpy.get("rz", 0.0)]
            location = cmd.get("location", "未知位置")
            print(f"原始姿态（RPY角度）：{[round(p, 3) for p in rpy_deg]}°")

            try:
                rotvec_rad = rpy_deg_to_rotvec_rad(rpy_deg)
                print(f"转换后姿态（旋转矢量弧度）：{[round(p, 6) for p in rotvec_rad]}")
            except Exception as e:
                print(f"❌ 姿态转换失败：{e}")
                raise

            target_pose = [x, y, z] + rotvec_rad
            print(f"执行直线运动（{location}），目标位姿：{[round(p, 3) for p in target_pose]}")
            
            # --- 预检查 (为简洁起见，此处省略) ---
            
            # 6. 发送MoveL指令
            try:
                await rc.MoveL(target_pose)
                print("✅ MoveL指令已发送至机器人")
                
                # 7. 等待运动完成：核心等待循环
                print(f"等待机器人到达目标位姿（位置阈值：{pos_threshold}mm，姿态阈值：{rot_threshold}rad）...")
                start_time = time()
                reached = False

                while not reached:
                    elapsed_time = time() - start_time
                    if elapsed_time > timeout:
                        # 打印失败信息前，清除状态信息
                        sys.stdout.write(f"\r{' ' * max_print_length}\r") 
                        sys.stdout.flush()
                        raise TimeoutError(f"等待超时（{timeout}秒），未到达目标位姿")

                    # 读取机器人状态
                    try:
                        status_json = await rc.getRobotStatus()
                        status_dict = json.loads(status_json)
                    except Exception as e:
                        await asyncio.sleep(0.2)
                        continue

                    # 提取实际位姿和运动状态
                    tcp_pose = status_dict.get("pose", {})
                    actual_pos = [tcp_pose.get("x", 0), tcp_pose.get("y", 0), tcp_pose.get("z", 0)]
                    actual_rot = [tcp_pose.get("roll", 0), tcp_pose.get("pitch", 0), tcp_pose.get("yaw", 0)]
                    actual_pose = actual_pos + actual_rot
                    
                    actual_joints = [status_dict["jointPositions"].get(f"J{i}", 0) for i in range(1, 7)]
                    is_moving = status_dict.get("isMoving", True)
                    
                    # 计算位置/姿态误差
                    pos_error = np.linalg.norm(np.array(actual_pos) - np.array(target_pose[:3]))
                    rot_error = np.linalg.norm(np.array(actual_rot) - np.array(target_pose[3:]))
                    
                    # 将弧度转为角度制
                    actual_joints_deg = [math.degrees(r) for r in actual_joints]
                    
                    # --- 优化后的打印逻辑（方案一）---
                    current_output = (
                        f"位姿：{[round(p, 4) for p in actual_pose]} | "
                        f"关节位置：{[round(p, 4) for p in actual_joints_deg]} |"
                        f"位置误差：{pos_error:.6f}mm | 姿态误差：{rot_error:.6f}rad"
                    )
                    
                    # 更新最大输出长度
                    max_print_length = max(max_print_length, len(current_output))
                    
                    # 构造输出行：\r 回车 + 内容 + 空格填充
                    output_line = f"\r{current_output}{' ' * (max_print_length - len(current_output))}"
                    
                    sys.stdout.write(output_line)
                    sys.stdout.flush() 
                    
                    # --- 打印逻辑结束 ---

                    # 检查到达条件：必须静止 AND 误差小于阈值
                    if not is_moving:
                        if pos_error < pos_threshold and rot_error < rot_threshold:
                            reached = True
                            # 打印成功信息前，清除状态信息
                            sys.stdout.write(f"\r{' ' * max_print_length}\r") 
                            sys.stdout.flush()
                            print("✅ 已到达目标位姿，继续执行下一条指令")
                        else:
                            await asyncio.sleep(0.1)
                    else:
                        await asyncio.sleep(0.1)

            except Exception as e:
                # 打印失败信息前，清除状态信息
                sys.stdout.write(f"\r{' ' * max_print_length}\r") 
                sys.stdout.flush()
                print(f"\n❌ 运动指令执行失败：{e}")
                raise

        elif cmd_type == "sleep":
            duration_ms = cmd.get("duration_ms", 0)
            duration_sec = duration_ms / 1000.0
            print(f"执行延时：{duration_sec}秒（{duration_ms}毫秒）")
            try:
                await asyncio.sleep(duration_sec) 
                print("✅ 延时完成")
            except Exception as e:
                print(f"❌ 延时执行失败：{e}")
                raise

        else:
            print(f"⚠️ 忽略未知指令类型：{cmd_type}")

    print("\n----- 所有轨迹指令执行完毕 -----")


async def run_test_cases(rc: RobotCore):
    """测试集合"""
    await test_init_tasks(rc)
    await test_robot_enable(rc)
    await test_move_L(rc)
    

# ==============================================================================
# 主执行入口
# ==============================================================================

def main():
    """主函数：组织测试流程，明确初始化→测试→清理的逻辑"""
    CODESYS_CONFIG = {
        "target_ip": "192.168.232.155",
        "port": 502, 
        "unit_id": 1 
    }

    print("开始初始化RobotCore实例...")
    try:
        rc = RobotCore(target_ip=CODESYS_CONFIG["target_ip"]) 
        print("RobotCore实例创建成功")
    except Exception as e:
        print(f"RobotCore初始化失败：{e}")
        return

    if not rc.connected:
        print("❌ 与Codesys Modbus服务器连接失败，请检查配置和网络。")
        return

    print("✅ 与Codesys Modbus服务器连接成功！")

    try:
        asyncio.run(run_test_cases(rc))
    except Exception as e:
        print(f"\n致命错误：测试集运行中断: {e}")
    finally:
        print("\n开始清理资源...")
        sleep(1)
        rc.stop()
        print("资源已清理，测试结束")


if __name__ == '__main__':
    # 强制使用 Windows 的 ProactorEventLoop，以确保兼容性
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError:
             pass 
    main()