'''
测试用例：MoveJ 关节运动（基于 CSV 文件的手动步进执行）
说明：
- 该测试用例通过读取 CSV 文件中的关节角度数据，逐步发送 MoveJ 指令给机器人。
- 用户可以在 CSV 文件中定义多组关节角度，机器人将依次执行这些指令。
- 该测试用例适用于需要批量处理关节运动的场景。
'''
import asyncio
import json
import csv
import math
import sys
from functools import partial
from time import sleep, time
from typing import Dict, List, Tuple

# 假设 RobotCore 和 ControlMode 在同一路径下
# (如果不在，请确保 PythonWorkFlow 在您的 PYTHONPATH 中)
try:
    from PythonWorkFlow.Core.RobotCore import RobotCore
    from PythonWorkFlow.Core.Basic import ControlMode
except ImportError:
    print("错误: 无法导入 RobotCore 或 ControlMode。")
    print("请确保 PythonWorkFlow 目录位于您的 PYTHONPATH 中，或者此脚本在正确的相对路径下运行。")
    sys.exit(1)


# ==============================================================================
# 辅助函数
# ==============================================================================

def deg_to_rad(degrees: List[float]) -> List[float]:
    """角度转弧度"""
    return [math.radians(d) for d in degrees]

async def wait_for_user_input(prompt: str = "按 Enter 键继续...") -> None:
    """异步等待用户输入（在单独线程中执行阻塞的 input()）"""
    print(f"\n{'-'*30}\n{prompt}\n{'-'*30}")
    # 使用 asyncio.to_thread 包装阻塞的 input() 调用
    await asyncio.to_thread(partial(input, ""))


def load_csv_data(file_path: str) -> List[Dict]:
    """
    读取 CSV 文件并解析关节角度数据。
    CSV 格式要求: Id, J1, J2, J3, J4, J5, J6, Info
    返回: 包含关节数据（角度制）的列表
    """
    commands = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # 读取表头

            # 验证和定位关节列
            joint_keys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
            if header[1:7] != joint_keys:
                print(f"⚠️ 警告：CSV表头顺序可能不标准 ({header[1:7]})，尝试按位置解析。")
            
            for row in reader:
                if len(row) < 7:
                    print(f"警告: 跳过不完整的行: {row}")
                    continue
                
                try:
                    # 提取关节角度（角度制）
                    joint_angles_deg = [float(angle) for angle in row[1:7]]
                    
                    commands.append({
                        "id": row[0],
                        "joint_angles_deg": joint_angles_deg,
                        "location": row[7] if len(row) > 7 else f"ID {row[0]}"
                    })
                except ValueError as e:
                    print(f"错误: 转换行数据失败 ({row})，跳过. 错误: {e}")
            
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        raise
    except Exception as e:
        print(f"❌ 错误: 读取CSV文件时发生未知错误: {e}")
        raise
        
    return commands


# ==============================================================================
# 异步测试用例 (MoveJ)
# ==============================================================================

async def test_init_tasks(rc: RobotCore):
    """测试初始化流程（init_tasks）的完整执行结果"""
    print("\n测试初始化流程（init_tasks）...")

    if not rc.connected:
        print("❌ 机器人未连接，无法执行初始化流程")
        return
    await asyncio.sleep(3.0) 
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

async def test_set_control_mode(rc: RobotCore):
    """设置控制模式为关节运动"""
    print("\n2. 调用SetControlMode...")
    try:
        await rc.SetControlMode(ControlMode.MoveJoint) 
        print("✅ 控制模式已设置为MoveJoint")
    except Exception as e:
        print(f"❌ 设置控制模式失败：{e}")
        raise

async def test_move_J(rc: RobotCore):
    """测试手动步进执行 CSV 文件中的关节运动指令"""
    print("\n3. 开始执行CSV文件中的关节运动指令...")
    
    # TODO 1. 读取CSV文件
    csv_file_path = "Communication/Tests/JointData20251031.csv" 
    try:
        joint_commands = load_csv_data(csv_file_path)
        if not joint_commands:
            print("⚠️ CSV文件中未找到关节运动指令数据")
            return
        print(f"✅ 成功读取CSV文件，共{len(joint_commands)}个目标点")
    except Exception as e:
        print(f"❌ 读取CSV文件失败：{e}")
        raise

    # 2. 按顺序手动执行每条指令
    for idx, cmd in enumerate(joint_commands, 1):
        location = cmd["location"]
        joints_deg = cmd["joint_angles_deg"]
        
        # 2.1 转换：角度 -> 弧度 (MoveJ 期望弧度输入)
        joints_rad = deg_to_rad(joints_deg)
        
        print(f"\n--- 目标点 {idx}/{len(joint_commands)} (位置: {location}) ---")
        print(f"目标关节角（度）：{[round(j, 3) for j in joints_deg]}")
        print(f"目标关节角（弧度）：{[round(j, 6) for j in joints_rad]}")

        # 2.2 等待用户按下 Enter 键
        await wait_for_user_input(
            prompt=f">>> 按 Enter 发送 MoveJ 指令 ({location}) <<<"
        )
        
        # 2.3 发送 MoveJ 指令
        try:
            # 传入弧度制的关节位置
            await rc.MoveJ(joints_rad) 
            print("✅ MoveJ 指令发送成功，机器人开始运动...")

            # 2.4 读取状态
            await asyncio.sleep(0.5) 
            status_json = await rc.getRobotStatus()
            status_dict = json.loads(status_json)
            
            # 显示当前状态
            is_moving = status_dict.get("isMoving", False)
            current_joints_rad = [status_dict["jointPositions"].get(f"J{i}", 0) for i in range(1, 7)]
            current_joints_deg = [round(math.degrees(r), 3) for r in current_joints_rad]
            
            print(f" [状态] 机器人移动状态：{'正在运动' if is_moving else '静止'}")
            print(f" [状态] 实际关节位置（度）：{current_joints_deg}")

        except Exception as e:
            print(f"\n❌ MoveJ 执行失败：{e}")
            raise

    print("\n----- 所有轨迹指令发送完毕 -----")
    
async def run_test_cases(rc: RobotCore):
    """测试用例集合"""
    # await test_init_tasks(rc) # 假设的初始化，如果 RobotCore 已处理，则注释掉
    await test_robot_enable(rc)
    await test_set_control_mode(rc) 
    await test_move_J(rc)

# ==============================================================================
# (新功能) 异步任务执行与清理
# ==============================================================================

async def run_test_and_cleanup(rc: RobotCore):
    """
    (新) 这是一个包裹函数，用于执行主测试，并确保在退出时执行异步清理（去使能）。
    """
    try:
        # 1. 执行主要的测试用例
        await run_test_cases(rc)
    
    except KeyboardInterrupt:
        print('\n(async) 收到 KeyboardInterrupt，正在中止测试并清理资源...')
        # 重新抛出异常，以便 main() 中的同步 finally 也能执行
        raise 
    except Exception as e:
        print(f"\n(async) 致命错误：测试集运行中断: {e}")
        # 重新抛出异常
        raise
    finally:
        # 2. (新功能) 异步清理：无论成功、失败还是中断，都尝试去使能
        print("\n(async) 开始异步清理 (去使能)...")
        if rc.connected:
            try:
                print("  - 正在发送去使能指令 (Power_Off)...")
                await rc._service.write_bool('Instructions.Power_On', False)
                print("  - 去使能指令发送成功")

            except AttributeError:
                # 如果 rc._service 不存在 (例如 RobotCore 封装了它)
                print("  - ⚠️ 警告：无法访问 'rc._service'。无法自动去使能。")
                print("  - 提示：最好的做法是在 RobotCore 中添加一个 'async def RobotDisable(self)' 方法，")
                print("  -       然后在 RobotCore 中调用 'await rc.RobotDisable()'。")
            except Exception as e:
                print(f"  - ❌ 异步清理（去使能）时发生错误：{e}")
        else:
            print("  - 机器人未连接，跳过去使能。")

# ==============================================================================
# 主执行入口
# =================================================S=============================

def main():
    """主函数：组织测试流程，明确初始化→测试→清理的逻辑"""
    
    # TODO 修改IP
    CODESYS_CONFIG = {
        "target_ip": "192.168.232.155",
        # "target_ip": "192.168.1.253",
        "port": 502, 
        "unit_id": 1 
    }

    print("开始初始化RobotCore实例...")
    rc = None # 先声明
    
    try:
        rc = RobotCore(target_ip=CODESYS_CONFIG["target_ip"])
        print("RobotCore实例创建成功")
    except Exception as e:
        print(f"RobotCore初始化失败：{e}")
        return

    if not rc.connected:
        print("❌ 与Codesys Modbus服务器连接失败，请检查配置和网络。")
        # 即使连接失败，也尝试调用 rc.stop() 来清理可能的半开连接
    
    # (新) 将同步的 rc.stop() 放入顶层 finally，确保它总能执行
    try:
        if not rc.connected:
            # 如果未连接，就不运行测试
            return 

        print("✅ 与Codesys Modbus服务器连接成功！")

        # Windows 平台设置事件循环策略
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        # (新) 调用包含异步清理的包裹函数
        asyncio.run(run_test_and_cleanup(rc))
        
    except KeyboardInterrupt:
        print('\n...主程序已中止')
    except Exception as e:
        # 这个异常是由 run_test_and_cleanup 重新抛出的
        print(f"\n...主程序因错误退出")
    finally:
        # (新) 同步清理：这是最后执行的代码
        print("\n(sync) 开始同步清理 (rc.stop)...")
        if rc:
            try:
                rc.stop()
            except Exception as e:
                print(f"  - ❌ rc.stop() 清理时发生错误：{e}")
        print("资源已清理，测试结束")


if __name__ == '__main__':
    main()