'''
测试用例：MoveJ 关节运动交互控制
说明：
- 该测试用例允许用户通过终端输入关节角度（度），并实时发送 MoveJ 指令给机器人。
- 用户可以连续输入多组关节角度，机器人将依次执行这些指令。
- 输入 'exit' 或 'quit' 可退出交互模式。
- 该测试用例适用于需要频繁调整关节位置的调试场景。
'''
import asyncio
import json
import csv
import math
import sys
from functools import partial
from time import sleep, time
from typing import Dict, List, Tuple

from PythonWorkFlow.Core.RobotCore import RobotCore
from PythonWorkFlow.Core.Basic import ControlMode 

# ==============================================================================
# 辅助函数
# ==============================================================================

def deg_to_rad(degrees: List[float]) -> List[float]:
    """
    Converts joint angles from degrees to radians.
    角度转弧度。
    """
    return [math.radians(d) for d in degrees]


def parse_joint_input(input_str: str) -> Tuple[bool, List[float]]:
    """
    Parses a string of joint angles (degrees) into a list of 6 floats.
    解析输入的关节角度字符串，并验证其是否包含6个有效数字。
    """
    # 替换常见的中文逗号和/或空格，然后按逗号分割
    cleaned_str = input_str.replace('，', ',').replace(' ', ',').strip()
    
    # 尝试分割并过滤空字符串
    try:
        parts = [float(p.strip()) for p in cleaned_str.split(',') if p.strip()]
    except ValueError:
        return False, []

    if len(parts) == 6:
        return True, parts
    else:
        return False, []


# ==============================================================================
# 异步测试用例 (初始化部分沿用 MoveJ 逻辑)
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
        # 核心设置：关节运动模式
        await rc.SetControlMode(ControlMode.MoveJoint) 
        print("✅ 控制模式已设置为MoveJoint")
    except Exception as e:
        print(f"❌ 设置控制模式失败：{e}")
        raise

async def run_interactive_control(rc: RobotCore):
    """
    Main interactive loop for continuous joint angle input and MoveJ execution.
    主交互循环，用于接收用户输入的关节角度并发送MoveJ指令。
    """
    print("\n==================================================")
    print("🤖 关节运动交互模式已启动 (MoveJoint)")
    print("   请按顺序输入 J1, J2, J3, J4, J5, J6 的角度值 (度)。")
    print("   示例输入: 10, -20, 30, 0, 45, -15")
    print("   输入 'exit' 或 'quit' 结束程序。")
    print("==================================================")

    # 持续循环等待输入
    while True:
        try:
            # 使用 asyncio.to_thread 包装阻塞的 input() 调用
            prompt = "\n>>> 请输入 6 个关节角度 (度, 逗号或空格分隔): "
            user_input = await asyncio.to_thread(partial(input, prompt))
            user_input = user_input.strip().lower()

            if user_input in ['exit', 'quit']:
                print("\n收到退出指令，正在中止交互控制...")
                break

            # 1. 解析和校验输入
            is_valid, joints_deg = parse_joint_input(user_input)

            if not is_valid:
                print("❌ 输入格式错误或数量不对。请确保输入了 6 个用逗号或空格分隔的数字。")
                continue

            # 2. 转换：角度 -> 弧度 (MoveJ 期望弧度输入)
            joints_rad = deg_to_rad(joints_deg)

            print(f"\n目标关节角（度）：{[round(j, 3) for j in joints_deg]}")
            print(f"目标关节角（弧度）：{[round(j, 6) for j in joints_rad]}")

            # 3. 发送 MoveJ 指令
            try:
                await rc.MoveJ(joints_rad) 
                print("✅ MoveJ 指令发送成功，机器人开始运动...")

                # 4. 实时读取并显示状态 (等待运动完成，约 1 秒后读取一次)
                print("正在等待机器人开始运动...")
                await asyncio.sleep(1.0) 
                status_json = await rc.getRobotStatus()
                status_dict = json.loads(status_json)
                
                is_moving = status_dict.get("isMoving", False)
                current_joints_rad = [status_dict["jointPositions"].get(f"J{i}", 0) for i in range(1, 7)]
                current_joints_deg = [round(math.degrees(r), 3) for r in current_joints_rad]
                
                print(f" [状态] 机器人移动状态：{'正在运动' if is_moving else '静止'}")
                print(f" [状态] 实际关节位置（度）：{current_joints_deg}")

            except Exception as e:
                print(f"\n❌ MoveJ 执行失败：{e}")
                # 运动失败，但继续循环等待下一次输入
                continue

        except KeyboardInterrupt:
            # 允许 Ctrl+C 退出
            print("\n收到 KeyboardInterrupt，正在中止交互控制...")
            break
        except Exception as e:
            # 捕获其他意外错误
            print(f"程序运行中发生意外错误: {e}")
            break

    print("\n----- 交互控制结束 -----")
    

async def run_test_cases(rc: RobotCore):
    """测试用例集合：按顺序执行初始化步骤和交互控制循环"""
    await test_init_tasks(rc)
    await test_robot_enable(rc)
    await test_set_control_mode(rc) 
    await run_interactive_control(rc) # 运行交互控制循环

# ==============================================================================
# 主执行入口
# ==============================================================================

def main():
    """主函数：组织测试流程，明确初始化→测试→清理的逻辑"""
    CODESYS_CONFIG = {
        "target_ip": "192.168.1.253",
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
        # Windows 平台设置事件循环策略
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        asyncio.run(run_test_cases(rc))
        
    except KeyboardInterrupt:
        print('\n收到 KeyboardInterrupt，正在中止测试并清理资源...')
    except Exception as e:
        print(f"\n致命错误：测试集运行中断: {e}")
    finally:
        print("\n开始清理资源...")
        try:
            rc.stop()
        except Exception as e:
            print(f"清理时发生错误：{e}")
        print("资源已清理，测试结束")


if __name__ == '__main__':
    main()
