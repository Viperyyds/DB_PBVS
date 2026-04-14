import asyncio
import os
import json
from unittest.mock import MagicMock, patch
from time import sleep

from PythonWorkFlow.Core.RobotCore import RobotCore
from PythonWorkFlow.Core.Basic import ControlMode

def main():
    """主函数：组织测试流程，明确初始化→测试→清理的逻辑"""
    # 1. 定义连接配置（单独提取，便于修改和维护）
    CODESYS_CONFIG = {
        # "target_ip": "192.168.232.165",
        # "target_ip": "192.168.1.105",
        "target_ip": "192.168.1.2",
        "port": 502, 
        "unit_id": 1 
    }

    # 2. 直接声明并初始化RobotCore实例（核心优化点）
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
    finally:
        # 5. 确保资源清理（无论测试成功与否）
        print("开始清理资源...")
        sleep(2)
        rc.stop()
        print("资源已清理，测试结束")


async def run_test_cases(rc: RobotCore):
    """测试用例集合"""
    
    # 测试0：初始化流程验证
    await test_init_tasks(rc)
    
    # 测试1：机器人使能
    await test_robot_enable(rc)

    # 测试2：设置关节运动模式
    await test_set_control_mode(rc)

    # 测试3：关节运动（MoveJ）
    await test_move_j(rc)
    
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
    print("\n2. 调用SetControlMode（设置为关节运动模式）...")
    try:
        await rc.SetControlMode(ControlMode.MoveJoint)
        print("✅ 控制模式设置完成（关节运动模式）")
    except Exception as e:
        print(f"❌ 设置控制模式失败：{e}")
        raise


async def test_move_j(rc: RobotCore):
    """测试关节运动（MoveJ）"""
    print("\n3. 调用MoveJ（向Codesys发送J1-J6目标位置）...")
    target_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 安全目标位置
    try:
        await rc.MoveJ(target_joints)
        print(f"✅ MoveJ指令已发送（目标位置：{target_joints}）")

        # 等待运动执行
        print("等待Codesys执行关节运动...（5秒后读取状态）")
        await asyncio.sleep(5)

        # 验证运动结果
        status_json = await rc.getRobotStatus()
        status_dict = json.loads(status_json)
   
        actual_joints = [
            status_dict["jointPositions"]["J1"],
            status_dict["jointPositions"]["J2"],
            status_dict["jointPositions"]["J3"],
            status_dict["jointPositions"]["J4"],
            status_dict["jointPositions"]["J5"],
            status_dict["jointPositions"]["J6"]
        ]
        print(f"当前关节位置（Codesys反馈）：{[round(p, 6) for p in actual_joints]}")

    except Exception as e:
        print(f"❌ 调用MoveJ失败：{e}")
        raise
    
# 调试用
async def read_joint_positions(rc: RobotCore):
    """辅助函数：通过read_直接读取当前关节位置"""
    positions = []
    for i in range(6):
        pos = await rc._service.read_real("Status.Joint_Actual_Position", i + 1)
        positions.append(pos)
    print(f"读取的关节位置：{positions}")
    return positions


if __name__ == '__main__':
    main()
