import asyncio
import time
from typing import List


from Core.RobotCore import *

async def test_moves(robot):

    if not robot.connected:
        raise RuntimeError(f"机器人未连接")

    # 定义三个关节目标位置（示例值，请根据实际机器人安全范围修改）
    pos_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pos_B = [0.2, 0.3, 0.2, 0.2, 0.4, 0.6]
    pos_C = [0.4, 0.05, 0.2, 0.1, 0.3, 0.1]

    print(f"=== MoveS 测试开始 ===")

    # 1. 发送第一个目标
    print(f"发送目标 A: {pos_A}")
    start_time = time.perf_counter()
    await robot.MoveS(pos_A)
    end_time = time.perf_counter()
    print(f"MoveS 调用耗时: { (end_time - start_time) * 1000:.2f} 毫秒")

    # 等待 0.5 秒后发送第二个目标（应打断第一个运动）
    await asyncio.sleep(0.5)
    print(f"发送目标 B: {pos_B}")
    start_time2 = time.perf_counter()
    await robot.MoveS(pos_B)
    end_time2 = time.perf_counter()
    print(f"MoveS 调用耗时: { (end_time2 - start_time2) * 1000:.2f} 毫秒")

    # 等待 1 秒后发送第三个目标（再次打断）
    await asyncio.sleep(1.0)
    print(f"发送目标 C: {pos_C}")
    await robot.MoveS(pos_C)

    await asyncio.sleep(3.0)
    print("=== MoveS 测试结束 ===")


async def main():
    robot = RobotCore('10.16.208.244')

    await test_moves(robot)


if __name__ == "__main__":
    asyncio.run(main())