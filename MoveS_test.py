import asyncio
import msvcrt
import time
from typing import List


from Core.RobotCore import *


def interpolate_joint_positions(start: List[float], end: List[float], num_points: int) -> List[List[float]]:
    """在两组关节角之间做线性插值，返回包含起终点的轨迹点。"""

    if len(start) != len(end):
        raise ValueError("起点和终点的关节维度不一致")
    if num_points < 2:
        raise ValueError("插值点数量至少需要 2 个")

    return [
        [
            start[j] + (end[j] - start[j]) * i / (num_points - 1)
            for j in range(len(start))
        ]
        for i in range(num_points)
    ]


def should_quit() -> bool:
    """非阻塞检查是否按下 q 键退出。"""

    while msvcrt.kbhit():
        key = msvcrt.getwch().lower()
        if key == "q":
            return True
    return False


async def test_moves(robot):
    if not robot.connected:
        raise RuntimeError("机器人未连接")

    pos_A = [0, 0, 0, 0, 0, 0]
    pos_C = [0.4, 0.05, 0.2, 0.1, 0.3, 0.1]
    num_points = 100

    trajectory = interpolate_joint_positions(pos_A, pos_C, num_points)

    print("=== MoveS 插值循环测试开始 ===")
    print(f"起点 A: {pos_A}")
    print(f"终点 C: {pos_C}")
    print(f"A 到 C 线性插值点数: {num_points}")
    print(f"A 到 C 轨迹总点数: {len(trajectory)}")
    print("运行过程中按 q 可退出循环。")

    cycle_idx = 0
    while True:
        if should_quit():
            print("检测到 q 键，程序退出。")
            break

        cycle_idx += 1
        print(f"=== 开始第 {cycle_idx} 轮轨迹下发 ===")

        for point_idx, target in enumerate(trajectory, start=1):
            if should_quit():
                print("检测到 q 键，程序退出。")
                return

            start_time = time.perf_counter()
            await robot.MoveS(target)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            await asyncio.sleep(0.08)

            print(
                f"[轮次 {cycle_idx:04d}] "
                f"点 {point_idx:03d}/{len(trajectory):03d} -> {target}, "
                f"MoveS 耗时: {elapsed_ms:.2f} ms"
            )
        break
    print("=== MoveS 插值循环测试结束 ===")
    robot.stop()



async def main():
    robot = RobotCore('192.168.1.253')
    await test_moves(robot)


if __name__ == "__main__":
    asyncio.run(main())
