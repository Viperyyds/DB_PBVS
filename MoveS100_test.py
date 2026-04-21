import asyncio
import time
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import List
from Core.RobotCore import RobotCore

async def test_moves_n_rounds(robot, num_rounds=1000):
    """执行指定轮数的 MoveS 调用测试，统计每次调用耗时"""

    if not robot.connected:
        raise RuntimeError("机器人未连接")

    durations = []  # 存储每轮耗时（毫秒）

    print(f"=== 开始 {num_rounds} 轮 MoveS 测试 ===")

    for i in range(num_rounds):
        # 生成六个关节目标位置，范围 [-0.5, 0.5]
        target = [random.uniform(-0.5, 0.5) for _ in range(6)]

        # 打印目标位置（保留四位小数）
        print(f"第 {i+1:4d} 轮目标: {[round(x, 4) for x in target]}")

        start_time = time.perf_counter()
        await robot.MoveS(target)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        durations.append(elapsed_ms)
        print(f"         耗时: {elapsed_ms:.2f} 毫秒")
        # 时间延迟
        await asyncio.sleep(0.05)

    # 计算统计值
    avg_duration = sum(durations) / len(durations)
    max_duration = max(durations)
    min_duration = min(durations)

    print("\n=== 测试完成 ===")
    print(f"测试轮数: {num_rounds}")
    print(f"平均耗时: {avg_duration:.2f} 毫秒")
    print(f"最大耗时: {max_duration:.2f} 毫秒")
    print(f"最小耗时: {min_duration:.2f} 毫秒")

    # 调用绘图函数
    plot_durations(durations, avg_duration, max_duration, num_rounds)


def plot_durations(durations: List[float], avg_duration: float, max_duration: float, num_rounds: int):
    """绘制耗时散点图，并标注平均值和最大值（支持中文）"""

    # ---------- 配置中文字体 ----------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示异常
    # ---------------------------------

    rounds = list(range(1, len(durations) + 1))

    plt.figure(figsize=(14, 7))

    # 散点图：每轮耗时
    plt.scatter(rounds, durations, color='steelblue', alpha=0.5, label='单次耗时', s=15, edgecolors='none')

    # 平均耗时水平线（红色虚线）
    plt.axhline(y=avg_duration, color='red', linestyle='--', linewidth=2)

    # 最大耗时水平线（绿色虚线）
    plt.axhline(y=max_duration, color='green', linestyle='--', linewidth=2)

    # ---------- 添加数值文本标注 ----------
    x_max = len(durations)  # x 轴最大值
    # 平均耗时标注（放在线右端偏上位置）
    plt.text(x_max * 0.98, avg_duration + (max_duration - avg_duration) * 0.03,
             f'平均: {avg_duration:.2f} ms', color='red', fontsize=11, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # 最大耗时标注（放在线右端偏下位置）
    plt.text(x_max * 0.98, max_duration - (max_duration - avg_duration) * 0.03,
             f'最大: {max_duration:.2f} ms', color='green', fontsize=11, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # ------------------------------------

    # 图例（手动构建以包含数值）
    plt.legend(handles=[
        plt.Line2D([0], [0], color='steelblue', marker='o', linestyle='', label='单次耗时'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'平均耗时: {avg_duration:.2f} ms'),
        plt.Line2D([0], [0], color='green', linestyle='--', label=f'最大耗时: {max_duration:.2f} ms')
    ], loc='best')

    plt.xlabel('测试轮次', fontsize=13)
    plt.ylabel('耗时 (毫秒)', fontsize=13)
    plt.title(f'MoveS 调用耗时分布 ({num_rounds} 轮)', fontsize=15)
    plt.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()


async def main():
    # 请根据实际机器人 IP 修改
    robot = RobotCore('192.168.1.253')
    # 运行 50 轮测试
    await test_moves_n_rounds(robot, num_rounds=50)


if __name__ == "__main__":
    asyncio.run(main())