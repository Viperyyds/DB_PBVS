import numpy as np
import time
import asyncio
from multiprocessing import Process, Pipe, Queue
from SCameraProcess import SCameraProcess
from get_flower_pose_ema_visp import get_flower_pose_ema_visp

async def main_test():
    # 1. 加载相同的配置文件（确保环境一致）
    flower_tag_board_params = np.load('flower_tag_cali_params_board_2025032418.npy', allow_pickle=True).item()
    stereo_calb_params = np.load('baser4096_camera_stereo_matlab_20250617.npy', allow_pickle=True).item()
    db_tag_ids = np.arange(7, 25)

    # 2. 启动相机后台进程
    image_data_queue = Queue(1)
    out_pipe, in_pipe = Pipe(True)
    s_cam_process = SCameraProcess(
        camera_calib_params=stereo_calb_params,
        flower_tag_board_params=flower_tag_board_params, 
        db_tag_id_lst=db_tag_ids,
        cam_exp_time=10000, 
        pipe=[out_pipe, in_pipe],
        result_queue=image_data_queue
    )
    s_cam_process.start()
    print(">>> 相机进程已启动，正在预热...")
    time.sleep(2) # 等待相机初始化

    # 3. 测试参数设置
    test_frames = 1000
    latencies = []      # 记录函数单次执行耗时
    intervals = []      # 记录两次获取到有效新数据的时间间隔
    last_valid_time = time.perf_counter()

    print(f">>> 开始测试，共计 {test_frames} 帧。请在相机前缓慢晃动靶标...")

    count = 0
    while count < test_frames:
        start_t = time.perf_counter()
        
        # --- 核心测试目标 ---
        # 这里的 alpha 设为 0.3 与你主程序一致
        cMo_raw = get_flower_pose_ema_visp(image_data_queue, alpha=0.3)
        # ------------------
        
        end_t = time.perf_counter()
        
        # 记录函数本身的计算耗时 (CPU时间)
        latencies.append((end_t - start_t) * 1000)

        if cMo_raw is not None:
            # 记录数据更新间隔 (反映真实的视觉刷新率)
            now = time.perf_counter()
            intervals.append((now - last_valid_time) * 1000)
            last_valid_time = now
            
            count += 1
            if count % 20 == 0:
                print(f"进度: {count}/{test_frames}...")

        # 模拟控制循环的极短休眠，防止 CPU 空转
        await asyncio.sleep(0.001)

    # 4. 统计结果
    print("" + "="*30)
    print("视觉性能测试报告")
    print("="*30)
    
    # 函数内部计算耗时
    avg_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    print(f"1. 函数执行耗时 (Internal Latency):")
    print(f"   - 平均: {avg_lat:.2f} ms")
    print(f"   - 最大: {max_lat:.2f} ms")
    print(f"   (注：此项反映 CPU 处理位姿的速度，通常很短)")

    # 数据更新周期
    if len(intervals) > 1:
        # 排除掉第一帧的异常间隔
        valid_intervals = intervals[1:]
        avg_int = np.mean(valid_intervals)
        max_int = np.max(valid_intervals)
        fps = 1000.0 / avg_int
        print(f"2. 视觉更新周期 (Data Update Interval):")
        print(f"   - 平均间隔: {avg_int:.2f} ms")
        print(f"   - 最大间隔: {max_int:.2f} ms")
        print(f"   - 实际帧率: {fps:.2f} FPS")
        print(f"   (注：此项是设置 dt_sim 的核心依据)")

        # 5. 给出建议值
        suggested_dt = max(avg_int * 1.2, max_int) / 1000.0
        print(f"3. 建议设置:")
        print(f"   >>> dt_sim = {suggested_dt:.3f} s <<<")
    else:
        print("未能获取足够的数据更新间隔，请检查靶标是否被识别")

    # 6. 清理
    s_cam_process.terminate()
    s_cam_process.join()
    print("测试结束，资源已释放。")

if __name__ == '__main__':
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        print("测试被用户中断")
