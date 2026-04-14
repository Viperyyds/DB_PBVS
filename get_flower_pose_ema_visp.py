import numpy as np
from visp.core import HomogeneousMatrix, PoseVector
from queue import Empty

# 全局变量用于存储滤波器的上一次状态
last_filtered_pose_vec_np = None
def get_flower_pose_ema_visp(image_queue, alpha=0.3):
    """
        使用指数平滑算法 (EMA) 获取靶标位姿，并返回 ViSP 的 HomogeneousMatrix。
        :param image_queue: 相机进程的数据队列 (Queue(1))
        :param alpha: 平滑系数 (推荐 0.2 - 0.5)
        :return: visp.core.HomogeneousMatrix 或 None
        """
    global last_filtered_pose_vec_np

    # 1. 获取最新数据并清空队列（Catch-up logic）
    latest_data = None
    while not image_queue.empty():
        try:
            latest_data = image_queue.get_nowait()
        except Empty:
            break

    if latest_data is None:
        # 如果当前没拿到新数据，返回上一次滤波后的位姿矩阵
        if last_filtered_pose_vec_np is not None:
            return HomogeneousMatrix(PoseVector(last_filtered_pose_vec_np.tolist()))
        return None

    # 解析 SCameraProcess 传出的原始数据 (假设 flower_cam_pose 是 4x4 numpy array)
    flower_cam_pose_np, _, _ = latest_data[:3]

    if flower_cam_pose_np is None:
        return None
    
    # 进行单位转换：将位移从毫米转换为米
    flower_cam_pose_np[0:3, 3] = flower_cam_pose_np[0:3, 3] / 1000.0 

    # 2. 转换为 ViSP 格式并提取 6 维位姿向量
    # cMo_raw: 原始观测到的相机到靶标位姿
    try:
        cMo_raw = HomogeneousMatrix(flower_cam_pose_np.tolist())
        # PoseVector 包含 [tx, ty, tz, theta_ux, theta_uy, theta_uz]
        # 其中旋转是以旋转向量(Angle-Axis)表示的，非常适合做线性平滑
        current_pose_vec = PoseVector(cMo_raw)
        current_vec_np = np.array([current_pose_vec[i] for i in range(6)])
    except Exception as e:
        print(f"转换 ViSP 矩阵失败: {e}")
        return None

    # 3. 指数加权移动平均 (EMA) 滤波
    if last_filtered_pose_vec_np is None:
        last_filtered_pose_vec_np = current_vec_np
    else:
        # EMA 公式
        last_filtered_pose_vec_np = alpha * current_vec_np + (1 - alpha) * last_filtered_pose_vec_np

    # 4. 将滤波后的向量重新封装为 ViSP 的 HomogeneousMatrix
    filtered_pose_vec = PoseVector()
    for i in range(6):
        filtered_pose_vec[i] = last_filtered_pose_vec_np[i]

    # 构造并返回最终的 ViSP 矩阵
    return HomogeneousMatrix(filtered_pose_vec)