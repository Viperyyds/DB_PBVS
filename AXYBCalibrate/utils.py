import numpy as np

def fk_delta_mDH(theta_input, delta_x, nominal_params):
    """
    fk_delta_mDH: 计算 mDH 正运动学
    theta_input: 1x6 关节角度 (度) - list 或 numpy array
    delta_x: 24x1 误差修正量 (顺序: theta, d, a, alpha) - list 或 numpy array
    nominal_params: 机器人的理论mdh参数 - numpy array
    """
    nom_alpha  = nominal_params[:, 0]
    nom_a      = nominal_params[:, 1]
    nom_d      = nominal_params[:, 2]
    nom_th_off = nominal_params[:, 3]

    theta_input = np.array(theta_input).flatten()
    delta_x = np.array(delta_x).flatten()

    theta_act = np.deg2rad(theta_input) + nom_th_off + delta_x[0:6]
    d_act = nom_d + delta_x[6:12]
    a_act = nom_a + delta_x[12:18]
    alpha_act = nom_alpha + delta_x[18:24]

    # 3. 计算连杆变换矩阵
    T = np.eye(4)
    for i in range(6):
        ct = np.cos(theta_act[i])
        st = np.sin(theta_act[i])
        ca = np.cos(alpha_act[i])
        sa = np.sin(alpha_act[i])
        a = a_act[i]
        d = d_act[i]

        Ti = np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])
        T = T @ Ti

    return T


def vec2htm(pose):
    """
    将 6维位姿向量 [x, y, z, Rx, Ry, Rz] (角度为度)
    转换为 4x4 齐次变换矩阵
    旋转顺序: Rz * Ry * Rx (固定坐标系/外旋)
    """
    # 确保输入是扁平的 numpy 数组，防止形状错误
    pose = np.array(pose).flatten()

    # 1. 提取位置
    x, y, z = pose[0], pose[1], pose[2]

    # 2. 提取角度并转弧度 (MATLAB 4,5,6 -> Python 3,4,5)
    rx = np.deg2rad(pose[3])
    ry = np.deg2rad(pose[4])
    rz = np.deg2rad(pose[5])

    # 3. 计算旋转矩阵 (为提高效率，先计算 sin/cos)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Rz 矩阵
    Rz_mat = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    # Ry 矩阵
    Ry_mat = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    # Rx 矩阵
    Rx_mat = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    # 4. 组合旋转矩阵 (Rz * Ry * Rx)
    # Python 中矩阵乘法必须用 @ 符号
    R = Rz_mat @ Ry_mat @ Rx_mat

    # 5. 构建齐次变换矩阵 T
    T = np.eye(4)  # 生成 4x4 单位矩阵
    T[:3, :3] = R  # 填充左上角 3x3 旋转部分
    T[:3, 3] = [x, y, z]  # 填充右上角 3x1 平移部分

    return T

def schmidt_orthogonalization(a):
    a = np.array(a, dtype=np.float64)
    m, n = a.shape
    if m < n:
        raise ValueError(
            'Error! The row is smaller than the column and cannot be calculated. Please transpose and re-enter')
    b = np.zeros((m, n))
    b[:, 0] = a[:, 0]
    for i in range(1, n):
        for j in range(i):
            # 计算点积
            numerator = np.dot(a[:, i], b[:, j])
            denominator = np.dot(b[:, j], b[:, j])
            b[:, i] = b[:, i] - (numerator / denominator) * b[:, j]
        b[:, i] = b[:, i] + a[:, i]
    for k in range(n):
        norm_val = np.linalg.norm(b[:, k])
        b[:, k] = b[:, k] / norm_val
    return b

def error_calculate_single_pose(T_true, T_est):
    """
    计算两个4x4齐次变换矩阵之间的位置和旋转误差

    参数:
      T_true: 4x4 numpy array (真实值/参考值)
      T_est:  4x4 numpy array (估计值/计算值)

    返回:
      position_error: 位置误差 (m, 假设输入单位为m)
      rot_angle_error: 旋转角度误差 (度)
    """
    T_true = np.array(T_true)
    T_est = np.array(T_est)

    # 提取旋转矩阵 (前3行前3列)
    R_true = T_true[0:3, 0:3]
    R_est = T_est[0:3, 0:3]

    # --- 计算位置误差 ---
    # 提取平移向量 (前3行第4列) 并计算欧几里得范数
    t_true = T_true[0:3, 3]
    t_est = T_est[0:3, 3]
    position_error = np.linalg.norm(t_est - t_true)

    # --- 计算旋转误差 ---
    # 计算相对旋转矩阵 R_err = R_true * R_est'
    R_err = R_true @ R_est.T

    # 计算迹 (Trace)
    tr = np.trace(R_err)

    # 截断数值 (Clamping)
    tr = np.clip(tr, -1.0, 3.0)

    # 计算旋转角 (弧度)
    # 公式: trace = 1 + 2cos(theta) -> theta = arccos((trace - 1) / 2)
    rot_angle_rad = np.arccos((tr - 1) / 2.0)
    # 转换为度
    rot_angle_error = np.degrees(rot_angle_rad)
    return position_error, rot_angle_error


def calculate_system_error(theta, tpro_pose, X, Y, delta_x, nominal_params):
    num = np.size(theta,0)
    sum_pos_err = 0
    sum_rot_err = 0
    for i in range(num):
        B_meas = vec2htm(tpro_pose[i])
        A_calc = fk_delta_mDH(theta[i],delta_x,nominal_params)
        rhs = A_calc @ X
        B_pred = np.linalg.solve(Y, rhs)
        p_err, r_err = error_calculate_single_pose(B_meas, B_pred)
        sum_pos_err += p_err
        sum_rot_err += r_err
    avg_pos_err = sum_pos_err / num
    avg_rot_err = sum_rot_err / num
    return avg_pos_err, avg_rot_err


def calc_error_vector(theta, X_params, T_true, nominal_params):
    # 获取样本数量
    Samplesize = theta.shape[0]
    FX = np.zeros(6 * Samplesize)
    T_est_all = np.zeros((Samplesize, 4, 4))
    I4 = np.eye(4)

    for k in range(Samplesize):
        # 计算正运动学
        T_est = fk_delta_mDH(theta[k], X_params, nominal_params)
        T_est_all[k] = T_est

        # 计算误差矩阵
        T_err = T_true[k] @ np.linalg.inv(T_est) - I4

        # 当前样本在 FX 中的起始索引
        idx = k * 6

        FX[idx: idx + 3] = T_err[0:3, 3]
        FX[idx + 3] = T_err[2, 1]
        FX[idx + 4] = T_err[0, 2]
        FX[idx + 5] = T_err[1, 0]

    return FX, T_est_all


def calc_numerical_jacobian(theta, X_current, T_true, FX_current, nominal_params):
    # 获取参数维度
    num_params = X_current.shape[0]

    # 获取残差维度
    num_residuals = FX_current.shape[0]

    # 初始化雅可比矩阵 (行: 残差数, 列: 参数数)
    J = np.zeros((num_residuals, num_params))

    delta = 1e-6

    for j in range(num_params):
        X_perturbed = X_current.copy()
        X_perturbed[j] += delta
        FX_perturbed, _ = calc_error_vector(theta, X_perturbed, T_true, nominal_params)
        J[:, j] = (FX_perturbed - FX_current) / delta

    return J

