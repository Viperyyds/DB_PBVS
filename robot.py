import numpy as np
import math
from visp.core import Matrix

class RobotMDH:
    def __init__(self, mdh_params):
        """
        初始化机器人 (使用 Modified DH 参数)
        :param mdh_params: 列表或数组，每一行代表一个关节的MDH参数。
        """
        self.mdh_params = np.array(mdh_params)
        self.dof = len(mdh_params)

    def _get_mdh_matrix(self, alpha, a, d, theta):
        """
        根据 Modified DH 公式计算变换矩阵 T_{i-1, i}
        公式: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
        """
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        return np.array([
            [1, 0, 0, 0],
            [0, ca, -sa, 0],
            [0, sa, ca, 0],
            [0, 0, 0, 1]
        ]) @ np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) @ np.array([
            [ct, -st, 0, 0],
            [st, ct, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) @ np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        # 你的原代码展开式也是对的，为了绝对的严谨，这里用矩阵相乘的形式展开，结果一致
        # ct = math.cos(theta)
        # st = math.sin(theta)
        # ca = math.cos(alpha)
        # sa = math.sin(alpha)
        # return np.array([
        #     [ct, -st, 0, a],
        #     [st * ca, ct * ca, -sa, -sa * d],
        #     [st * sa, ct * sa, ca, ca * d],
        #     [0, 0, 0, 1]
        # ])

    def get_forward_kinematics(self, q):
        """
        计算正运动学 (MDH)
        :param q: 当前真实的关节角度 (弧度)
        :return:
            fMe: 末端相对于基座的变换矩阵 (4x4 numpy array)
            T_abs_list: 每个坐标系相对于基座的变换矩阵列表
        """
        T_abs_list = []
        T_curr = np.eye(4)  # T_{0,0} 基座坐标系

        for i in range(self.dof):
            # MDH 参数提取: alpha_{i-1}, a_{i-1}, d_i, offset_i
            alpha = self.mdh_params[i][0]
            a = self.mdh_params[i][1]
            d = self.mdh_params[i][2]
            offset = self.mdh_params[i][3]

            theta = q[i] + offset

            # 计算 T_{i-1, i}
            T_i = self._get_mdh_matrix(alpha, a, d, theta)

            # 更新全局变换 T_{0, i} = T_{0, i-1} * T_{i-1, i}
            T_curr = np.dot(T_curr, T_i)
            T_abs_list.append(T_curr)

        fMe = T_curr
        return fMe, T_abs_list

    def get_fJe(self, q):
        """
        计算基座标系下的几何雅可比矩阵 (Base Frame Jacobian)
        针对 MDH 参数的特殊处理
        """
        fMe, T_abs_list = self.get_forward_kinematics(q)
        fJe = np.zeros((6, self.dof))

        # 末端位置 (p_e) - 提取自 T_{0, n}
        p_e = fMe[0:3, 3]

        for i in range(self.dof):
            T_0_i = T_abs_list[i]

            # 旋转轴 z_i (Frame i 的 Z轴在基座标系下的表示)
            z_i = T_0_i[0:3, 2]  # 旋转矩阵的第三列

            # 关节位置 p_i (Frame i 的原点在基座标系下的表示)
            p_i = T_0_i[0:3, 3]

            # 线性速度部分 (前3行)
            fJe[0:3, i] = np.cross(z_i, p_e - p_i)

            # 角速度部分 (后3行)
            fJe[3:6, i] = z_i

        return fJe

    def get_eJe(self, q):
        """
        获取末端执行器坐标系下的雅可比矩阵
        :param q: 当前真实的关节角度 (list or np.array)
        :return: 6xN visp.core.Matrix
        """
        # 1. 计算基座标系下的雅可比
        fJe = self.get_fJe(q)

        # 2. 计算基座到末端的变换矩阵 fMe
        fMe, _ = self.get_forward_kinematics(q)

        # 3. 提取旋转矩阵 fRe (Base -> End-Effector)
        fRe = fMe[0:3, 0:3]

        # 4. 构建速度变换矩阵 eVf (将基座系下的速度转换到末端系下)
        eRf = fRe.T

        eVf = np.zeros((6, 6))
        eVf[0:3, 0:3] = eRf
        eVf[3:6, 3:6] = eRf
        eJe_np = np.dot(eVf, fJe)

        # 返回 ViSP 格式的矩阵
        return Matrix(eJe_np.tolist())


# ==========================================
# 独立测试模块
# ==========================================
if __name__ == "__main__":
    # 你的机器人的 MDH 参数
    duban_mdh_params = [
        [0, 0, 0.3085, 0],
        [-np.pi / 2, 0, 0, -np.pi / 2],
        [0, 0.3, 0, np.pi/2],
        [np.pi / 2, 0, 0.6865, 0],
        [-np.pi / 2, 0, 0, 0],
        [np.pi / 2, 0, 0.2649, 0],
    ]

    # 实例化机器人运动学模型
    robot = RobotMDH(duban_mdh_params)

    # 假设从真实机器人读取到的当前关节角度 (6自由度)
    q_current_deg = np.array([22.5, 67.3, -45.5, 42.6, 35.2, 75.5])
    q_current_rad = np.deg2rad(q_current_deg)

    # 计算末端坐标系雅可比
    eJe = robot.get_eJe(q_current_rad)

    print("计算出的末端雅可比矩阵 eJe (ViSP Matrix 格式):")
    print(eJe)
