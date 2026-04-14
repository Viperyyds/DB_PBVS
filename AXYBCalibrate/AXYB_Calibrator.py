import numpy as np
import sympy
from sympy import Matrix, BlockMatrix, Identity, Symbol
from cvxopt import solvers
from lmi_sdp import LMI, to_cvxopt


def solve_axyb_sdp(rho1, rho2, R1):
    """
    求解 AX=YB 手眼标定问题的 SDP (半正定规划) 方法
    参数:
    rho1, rho2, R1: 从之前的 QR 分解步骤获得的 numpy 数组
    返回:
    HTMx: 4x4 变换矩阵 (X)
    HTMy: 4x4 变换矩阵 (Y)
    error: 标定残差
    """

    # 1. 数据类型转换
    # 统一转为 float numpy 数组，再转 sympy Matrix，避免 numpy scalar 与 sympy 混算报错。
    rho1_arr = np.asarray(rho1, dtype=float)
    rho2_arr = np.asarray(rho2, dtype=float)
    R1_arr = np.asarray(R1, dtype=float)
    rho1_sym = Matrix(rho1_arr.tolist())
    R1_sym = Matrix(R1_arr.tolist())

    # 计算 rho2 的范数平方
    rho2_norm_sqr_wls = float(np.linalg.norm(rho2_arr) ** 2)

    # 2. 定义符号变量 (SymPy)
    n_beta = 24
    # 定义 beta1 到 beta24
    beta_symbs = sympy.Matrix([Symbol(f'beta{i + 1}', real=True) for i in range(n_beta)])
    u = Symbol('u')  # 优化目标变量

    # 3. 构建 LMI (线性矩阵不等式)
    # 构造矩阵 U_rho:
    # [ u - ||rho2||^2     (R1*beta - rho1)^T ]
    # [ (R1*beta - rho1)        I             ]

    I_beta = Identity(n_beta)

    # 注意: 这里混合了 numpy 矩阵和 sympy 符号，lmi_sdp 库通常能处理这种情况
    # 也可以显式将 numpy 数组转为 sympy Matrix: Matrix(R1_ols)

    term_11 = Matrix([u - sympy.Float(rho2_norm_sqr_wls)])
    term_12 = (R1_sym * beta_symbs - rho1_sym).T
    term_21 = (R1_sym * beta_symbs - rho1_sym)
    term_22 = I_beta

    # 构建块矩阵并展开
    U_rho = BlockMatrix([[term_11, term_12],
                         [term_21, term_22]])
    U_rho = U_rho.as_explicit()

    # 4. 设置优化问题
    lmis = [LMI(U_rho)]
    variables = [u] + list(beta_symbs)  # 变量列表: [u, beta1, ..., beta24]
    objf = u  # 目标函数: 最小化 u

    # 5. 调用 CVXOPT 求解
    # 禁止输出求解过程 (可选)
    solvers.options['show_progress'] = False

    # 转换为 cvxopt 格式
    c, Gs, hs = to_cvxopt(objf, lmis, variables)

    # 求解 SDP
    sol = solvers.sdp(c, Gs=Gs, hs=hs)

    # 6. 提取结果
    # sol['x'] 是一个列向量，包含了所有变量的值
    # 索引 0 是 u, 索引 1~24 是 beta1~beta24
    res_vec = np.array(sol['x']).flatten()

    error_val = res_vec[0]  # u
    beta_val = res_vec[1:]  # beta 向量

    # 7. 还原参数 (Rx, Ry, tx, ty)
    # 根据原脚本切片逻辑:
    # Rx: beta 1:10 (即数组索引 0:9)
    # Ry: beta 10:19 (即数组索引 9:18)
    # tx: beta 19:22 (即数组索引 18:21)
    # ty: beta 22:25 (即数组索引 21:24)

    Rx = beta_val[0:9].reshape(3, 3)
    Ry = beta_val[9:18].reshape(3, 3)
    tx = beta_val[18:21]
    ty = beta_val[21:24]

    # 8. 构建齐次变换矩阵
    HTMx = np.eye(4)
    HTMx[:3, :3] = Rx
    HTMx[:3, 3] = tx

    HTMy = np.eye(4)
    HTMy[:3, :3] = Ry
    HTMy[:3, 3] = ty

    return HTMx, HTMy, error_val
