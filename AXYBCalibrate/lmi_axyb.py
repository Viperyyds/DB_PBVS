import numpy as np
from .AXYB_Calibrator import *
from .utils import schmidt_orthogonalization

def LMI_AXYB(A, B):
    n = A.shape[0]
    H = np.zeros((12 * n, 24))
    omega = np.zeros((12 * n, 1))
    for i in range(n):
        Ra = A[i, 0:3, 0:3]
        Rb = B[i, 0:3, 0:3]
        ta = A[i, 0:3, 3].reshape(3, 1)  # 强制转为列向量 (3,1)
        tb = B[i, 0:3, 3].reshape(3, 1)  # 强制转为列向量 (3,1)

        k_tb = np.kron(np.eye(3), tb.T)
        row1 = np.hstack([
            np.kron(Ra, Rb),  # (9, 9)
            -np.eye(9),  # (9, 9)
            np.zeros((9, 3)),  # (9, 3)
            np.zeros((9, 3))  # (9, 3)
        ])
        row2 = np.hstack([
            np.zeros((3, 9)),  # (3, 9)
            k_tb,  # (3, 9)  <-- 确保此前 k_tb 形状已修正为 (3,9)
            -Ra,  # (3, 3)
            np.eye(3)  # (3, 3)
        ])
        Hi = np.vstack([row1, row2])
        ome_i = np.vstack([np.zeros((9, 1)), ta])
        H[12 * i: 12 * (i + 1), :] = Hi
        omega[12 * i: 12 * (i + 1), :] = ome_i
    Q, R = np.linalg.qr(H, mode='complete')
    Q1 = Q[:, 0:24]
    Q2 = Q[:, 24:]
    R1 = R[0:24, :]

    rho1 = Q1.T @ omega
    rho2 = Q2.T @ omega
    # 进行AX=YB标定
    Hx, Hy, _ = solve_axyb_sdp(rho1, rho2, R1)
    # 进行施密特正交化
    Rx = Hx[0:3, 0:3]
    tx = Hx[0:3, 3]
    Ry = Hy[0:3, 0:3]
    ty = Hy[0:3, 3]
    if np.linalg.norm(Rx @ Rx.T - np.eye(3), ord=2) > 1e-1:
        Rx = schmidt_orthogonalization(Rx)
    if np.linalg.norm(Ry @ Ry.T - np.eye(3), ord=2) > 1e-1:
        Ry = schmidt_orthogonalization(Ry)
    Hx_correct = np.eye(4)
    Hx_correct[0:3, 0:3] = Rx
    Hx_correct[0:3, 3] = tx

    Hy_correct = np.eye(4)
    Hy_correct[0:3, 0:3] = Ry
    Hy_correct[0:3, 3] = ty

    return Hx_correct, Hy_correct
