import numpy as np
import cv2
from AXYBCalibrate.lmi_axyb import LMI_AXYB


def orthonormalize_transform(T):
    """Project rotation part of a 4x4 transform to SO(3) using SVD."""
    T_fix = T.copy()
    R = T_fix[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    T_fix[:3, :3] = R_ortho
    return T_fix


def refine_translation_after_rotation_fix(bMe_list, cMo_list, eMo, bMc):
    """Re-estimate translations with fixed rotations using least squares."""
    R_x = eMo[:3, :3]
    R_y = bMc[:3, :3]
    lhs_blocks = []
    rhs_blocks = []

    for bMe, cMo in zip(bMe_list, cMo_list):
        R_a = bMe[:3, :3]
        t_a = bMe[:3, 3]
        t_b = cMo[:3, 3]

        # From bMe * X = Y * cMo:
        # R_a * t_x - t_y = R_y * t_b - t_a
        lhs_blocks.append(np.hstack([R_a, -np.eye(3)]))
        rhs_blocks.append((R_y @ t_b - t_a).reshape(3, 1))

    lhs = np.vstack(lhs_blocks)
    rhs = np.vstack(rhs_blocks)
    sol, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

    t_x = sol[:3, 0]
    t_y = sol[3:, 0]

    eMo_fix = eMo.copy()
    bMc_fix = bMc.copy()
    eMo_fix[:3, 3] = t_x
    bMc_fix[:3, 3] = t_y
    return eMo_fix, bMc_fix

def calculate_error(bMe_list, cMo_list, eMo, bMc):
    err_t, err_r = [], []
    for bMe, cMo_meas in zip(bMe_list, cMo_list):
        cMo_est = np.linalg.inv(bMc) @ bMe @ eMo
        err_t.append(np.linalg.norm(cMo_est[:3, 3] - cMo_meas[:3, 3]) * 1000)
        R_diff = cMo_est[:3, :3] @ cMo_meas[:3, :3].T
        angle, _ = cv2.Rodrigues(R_diff)
        err_r.append(np.linalg.norm(angle) * 180 / np.pi)
    return np.mean(err_t), np.std(err_t), np.mean(err_r), np.std(err_r)


def check_data_consistency(bMe_list, cMo_list):
    print("" + " >> > 开始进行数据一致性校验(相对旋转角应几乎相等) << < ")
    for i in range(1, len(bMe_list)):
        # 机器人末端两次动作之间的相对运动
        R_bMe1 = bMe_list[i - 1][:3, :3]
        R_bMe2 = bMe_list[i][:3, :3]
        rel_R_robot = R_bMe1.T @ R_bMe2
        angle_robot, _ = cv2.Rodrigues(rel_R_robot)
        deg_robot = np.linalg.norm(angle_robot) * 180 / np.pi

        # 相机观测到靶标的两次相对运动
        R_cMo1 = cMo_list[i - 1][:3, :3]
        R_cMo2 = cMo_list[i][:3, :3]
        # 注意眼在手外 (Eye-to-Hand) 的相对旋转计算方式
        rel_R_cam = R_cMo1 @ R_cMo2.T
        angle_cam, _ = cv2.Rodrigues(rel_R_cam)
        deg_cam = np.linalg.norm(angle_cam) * 180 / np.pi

        diff = abs(deg_robot - deg_cam)
        print(f"动作 {i}: 机器人转了 {deg_robot:6.2f}° | 视觉看到转了 {deg_cam:6.2f}° | 差值: {diff:5.2f}°")

    print(">>> 校验结束.如果差值普遍大于 3~5度,数据本身存在问题 <<<")


def test_offline_calibration(npz_file_path):
    data = np.load(npz_file_path)
    bMe_list = data['bMe_list']
    cMo_list = data['cMo_list']
    print(f"成功加载 {len(bMe_list)} 组位姿数据\n")
    check_data_consistency(bMe_list, cMo_list)
    # 加载A矩阵
    A = np.zeros((len(bMe_list),4,4))
    for i in range(len(bMe_list)):
        A[i] = bMe_list[i]
    # 加载B矩阵
    B = np.zeros((len(bMe_list),4,4))
    for i in range(len(bMe_list)):
        B[i] = cMo_list[i]
    X, Y = LMI_AXYB(A, B)
    X = orthonormalize_transform(X)
    Y = orthonormalize_transform(Y)
    X, Y = refine_translation_after_rotation_fix(bMe_list, cMo_list, X, Y)
    print(f'eMo:{X}')
    print(f'bMc:{Y}')
    # 计算误差
    err_t_mean, err_t_std, err_r_mean, err_r_std = calculate_error(bMe_list, cMo_list, X, Y)
    print(f"平均位置误差: {err_t_mean:.2f} ± {err_t_std:.2f} mm")
    print(f"平均旋转误差: {err_r_mean:.2f} ± {err_r_std:.2f} deg")

if __name__ == "__main__":
    test_offline_calibration('calib_raw_data_20260318_191928.npz')
