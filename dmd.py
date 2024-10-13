import numpy as np


def dmd(X, r):
    # 获取矩阵 X 的维度
    m, n = X.shape

    # 将 X 分为两个时间步长的矩阵
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # 计算 SVD
    U, S, VT = np.linalg.svd(X1, full_matrices=False)

    # 提取前 r 个奇异值和向量
    U_r = U[:, :r]
    S_r = S[:r]
    VT_r = VT[:r, :]

    # 构建 A_tilde
    A_tilde = U_r.T @ X2 @ VT_r.T @ np.diag(1 / S_r)

    # 计算 DMD 模式和特征值
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    Phi = X2 @ VT_r.T @ np.diag(1 / S_r) @ eigenvectors

    return Phi, eigenvalues


def reconstruct_dmd(X, Phi, eigenvalues, r):
    # 使用 DMD 模式和特征值重构数据
    m, n = X.shape
    omega = np.log(eigenvalues)
    time_dynamics = np.zeros((r, n), dtype=complex)
    for i in range(n):
        time_dynamics[:, i] = np.exp(omega * i)
    X_dmd = Phi[:, :r] @ time_dynamics
    return np.real(X_dmd)  # 返回实部