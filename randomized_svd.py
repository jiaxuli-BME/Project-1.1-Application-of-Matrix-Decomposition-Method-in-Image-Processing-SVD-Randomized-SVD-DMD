import numpy as np


def rsvd(X, r, q, p):
    # 获取矩阵 X 的列数
    ny = X.shape[1]

    # 生成随机矩阵 P
    P = np.random.randn(ny, r + p)

    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=False)
    U = Q @ UY

    return U, S, VT


def reconstruct_rsvd(X, U, S, VT, r):
    return U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]