import numpy as np

def svd(X, r):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    return U[:, :r], S[:r], VT[:r, :]

def reconstruct_svd(X, U, S, VT, r):
    return U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]