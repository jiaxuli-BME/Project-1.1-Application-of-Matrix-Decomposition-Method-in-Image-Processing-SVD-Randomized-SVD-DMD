import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import time
from svd import svd, reconstruct_svd
from randomized_svd import rsvd, reconstruct_rsvd
from dmd import dmd, reconstruct_dmd

# Load and preprocess the image
A = imread(r"E:\HuaweiMoveData\Users\HUAWEI\Desktop\IMG_5616.jpg")
X = np.mean(A, axis=2)  # Convert RGB -> grayscale

# Measure time for Deterministic SVD
start_time = time.time()
U, S, VT = svd(X, 40)
det_time = time.time() - start_time

# Parameters for randomized SVD
r = 40  # Target rank
q = 1  # Power
p = 5  # Oversampling parameter

# Measure time for Randomized SVD
start_time = time.time()
rU, rS, rVT = rsvd(X, r, q, p)
rand_time = time.time() - start_time

# Measure time for DMD
start_time = time.time()
Phi, eigenvalues = dmd(X, r)
dmd_time = time.time() - start_time

print(f"Deterministic SVD Time: {det_time:.4f} seconds")
print(f"Randomized SVD Time: {rand_time:.4f} seconds")
print(f"DMD Time: {dmd_time:.4f} seconds")

# Reconstruct images
XSVD = reconstruct_svd(X, U, S, VT, r)
XrSVD = reconstruct_rsvd(X, rU, rS, rVT, r)
Xdmd = reconstruct_dmd(X, Phi, eigenvalues, r)

# Calculate errors
errSVD = np.linalg.norm(X - XSVD, ord='fro') / np.linalg.norm(X, ord='fro')
errrSVD = np.linalg.norm(X - XrSVD, ord='fro') / np.linalg.norm(X, ord='fro')
errdmd = np.linalg.norm(X - Xdmd, ord='fro') / np.linalg.norm(X, ord='fro')

# Plot
fig, axs = plt.subplots(1, 4, figsize=(32, 12))

plt.set_cmap('gray')
axs[0].imshow(X)
axs[0].axis('off')
axs[0].set_title('Original')

axs[1].imshow(XSVD)
axs[1].axis('off')
axs[1].set_title(f'SVD (Error: {errSVD:.4f}, Time: {det_time:.4f}s)')

axs[2].imshow(XrSVD)
axs[2].axis('off')
axs[2].set_title(f'Randomized SVD (Error: {errrSVD:.4f}, Time: {rand_time:.4f}s)')

axs[3].imshow(Xdmd)
axs[3].axis('off')
axs[3].set_title(f'DMD (Error: {errdmd:.4f}, Time: {dmd_time:.4f}s)')

plt.tight_layout()
plt.show()

# Plot error comparison
plt.figure()
ranks = [10, 20, 30, 40, 50, 60, 70, 80]
det_errors = []
rand_errors = []
dmd_errors = []

for rank in ranks:
    det_errors.append(np.linalg.norm(X - reconstruct_svd(X, U, S, VT, rank), ord='fro') / np.linalg.norm(X, ord='fro'))
    rand_errors.append(np.linalg.norm(X - reconstruct_rsvd(X, rU, rS, rVT, rank), ord='fro') / np.linalg.norm(X, ord='fro'))
    Phi, eigenvalues = dmd(X, rank)
    dmd_errors.append(np.linalg.norm(X - reconstruct_dmd(X, Phi, eigenvalues, rank), ord='fro') / np.linalg.norm(X, ord='fro'))

plt.plot(ranks, det_errors, label='Deterministic SVD')
plt.plot(ranks, rand_errors, label='Randomized SVD')
plt.plot(ranks, dmd_errors, label='DMD')
plt.xlabel('Rank')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Comparison of Errors')
plt.show()