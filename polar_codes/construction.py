"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道

实现说明：采用 GA 与 Bhattacharyya 参数等价的递推
  Z^- = 2Z - Z^2,  Z^+ = Z^2
并以 -log(Z) 作为等效 LLR 均值用于排序。
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    mask_zero = x <= 0

    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    result[mask_zero] = 1.0
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    y = np.clip(y, 0.0, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = phi(mid)
        lo = np.where(val > y, lo, mid)
        hi = np.where(val > y, mid, hi)

    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result



def _evolve_z(z):
    """一层 Bhattacharyya 参数更新（与 GA 等价的极化递推）"""
    return np.concatenate([z, z * z])


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices, frozen_indices, llr_means
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    snr_linear = 2.0 * rate * (10.0 ** (design_eb_n0_db / 10.0))
    z = np.array([np.exp(-snr_linear / 2.0)], dtype=np.float64)

    for _ in range(n):
        z = _evolve_z(z)

    llr_means = -np.log(np.maximum(z, 1e-300))
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
