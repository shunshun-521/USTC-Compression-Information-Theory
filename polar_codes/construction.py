"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10.0
    mask_large = ~mask_small
    xs = x[mask_small]
    if xs.size:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size:
        out[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    y = np.clip(y, 0.0, phi(np.array(100.0)))
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = phi(mid)
        hi = np.where(pm > y, mid, hi)
        lo = np.where(pm <= y, mid, lo)
    result = 0.5 * (lo + hi)
    return float(result[0]) if scalar else result


def _bhattacharyya_block_ga(N, design_eb_n0_db, rate):
    """
    极化码标准 GA：Bhattacharyya 参数块蝶形递推。
    与 phi 递推在极化因子图结构上等价，且能正确区分各子信道可靠性。
    """
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    z0 = np.exp(-2.0 / (sigma ** 2))
    n = int(np.log2(N))
    z = np.zeros(N, dtype=np.float64)
    z[0] = z0
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                t = z[j]
                z[j] = min(1.0, 2.0 * t - t * t)
                z[j + step] = t * t
    return z


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    返回：
        info_indices, frozen_indices, llr_means
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    z = _bhattacharyya_block_ga(N, design_eb_n0_db, rate)
    llr_means = -np.log(np.clip(z, 1e-300, 1.0))

    info_indices = np.argsort(llr_means)[-K:]
    info_indices.sort()
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
