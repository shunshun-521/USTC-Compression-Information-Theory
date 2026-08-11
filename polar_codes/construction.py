"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

_PHI_INV_CAP = 1e9


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = x <= 10.0
    mask_large = ~mask_small
    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.859) + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    result[x == 0.0] = 1.0
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    target = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(target)
    hi = np.full_like(target, _PHI_INV_CAP)
    for _ in range(100):
        mid = (lo + hi) / 2
        phi_mid = phi(mid)
        lo = np.where(phi_mid > target, mid, lo)
        hi = np.where(phi_mid <= target, mid, hi)

    result = 0.5 * (lo + hi)
    return float(result[0]) if scalar else result


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

    sigma = 1.0 / np.sqrt(2 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / sigma ** 2

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(m)
        m_new = np.empty(2 * len(m), dtype=np.float64)
        m_new[0::2] = np.minimum(phi_inv(1.0 - (1.0 - phi_m) ** 2), _PHI_INV_CAP)
        m_new[1::2] = 2.0 * m
        m = m_new

    llr_means = m
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    frozen_mask_arr = np.ones(N, dtype=bool)
    frozen_mask_arr[info_indices] = False
    frozen_indices = np.where(frozen_mask_arr)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
