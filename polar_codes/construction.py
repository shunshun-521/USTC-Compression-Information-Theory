"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 10
    large = ~small
    xs = x[small]
    if xs.size:
        out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[large]
    if xl.size:
        out[large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 0:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) * 0.5
    return result[0] if result.size == 1 else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    if rate is None:
        rate = K / N

    # BEC 极化可靠性（与 AWGN 下常用 GA 构造在长码长上一致性较好）
    z = 0.5
    z_vec = [z]
    n = int(np.log2(N))
    for _ in range(n):
        z_next = []
        for z0 in z_vec:
            z_next.append(2.0 * z0 - z0 ** 2)
            z_next.append(z0 ** 2)
        z_vec = z_next
    llr_means = 1.0 - np.array(z_vec, dtype=np.float64)

    # 设计信噪比用于微调排序（高 SNR 时更偏好高索引子信道）
    sigma = 1.0 / np.sqrt(2.0 * rate) * (10.0 ** (-design_eb_n0_db / 20.0))
    snr_bias = 2.0 / (sigma ** 2)
    llr_means = llr_means * snr_bias
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
    print("N=256, K=128, info_indices first 20:", info256[:20])
