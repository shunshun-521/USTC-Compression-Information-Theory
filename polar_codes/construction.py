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
    small = (x > 0) & (x < 10)
    large = x >= 10
    out[small] = np.exp(-0.4527 * np.power(x[small], 0.86) + 0.0218)
    xs = x[large]
    out[large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    求 x 使得 phi(x) = y
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm > y, hi, mid)
    return (lo + hi) * 0.5


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    # 设计 SNR 下 GA 初始 LLR 均值：m0 = 2/sigma^2, sigma 由 BPSK-AWGN 与码率决定
    sigma = eb_n0_to_sigma(design_eb_n0_db, rate)
    m0 = 2.0 / (sigma ** 2)
    llr_means = np.array([m0], dtype=np.float64)

    for _ in range(n):
        half = len(llr_means)
        m_new = np.empty(2 * half, dtype=np.float64)
        pm = phi(llr_means)
        m_new[0::2] = phi_inv(1.0 - (1.0 - pm) ** 2)
        m_new[1::2] = 2.0 * llr_means
        llr_means = m_new

    info_indices = np.argsort(-llr_means)[:K]
    info_indices.sort()
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return info_indices, frozen_indices, llr_means


def eb_n0_to_sigma(eb_n0_db, rate):
    """与 channel.eb_n0_to_sigma 相同公式，避免循环导入时 construction 可独立验证。"""
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info:", info256[:20])
