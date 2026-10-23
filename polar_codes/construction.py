"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

_PHI_INV_CAP = 1.0e9


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_lo = x < 10
    mask_hi = ~mask_lo
    xl = x[mask_lo]
    xh = x[mask_hi]
    out[mask_lo] = np.exp(-0.4527 * np.power(xl, 0.859) + 0.0218)
    out[mask_hi] = np.sqrt(np.pi / xh) * np.exp(-xh / 4.0) * (1.0 - 10.0 / (7.0 * xh))
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    target = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(target)
    hi = np.full_like(target, _PHI_INV_CAP)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        greater = phi(mid) > target
        lo = np.where(greater, mid, lo)
        hi = np.where(greater, hi, mid)
    return 0.5 * (lo + hi)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    设计点采用 BPSK-AWGN 等效均值 LLR：m0 = 4 * 10^(Eb/N0/10)（与 sigma=1/sqrt(2R*10^(Eb/10)) 时
    m0=2/sigma^2 在 R=1/2 时一致；一般 R 下以标准 GA 设计为准）。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    # 设计点 Eb/N0（dB）；等效 LLR 均值 m0 = 4 * 10^(Eb/N0/10)（标准 BPSK-AWGN GA）
    m0 = 4.0 * (10.0 ** (design_eb_n0_db / 10.0))

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        upper = phi_inv(1.0 - (1.0 - phi(means)) ** 2)
        lower = 2.0 * means
        means = np.empty(2 * means.size, dtype=np.float64)
        means[0::2] = upper
        means[1::2] = lower

    llr_means = means
    order = np.argsort(-llr_means, kind="stable")
    info_indices = np.sort(order[:K])
    frozen_indices = np.sort(order[K:])

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])
