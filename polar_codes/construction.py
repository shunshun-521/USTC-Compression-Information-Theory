"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

_PHI_THRESHOLD = 10.0
_PHI_INV_CAP = 1.0e9


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x <= _PHI_THRESHOLD
    xs = x[mask_small]
    if xs.size:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.859) + 0.0218)
    xl = x[~mask_small]
    if xl.size:
        out[~mask_small] = (
            np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
        )
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）。"""
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
    design_eb_n0_db 为信息比特 Eb/N0；内部转换为 Es/N0 进行密度演化。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    # BPSK：Es/N0(dB) = Eb/N0(dB) + 10*log10(R)
    design_es_n0_db = design_eb_n0_db + 10.0 * np.log10(rate)
    m0 = 4.0 * (10.0 ** (design_es_n0_db / 10.0))

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        upper = np.minimum(phi_inv(1.0 - (1.0 - phi(m)) ** 2), _PHI_INV_CAP)
        lower = 2.0 * m
        m_new = np.empty(2 * m.size, dtype=np.float64)
        m_new[0::2] = upper
        m_new[1::2] = lower
        m = m_new

    llr_means = m
    order = np.argsort(-llr_means)
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
