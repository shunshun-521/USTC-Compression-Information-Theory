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
    result = np.empty_like(x)
    small = x <= _PHI_THRESHOLD
    xs = x[small]
    xl = x[~small]
    result[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    result[~small] = (
        np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    )
    result[x == 0.0] = 1.0
    return result


def phi_inv(y):
    """phi 函数的数值逆（二分法）。"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    target = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(target)
    hi = np.full_like(target, _PHI_INV_CAP)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        greater = phi(mid) > target
        lo = np.where(greater, mid, lo)
        hi = np.where(greater, hi, mid)
    result = 0.5 * (lo + hi)
    return result.item() if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    # 设计点 LLR 均值：m0 = 4 * R * 10^(Eb/N0/10)
    m0 = 4.0 * rate * (10 ** (design_eb_n0_db / 10.0))

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(means)
        upper = np.minimum(phi_inv(1.0 - (1.0 - phi_m) ** 2), _PHI_INV_CAP)
        lower = 2.0 * means
        new_means = np.empty(2 * len(means), dtype=np.float64)
        new_means[0::2] = upper
        new_means[1::2] = lower
        means = new_means

    llr_means = means
    info_indices = np.sort(np.argsort(llr_means, kind="stable")[-K:])
    frozen_indices = np.setdiff1d(np.arange(N), info_indices)

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
