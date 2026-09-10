"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = x < 10.0
    xs = x[mask_small]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~mask_small]
    result[~mask_small] = np.sqrt(np.pi / xl) * (1.0 - 10.0 / (7.0 * xl)) * np.exp(-xl / 4.0)
    return result


def _phi_scalar(x):
    if x < 10.0:
        return float(np.exp(-0.4527 * (x ** 0.86) + 0.0218))
    return float(np.sqrt(np.pi / x) * (1.0 - 10.0 / (7.0 * x)) * np.exp(-x / 4.0))


def phi_inv(y):
    """phi 函数的数值逆（二分法，phi 随 x 单调递减）"""
    y = np.asarray(y, dtype=np.float64)
    flat = y.ravel()
    out = np.empty_like(flat)
    for idx, target in enumerate(flat):
        lo, hi = 0.0, 10000.0
        for _ in range(80):
            mid = (lo + hi) * 0.5
            if _phi_scalar(mid) > target:
                lo = mid
            else:
                hi = mid
        out[idx] = (lo + hi) * 0.5
    return out.reshape(y.shape)


def logQ_Borjesson(x):
    """Borjesson 近似的 log Q 函数，用于 GA 可靠性排序"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = np.abs(np.asarray(x, dtype=np.float64))
    return -np.log((1.0 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（分块 GA + logQ 可靠性度量）。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    eb_no_linear = rate * (10.0 ** (design_eb_n0_db / 10.0))
    z0 = np.full(N, 4.0 * eb_no_linear, dtype=np.float64)

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 2 ** j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    reliabilities = np.array([
        logQ_Borjesson(0.707 * np.sqrt(max(llr_means[i], 1e-15)))
        for i in range(N)
    ])

    order = np.argsort(reliabilities, kind='mergesort')
    info_indices = np.sort(order[:K])
    frozen_indices = np.sort(order[K:])
    return info_indices, frozen_indices, reliabilities


if __name__ == '__main__':
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print('N=8, K=4, Eb/N0=2.5dB')
    print('info_indices:', info8)
    print('frozen_indices:', frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256, K=128, first 20 info_indices:', info256[:20])
