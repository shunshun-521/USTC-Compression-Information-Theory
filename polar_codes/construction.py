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
    result = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    xl = x[mask_large]
    result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    result[mask_large] = (
        np.sqrt(np.pi / xl)
        * np.exp(-xl / 4.0)
        * (1.0 - 10.0 / (7.0 * xl))
    )
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def _log_q_borjesson(x):
    """Borjesson Q 函数对数，用于 GA 信道排序。"""
    from scipy.special import erfc

    q = 0.5 * erfc(x / np.sqrt(2.0))
    q = np.clip(q, 1e-300, 1.0)
    return np.log(q)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices: 长度为 K 的数组，信息位在 u 向量中的索引（从 0 开始）
        frozen_indices: 长度为 N-K 的数组，冻结位索引
        llr_means: 长度为 N 的数组，每个极化信道的等效 LLR 均值
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert (1 << n) == N, "N must be a power of 2"

    eb_linear = 10 ** (design_eb_n0_db / 10.0) * rate
    z0 = 4.0 * eb_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for j in range(1, n + 1):
        block = 1 << j
        half = block // 2
        for t in range(0, N, block):
            for s in range(half):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + half, j - 1]
                z[k, j] = phi_inv(1.0 - (1.0 - phi(z_top)) * (1.0 - phi(z_bottom)))
                z[k + half, j] = z_top + z_bottom

    llr_means = z[:, n]
    reliability = np.array(
        [_log_q_borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)]
    )
    info_indices = np.sort(np.argsort(reliability)[-K:])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
