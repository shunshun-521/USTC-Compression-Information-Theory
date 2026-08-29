"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from encoder import bit_reversal_permutation


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_small = x < 10.0
    mask_large = ~mask_small
    xs = x[mask_small]
    if xs.size > 0:
        result[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size > 0:
        result[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    return result


def phi_inv(y):
    """
    phi 函数的数值逆（二分法）
    """
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = np.array([y])
    y = np.clip(y, 0.0, 1.0 - 1e-12)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 1e4)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) * 0.5
    return float(result[0]) if scalar else result


def _ga_bad_branch(L):
    """f 分支（W^-）的 GA 更新。"""
    L = np.asarray(L, dtype=np.float64)
    p = phi(L)
    y = 1.0 - (1.0 - p) ** 2
    result = np.empty_like(L)
    small = p < 1e-4
    if np.any(~small):
        result[~small] = phi_inv(y[~small])
    if np.any(small):
        result[small] = L[small]
    return result


def _bec_info_indices(N, K, z0):
    """BEC Bhattacharyya 辅助构造，保证 Arikan 信道索引下可靠度排序正确。"""
    z = float(z0)
    zs = [z]
    for _ in range(int(np.log2(N))):
        nxt = []
        for val in zs:
            nxt.append(2.0 * val - val * val)
            nxt.append(val * val)
        zs = nxt
    z_arr = np.asarray(zs, dtype=np.float64)
    return np.sort(np.argsort(z_arr)[:K])


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
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    # GA 递归：m_new[2i-1]=f 分支, m_new[2i]=g 分支（1-based）
    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        bad = _ga_bad_branch(means)
        new_means = np.empty(2 * len(means), dtype=np.float64)
        new_means[0::2] = bad
        new_means[1::2] = 2.0 * means
        means = new_means

    # 蝶形展开顺序映射到 Arikan 信道索引
    br = bit_reversal_permutation(N)
    llr_means = means[br]

    # 信息位集合：BEC 等效 Bhattacharyya 辅助（z0 = phi(m0)）保证大码长下排序正确
    z0 = float(phi(np.array([m0]))[0])
    info_indices = _bec_info_indices(N, K, z0)
    all_idx = np.arange(N)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = all_idx[frozen_mask]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
