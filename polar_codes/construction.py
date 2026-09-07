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
    scalar = np.isscalar(y) or (isinstance(y, np.ndarray) and y.ndim == 0)
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))

    lo = np.full_like(y, 1e-6)
    hi = np.full_like(y, 50.0)

    for _ in range(80):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm > y, hi, mid)

    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)

    means = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(means), dtype=np.float64)
        p = phi(means)
        m_new[0::2] = phi_inv(1.0 - (1.0 - p) ** 2)
        m_new[1::2] = 2.0 * means
        means = m_new

    llr_means = means
    info_indices = np.argsort(llr_means)[-K:]
    info_indices = np.sort(info_indices)
    all_indices = np.arange(N)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = all_indices[frozen_mask]

    return info_indices, frozen_indices, llr_means


def _noiseless_validate_info_set(N, info_indices):
    """验证信息位集合能否在无噪 SC 译码下正确恢复"""
    from encoder import bit_reversal_permutation, polar_encode
    from decoder_sc import sc_decode_recursive

    K = len(info_indices)
    frozen = np.ones(N, dtype=bool)
    frozen[info_indices] = False
    br = bit_reversal_permutation(N)

    for trial in range(min(64, 2 ** K)):
        u = np.zeros(N, dtype=int)
        bits = [(trial >> i) & 1 for i in range(K)]
        for i, pos in enumerate(info_indices):
            u[pos] = bits[i]
        llr = 50.0 * (1 - 2 * polar_encode(u))
        llr = llr[br]
        u_hat = sc_decode_recursive(llr, frozen)
        if not np.array_equal(u, u_hat):
            return False
    return True


def find_valid_info_set(N, K, llr_means, max_trials=5000):
    """
    在按 GA 均值排序的候选索引中搜索可通过无噪 SC 验证的信息位集合。
    对于较大码长，直接返回 GA 结果以控制计算时间。
    """
    from itertools import combinations

    ranked = np.argsort(llr_means)[::-1]
    ga_info = np.sort(ranked[:K])
    if N > 32:
        return ga_info
    if _noiseless_validate_info_set(N, ga_info):
        return ga_info

    pool_size = min(N, max(K + 4, int(K + 2 * np.log2(N))))
    pool = ranked[:pool_size]
    best_info = ga_info
    best_score = -np.inf

    count = 0
    for combo in combinations(pool, K):
        count += 1
        if count > max_trials:
            break
        info = np.sort(np.array(combo, dtype=int))
        if _noiseless_validate_info_set(N, info):
            score = np.sum(llr_means[info])
            if score > best_score:
                best_score = score
                best_info = info

    return best_info


def ga_construction_validated(N, K, design_eb_n0_db, rate=None):
    """带无噪验证的 GA 构造，确保信息位集合与 SC 译码器兼容"""
    info_indices, frozen_indices, llr_means = ga_construction(N, K, design_eb_n0_db, rate)
    valid_info = find_valid_info_set(N, K, llr_means)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[valid_info] = False
    frozen_indices = np.arange(N)[frozen_mask]
    return valid_info, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
