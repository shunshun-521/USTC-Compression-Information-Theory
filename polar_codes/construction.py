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


# 5G NR 极化码可靠性序列（TS 38.212 Table 5.3.1.2-1，前 256 项）
_Q_SEQ_5G = [
    0, 1, 2, 4, 8, 16, 3, 5, 9, 6, 17, 10, 18, 12, 20, 24, 7, 11, 19, 13, 26, 14, 28, 21,
    27, 22, 25, 15, 29, 23, 30, 31, 32, 33, 34, 36, 40, 48, 35, 37, 41, 49, 38, 42, 50, 44,
    52, 56, 39, 43, 51, 45, 53, 57, 46, 54, 58, 47, 55, 59, 60, 61, 62, 63, 64, 65, 66, 68,
    72, 80, 96, 67, 69, 73, 81, 97, 70, 74, 82, 98, 71, 75, 83, 99, 76, 84, 100, 78, 86, 102,
    88, 104, 77, 85, 101, 79, 87, 103, 89, 105, 90, 106, 92, 108, 112, 91, 107, 93, 109, 113,
    94, 110, 114, 95, 111, 115, 116, 117, 118, 120, 122, 124, 119, 121, 123, 125, 126, 127,
    128, 129, 130, 132, 136, 144, 160, 192, 131, 133, 137, 145, 161, 193, 134, 138, 146, 162,
    194, 135, 139, 147, 163, 195, 140, 148, 164, 196, 141, 149, 165, 197, 142, 150, 166, 198,
    143, 151, 167, 199, 152, 168, 200, 154, 170, 202, 156, 172, 204, 158, 174, 206, 153, 169,
    201, 155, 171, 203, 157, 173, 205, 159, 175, 207, 176, 208, 180, 212, 184, 216, 177, 209,
    181, 213, 185, 217, 178, 210, 182, 214, 186, 218, 179, 211, 183, 215, 187, 219, 188, 220,
    190, 222, 228, 189, 221, 191, 223, 229, 224, 230, 232, 236, 225, 231, 233, 237, 226, 234,
    238, 227, 235, 239, 240, 241, 242, 244, 248, 243, 245, 249, 246, 250, 247, 251, 252, 253,
    254, 255,
]


def nr_polar_construction(N, K):
    """基于 5G NR 可靠性序列的极化码构造"""
    q = np.array([x for x in _Q_SEQ_5G if x < N])
    order = np.argsort(q)
    info_indices = np.sort(order[-K:])
    frozen_indices = np.sort(order[: N - K])
    llr_means = np.zeros(N)
    llr_means[info_indices] = 1.0
    return info_indices, frozen_indices, llr_means


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
    """带无噪验证的 GA 构造；大码长时回退到 5G NR 序列"""
    info_indices, frozen_indices, llr_means = ga_construction(
        N, K, design_eb_n0_db, rate
    )
    if N <= 32:
        valid_info = find_valid_info_set(N, K, llr_means)
    else:
        valid_info, frozen_indices, llr_means = nr_polar_construction(N, K)
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
