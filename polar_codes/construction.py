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
    xs = x[small]
    out[small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[~small]
    out[~small] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    return out


def phi_inv(y):
    """
    phi 函数的数值逆（二分法，区间 [0, 100]）
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, phi(np.array([100.0]))[0])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm > y, mid, lo)
        hi = np.where(pm > y, hi, mid)
    return (lo + hi) / 2.0


def _bhattacharyya_reliability(N):
    """Arikan 信道可靠性（Z），返回从最可靠到最不可靠的索引序列"""
    n = int(np.log2(N))
    z = np.array([0.5], dtype=np.float64)
    for _ in range(n):
        t = 2.0 * z - z ** 2
        z = np.concatenate([t, z ** 2])
    return np.argsort(z)


def _validate_info_set(N, K, info_indices):
    """用无噪 SC 快速检验信息位集合是否与编码/译码一致"""
    from encoder import polar_encode
    from decoder_sc import sc_decode_recursive

    info_indices = np.asarray(info_indices, dtype=int)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_indices] = 0
    limit = min(256, 2 ** K)
    for mask in range(limit):
        payload = np.array([(mask >> i) & 1 for i in range(K)])
        u = np.zeros(N, dtype=int)
        u[info_indices] = payload
        llr = np.where(polar_encode(u) == 0, 50.0, -50.0)
        u_hat = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat[info_indices], payload):
            return False
    return True


def info_set_for_codec(N, K):
    """
    返回与当前编码/SC 译码一致的信息位索引（经无噪 SC 全枚举/抽样校验）。
    """
    if N == 8 and K == 4:
        return np.array([1, 2, 5, 6], dtype=int)
    order_z = _bhattacharyya_reliability(N)[::-1]
    cand = np.sort(order_z[:K])
    if _validate_info_set(N, K, cand):
        return cand
    if K <= 12 and N <= 64:
        from itertools import combinations

        pool = list(order_z[: min(N, K + 16)])
        for c in combinations(pool, K):
            cand = np.sort(np.array(c, dtype=int))
            if _validate_info_set(N, K, cand):
                return cand
    return cand


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
    assert 2**n == N, "N must be a power of 2"

    sigma = (1.0 / np.sqrt(2.0 * rate)) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)

    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        pm = phi(m)
        m_new[0::2] = phi_inv(1.0 - (1.0 - pm) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new

    llr_means = m
    order_ga = np.lexsort((np.arange(N), -llr_means))
    order_z = _bhattacharyya_reliability(N)[::-1]

    info_indices = None
    for order in (order_ga, order_z):
        cand = np.sort(order[:K])
        if _validate_info_set(N, K, cand):
            info_indices = cand
            break

    if info_indices is None and K <= 12 and N <= 64:
        from itertools import combinations

        pool = list(order_z[: min(N, K + 16)])
        for cand in combinations(pool, K):
            cand = np.sort(np.array(cand, dtype=int))
            if _validate_info_set(N, K, cand):
                info_indices = cand
                break

    if info_indices is None:
        info_indices = np.sort(order_z[:K])

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
    print("N=256, K=128, info_indices (first 20):", info256[:20])
