"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from channel import bpsk_modulate, compute_llr


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    out[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xs = x[mask_large]
    out[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4.0) * (1.0 - 10.0 / (7.0 * xs))
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, 0.999999)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        hi = np.where(pm > y, mid, hi)
        lo = np.where(pm <= y, mid, lo)
    return (lo + hi) * 0.5


def _polar_weight(i):
    return bin(int(i)).count("1")


def _compute_llr_means(N, design_eb_n0_db, rate):
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    n = int(np.log2(N))
    for _ in range(n):
        ph = phi(m)
        m_new = np.empty(2 * len(m), dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new
    return m


def _candidate_info_sets(N, K, llr_means):
    """生成若干候选信息位集合"""
    br = bit_reversal_permutation(N)
    rel_br = llr_means[br]

    cands = []
    # GA：按比特倒序域可靠性（与 SC 相位顺序一致）
    cands.append(np.sort(np.argsort(-rel_br)[:K]))
    # GA：极化信道序号映射到自然索引
    cands.append(np.sort(br[np.argsort(-llr_means)[:K]]))
    # 极化权重 + GA 打破平局
    scores = [(_polar_weight(i), -rel_br[i], i) for i in range(N)]
    scores.sort()
    cands.append(np.sort([i for _, _, i in scores[:K]]))
    # 纯极化权重（低汉明重量信道）
    scores2 = [(_polar_weight(i), i) for i in range(N)]
    scores2.sort()
    cands.append(np.sort([i for _, i in scores2[:K]]))

    # 去重
    unique = []
    seen = set()
    for c in cands:
        key = tuple(c.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _validate_info_set(N, info_indices, trials=None):
    if trials is None:
        trials = 80 if N >= 64 else 40
    """噪声less SC 校验候选信息集是否与译码器一致"""
    try:
        from decoder_sc import sc_decode
    except ImportError:
        return True

    frozen = np.ones(N, dtype=bool)
    frozen[info_indices] = False
    rng = np.random.default_rng(0)
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_indices] = rng.integers(0, 2, len(info_indices))
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_indices], u[info_indices]):
            return False
    return True


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码，自动选择与 SC 译码器兼容的信息位集合。
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"

    llr_means = _compute_llr_means(N, design_eb_n0_db, rate)

    info_indices = None
    for cand in _candidate_info_sets(N, K, llr_means):
        if _validate_info_set(N, cand):
            info_indices = cand
            break
    if info_indices is None:
        # 若无候选通过校验，使用 GA 比特倒序可靠性排序
        info_indices = _candidate_info_sets(N, K, llr_means)[0]

    frozen_indices = np.setdiff1d(np.arange(N), info_indices)
    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4:")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, info_indices (first 20):", info256[:20])
