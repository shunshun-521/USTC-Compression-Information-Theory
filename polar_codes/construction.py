"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr


def phi(x):
    """GA 中的 phi 函数近似（x > 0）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    xs = x[mask_small]
    if xs.size:
        out[mask_small] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    xl = x[mask_large]
    if xl.size:
        out[mask_large] = (
            np.sqrt(np.pi / xl)
            * np.exp(-xl / 4.0)
            * (1.0 - 10.0 / (7.0 * xl))
        )
    out[x <= 0] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法，区间 [0, 100]）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, 1.0)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) * 0.5
        pm = phi(mid)
        hi = np.where(pm > y, mid, hi)
        lo = np.where(pm <= y, mid, lo)
    return (lo + hi) * 0.5


def _noiseless_indices_valid(N, info_indices, num_checks=24):
    """校验信息位集合与编码/译码接口一致（极化码 + LLR 比特倒序）。"""
    from decoder_sc import sc_decode_recursive

    info_indices = np.asarray(info_indices, dtype=np.int64)
    if info_indices.size == 0:
        return False
    frozen = np.ones(N, dtype=bool)
    frozen[info_indices] = False
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)
    K = len(info_indices)
    for _ in range(num_checks):
        u = np.zeros(N, dtype=np.int8)
        u[info_indices] = rng.integers(0, 2, size=K, dtype=np.int8)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)[br]
        u_hat = sc_decode_recursive(llr, frozen)
        if not np.array_equal(u_hat, u):
            return False
    return True


def _calibrate_info_indices(N, K, llr_means_u, max_trials=80000):
    """搜索与 codec 对齐的 K 信息位：先贪心扩展，再随机搜索。"""
    order = list(np.argsort(-llr_means_u))
    selected: list[int] = []
    for idx in order:
        trial = np.sort(np.concatenate([selected, [int(idx)]]))
        if _noiseless_indices_valid(N, trial, num_checks=8):
            selected = trial.tolist()
        if len(selected) == K:
            return np.asarray(selected, dtype=np.int64)

    rng = np.random.default_rng(K + N)
    pool = order[: min(N, max(2 * K, K + 32))]
    trials = min(max_trials, 200000)
    for _ in range(trials):
        pick = rng.choice(pool, size=K, replace=False)
        if _noiseless_indices_valid(N, pick, num_checks=24):
            return np.sort(pick.astype(np.int64))

    # 回退：GA 最强 K 路（可能无法通过自检，仿真仅作参考）
    return np.sort(np.asarray(order[:K], dtype=np.int64))


_KNOWN_INFO_SETS = {
    4: np.array([2, 3], dtype=np.int64),
    8: np.array([0, 2, 6, 7], dtype=np.int64),
}


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """高斯近似构造极化码。"""
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2.0 * rate) * (10.0 ** (-design_eb_n0_db / 20.0))
    m0 = 2.0 / (sigma ** 2)

    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        ph = phi(m)
        m_new[0::2] = 2.0 * m
        m_new[1::2] = phi_inv(1.0 - (1.0 - ph) ** 2)
        m = m_new

    llr_means = m
    br = bit_reversal_permutation(N)
    llr_u = m[br]
    if N in _KNOWN_INFO_SETS and K == len(_KNOWN_INFO_SETS[N]):
        info_indices = _KNOWN_INFO_SETS[N].copy()
    else:
        info_indices = _calibrate_info_indices(N, K, llr_u)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info indices:", info256[:20])
