"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    mask_zero = x <= 0
    out[mask_small] = np.exp(-0.4527 * np.power(x[mask_small], 0.86) + 0.0218)
    xl = x[mask_large]
    out[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    out[mask_zero] = 1.0
    return out


def phi_inv(y):
    """phi 函数的数值逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    y = np.clip(y, 1e-12, 0.999999)
    lo = np.zeros_like(y)
    hi = np.full_like(y, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        pm = phi(mid)
        lo = np.where(pm < y, mid, lo)
        hi = np.where(pm >= y, mid, hi)
    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None, probe_trials=None):
    """
    高斯近似构造极化码；默认通过 SC 单比特探测对齐信息位索引。

    返回 info_indices, frozen_indices, llr_means
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2.0 * rate) * 10 ** (-design_eb_n0_db / 20.0)
    m = np.array([2.0 / (sigma ** 2)], dtype=np.float64)

    for _ in range(n):
        m_new = np.empty(2 * len(m), dtype=np.float64)
        for i in range(len(m)):
            m_new[2 * i] = phi_inv(1.0 - (1.0 - phi(m[i])) ** 2)
            m_new[2 * i + 1] = 2.0 * m[i]
        m = m_new

    llr_means = m
    info_indices = np.sort(np.argsort(llr_means)[-K:])

    if probe_trials is None:
        if N <= 64:
            probe_trials = 50
        elif N <= 256:
            probe_trials = 30
        elif N <= 512:
            probe_trials = 15
        else:
            probe_trials = 8

    if probe_trials > 0:
        info_indices = _probe_info_indices(
            N, K, design_eb_n0_db, rate, probe_trials
        )

    all_idx = np.arange(N)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = all_idx[frozen_mask]
    return info_indices, frozen_indices, llr_means


def _probe_info_indices(N, K, design_eb_n0_db, rate, trials):
    """单比特探测，与 SCD 译码器对齐选择信息位。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode
    from encoder import polar_encode

    sigma = eb_n0_to_sigma(design_eb_n0_db, rate)
    rng = np.random.default_rng(0)
    err_rate = np.zeros(N)
    for i in range(N):
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[i] = False
        err = 0
        for _ in range(trials):
            u = np.zeros(N, dtype=int)
            u[i] = int(rng.integers(0, 2))
            y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
            llr = compute_llr(y, sigma)
            u_hat = sc_decode(llr, frozen_bits)
            err += int(u_hat[i] != u[i])
        err_rate[i] = err / trials
    return np.sort(np.argsort(err_rate)[:K])


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
