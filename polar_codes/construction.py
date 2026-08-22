"""
极化码构造：高斯近似（GA）方法
"""
import numpy as np


def phi(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    mask_low = x < 10
    mask_high = ~mask_low
    result[mask_low] = np.exp(-0.4527 * np.power(x[mask_low], 0.86) + 0.0218)
    xh = x[mask_high]
    result[mask_high] = np.sqrt(np.pi / xh) * np.exp(-xh / 4.0) * (1.0 - 10.0 / (7.0 * xh))
    return result


def phi_inv(y):
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
    return result.item() if scalar else result


def _ga_array_doubling(N, K, design_eb_n0_db, rate):
    """数组倍增法 GA（用于报告核对）"""
    n = int(np.log2(N))
    sigma = 1.0 / np.sqrt(2.0 * rate) * 10.0 ** (-design_eb_n0_db / 20.0)
    m0 = 2.0 / (sigma ** 2)
    m = np.array([m0], dtype=np.float64)
    for _ in range(n):
        phi_m = phi(m)
        m_new = np.empty(2 * len(m), dtype=np.float64)
        m_new[0::2] = phi_inv(1.0 - (1.0 - phi_m) ** 2)
        m_new[1::2] = 2.0 * m
        m = m_new
    llr_means = m
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_indices = np.where(np.isin(np.arange(N), info_indices, invert=True))[0]
    return info_indices, frozen_indices, llr_means


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回数组倍增法结果（与实验报告核对一致）。
    仿真时冻结位通过 polar_wrapper.construct_code 获取。
    """
    if rate is None:
        rate = K / N
    return _ga_array_doubling(N, K, design_eb_n0_db, rate)


def get_simulation_frozen_bits(N, K, design_eb_n0_db):
    """获取与译码器兼容的冻结位（参考 GA 树形构造）"""
    from polar_wrapper import construct_code
    _, info_idx, _ = construct_code(N, K, design_eb_n0_db, "ga")
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    return frozen_bits, info_idx


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
