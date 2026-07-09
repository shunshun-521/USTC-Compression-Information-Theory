"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA phi 函数（Trifonov 近似）"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask = x < 10
    if np.any(mask):
        xs = x[mask]
        out[mask] = np.exp(-0.4527 * np.power(xs, 0.86) + 0.0218)
    if np.any(~mask):
        xl = x[~mask]
        out[~mask] = (
            np.sqrt(np.pi / xl)
            * (1.0 - 10.0 / (7.0 * xl))
            * np.exp(-xl / 4.0)
        )
    return out


def _phi_residual(x, val):
    return phi(x) - val


def phi_inv(y):
    """phi 函数逆（二分法）"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = np.array([y], dtype=np.float64)
    out = np.empty_like(y, dtype=np.float64)
    for idx, target in enumerate(y.flat):
        lo, hi = 0.0, 10000.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _phi_residual(mid, target) * _phi_residual(lo, target) < 0:
                hi = mid
            else:
                lo = mid
        out.flat[idx] = (lo + hi) / 2.0
    return out.item() if scalar else out


def _log_q_borjesson(x):
    """Borjesson 近似的 log Q 函数"""
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2.0 * np.pi)
    x = float(x)
    if x < 0:
        x = -x
        y = -np.log((1.0 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
        return float(np.log(1.0 - np.exp(y)))
    y = -np.log((1.0 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
    return float(y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（因子树 GA）。

    参数：
        N: 码长（必须是 2 的幂）
        K: 信息位数
        design_eb_n0_db: 设计信噪比 Eb/N0（dB）
        rate: 码率 R=K/N，若为 None 则自动计算

    返回：
        info_indices: 信息位索引
        frozen_indices: 冻结位索引
        llr_means: 各信道等效 LLR 均值
    """
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    design_snr = (10.0 ** (design_eb_n0_db / 10.0)) * rate
    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = 4.0 * design_snr

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
    reliabilities = np.array(
        [_log_q_borjesson(0.707 * np.sqrt(max(llr_means[i], 0.0))) for i in range(N)],
        dtype=np.float64,
    )
    frozen_indices = np.argsort(reliabilities, kind="mergesort")[K:]
    frozen_indices = np.sort(frozen_indices)
    info_mask = np.ones(N, dtype=bool)
    info_mask[frozen_indices] = False
    info_indices = np.where(info_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, info_indices (first 20):", info256[:20])
