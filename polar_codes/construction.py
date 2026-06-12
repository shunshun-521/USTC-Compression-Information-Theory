"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0），支持向量化。
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)

    out = np.empty_like(x)
    for i, xi in enumerate(x.flat):
        if xi < 10:
            out.flat[i] = np.exp(-0.4527 * (xi ** 0.86) + 0.0218)
        else:
            out.flat[i] = np.sqrt(np.pi / xi) * (1.0 - 10.0 / (7.0 * xi)) * np.exp(-xi / 4.0)

    return out.item() if scalar else out.reshape(x.shape)


def _ref_phi(x):
    x = float(x)
    if x < 10:
        return float(np.exp(-0.4527 * (x ** 0.86) + 0.0218))
    return float(np.sqrt(3.14159 / x) * (1.0 - 10.0 / (7.0 * x)) * np.exp(-x / 4.0))


def _ref_phi_residual(x, val):
    return _ref_phi(x) - float(val)


def _ref_phi_inv(y):
    val = float(y)
    a, b = 0.0, 10000.0
    c = a
    while (b - a) >= 0.01:
        c = (a + b) / 2.0
        if _ref_phi_residual(c, val) == 0.0:
            break
        if _ref_phi_residual(c, val) * _ref_phi_residual(a, val) < 0:
            b = c
        else:
            a = c
    return c


def _ref_logQ_Borjesson(x):
    a = 0.339
    b = 5.510
    half_log2pi = 0.5 * np.log(2 * np.pi)
    x = float(x)
    if x < 0:
        x = -x
        y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
        return float(np.log(1.0 - np.exp(y)))
    y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2.0) - half_log2pi
    return float(y)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码（因子树布局，与 SC 译码器一致）。
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    snr_linear = 2.0 * rate * (10.0 ** (design_eb_n0_db / 10.0))
    z0 = 4.0 * snr_linear

    z = np.zeros((N, n + 1), dtype=np.float64)
    z[:, 0] = z0

    for stage in range(1, n + 1):
        block = 2 ** stage
        half = block // 2
        for base in range(0, N, block):
            for offset in range(half):
                k = base + offset
                z_top = z[k, stage - 1]
                z_bottom = z[k + half, stage - 1]
                z[k, stage] = _ref_phi_inv(
                    1.0 - (1.0 - _ref_phi(z_top)) * (1.0 - _ref_phi(z_bottom))
                )
                z[k + half, stage] = z_top + z_bottom

    llr_means = z[:, n]
    metric = np.array([
        _ref_logQ_Borjesson(0.707 * np.sqrt(llr_means[i])) for i in range(N)
    ])

    frozen_indices = np.sort(np.argsort(metric, kind="mergesort")[K:])
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
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
