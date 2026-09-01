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
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10
    mask_zero = x <= 0

    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xl = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    result[mask_zero] = 1.0
    return result


def _phi_derivative(x):
    x = float(x)
    if 0 < x <= 10:
        return -0.4527 * 0.86 * (x ** (-0.14)) * float(phi(x))
    return float(
        np.sqrt(np.pi) * np.exp(-x / 4.0)
        * ((15.0 / 7.0) * (x ** (-2.5)) - (1.0 / 7.0) * (x ** (-1.5)) - (1.0 / 4.0) * (x ** (-0.5)))
    )


def phi_inv(y):
    """phi 函数的数值逆（Newton 迭代，小值区间用闭式近似）。"""
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 0
    if scalar:
        y = y.reshape(1)

    result = np.empty_like(y, dtype=np.float64)
    for idx, val in enumerate(y.flat):
        val = float(val)
        if 0.0388 <= val <= 1.0221:
            result.flat[idx] = ((0.0218 - np.log(val)) / 0.4527) ** (1.0 / 0.86)
            continue
        x0 = 0.0388 if val < 0.0388 else 10.0
        for _ in range(40):
            fx = float(phi(x0)) - val
            dfx = _phi_derivative(x0)
            if abs(dfx) < 1e-12:
                break
            x1 = x0 - fx / dfx
            if abs(x1 - x0) < 1e-8:
                x0 = x1
                break
            x0 = x1
        result.flat[idx] = x0

    return result[0] if scalar else result


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """
    高斯近似构造极化码。
    返回 info_indices, frozen_indices, llr_means
    """
    if rate is None:
        rate = K / N

    n = int(np.log2(N))
    sigma2 = (1.0 / (2.0 * rate)) * (10 ** (-design_eb_n0_db / 10.0))
    llr0 = 2.0 / sigma2

    llri = [0.0] * N
    llri[0] = llr0
    m = 1
    while m <= N // 2:
        llrcopy = llri.copy()
        for k in range(m):
            llrcopy[2 * k] = float(phi_inv(1.0 - (1.0 - phi(llri[k])) ** 2))
            llrcopy[2 * k + 1] = llri[k] * 2.0
        llri = llrcopy
        m *= 2

    llr_means = np.array(llri, dtype=np.float64)
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == '__main__':
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print('N=8, K=4, Eb/N0=2.5dB')
    print('info_indices:', info)
    print('frozen_indices:', frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print('N=256, K=128, info_indices (first 20):', info256[:20])
