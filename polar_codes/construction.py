"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道
"""
import numpy as np


def phi(x):
    """GA 中的 phi 函数近似（x > 0）。"""
    x = float(x)
    if 0 <= x <= 10:
        return float(np.exp(-0.4527 * (x ** 0.859) + 0.0218))
    if x > 10:
        return float(
            np.sqrt(np.pi / x) * np.exp(-x / 4.0) * (1.0 - 10.0 / (7.0 * x))
        )
    raise ValueError("phi expects x >= 0")


def _phi_derivative(x):
    if 0 <= x <= 10:
        return -0.4527 * 0.86 * (x ** (-0.14)) * phi(x)
    return float(
        np.sqrt(np.pi)
        * np.exp(-x / 4.0)
        * ((15.0 / 7.0) * (x ** (-2.5)) - (1.0 / 7.0) * (x ** (-1.5)) - (0.25) * (x ** (-0.5)))
    )


def phi_inv(y):
    """phi 函数的数值逆（与标准 GA 参考实现一致）。"""
    y = float(y)
    if 0.0388 <= y <= 1.0221:
        return float(((0.0218 - np.log(y)) / 0.4527) ** (1.0 / 0.86))

    x0 = 0.0388
    x1 = x0 - ((phi(x0) - y) / _phi_derivative(x0))
    while abs(x1 - x0) >= 1e-3:
        x0 = x1
        x1 = x1 - ((phi(x1) - y) / _phi_derivative(x1))
        if x1 > 1e2:
            break
    return float(x1)


def ga_construction(N, K, design_eb_n0_db, rate=None):
    """高斯近似构造极化码。"""
    if rate is None:
        rate = K / N
    n = int(np.log2(N))
    assert 2 ** n == N, "N must be a power of 2"

    sigma2 = (1.0 / (2.0 * rate)) * (10 ** (-design_eb_n0_db / 10.0))
    llr = 2.0 / sigma2

    llri = np.zeros(N, dtype=np.float64)
    llri[0] = llr
    m = 1
    while m <= N // 2:
        llrcopy = llri.copy()
        for k in range(m):
            llr_temp = llri[k]
            llrcopy[k * 2] = phi_inv(1.0 - (1.0 - phi(llr_temp)) ** 2)
            llrcopy[2 * k + 1] = llr_temp * 2.0
        llri = llrcopy
        m *= 2

    llr_means = llri
    info_indices = np.sort(np.argsort(llr_means)[-K:])
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False
    frozen_indices = np.where(frozen_mask)[0]

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info8)
    print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
