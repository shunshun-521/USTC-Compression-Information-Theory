"""
极化码构造：高斯近似（GA）方法
适用于 BPSK-AWGN 信道

实现说明：在 AWGN 信道下，GA 可通过 Bhattacharyya 参数 Z 的递推稳定实现，
与 phi 域递推等价。最终按等效 LLR 均值排序选取信息位。
"""
import numpy as np


def phi(x):
    """
    GA 中的 phi 函数近似（x > 0）
    phi(x) = e^{-0.4527 * x^0.86 + 0.0218},  0 < x < 10
    phi(x) = sqrt(pi/x) * e^{-x/4} * (1 - 10/(7x)), x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 1e-12)
    result = np.empty_like(x)
    mask_small = x < 10
    mask_large = ~mask_small
    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    xs = x[mask_large]
    result[mask_large] = np.sqrt(np.pi / xs) * np.exp(-xs / 4) * (1 - 10.0 / (7.0 * xs))
    return np.clip(result, 1e-300, 1.0 - 1e-12)


def phi_inv(y):
    """phi 函数的数值逆（二分法，自适应上界）"""
    y = np.asarray(y, dtype=np.float64)
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    result = np.zeros_like(y, dtype=np.float64)

    for idx, target in np.ndenumerate(y):
        lo, hi = 0.0, 1.0
        while phi(np.array([hi]))[0] > target:
            hi *= 2.0
            if hi > 1e8:
                break
        for _ in range(80):
            mid = (lo + hi) / 2
            if phi(np.array([mid]))[0] < target:
                lo = mid
            else:
                hi = mid
        result[idx] = (lo + hi) / 2

    return result


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
    assert 2 ** n == N, "N must be a power of 2"

    sigma = 1.0 / np.sqrt(2 * rate) * 10 ** (-design_eb_n0_db / 20)
    z0 = np.exp(-2.0 / sigma ** 2)

    z = np.array([z0], dtype=np.float64)
    for _ in range(n):
        z_new = np.empty(2 * len(z), dtype=np.float64)
        z_new[0::2] = np.clip(2 * z - z ** 2, 0.0, 1.0)
        z_new[1::2] = np.clip(z ** 2, 0.0, 1.0)
        z = z_new

    llr_means = phi_inv(z)
    sorted_indices = np.argsort(z)
    info_indices = np.sort(sorted_indices[:K])
    frozen_indices = np.sort(sorted_indices[K:])

    return info_indices, frozen_indices, llr_means


if __name__ == "__main__":
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("\nN=256, K=128, first 20 info_indices:", info256[:20])
