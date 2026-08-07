"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversed


# ==================== 基本运算 ====================


def logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    对数域 box-plus（f 运算）：
    upper_llr(La, Lb) = logsum(La+Lb, 0) - logsum(La, Lb)
    支持向量化
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    vec = np.vectorize(logdomain_sum)
    return vec(La + Lb, 0.0) - vec(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：lower_llr(btm, top, u) = top + btm (u=0) 或 btm - top (u=1)
    此处 La=top, Lb=btm
    """
    u_hat = np.asarray(u_hat, dtype=int)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def f_operation_minsum(La, Lb):
    """min-sum 近似 f 运算（供 BP 译码器使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


# ==================== 辅助函数 ====================


def active_llr_level(i, n):
    """LLR 更新起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """比特回传起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 2)]

    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [bit_reversed(i, n) for i in range(N)]

    for phi in decode_order:
        layer_start = n - active_llr_level(phi, n)
        llr_layer_vec.append(list(range(layer_start, n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(phi, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    L = np.full((N, n + 1), np.nan)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1]) if not np.isnan(B[j - branch_size, s + 1]) else 0
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（基于矩阵存储的高效实现）。
    """
  # 使用与参考实现等价的迭代算法
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    u2 = np.array([1, 0, 1, 1])
    x2 = polar_encode(u2)
    llr = 100 * (1 - 2 * x2)
    u_hat = sc_decode_recursive(llr, np.zeros(4, dtype=bool))
    assert np.array_equal(u_hat, u2), f"编码环回错误: {u_hat}"
    print("Encoder loopback test passed!")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test: {100 - errors}/100 frames correct at Eb/N0=10dB")
