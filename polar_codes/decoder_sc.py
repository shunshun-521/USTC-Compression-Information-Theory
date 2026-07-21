"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed


def logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    对数域精确 f 运算（boxplus）。
    同时提供向量化 min-sum 近似接口。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return logdomain_sum(float(La) + float(Lb), 0.0) - logdomain_sum(float(La), float(Lb))
    result = np.empty_like(La, dtype=np.float64)
    flat_a = La.ravel()
    flat_b = Lb.ravel()
    for i in range(flat_a.size):
        a, b = flat_a[i], flat_b[i]
        result.ravel()[i] = logdomain_sum(a + b, 0.0) - logdomain_sum(a, b)
    return result.reshape(La.shape)


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
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
    """预计算非递归 SC 译码所需的辅助向量（保留接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversed(phi, n)
        layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(layers)

        bit_layers = []
        if l >= N // 2:
            bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def rec(llr_node, frz):
        if len(llr_node) == 1:
            return np.array([0 if frz[0] else (0 if llr_node[0] >= 0 else 1)])
        half = len(llr_node) // 2
        ul = rec(f_operation(llr_node[:half], llr_node[half:]), frz[:half])
        ur = rec(g_operation(llr_node[:half], llr_node[half:], ul), frz[half:])
        return np.concatenate([ul, ur])

    return rec(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（基于因子图对数域更新）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    C = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], C[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            C[l, n] = 0
        else:
            C[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    C[j - branch_size, s - 1] = C[j, s] ^ C[j - branch_size, s]
                    C[j, s - 1] = C[j, s]

    u_hat = C[:, n].copy()
    return u_hat
