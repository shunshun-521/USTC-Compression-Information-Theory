"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_exact(La, Lb):
    """精确 log-domain box-plus（验证用）。"""
    return np.asarray(np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb), dtype=np.float64)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits, use_min_sum=True):
    """
    递归 SC 译码（含极化码部分和 beta 回传）。
    frozen_bits: True/1 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)
    f_fn = f_operation if use_min_sum else f_exact

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.uint8)

        half = n // 2
        La = llr_node[:half]
        Lb = llr_node[half:]
        beta_upper = decode_node(f_fn(La, Lb), bit_offset)
        llr_lower = g_operation(La, Lb, beta_upper)
        beta_lower = decode_node(llr_lower, bit_offset + half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 所需层索引。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layer_vec.append(list(range(n - 1, -1, -1)))
            bit_layer_vec.append([])
            continue
        l = 0
        p = phi
        while p % 2 == 1:
            l += 1
            p >>= 1
        llr_layer_vec.append(list(range(l - 1, -1, -1)) if l > 0 else [])
        bit_layer_vec.append(list(range(l)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits, cache=None, use_min_sum=True):
    """非递归 SC（分层 LLR / 部分和更新）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    if cache is None:
        cache = precompute_sc_indices(N)
    lambda_offset, llr_layer_vec, bit_layer_vec = cache
    f_fn = f_operation if use_min_sum else f_exact

    L = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    L[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    def update_llr(layer):
        step = lambda_offset[layer]
        for i in range(0, N, 2 * step):
            La = L[layer + 1, i: i + step]
            Lb = L[layer + 1, i + step: i + 2 * step]
            L[layer, i: i + step] = f_fn(La, Lb)
            u_partial = C[layer, i: i + step]
            L[layer, i + step: i + 2 * step] = g_operation(La, Lb, u_partial)

    def update_bits(layer):
        step = lambda_offset[layer]
        for i in range(0, N, 2 * step):
            C[layer + 1, i: i + step] = (
                C[layer, i: i + step] ^ C[layer, i + step: i + 2 * step]
            )
            C[layer + 1, i + step: i + 2 * step] = C[layer, i + step: i + 2 * step]

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            update_llr(layer)
        u_hat[phi] = 0 if frozen_bits[phi] or L[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            update_bits(layer)

    return u_hat


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """主 SC 译码入口（递归 + min-sum f）。"""
    return sc_decode_recursive(llr_ch, frozen_bits, use_min_sum=True)
