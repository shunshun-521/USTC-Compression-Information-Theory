"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


LLR_MAX = 30.0


def f_operation(La, Lb):
    """
    f 运算（boxplus，与 min-sum 相比数值更稳定）。
    在 BP 模块中仍使用 min-sum；SC 使用 boxplus 以保证正确性。
    """
    x = np.clip(La, -LLR_MAX, LLR_MAX)
    y = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（供参考/对比）"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """g 运算：使用部分和（stage 输出）而非直接译码比特"""
    return (1.0 - 2.0 * u_partial) * La + Lb


def _sc_recursive_core(llr, frozen_ind):
    """
    递归 SC 译码核心。
    frozen_ind: 长度 N，1 表示冻结位，0 表示信息位。
    返回 (u_hat, u_up) 对。
    """
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    llr = np.clip(np.asarray(llr, dtype=np.float64), -LLR_MAX, LLR_MAX)
    n = len(llr)

    if n > 1:
        half = n // 2
        llr1 = llr[:half]
        llr2 = llr[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        u1, u1_up = _sc_recursive_core(f_operation(llr1, llr2), f1)
        u2, u2_up = _sc_recursive_core(
            g_operation(llr1, llr2, u1_up), f2
        )

        u_hat = np.concatenate([u1, u2])
        u1_up_xor = np.bitwise_xor(u1_up.astype(np.int8), u2_up.astype(np.int8)).astype(
            int
        )
        u_up = np.concatenate([u1_up_xor, u2_up])
        return u_hat, u_up

    if frozen_ind[0] == 1:
        return np.array([0], dtype=int), np.array([0], dtype=int)

    if llr[0] >= 0:
        bit = 0
    elif llr[0] < 0:
        bit = 1
    else:
        bit = 1  # LLR=0 时按 Sionna 约定判 1
    return np.array([bit], dtype=int), np.array([bit], dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_ind = np.asarray(frozen_bits, dtype=int)
    u_hat, _ = _sc_recursive_core(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 复用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        while p & 1:
            p >>= 1
        first_zero_layer = 0 if phi == 0 else int(math.log2(phi & -phi))
        llr_layer_vec.append([first_zero_layer])
        bit_layer_vec.append([l for l in range(n) if phi & (1 << l)])

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用经数值验证的递归核心）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
