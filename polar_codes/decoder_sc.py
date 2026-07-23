"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala Permuted SCD）
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _frozen_mask_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        return set(np.where(frozen_bits.astype(bool))[0])
    return set(np.where(frozen_bits)[0])


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_set = _frozen_mask_to_set(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（Vangala Permuted SCD）====================


def precompute_sc_indices(N):
    """预计算 Permuted SCD 的辅助索引。"""
    n = int(math.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = [list(range(n - _active_llr_level(l, n), n)) for l in decode_order]
    bit_layer_vec = [
        list(range(n, n - _active_bit_level(l, n), -1)) for l in decode_order
    ]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _frozen_mask_to_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - half, s], L[j, s], B[j - half, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 1 << s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def hard_decision(llr):
    return (llr < 0).astype(int)
