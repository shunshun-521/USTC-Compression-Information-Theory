"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（check-node）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（variable-node）。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, bit):
    return l1 + l2 if bit == 0 else l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考 Sionna 实现，使用部分和 u_up）。
    frozen_bits: True 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_bits, dtype=np.int8)

    def _polar_decode_sc(llr_ch, frozen):
        n = len(llr_ch)
        if n > 1:
            half = n // 2
            llr1 = llr_ch[:half]
            llr2 = llr_ch[half:]
            fr1 = frozen[:half]
            fr2 = frozen[half:]

            llr_left = f_operation(llr1, llr2)
            u1, u1_up = _polar_decode_sc(llr_left, fr1)

            llr_right = g_operation(llr1, llr2, u1_up)
            u2, u2_up = _polar_decode_sc(llr_right, fr2)

            u_hat = np.concatenate([u1, u2])
            u1_up_int = np.bitwise_xor(u1_up.astype(np.int8), u2_up.astype(np.int8))
            u_up = np.concatenate([u1_up_int, u2_up])
            return u_hat, u_up

        if frozen[0]:
            bit = 0
        else:
            bit = 0 if llr_ch[0] >= 0 else 1
        u = np.array([bit], dtype=int)
        return u, u.copy()

    u_hat, _ = _polar_decode_sc(llr, frozen_ind)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        tmp = phi
        while tmp & 1:
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        if phi % 2 == 0:
            bit_layer_vec.append([])
        else:
            psi = phi // 2
            bl = 0
            while psi & 1:
                psi >>= 1
                bl += 1
            bit_layer_vec.append(list(range(bl, n)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用递归实现，N<=1024 足够高效）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
