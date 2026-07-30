"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（用于 BP 等场景）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _f_boxplus(La, Lb):
    """SC 译码使用的精确 log-domain f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _lower_llr(l1, l2, b):
    """g 运算的标量形式（与参考实现一致）。"""
    return (l1 + l2) if b == 0 else (l1 - l2)


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


def _permute_channel_llrs(llr_ch):
    """将信道 LLR 重排为译码树所需顺序（与编码端比特倒序对应）。"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _frozen_set_from_array(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(bool))[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _permute_channel_llrs(llr)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = _frozen_set_from_array(frozen_bits)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_block(llr_node, bit_start, length):
        if length == 1:
            idx = bit_start
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_left = _f_boxplus(llr_node[:half], llr_node[half:])
        decode_block(llr_left, bit_start, half)
        u_left = u_hat[bit_start:bit_start + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, bit_start + half, half)

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    reordered_llr = np.zeros(N, dtype=np.float64)
    for i, idx in enumerate(decode_order):
        reordered_llr[i] = llr[idx]

    decode_block(reordered_llr, 0, N)
    result = np.zeros(N, dtype=np.int8)
    for i, idx in enumerate(decode_order):
        result[idx] = u_hat[i]
    return result


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        br_phi = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(br_phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(br_phi, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = _permute_channel_llrs(llr_ch)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_set_from_array(frozen_bits)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for phi in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(phi, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phi, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

        if phi in frozen_set:
            B[phi, n] = 0
        else:
            B[phi, n] = 0 if L[phi, n] >= 0 else 1

        if phi >= N // 2:
            for s in range(n, n - _active_bit_level(phi, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(phi, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int8)
