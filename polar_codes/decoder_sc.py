"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    scalar = np.isscalar(La) and np.isscalar(Lb)
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a[sign_a == 0] = 1.0
    sign_b[sign_b == 0] = 1.0
    out = sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    scalar = np.isscalar(La) and np.isscalar(Lb) and np.isscalar(u_hat)
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    u_hat = np.atleast_1d(np.asarray(u_hat, dtype=np.float64))
    out = (1.0 - 2.0 * u_hat) * La + Lb
    return float(out[0]) if scalar else out


def _active_llr_level(i, n):
    """返回 LLR 更新起始层（与 mcba1n active_llr_level 一致）。"""
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
    """返回比特回传起始层。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，块分割结构）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        llr_right = g_operation(
            llr_node[:half], llr_node[half:], u_hat[bit_offset:bit_offset + half]
        )
        decode_node(llr_right, bit_offset + half)

    br = bit_reversal_permutation(N)
    decode_node(llr[br], 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    offset = 0
    for layer in range(n + 1):
        lambda_offset[layer] = offset
        offset += 1 << layer

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(_bit_reversed(phi, n), n), n)))
        if phi == N - 1:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layer_vec.append(list(range(n - _active_bit_level(_bit_reversed(phi, n), n))))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    与含比特倒序置换的编码器配套：信道 LLR 按传输顺序输入。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int64)
    for i in range(N):
        L[i, 0] = llr_ch[br[i]]

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start_layer = n - _active_llr_level(l, n)

        for s in range(start_layer, n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
