"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSC 实现）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 时 La+Lb，b=1 时 La-Lb"""
    return ((1 - 2 * u_hat) * La + Lb) if np.isscalar(u_hat) else (
        (1 - 2 * u_hat) * La + Lb
    )


def _active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位起）"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从高位起）"""
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
    """f 运算（min-sum）"""
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    """g 运算：l1=bottom, l2=top"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode(L, layer_n, bit_pos):
        if layer_n == 0:
            if frozen_bits[bit_pos]:
                u_hat[bit_pos] = 0
            else:
                u_hat[bit_pos] = 0 if L[0] >= 0 else 1
            return
        half = 2 ** (layer_n - 1)
        L_left = np.array([_upper_llr(L[i], L[i + half]) for i in range(half)])
        decode(L_left, layer_n - 1, bit_pos)
        u_left = u_hat[bit_pos:bit_pos + half]
        L_right = np.array([
            _lower_llr(L[i + half], L[i], u_left[i]) for i in range(half)
        ])
        decode(L_right, layer_n - 1, bit_pos + half)

    decode(llr, n, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSC SC 译码（高效实现）。
    按比特倒序索引遍历译码树，与标准极化码因子图一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容接口）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f'0{n}b')
        llr_layer_vec.append([layer for layer in range(n) if phi_bin[n - 1 - layer] == '0'])
        bit_layer_vec.append([layer for layer in range(n) if phi_bin[n - 1 - layer] == '1'])
    return list(range(n + 1)), llr_layer_vec, bit_layer_vec
