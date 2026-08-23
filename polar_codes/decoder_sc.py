"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位计）。"""
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
    """找到 i 的二进制表示中第一个 0 的位置（从高位计）。"""
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
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << phi for phi in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        br = bit_reversed(phi, n)
        llr_layer_vec.append(
            list(range(n - active_llr_level(br, n), n))
        )
        bit_layer_vec.append(
            list(range(n, n - active_bit_level(br, n), -1))
            if br >= N // 2
            else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, frozen_bits):
    """更新 LLR 树（非递归）。"""
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    """比特回传（非递归）。"""
    if l < len(B) // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2**s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
  # 编码器输出含比特倒序，信道 LLR 需对应映射到译码树叶子
    L[:, 0] = np.array([llr_ch[bit_reversed(i, n)] for i in range(N)])

    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed(phi, n)
        _update_llrs(L, B, l, n, frozen_bits)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n)
        u_hat[l] = B[l, n]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    decode_order = [bit_reversed(i, n) for i in range(N)]
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = np.array([llr[bit_reversed(i, n)] for i in range(N)])

    for phi in range(N):
        l = decode_order[phi]
        _update_llrs(L, B, l, n, frozen_bits)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)
        u_hat[l] = B[l, n]

    return u_hat
