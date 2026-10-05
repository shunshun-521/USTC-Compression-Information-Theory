"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus min-sum）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 域）"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed_int(x, n):
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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_int(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_br, frozen_indices):
    """
    非递归 SC（Permuted SCD，LLR 已做比特倒序）
    frozen_indices: 冻结位在自然顺序下的索引集合
    """
    llr_br = np.asarray(llr_br, dtype=np.float64)
    N = len(llr_br)
    n = int(np.log2(N))
    frozen = set(int(i) for i in frozen_indices)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_br

    for i in range(N):
        l = _bit_reversed_int(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_br, frozen_indices):
    """递归 SC（与 nonrecursive 等价的参考实现，LLR 已比特倒序）"""
    llr_br = np.asarray(llr_br, dtype=np.float64)
    frozen = set(int(i) for i in frozen_indices)
    N = len(llr_br)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(L, offset, depth):
        if depth == 0:
            idx = offset
            if idx in frozen:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if L[0] >= 0 else 1
            return

        half = 1 << (depth - 1)
        L_left = np.empty(half, dtype=np.float64)
        for i in range(half):
            L_left[i] = f_operation(L[i], L[i + half])
        decode_block(L_left, offset, depth - 1)

        L_right = np.empty(half, dtype=np.float64)
        for i in range(half):
            L_right[i] = g_operation(L[i], L[i + half], u_hat[offset + i])
        decode_block(L_right, offset + half, depth - 1)

    decode_block(llr_br, 0, n)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口。
    frozen_bits: 长度 N，1/True 表示冻结位（自然顺序，与 ga_construction 一致）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_indices = np.where(frozen_bits.astype(bool))[0]
    llr_br = llr_ch[br]
    return sc_decode_nonrecursive(llr_br, frozen_indices)
