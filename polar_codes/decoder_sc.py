"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 置换 SC 风格）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def prepare_channel_llr(llr_ch):
    """对信道 LLR 做比特倒序置换，与编码器输出顺序对齐。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    m = np.maximum(a, b)
    return m + np.log1p(np.exp(-np.abs(a - b)))


def _upper_llr_exact(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（min-sum 近似）。"""
    llr = prepare_channel_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
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


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N / 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_min_sum=True):
    """
    非递归 SC 译码主函数（置换 SC，比特按倒序索引译码）。
    """
    llr_ch = prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    if use_min_sum:
                        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    b_top = int(B[j - branch_size, s + 1])
                    if use_min_sum:
                        L[j, s + 1] = (
                            L[j, s] + L[j - branch_size, s]
                            if b_top == 0
                            else L[j, s] - L[j - branch_size, s]
                        )
                    else:
                        L[j, s + 1] = _lower_llr_exact(
                            L[j, s], L[j - branch_size, s], b_top
                        )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode

    for N in [64, 256]:
        K = N // 2
        info, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info] = 0
        ok = True
        for _ in range(50):
            u = np.zeros(N, dtype=int)
            u[info] = np.random.randint(0, 2, K)
            llr = (1.0 - 2.0 * polar_encode(u)) * 100.0
            if not np.array_equal(sc_decode(llr, frozen_bits), u):
                ok = False
                break
        print(f"N={N} SC decode ok: {ok}")
